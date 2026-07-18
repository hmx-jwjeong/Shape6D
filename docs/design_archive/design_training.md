# Shape6D 학습 파이프라인 상세 설계 (M2-A/B/C/D 실행 스펙, v0.1)

기반 코드 분석: `/Users/jaewoo/Documents/SAM-6D/SAM-6D/Pose_Estimation_Model/config/base.yaml`, `provider/training_dataset.py`, `utils/loss_utils.py` (전문 확인 완료).
학습 HW 전제: RTX PRO 6000 Blackwell 96GB × 1, Ubuntu PC. 모든 수치 추정은 "추정"으로 표기하고 근거를 병기한다.

---

## 0. SAM-6D 학습 설정 요약 (승계/변경의 기준점)

| 항목 | SAM-6D 원본 (코드 확인값) | Shape6D 변경 |
|---|---|---|
| 데이터 | MegaPose-GSO + MegaPose-ShapeNetCore, `train_pbr_web` 샤드 (`key_to_shard.json`으로 키→샤드 매핑, 샤드는 **풀린 상태**의 개별 파일 접근: `.rgb.jpg/.depth.png/.gt.json/.gt_info.json/.mask_visib.json/.camera.json`) | 동일 셋 승계 + LiDAR 희소화·기하 증강을 로더에서 온더플라이 적용 |
| 오브젝트 측 입력 | RGB 템플릿 2뷰(`_get_template`, `rgb_i.png/xyz_i.npy/mask_i.png`, 각 5000pt) | **CAD 포인트+노멀 직접** (S0 캐시 npz) — 템플릿 로딩 경로 제거 |
| 관측 측 | dense depth 역투영 2048pt (부족 시 복원추출), outlier 제거(반경 1.2×), mask dilate 50% | 동일 골격 + **LiDAR 희소화 마스크를 먼저 적용 후** 2048pt 샘플(패딩 포함) |
| 색 증강 | gdrnpp imgaug 시퀀스 p=0.8 | 승계 + 재도장 등가 증강 추가 (§2-②) |
| 기하 증강 | 랜덤 회전(양측), shift ±0.01m + per-point σ=1mm | 승계 + 범프/부착물(§2-③), 거리 노이즈는 LiDAR 모델로 대체 |
| 옵티마이저 | Adam lr 1e-4, betas (0.5, 0.999), wd 0 | AdamW lr 2e-4(bs 스케일 반영), betas (0.9, 0.999), wd 1e-4 — (0.5, 0.999)는 GAN 유래 관성으로 보이며 승계 근거 없음. 파일럿에서 원 세팅과 A/B |
| 스케줄 | WarmupCosineLR, 600k iter, warmup 1000 | 동일 구조, iter 수는 §5의 "샘플 수 등가" 기준으로 환산 |
| 배치 | bs 28, workers 24, 15 epoch(가상 — `iters_per_epoch`로 재분배) | bs 96 (§5), workers 24~32 |
| loss | coarse/fine 각 3층 deep supervision, 양방향 CE, dis_thres 0.15, 총합 clamp 100 | 승계 + **대칭군 최소화** (§3) |

---

## 1. 데이터

### 1.1 구성·규모

| 셋 | 내용 | 규모 (추정) | 근거 |
|---|---|---|---|
| MegaPose-GSO `train_pbr_web` | GSO 스캔 물체 약 1,000종, PBR 클러터 렌더 | 약 1M 이미지 | MegaPose/BOP 배포 관례. 다운로드 후 `key_to_shard.json` 키 수로 실측 확정할 것 |
| MegaPose-ShapeNetCore `train_pbr_web` | ShapeNetCore 다수 카테고리 CAD | 약 1M 이미지 | 상동 |
| 모델 메시 | `Google_Scanned_Objects/`, `shapenetcorev2/` | 수십 GB | S0 캐시 생성용, 학습 루프에서는 미접근 |
| 오염 검증용 재렌더 서브셋 (신규) | GSO 100종 × 재질/텍스처 스왑 재렌더 (BlenderProc) | 5만 이미지, 약 30GB (추정: 0.6MB/장) | §2-② 오프라인 보강분 — 학습 혼입이 아니라 **평가·회귀 전용** |

- **디스크 요구량: 총 약 1.2~1.5TB (추정: 이미지당 rgb jpg ~0.1MB + depth png 16bit ~0.3MB + json 3종 ~0.2MB ≈ 0.6MB × 2M장 + 메시/템플릿).** NVMe SSD 2TB 이상 필수. 다운로드 후 `du -sh`로 확정하고 본 표를 갱신한다.
- min_visib_fract 0.1 / min_px_count_visib 512 필터 승계. 단 Shape6D는 희소화 후 물체 위 포인트가 추가로 줄므로, **희소화 후 유효 포인트 < 64인 샘플은 로더에서 재추첨**(`_rand_another` 경로 재사용).

### 1.2 저장 전략: webdataset tar 유지 vs 풀린 파일

SAM-6D 방식(풀린 개별 파일)은 파일 수가 이미지당 6개 × 2M = **1,200만 파일**로 inode·rsync·백업이 고통스럽지만, 랜덤 액세스(인덱스 셔플)가 자유롭고 코드 이식이 즉시 된다. 권장:

- **1차(M2-A): SAM-6D 방식 그대로 승계** (풀린 파일 + NVMe). 이식 리스크 최소화가 우선. ext4 inode 수 확인(`mkfs` 시 `-N` 여유).
- **2차(로더가 병목으로 실측되면): 자체 팩 포맷으로 재패킹** — 샘플당 1개 `.npz`(rgb jpg bytes, depth png bytes, K, gt R/t, obj_id, mask RLE)로 묶어 파일 수를 1/6로. webdataset 순차 스트리밍은 **채택하지 않음** (셔플 버퍼 RAM 부담 + `num_img_per_epoch` 랜덤 서브샘플 로직과 충돌).

### 1.3 오브젝트 측 캐시 (신규 — 템플릿 경로 대체)

S0 온보딩 파이프라인을 학습 데이터 전 물체에 일괄 실행하여 캐시:

```
Data/Shape6D-ObjCache/{dataset}/{obj_key}.npz
  ├ pts_dense    float32 [2048, 3]   # Poisson disk, 물체 좌표계(m)
  ├ nrm_dense    float32 [2048, 3]
  ├ pts_sparse_idx int32 [196]       # dense에서의 FPS 인덱스 (오프라인 계산)
  ├ diameter     float32 []          # 메트릭 직경(m)
  ├ sym_rots     float32 [G, 3, 3]   # 대칭군 이산화 (§3.3), G ≤ 16, 항등 포함
  └ meta         json str            # 소스 경로, 스케일, 대칭 검출 로그
```

- 전 물체 캐시 크기: (2048×6+196)×4B ≈ 50KB/물체 × 약 5만 물체(ShapeNet 포함 추정) ≈ **2.5GB → 학습 시작 시 RAM에 전량 로드** (로더 워커 간 공유, `torch.Tensor.share_memory_()` 또는 워커 fork 전 로드). 이것이 SAM-6D의 템플릿 디스크 I/O(이미지 3장 × 2뷰/샘플)를 완전히 제거한다.
- ShapeNet 메시 품질(비다양체·비닫힘)로 노멀 추정 실패 물체는 캐시 생성 단계에서 제외 목록 `bad_objects.json`으로 관리 — 로더는 해당 obj_id 샘플을 skip.

### 1.4 로더 병목 대책

샘플당 CPU 비용 분해 (추정, 근거: 각 단계의 통상 단가):

| 단계 | 비용/샘플 (추정) |
|---|---|
| jpg decode (crop 후 리사이즈 224²) | 3~6ms |
| depth png 16bit decode | 3~5ms |
| imgaug 색 증강 (p=0.8) | 15~30ms — **최대 항목** |
| LiDAR 희소화 + 범프 (§2, numpy) | 3~8ms |
| 역투영·샘플링·텐서화 | 3~5ms |
| 합계 | 약 30~50ms/샘플 → 워커당 20~30 샘플/s |

- **요구 처리량**: bs 96 × 목표 0.8 iter/s ≈ 77 샘플/s → **워커 8개면 이론상 충분하나, 지터·GC 감안 24~32 워커** (SAM-6D도 bs28에 24 워커).
- **요구 CPU/RAM (추정)**: 물리 코어 ≥ 16(권장 32스레드 이상), RAM ≥ 64GB (워커 32 × 프로세스 상주 ~1GB + 오브젝트 캐시 2.5GB + 페이지 캐시 여유; 여유 있으면 128GB). D3-a 확인 시 이 표와 대조.
- `prefetch_factor=4`, `persistent_workers=True`, `pin_memory=True` (SAM-6D는 False였음 — GPU 전송이 커진 bs 96에선 True가 유리, 파일럿에서 확인).
- imgaug가 병목으로 실측되면: imgaug → albumentations 등가 시퀀스 포팅(2~3× 빠름, 구현 필요) 또는 색 증강 확률 하향. **선측정 후 교체** 원칙.
- 디스크: 77 샘플/s × 0.6MB ≈ 46MB/s 순차+랜덤 혼합 — NVMe면 무시 가능. HDD 불가.

---

## 2. 증강 구현 스펙

적용 순서(로더 내, `read_data` 개조):

```
depth 로드 → ③ 범프/부착물 (depth 레벨) → ① LiDAR 희소화 마스크 + 노이즈
→ 역투영·유효포인트 샘플링 → ② RGB 색/재도장 증강 → 회전/시프트 증강(승계)
(④ modality dropout은 로더가 아니라 모델 forward에서 — 아래 참조)
```

### ① LiDAR 희소화 시뮬레이션

**목표 분포**: "dense·노이즈"가 아닌 "희소·고정밀". 물체 위 유효 포인트 수 `N_obj ~ Uniform[100, 2000]` (D2-a 미확정 구간의 파라미터화 — config에서 `[n_obj_min, n_obj_max]`로 노출, 실측 후 즉시 좁힌다).

**알고리즘** (crop 좌표계가 아닌 **원 이미지 좌표계**에서 패턴 생성 — 스캔 패턴은 센서 고정이므로 crop과 무관해야 함. 1280×800 실기 해상도와 MegaPose 렌더 해상도의 차이는 패턴 파라미터를 픽셀이 아닌 **각도/이미지비율 단위**로 정의해 흡수):

```python
def make_lidar_mask(H, W, depth, pattern, rng):  # returns bool [H, W]
    if pattern == "scanline":
        # 주사선: 수평(또는 ±θ 기울기) 라인 묶음
        n_lines   = rng.integers(24, 128)          # 수직 시야 내 라인 수
        tilt_deg  = rng.uniform(-5, 5)
        phase     = rng.uniform(0, 1)              # 라인 오프셋
        along_step= rng.uniform(1.5, 6.0)          # 라인 방향 샘플 간격(px, 근거리 기준)
    elif pattern == "rosette":
        # 로제트(리사주/장미곡선): u(t)=cx+A·cos(a·t+φ), v(t)=cy+B·sin(b·t)
        a, b      = rng.choice([(3,2),(5,4),(7,6)])
        n_pts     = rng.integers(20_000, 120_000)  # 프레임당 발사 수
    # 패턴 점 집합 P를 이미지에 래스터화
    mask = rasterize(P, H, W)
    # 거리 의존 밀도: 각 패턴 점을 확률 p_keep = clip((z_ref / z)^2, 0.05, 1)로 유지
    #   근거: 고정 각도 샘플링에서 표면 위 포인트 밀도는 1/z^2 에 비례
    mask &= (rng.random((H, W)) < (z_ref / np.maximum(depth, 1e-3))**2)
    return mask
```

이후 물체 위 포인트 수 강제:

```python
m_obj = mask & obj_mask
if m_obj.sum() > N_obj_target:  # 랜덤 서브샘플로 정확히 N_obj_target 개
    m_obj = subsample(m_obj, N_obj_target, rng)
elif m_obj.sum() < max(64, 0.5 * N_obj_target):
    # 패턴이 우연히 물체를 비켜간 경우: 패턴 재추첨 1회 → 그래도 <64면 샘플 폐기(재추첨)
```

- 패턴 종류 확률: scanline 0.5 / rosette 0.3 / **균일 랜덤 드롭** 0.2 (특정 스캐너 패턴 과적합 방지용 무패턴 대조군).
- **빔 발산 엣지 노이즈**: 깊이 불연속(3×3 창 내 `max(z)−min(z) > 0.03m`, 추정 임계)에 접한 유효 포인트에 대해 — 확률 0.5로 드롭(엣지 리턴 소실), 확률 0.25로 z를 전경/배경 혼합값 `z' = α·z_fg + (1−α)·z_bg, α~U(0.2,0.8)`로 교란(혼합 픽셀). 근거: LiDAR 빔 풋프린트가 엣지에 걸릴 때의 전형 거동.
- **거리 노이즈**: 유효 포인트에 `z += N(0, σ)`, **σ ~ U[2mm, 5mm]** (제안. 근거: 산업용 단파장 LiDAR/ToF의 근거리 1σ 정밀도 통상 수 mm — D2-a에서 기종 스펙 확정 시 교체). SAM-6D의 per-point 1mm 노이즈(`0.001*randn`)는 이것으로 대체.
- 출력: 기존 `choose`(마스크 내 유효 픽셀 인덱스) 산출 로직에서 `mask>0` 대신 `m_obj`를 사용. `n_sample_observed_point=2048` 샘플링은 승계하되, `N_obj < 2048`이면 **복원추출 대신 제로패딩 + 유효길이 `n_valid` 전달** — 복원추출(SAM-6D 방식)은 중복점이 attention 통계를 왜곡하므로, 인코더가 마스크/패딩을 받게 설계한다 (grid-conv 인코더의 유효포인트 마스크 채널과 정합).

### ② 외형 랜덤화 (재도장 등가물)

- gdrnpp 시퀀스(p=0.8) 승계.
- **추가 A — 물체 영역 전면 재색상 (로더, p=0.3)**: `rgb[mask] = blend(rgb[mask], solid_or_gradient_color, α~U(0.5,1.0))` + 저주파 명암 보존(원본의 L 채널 blur를 곱해 음영 유지). "도장은 바뀌어도 음영은 남는다"를 근사. HSV 회전 전면 적용(p=0.2)도 병용.
- **추가 B — 재질 스왑 재렌더 (오프라인)**: BlenderProc으로 GSO 서브셋 재질·텍스처 스왑 재렌더(§1.1). **학습 혼입은 하지 않고 평가 전용**으로 시작 — 로더 레벨 근사(A)로 충분한지를 이 셋에서 검증하고, 부족할 때만 학습 혼입으로 승격(재렌더 비용·파이프라인 복잡도 때문에 2단계로).
- 근거: modality dropout(④)이 이미 "RGB 완전 불신" 극단을 커버하므로, ②의 역할은 "RGB가 있되 온보딩과 다름" 중간 지대 학습이다.

### ③ 국소 기하 범프/부착물 — **depth 레벨 채택**

- **택1 근거**: 메시 레벨(메시 변형 후 재렌더)은 물리적으로 정확하지만 샘플당 렌더가 필요해 로더 처리량(§1.4)을 파괴한다. depth 레벨은 수 ms에 온더플라이 적용 가능하고, 우리가 원하는 학습 신호 — "CAD에 없는 국소 융기 포인트는 대응 없음(배경 라벨)으로 처리되어야 한다" — 는 depth 교란만으로 정확히 만들어진다 (loss의 dis_thres 메커니즘이 자동으로 해당 포인트를 background 라벨화, §3.1).
- 구현 (p=0.3):

```python
def add_bumps(depth, obj_mask, diameter, rng):
    for _ in range(rng.integers(1, 4)):                 # 융기/부착물 1~3개
        cy, cx = random_point_in(obj_mask, rng)
        r_px   = diameter_px * rng.uniform(0.03, 0.10)  # 물체 투영 직경의 3~10%
        h      = rng.uniform(0.002, 0.008)              # 높이 2~8mm (용접 스패터·퇴적 스케일, 추정)
        blob   = h * gaussian_kernel2d(r_px)            # 또는 Perlin 패치 (거친 표면)
        depth[disk(cy,cx,r_px)] -= blob                 # 카메라 쪽으로 융기
    return depth
```

- 부착물(이물) 변형: 확률 0.1로 blob 높이를 1~3cm로 키우고 해당 영역을 obj_mask에 포함 — "물체에 붙은 이물까지 마스크로 들어온" 상황.

### ④ modality dropout p=0.3 — **모델 forward 내 구현**

- **위치 근거**: 로더에서 RGB를 지우면 그 샘플의 RGB 증강·전처리 비용이 낭비되고, distillation(§4)에서 "같은 샘플의 RGB 유/무 페어"를 만들 수 없다. 모델 안에서 끊으면 배치 내 샘플 단위 제어와 추론 시 그래프 불변(TRT 영향 없음 — 학습 전용 분기)이 보장된다.
- 구현: RGB 보조 branch 출력 `f_rgb [B, N, d_rgb]`에 대해 샘플별 베르누이 마스크 `m ~ Bern(1−0.3) [B]`를 뽑아 `f_rgb = m·f_rgb + (1−m)·e_norgb` (`e_norgb`는 학습되는 "RGB 없음" 임베딩, `[d_rgb]`). **추론 시 RGB 결측 프레임에도 동일 임베딩을 사용** — 학습·추론 일관.
- 역방향(depth 밀도 dropout)은 ①의 `N_obj` 하한(100pt)이 이미 담당 — 별도 스위치 불요.

### 증강 온/오프 검증 방법 (파일럿에서 수행)

10% 서브셋(§5.4) 동일 시드·동일 스케줄로 아래 스위치별 학습 → 3축 평가 (클린 BOP 서브셋 AR / 재도장 프로토콜 AR / 희소도 스윕 AR@N_obj∈{100,300,1000,2000}):

| 런 | 스위치 | 기대 신호 (기대와 다르면 증강 구현 버그 의심) |
|---|---|---|
| base | 전부 off (SAM-6D 증강만) | 클린 AR 최고 근접, 희소·재도장에서 최저 |
| +① | LiDAR 희소화 | 희소 AR 대폭↑, 클린 AR 소폭↓ 허용(≤1pt) |
| +①② | +외형 랜덤화 | 재도장 AR↑, 클린 동등 |
| +①②③ | +범프 | 오염(범프) 프로토콜 AR↑, fine_fg_num이 범프 샘플에서 감소하는지 로그로 직접 확인 |
| +①②③④ | +modality dropout | RGB 차단 평가 AR 낙폭 ≤5pt 달성 여부 — M2 게이트 지표 |

학습 중 즉시 확인 가능한 프록시: `coarse_acc/fine_acc/fine_dis`(loss_utils가 이미 산출)를 **증강 태그별로 분리 로깅**(§6) — 예: 범프 적용 샘플의 fine_dis가 비적용 대비 크게 나쁘면 내성 미형성.

---

## 3. Loss

### 3.1 `compute_correspondence_loss` 분석 요약 (코드 확인)

- GT 생성: 관측점을 GT 포즈로 물체 좌표계로 되돌린 `gt_pts = (pts1 − t)·R` (`[B, N1, 3]`)과 모델점 `pts2` (`[B, N2, 3]`)의 전쌍 거리 `dis_mat [B, N1, N2]`에서 **양방향 최근접**을 취해, 거리 ≤ `dis_thres`(0.15, pts와 동일 미터계 단위)면 "최근접 인덱스+1", 아니면 **0 = 배경 빈(bin)** 라벨. 즉 각 attention 행렬은 (N+1) 클래스 분류이고 배경 빈이 첫 슬롯.
- loss: 매칭 transformer의 **각 층 출력 attention마다** (deep supervision, coarse 3층 + fine 3층) 양방향 CrossEntropy 평균 `0.5·(l1+l2)`.
- 부가 지표: 최종 층 기준 `acc`(라벨 일치율), `fg_num`(전경 예측 수), `dis`(예측 대응점의 GT 대비 평균 거리).
- 집계(`Loss.forward`): coarse/fine 전 층 loss 단순 합산, `clamp(max=100)` 후 배치 평균. **coarse:fine 가중 = 층수 비례로 사실상 1:1.**

### 3.2 대칭 처리 현황 — **미처리 확인**

GT는 단일 `(R, t)`뿐이며 `dis_mat`도 단일 포즈 기준. 대칭 물체(원기둥·정다각 부품 다수 — GSO/ShapeNet에 흔함)에서 등가 포즈로 관측된 샘플은 "정답인데 오답 라벨"이 되어 **모순 그라디언트**를 만든다. M2-A3의 "가정 금지" 확인 과제 결과: **미처리 확정 → 대칭군 최소 대응 loss로 교체 필요.**

### 3.3 대칭군 최소 대응 loss 설계

S0 캐시의 `sym_rots [G, 3, 3]` (자동 검출: 이산 회전 대칭 + 연속 축대칭은 12분할 이산화, G ≤ 16 상한, 항등 포함) 사용.

```python
def symmetry_aware_labels(pts1, pts2, gt_R, gt_t, sym_rots, dis_thres):
    """
    pts1: [B, N1, 3] 관측(카메라계, m) / pts2: [B, N2, 3] 모델(물체계)
    sym_rots: [B, G, 3, 3] (물체별 가변 G는 항등으로 패딩 + 유효마스크)
    returns labels1 [B, N1], labels2 [B, N2], g_star [B]
    """
    gt_pts = einsum('bnd,bde->bne', pts1 - gt_t[:, None], gt_R)     # [B, N1, 3]
    # 각 대칭원소 g에 대해: 모델점을 S_g로 돌린 뒤 거리행렬
    pts2_g  = einsum('bgde,bne->bgnd', sym_rots, pts2)              # [B, G, N2, 3]
    dis     = pairwise_distance(gt_pts[:, None].expand(-1, G, -1, -1),
                                pts2_g)                             # [B, G, N1, N2]
    # 샘플 단위 선택: 관측점 최근접거리 합이 최소인 g* 하나를 고른다
    cost    = dis.min(dim=3).values.clamp(max=dis_thres).sum(dim=2) # [B, G]
    g_star  = cost.argmin(dim=1)                                    # [B]
    dis_g   = dis[arange(B), g_star]                                # [B, N1, N2]
    # 이하 SAM-6D 원 로직과 동일 (양방향 최근접 + 배경 빈)
    ...
```

- **설계 선택 — "per-포인트 min"이 아니라 "per-샘플 g\* 선택"**: 포인트별로 다른 g를 허용하면 라벨이 비일관(하나의 강체 대응이 아님)해져 매칭 헤드가 물리적으로 불가능한 대응을 학습한다. 샘플당 단일 g\*로 강체 일관성을 유지 — FoundationPose/GDR-Net 계열의 sym-aware ADD-S loss와 같은 원리를 대응 라벨에 적용한 것.
- 비용: `dis` 텐서 `[B, 16, 2048, 2048]` fp16 ≈ B×134M 원소 — **fine에서 G 전체는 과대**. 실구현은 2단: coarse 해상도(196pt 모델 측)로 g\* 선택(`[B,16,2048,196]` ≈ 저렴) → fine 라벨은 g\* 하나로만 계산. GPU에서 loss 직전 no_grad로 산출.
- 라벨 계산은 매 forward 시 GPU에서 수행(회전 증강이 로더에서 이미 반영된 `rotation_label` 사용 — 증강과 순서 충돌 없음).

### 3.4 coarse/fine 가중

- 기본 승계: 전 층 합산 1:1, `clamp 100` 유지 (초기 발산 방어 — 코드 확인된 원 설계 의도).
- 제안 가중 (config 노출): `w_coarse=1.0, w_fine=1.0`, 층별 균등. 파일럿에서 `w_fine ∈ {1.0, 2.0}` A/B — 근거: Shape6D는 S4 ICP가 최종 폴리싱을 담당하므로 coarse의 가설 품질(top-k 진입률)이 더 중요할 수 있고, 반대로 fine 축소(2048→1024)의 손실 보전에는 w_fine↑가 유리 — 데이터로 결정.
- distill 항 추가 시 총 loss: `L = w_c·L_coarse + w_f·L_fine + λ_feat·L_feat + λ_KL·L_KL` (§4).

---

## 4. Distillation (M2-D)

### 4.1 teacher 구성

| teacher | 역할 | 제약 |
|---|---|---|
| **Shape6D-full (M2-C1: 선정 인코더 + 256d·3블록·fine 2048pt)** | 주 teacher — 학생과 동일 모달리티·동일 입력 분포 | 학생: hidden 192→128, 블록 2, fine 1024pt 순차 축소 |
| SAM-6D 원본 PEM (ViT-B RGB) | 보조 teacher — 특징 공간이 달라 feature-level 불가. **유사도행렬-level만**, RGB 가용·클린 샘플에 한정 | 가중 0.2 고정(추정 초기값). 파일럿에서 기여 없으면 제거 — RGB teacher가 기하 주도 학생에게 A1 위배 편향을 주입할 위험이 있어 "있으면 좋고 없으면 그만" 취급 |

### 4.2 feature-level — 어느 층인가

각 스테이지의 **최종 층 출력만** distill (전 층 distill은 학생 블록 수가 달라(3→2) 층 정렬이 자의적이고, 대응 loss가 이미 층별 deep supervision을 제공):

| 위치 | teacher 텐서 | student 텐서 | loss |
|---|---|---|---|
| 인코더 출력 (장면 측) | `f_m^T [B, 2048, 256]` | `proj(f_m^S) [B, 2048, 256]` (`proj`: 학생 d→256 linear, distill 전용) | cosine distance 평균 |
| coarse transformer 최종층 | `f1_c^T [B, 197, 256]` (bg 토큰 포함) | 상동 projection | cosine |
| fine transformer 최종층 | `f1_f^T [B, 2049, 256]` | 학생은 1025 토큰 — **teacher를 학생의 포인트 인덱스로 서브샘플해 정렬** (fine 1024pt는 2048의 부분집합으로 샘플링하도록 로더 시드 공유) | cosine |

### 4.3 유사도행렬-level KL

매칭의 본질은 특징 자체가 아니라 **할당 분포**이므로 이것이 주 distill 신호:

```
P^T = softmax(sim^T / (τ_d)) , P^S = softmax(sim^S / τ_d),  τ_d = 2·temp = 0.2
L_KL = 0.5·[ KL(P^T_row || P^S_row) + KL(P^T_col || P^S_col) ]   # 행/열(양방향) 평균
```

- coarse: `sim [B, 197, 197]` / fine: `[B, 1025(2049서브샘플), 1025]` — 배경 빈 포함.
- SAM-6D teacher는 이 항만, 좌표 정렬을 위해 동일 포인트 샘플 시드 필수.

### 4.4 가중 스케줄

```
L_total = L_task(GT, §3) + λ_feat(t)·L_feat + λ_KL·L_KL
λ_feat(t) = 1.0 → 0.1  (전체 iter의 50%까지 cosine decay 후 고정)
λ_KL      = 1.0 (상수)
```

근거(추정): 초기엔 feature 모방으로 빠른 워밍업, 후반엔 GT-task와 할당 분포 위주로 수렴시키는 통상 관례. 파일럿에서 `λ_KL ∈ {0.5, 1.0, 2.0}` 1축 스윕. distillation 런의 iter 수는 본학습의 1/3 (샘플 수 등가 ~5.6M, 근거: 초기화가 아닌 축소이므로 풀 스케줄 불요 — 추정, AR 수렴 곡선으로 조기 종료).

---

## 5. 96GB 1장 학습 계획

### 5.1 모델 연산·메모리 개산 (bf16, 후보 인코더별)

공통(매칭부, full 사이즈 기준): coarse 196+1 토큰 × 256d × 3블록(표준 attn) + fine 2048+1 토큰 × 256d × 3블록(linear attn), 양측(장면/모델) 처리.

활성화 메모리 개산 근거: 토큰 수 N, 폭 d, 블록당 저장 텐서를 어텐션+FFN(4d) 합쳐 약 `N·d·16` 원소로 근사(qkv, attn out, FFN 중간 4d×2, residual 등), bf16 2B.

| 구성요소 | 근사 | 활성화/샘플 (추정) |
|---|---|---|
| fine 매칭 3블록 × 양측(2049+2049 토큰) | 4098·256·16·2B·3 | ≈ 100MB |
| coarse 매칭 3블록 × 양측(392 토큰) | 무시 가능 | ≈ 10MB |
| (a) image-grid conv 인코더: 224²×(XYZ+노멀+마스크+RGB feat) 입력, 채널 64→256 다운샘플 4단 | 224²·64 첫 단이 지배, 피처맵 ~20장 등가 | ≈ 130~250MB |
| (b) PointNet++ lite: SA 2단, radius 그룹 K=32 → `[2048, 32, C]` 그룹 텐서 | 2048·32·(64+128)·2B × 중간 수 벌 | ≈ 60~120MB |
| (c) sparse conv: crop 복셀 ~1만 액티브 | 최소 | ≈ 30~60MB |
| RGB 보조 branch (소형 CNN 224²) | (a)의 절반 이하 | ≈ 50MB |
| loss/대칭 라벨(§3.3, no_grad) | 일시 버퍼 | ≈ 30MB |

**샘플당 활성화 합계 추정: (a) ≈ 300~440MB / (b) ≈ 250~350MB / (c) ≈ 220~300MB.**
파라미터+옵티마이저: 전체 ~30M 파라미터 추정(인코더 5~15M + 매칭 ~15M) × AdamW(fp32 m,v + fp32 master + bf16) ≈ 0.6GB — 무시 수준.

| 인코더 | 안전 bs (96GB, 활성화 80GB 예산 기준 추정) | 채택 bs |
|---|---|---|
| (a) grid conv | ~180 | **96** (로더·distill 여유, grad accum 불요) |
| (b) PN++ lite | ~220 | 96 |
| (c) sparse conv | ~260 | 96 |

- bs 96 통일 (비교 공정성). grad accum은 불요하나 config에 `grad_accum: 1` 필드를 두어 OOM 시 bs 48×2로 즉시 전환 가능하게. teacher 동시 상주(distill, ~2GB + teacher forward 활성화 no_grad ~수 GB)도 96GB에서 문제 없음.
- 정밀도: **bf16 autocast + fp32 master weights**, 유사도행렬 softmax/CE와 SVD(가설 생성은 학습 loss 미포함이므로 해당 없음)는 fp32 캐스팅. GradScaler 불요(bf16).
- lr: bs 28→96 (×3.4) 반영 `1e-4 → 2e-4` (sqrt 스케일링 ≈ ×1.85, 보수적 반올림. 추정 — 파일럿에서 1e-4와 A/B).

### 5.2 처리 속도와 소요 시간 (전부 추정 — 가정 명시)

- 가정 1: 샘플당 forward+backward FLOPs ≈ 30~50 GFLOPs (fine attn 양측 + 인코더, backward ×2 포함).
- 가정 2: RTX PRO 6000 Blackwell bf16 실효 처리량을 대형 dense 워크로드 기준 200~400 TFLOPS로 가정(피크 대비 30~50% 실효, 통상 경험칙).
- → 연산 한계 ≈ 4,000~13,000 샘플/s이나, **실제 상한은 로더(§1.4)와 커널 launch 오버헤드**로 훨씬 낮게 잡는다: **보수 가정 0.6~1.0 iter/s @ bs96 (≈ 60~96 샘플/s)** — 이 가정은 파일럿 첫 1시간 실측으로 즉시 교정하고 이후 모든 일정에 반영한다.

| 런 | 샘플 수 | 소요 (0.8 iter/s 가정) |
|---|---|---|
| **파일럿 1런** (10% 서브셋: SAM-6D 등가의 10% = 1.68M 샘플 ≈ 17.5k iter @bs96) | 1.68M | **≈ 6시간/런** → 인코더 3후보 + 증강 5스위치 + loss/lr A/B ≈ 10~12런 ≈ 3~4일 GPU 점유 (M2-B 2주 예산 내 여유) |
| **본학습 M2-C1** (600k iter @bs28 등가 = 16.8M 샘플 = 175k iter @bs96) | 16.8M | **≈ 61시간 ≈ 2.5일**; 로더 병목·평가 중단·재시작 마진 ×2 → **약 5~7일** |
| distill 1단계(§4.4, 1/3 스케줄) × 2~3단(192→128, 블록 3→2, fine 축소) | 각 5.6M | 각 ≈ 1~2일, 총 ≈ 5일 |

주: 02 계획서의 "본학습 3~4주"는 GPU 벽시계가 아니라 재런·분석 포함 캘린더 기간 — 위 추정과 모순 없음. 1런이 일주일 안이므로 **본학습 2회전(버그 발견 후 재학습) 여유가 실제로 존재**한다는 것이 96GB 단일 카드 계획의 핵심 결론.

### 5.3 체크포인트/재개 전략

- 5k iter마다 저장, 최근 5개 롤링 + 25k마다 영구 보존 + `best_AR` 별도. 내용: `model / ema_model(선택) / optimizer / scheduler / iter / rng(파이썬·numpy·torch·cuda) / dataset.img_idx / cfg 스냅샷 / git hash`.
- SAM-6D의 `reset()` 에폭 재추첨 로직 승계 시 `img_idx`를 반드시 체크포인트에 포함(미포함 시 재개 후 데이터 분포 재현 불가 — 원 코드에는 없음, 신규 구현).
- 저장 소요 ~수 초(수백 MB) — 비동기 저장 불요. 디스크 예산: 영구 7개 × ~0.7GB ≈ 5GB/런.
- 전원/드라이버 크래시 대비: 학습 스크립트는 `--resume auto`(최신 체크포인트 자동 탐색) 기본값.

### 5.4 파일럿(10% 서브셋) 구성

- 서브셋 정의: `key_to_shard.json` 키의 해시 하위 10% (물체 편중 방지를 위해 이미지 단위 해시 — 물체는 전 종 노출). 고정 파일 `train/subsets/pilot10.json`으로 커밋.
- 파일럿 판정 지표(M2-B 게이트): 서브셋 학습 후 BOP 미니 하네스(§6) AR + 희소도 스윕 + 처리 ms — 02 문서 E1 탈락 조건 그대로.

---

## 6. 평가 하네스 연동

### 6.1 학습 중 주기 평가

| 주기 | 내용 | 구현 |
|---|---|---|
| 500 iter | val 배치(고정 512샘플, 학습 미사용 홀드아웃 샤드) — loss/acc/fine_dis, **증강 태그별 분해**(클린/희소N구간별/범프/재색상/RGB-drop) | 학습 프로세스 내 인라인 |
| 10k iter | **BOP 미니**: YCB-V·T-LESS·ITODD 각 150이미지 고정 서브셋, **GT 마스크 입력으로 S3 단독 평가**(S1·S2 성능과 절연) → bop_toolkit AR(VSD/MSSD/MSPD) | 별도 프로세스(같은 GPU, 학습 일시정지 또는 유휴 시간대 실행 — 96GB라 동시 상주 가능, 동시 실행 우선) |
| 25k iter | **오염 프로토콜 회귀**(M0-3 산출물): 동일 미니 서브셋의 재도장/녹·분진/범프 합성판 AR + **RGB 차단(회색 입력) AR** + 희소도 스윕 N_obj∈{100,300,1000,2000} AR | 상동 |

- 게이트 자동화: `AR_repaint − AR_clean ≥ −3pt`, `AR_gray − AR_clean ≥ −5pt` 미달 시 W&B alert (M2 게이트 지표의 조기 감시).

### 6.2 로깅 항목 (W&B 주, tensorboard 병행 미러)

- loss: `coarse_loss{0,1,2}`, `fine_loss{0,1,2}`, `L_feat`, `L_KL`, total (전부 loss_utils 키 승계).
- 품질 프록시: `coarse_acc`, `fine_acc`, `fine_fg_num`, `fine_dis` + 증강 태그별 분해판.
- 대칭: `g_star` 분포(항등 비율 — 비항등 선택률이 0이면 대칭 loss 미작동 신호).
- 시스템: iter/s, 샘플/s, 로더 대기율(fetch wait / step time), VRAM peak, grad norm, lr.
- 증강 실측 분포: 배치 내 N_obj 히스토그램, 패턴 종류 비율, 범프/재색상 적용률 (설정값과 실측의 괴리 감시).
- 평가: 셋별·시나리오별 AR 시계열, best 체크포인트 포인터.
- 아티팩트: cfg 스냅샷, `pilot10.json`, `bad_objects.json`, 체크포인트(영구본만).

---

## 7. 학습 config YAML 초안 (`train/config/shape6d_pem_base.yaml`)

```yaml
project: Shape6D-PEM
run_name: null            # 미지정 시 자동 생성
seed: 1
output_dir: runs/${run_name}

optimizer:
  type: AdamW
  lr: 2.0e-4              # bs96 스케일 반영 (파일럿에서 1e-4 A/B)
  betas: [0.9, 0.999]
  eps: 1.0e-6
  weight_decay: 1.0e-4

lr_scheduler:
  type: WarmupCosineLR
  max_iters: 175000       # = SAM-6D 600k@bs28 샘플 수 등가 @bs96
  warmup_iters: 1000
  warmup_factor: 0.001

precision:
  amp_dtype: bf16
  fp32_ops: [softmax_matching, loss]

model:
  coarse_npoint: 196
  fine_npoint: 2048        # full 학습. 압축 단계에서 1024
  encoder:
    type: grid_conv        # grid_conv | pointnet2_lite | sparse_conv
    in_channels: 7         # XYZ(3)+normal(3)+valid_mask(1); RGB feat는 별도 branch
    widths: [64, 128, 256, 256]
    out_dim: 256
    domain_flag_embed: true    # 장면/CAD 공유 인코더 + 도메인 플래그
  rgb_branch:
    type: small_cnn        # none | small_cnn | dino_s_distilled
    out_dim: 64
    modality_dropout_p: 0.3
  geo_embedding:           # SAM-6D 승계
    sigma_d: 0.2
    sigma_a: 15
    angle_k: 3
    reduction_a: max
    hidden_dim: 256
  coarse_point_matching:   # SAM-6D 승계 + 가설 축소
    nblock: 3
    hidden_dim: 256
    out_dim: 256
    temp: 0.1
    sim_type: cosine
    normalize_feat: true
    loss_dis_thres: 0.15
    nproposal1: 2000       # 6000 -> 2000
    nproposal2: 100        # 300  -> 100
    topk_hypotheses: 3
  fine_point_matching:
    nblock: 3
    hidden_dim: 256
    out_dim: 256
    pe_radius1: 0.1
    pe_radius2: 0.2
    focusing_factor: 3
    temp: 0.1
    sim_type: cosine
    normalize_feat: true
    loss_dis_thres: 0.15

loss:
  w_coarse: 1.0
  w_fine: 1.0
  clamp_max: 100.0
  symmetry:
    enable: true
    max_group: 16          # 연속축 12분할 + 항등, 상한
    select_level: coarse   # g* 선택은 coarse 해상도에서 (§3.3)

distill:                   # M2-D에서 enable
  enable: false
  teacher_ckpt: null
  sam6d_teacher_ckpt: null # 보조 teacher (유사도행렬 KL only)
  sam6d_weight: 0.2
  feat_layers: [encoder_out, coarse_last, fine_last]
  feat_loss: cosine
  lambda_feat: {init: 1.0, final: 0.1, decay_until_frac: 0.5}
  lambda_kl: 1.0
  kl_temp: 0.2

train_dataset:
  data_dir: ../Data/MegaPose-Training-Data
  obj_cache_dir: ../Data/Shape6D-ObjCache
  subset_file: null        # 파일럿: train/subsets/pilot10.json
  img_size: 224
  n_sample_observed_point: 2048   # 패딩 방식 (복원추출 아님, n_valid 전달)
  n_sample_model_point: 2048
  min_visib_fract: 0.1
  min_px_count_visib: 512
  min_valid_obj_points: 64  # 희소화 후 하한, 미달 샘플 재추첨
  shift_range: 0.01
  rgb_mask_flag: true
  dilate_mask: true
  augment:
    color_gdrnpp_p: 0.8
    repaint:
      p: 0.3
      alpha_range: [0.5, 1.0]
      keep_shading: true
      hsv_rotate_p: 0.2
    lidar_sparsify:
      enable: true
      n_obj_range: [100, 2000]     # D2-a 확정 시 갱신
      pattern_probs: {scanline: 0.5, rosette: 0.3, random_drop: 0.2}
      scanline: {n_lines: [24, 128], tilt_deg: [-5, 5], along_step_px: [1.5, 6.0]}
      rosette: {ab_pairs: [[3,2],[5,4],[7,6]], n_pts: [20000, 120000]}
      dist_density: {z_ref: 1.0, p_keep_min: 0.05}
      edge_noise: {grad_thres_m: 0.03, drop_p: 0.5, mix_p: 0.25, mix_alpha: [0.2, 0.8]}
      range_noise_sigma_m: [0.002, 0.005]
    bumps:
      p: 0.3
      n_range: [1, 3]
      radius_frac: [0.03, 0.10]
      height_m: [0.002, 0.008]
      attachment_p: 0.1
      attachment_height_m: [0.01, 0.03]

train_dataloader:
  bs: 96
  grad_accum: 1
  num_workers: 28
  prefetch_factor: 4
  persistent_workers: true
  pin_memory: true
  shuffle: true
  drop_last: true

checkpoint:
  every_iters: 5000
  keep_last: 5
  permanent_every: 25000
  resume: auto

eval:
  val_batch_every: 500
  val_holdout_shards: 2
  bop_mini_every: 10000
  bop_mini: {datasets: [ycbv, tless, itodd], n_images: 150, use_gt_masks: true}
  contamination_every: 25000
  contamination_variants: [repaint, rust_dust, bumps, gray_rgb]
  sparsity_sweep_n_obj: [100, 300, 1000, 2000]
  alerts: {repaint_drop_pt: 3.0, gray_drop_pt: 5.0}

logging:
  backend: [wandb, tensorboard]
  wandb_project: shape6d-pem
  scalars_every: 50
  log_aug_stats: true
  log_symmetry_gstar: true
```

### 주요 모듈 인터페이스 (train/ 신규 구현부)

```python
class LiDARSparsifier:
    def __init__(self, cfg): ...
    def __call__(self, depth: np.ndarray, obj_mask: np.ndarray, K: np.ndarray,
                 rng: np.random.Generator) -> tuple[np.ndarray, dict]:
        """returns (valid_mask HxW bool, stats {n_obj, pattern, sigma})"""

class BumpAugmentor:
    def __call__(self, depth: np.ndarray, obj_mask: np.ndarray,
                 diameter_px: float, rng) -> tuple[np.ndarray, np.ndarray]:
        """returns (depth', obj_mask')  # 부착물 시 마스크 확장"""

class SymmetryAwareCorrespondenceLoss(nn.Module):
    def forward(self, end_points, atten_list, pts1, pts2, gt_r, gt_t,
                sym_rots, sym_valid, dis_thres, loss_str) -> dict: ...

class DistillLoss(nn.Module):
    def __init__(self, teacher: nn.Module, cfg): ...
    def forward(self, student_feats: dict, student_sims: dict, batch) -> dict: ...

class Shape6DTrainDataset(torch.utils.data.Dataset):
    """SAM-6D provider 포팅판. _get_template 제거, obj_cache(npz, RAM 상주) 사용.
       __getitem__ 반환: pts[2048,3], n_valid[], valid_pad_mask[2048],
       rgb[3,224,224], rgb_choose[2048], grid_xyz[7,224,224](grid_conv용),
       model_pts[2048,3], model_nrm[2048,3], sym_rots[16,3,3], sym_valid[16],
       rotation_label[3,3], translation_label[3], K[3,3], aug_tags[dict]"""
```

### 미확정/구현 필요 표기 정리

- D2-a(물체 위 포인트 수 실측) 확정 전까지 `n_obj_range=[100,2000]` 유지 — 확정 즉시 config 1곳만 수정.
- imgaug→albumentations 포팅: **구현 필요** (병목 실측 시에만).
- 대칭 자동 검출기(S0 `symmetry.py`): **구현 필요** — 학습용은 MegaPose 모델 메타에 대칭 정보가 없으므로 자체 검출 결과를 캐시에 굽는다.
- 모든 소요 시간·VRAM·iter/s 수치는 추정이며, 파일럿 첫 실행 실측으로 본 문서의 §5 표를 갱신하는 것이 M2-B 첫 작업이다.