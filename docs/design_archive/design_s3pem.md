모든 지정 파일(SAM-6D PEM 5개 모듈 + model_utils + config + Shape6D 설계 문서 2종)을 읽고 분석을 완료했다. 아래가 S3 Shape6D-PEM 상세 설계 산출물이다.

---

# Shape6D-PEM (S3) 네트워크 상세 설계 (v0.1)

기반 코드: `/Users/jaewoo/Documents/SAM-6D/SAM-6D/Pose_Estimation_Model/` — `model/pose_estimation_model.py`, `model/feature_extraction.py`, `model/coarse_point_matching.py`, `model/fine_point_matching.py`, `model/transformer.py`, `utils/model_utils.py`, `config/base.yaml`

## 0. SAM-6D PEM 구조 요약과 개조 지점 총괄

SAM-6D PEM의 데이터 흐름 (코드 확인 기준):

```
rgb crop 224² ──ViT-B(MAE)──▶ per-pixel feat (B,256,224,224)
                              └─ rgb_choose gather ─▶ dense_fm (B,2048,256)
depth ─▶ pts (B,2048,3) = dense_pm      템플릿 2뷰 ─▶ dense_po/fo (B,2048,3/256)
     │
     ├─ FPS 196 ─▶ sparse_pm/fm + geo_embedding(RPE용, 197² 포함 bg)
     ├─ CoarsePointMatching(3블록, RPE self + vanilla cross, 256d)
     │    └─ compute_coarse_Rt: 유사도→6000 가설→300 정제→best 1 (R,t)
     └─ FinePointMatching(3블록 SparseToDense, linear attn, 256d, PE=QueryAndGroup)
          └─ compute_fine_Rt: soft assignment → weighted SVD → pred_R/t/score
```

| # | 개조 지점 | 파일:라인 | Shape6D 조치 |
|---|---|---|---|
| 1 | ViT-B RGB 특징 추출 | `feature_extraction.py:122-181` | 기하 주도 인코더로 교체(§2) + RGB 보조 branch(§3) |
| 2 | 오브젝트 측 온라인 특징 (`get_obj_feats`, 템플릿 2뷰) | `feature_extraction.py:170-181` | S0 오프라인 캐시로 전량 이동(§4) |
| 3 | 오브젝트 측 FPS·geo embedding 매 프레임 재계산 | `pose_estimation_model.py:34-37` | S0 캐시 로드로 대체 |
| 4 | `furthest_point_sample`/`gather_operation` (pointnet2 CUDA) | `model_utils.py:53-66`, `transformer.py:654` | voxel 다운샘플(전처리) + `torch.gather` (§1, §8) |
| 5 | fine PE의 `QueryAndGroup` (pointnet2 CUDA) | `fine_point_matching.py:90-125` | 거리행렬 + masked top-k gather로 재구현 (§5.4) |
| 6 | hidden 256 / nblock 3 | `base.yaml` | 192(→128) / 2, RPE·geo embedding 차원 연동 (§5.1) |
| 7 | `compute_coarse_Rt` best-1 출력 | `model_utils.py:187-246` | 2000/100 축소 + pose-NMS top-k=3 (§5.3) |
| 8 | `torch.svd`, `torch.searchsorted` | `model_utils.py:219,344` | TRT 비호환 — §8에서 대체 |

---

## 1. 입력 규약

### 1.1 장면 측 입력

전제: 마스크 crop 내 **LiDAR 유효 포인트 P_m = 수백~수천 (가변, D2-a 미확정)**. RGB 1280×800에서 마스크 bbox를 정방 crop → 224×224 리사이즈 (SAM-6D `data_utils.py` 크롭 규약 승계).

| 텐서 | shape | dtype | 설명 |
|---|---|---|---|
| `rgb` | (B, 3, 224, 224) | fp32 | 정규화 RGB crop |
| `geo_maps` | (B, 8, 224, 224) | fp32 | XYZ(3, 마스크 유효점 centroid 기준 중심화 + radius 정규화) + 노멀 추정(3) + 유효마스크(1) + 정규화 깊이(1). LiDAR 미존재 픽셀은 0 |
| `pts` (=dense_pm) | (B, 1024, 3) | fp32 | LiDAR 유효 3D 포인트, fine용 N_f=1024 고정 |
| `pixel_uv` | (B, 1024, 2) | fp32 | 각 포인트의 crop 내 정규화 픽셀좌표 [-1,1] — `rgb_choose` 대체 (grid_sample용) |
| `valid_count` | (B,) | int32 | 실제 유효 포인트 수 P_m (로깅·신뢰도용) |
| `fps_idx_m` | (B, 196) | int32 | coarse 196pt 인덱스 — **전처리에서 계산해 엔진 입력으로 주입** |
| `flags` | (B, 2) | fp32 | [rgb_valid_flag, domain_flag(장면=1)] |

**P_m 가변성 처리 — 권장: 고정 N_f=1024 + 중복 샘플(sampling with replacement) + 미세 jitter.**

- 규칙: P_m ≥ 1024 → voxel-grid 다운샘플(목표 개수 근사) 후 랜덤 보충으로 정확히 1024. P_m < 1024 → 복원추출로 1024 채움 + 중복 좌표에 σ≈0.3mm 가우시안 jitter(중복점 간 거리 0으로 인한 geo embedding kNN 퇴화 방지 — `transformer.py:319`의 topk가 중복점을 이웃으로 선택하면 ref_vector=0이 되는 문제).
- 근거: ① SAM-6D 자체 관행(`run_inference_custom.py:223-226`이 부족 시 복원추출) — 검증된 경로, 매칭·loss 코드 무개조. ② TRT 정적 shape 유지(A4). ③ 중복점은 attention/Procrustes에서 "가중치 2배"와 등가로 통계적으로 무해. 
- 기각안: 패딩+key_mask — RPE attention은 `key_masks`를 지원하나(`transformer.py:369-398`) `LinearAttention`(`transformer.py:518-564`)은 마스크 경로가 없어 개조 필요 + 정적 shape 이점도 없음. 중복 샘플 대비 이득이 없어 ablation 항목으로만 유지.
- coarse 196pt: `sample_pts_feats`의 FPS(`model_utils.py:58`) 대신 **전처리(CPU/경량 CUDA)에서 voxel-grid 다운샘플로 196개 인덱스 계산 후 엔진 입력** `fps_idx_m`으로 주입. P_m<196이어도 중복 인덱스로 196 고정. 커스텀 op가 엔진 밖으로 나가므로 TRT 문제 소멸.

**rgb_choose 대응:** SAM-6D는 224² 업샘플 특징에서 `torch.gather`(`model_utils.py:69-81`)로 유효 픽셀을 선택한다. Shape6D는 인코더 출력이 stride-4(56×56)이므로 224 업샘플 없이 `F.grid_sample(feat56, pixel_uv)`로 서브픽셀 보간 gather — 유효(LiDAR 존재) 픽셀 좌표만 `pixel_uv`에 담기므로 "LiDAR 유효 픽셀만 선택"이 자동 충족된다. grid_sample은 ONNX/TRT(8.5+) 지원.

### 1.2 학습 배치
B=32 기준(SAM-6D bs28@600k iter 참조; RTX PRO 6000 96GB에서 본 모델은 B=64 이상 가능할 것 — 추정, 모델이 ViT-B 대비 ~1/15 크기이므로). 학습 시 오브젝트 측도 forward 통과(§4), GT는 SAM-6D와 동일한 `rotation_label`/`translation_label`.

---

## 2. 기하 인코더 3후보 layer-level 스펙

공통 출력 규약: `dense_fm (B, 1024, 256)` — 매칭부 `input_dim=256` 유지(in_proj만 hidden으로 축소하므로 인코더-매칭 인터페이스는 SAM-6D와 동일).

### (a) image-grid conv — 우선순위 1 (권장)

**입력 채널(10ch):** `geo_maps` 8ch + flags 2ch(상수 브로드캐스트: rgb_valid, domain). RGB는 별도 branch(§3)로 분리 — 기하 스트림을 오염시키지 않고 dropout 절단점을 명확히 하기 위함.

**백본 (ResNet18-lite + FPN-lite, 출력 stride 4):**

| 레이어 | 구성 | 출력 해상도 | 채널 |
|---|---|---|---|
| Stem | 3×3 conv s2 (partial conv), BN, ReLU | 112² | 32 |
| Stage1 | BasicBlock ×2 (첫 블록 s2, partial conv) | 56² | 48 |
| Stage2 | BasicBlock ×2 (s2) | 28² | 96 |
| Stage3 | BasicBlock ×2 (s2) | 14² | 192 |
| Up1 | 2× bilinear up + concat(Stage2) + 3×3 conv | 28² | 128 |
| Up2 | 2× bilinear up + concat(Stage1) + 3×3 conv | 56² | 96 |
| Head | 1×1 conv | 56² | 256 |

파라미터 ~2.0M, ~1.6 GMACs(≈3.2 GFLOPs) — 추정: 레이어별 k²·C_in·C_out·H·W 합산. ViT-B(86M, 17.6 GFLOPs) 대비 파라미터 1/43, FLOPs 1/5.5.

**희소 유효픽셀 처리 — 권장: 초기 2개 레이어 partial conv + 마스크 채널 유지.** 유효 밀도가 224²의 1~2%(수백 pt)일 수 있어 단순 0채움은 초기 conv 응답을 희석시킨다. Partial conv는 표준 op 조합으로 표현 가능(TRT 안전):
```
y = conv(x ⊙ m) · (k²·C / (conv_ones(m) + ε)),   m' = (conv_ones(m) > 0)
```
Stage2부터는 수용영역이 충분히 넓어져 일반 conv + 마스크 채널로 전환. 대안(ablation): 전처리에서 마스크 내 XYZ pull-push 조밀화 후 일반 conv — 파일럿(M2-B)에서 비교.

**노멀 추정(전처리, 투영 이웃 기반, 표준 op만):** 유효마스크 가중 박스필터로 창(15×15, 희소율에 따라 21×21) 내 1·x·y·z·xx·xy·… 9개 모멘트 맵 계산 → 픽셀별 3×3 공분산 → 최소 고유벡터 = 노멀. 3×3 고유분해는 해석적(Cardano) 공식으로 텐서 op 구현(구현 필요, 난이도 낮음). 뷰 방향으로 부호 통일. 이웃 유효점 <5인 픽셀은 노멀 0 + (노멀유효) 비트를 유효마스크에 반영.

**가설/탈락 조건(설계 문서 승계):** crop 내 심한 깊이 불연속(2D 이웃 ≈ 3D 이웃 근사 붕괴)에서 (b) 대비 열세면 탈락. 완화책: XYZ 채널이 절대 3D 좌표를 담으므로 네트워크가 불연속을 학습으로 인지 가능.

### (b) PointNet++ lite (커스텀 op 제거판)

커스텀 op 제거 방법: **FPS → voxel-grid 다운샘플(전처리, 인덱스 입력화)**, **ball query → N≤1024이므로 전체 pairwise 거리행렬(1024², 1M원소 — 무시 가능) + radius 밖 +inf 마스킹 + TopK(nsample) + Gather**. 전부 ONNX 표준 op.

| 레이어 | centers | radius(정규화 좌표) | nsample | MLP |
|---|---|---|---|---|
| SA1 | 512 (voxel) | 0.1 | 16 | [6+3, 64, 64, 128] |
| SA2 | 196 (voxel, coarse 인덱스와 공유) | 0.25 | 16 | [128+3, 128, 128, 256] |
| FP2→1 | 512 | 3-NN 역거리 보간(거리행렬+TopK) | — | [256+128, 256] |
| FP1→0 | 1024 | 3-NN | — | [256+6, 256, 256] |

입력은 pts+노멀(전처리 §(a)와 동일 방식). 파라미터 ~0.35M, ~0.5 GMACs (추정: center×nsample×MLP MAC 합산). FLOPs는 최소지만 gather 지배적이라 TRT 실효 효율 낮음 — 레이턴시는 (a)와 동급이거나 열세 가능(추정). 희소·저밀도에서 표현력 우위 가능성이 채택 조건(D2-a가 캐스팅보트).

### (c) sparse conv

voxel 크기 2~4mm, SubMConv3d 스택: [32,64,128,256] 4단계 × 2블록, stride 2 다운 3회 + sparse FP 업샘플. 파라미터 ~3.5M(추정: 3³ 커널 × 채널곱 합산). active site 수 ≈ P_m이라 연산량 극소.

**TRT 경로 리스크(1급):** spconv의 TRT 배포는 mmdeploy/CenterPoint 계열 커스텀 플러그인에 의존하며 Jetson·최신 TRT 버전 호환이 보장되지 않음(존재는 하나 유지보수 리스크 — "검증 필요"). M2-B 탈락 조건 "기한 내 배포 경로 미확보 시 즉시 제외" 유지. 표현력 상한 확인용 참조 후보로만.

### 후보 비교 요약 (전부 추정)

| 후보 | Params | GMACs | TRT 직행성 | 희소 내성 | 판정 |
|---|---|---|---|---|---|
| (a) grid conv | 2.0M | 1.6 | ◎ (표준 op만) | ○ (partial conv) | **기본안** |
| (b) PN++ lite | 0.35M | 0.5 | ○ (gather 다수) | ◎ | 정확도 기준점 |
| (c) sparse conv | 3.5M | ~0.3 | △ (플러그인 의존) | ◎ | 상한 확인용 |

---

## 3. RGB 보조 branch

**구조 — 1차: 경량 CNN (E2 실험으로 distilled DINOv2-S와 비교).**

| 레이어 | 구성 | 출력 |
|---|---|---|
| Stem | 3×3 s2, 16ch | 112² |
| Block1-3 | inverted residual(MBConv, expand 4) s2,s1,s1 | 56², 48ch |
| Head | 1×1 conv → 64ch | 56²×64 |

파라미터 ~0.4M, ~0.25 GMACs (추정). DINOv2-S distill판(ViT-S 21M)은 유사 형상 변별 상황에서만 정당화 — E2 판정.

**융합 — concat + proj + 플래그 조건 게이트:**
```python
F_geo = grid_sample(geo_feat56, pixel_uv)          # (B,1024,256)
F_rgb = grid_sample(rgb_feat56, pixel_uv)          # (B,1024,64)
g     = sigmoid(W_g @ [rgb_valid_flag])            # (B,1) 학습형 스칼라 게이트
dense_fm = Linear(320→256)(concat(F_geo, g·F_rgb)) # (B,1024,256)
```
근거: concat+proj는 SAM-6D 매칭부 입력 규약(256d)을 유지하며, 게이트가 dropout 플래그와 추론 시 동적 가중을 단일 메커니즘으로 통일한다.

**Modality dropout (학습):** 배치별 확률 p=0.3으로 `F_rgb ← 0`, `rgb_valid_flag ← 0` (branch forward 자체를 건너뛰지 않고 출력 0 치환 — 그래프 정적 유지). 역방향(depth 희소화 dropout)은 M2-A2 증강에서 담당. 플래그가 인코더 입력 채널(§2)과 융합 게이트 양쪽에 들어가므로 네트워크가 "RGB 없음" 상태를 명시적으로 인지.

**추론 시 동적 가중:** 기본은 rgb_valid_flag ∈ {0,1} 이진(결정론적, TRT 친화). S2의 DINOv2 점수 기반 연속값 α∈[0,1] 주입은 인터페이스만 열어두고(플래그가 이미 연속 입력) 기본 비활성 — 보정되지 않은 신뢰도를 특징 게이트에 넣는 것은 검증 전엔 리스크.

---

## 4. CAD 측 인코더와 S0 캐시

**권장: 인코더 가중치 공유 + 도메인 플래그. (a) 채택 시 CAD는 42뷰 depth 렌더로 동일 인코더 통과 → 멀티뷰 특징 융합.**

- (a) grid-conv 선택 시 CAD 포인트를 2D conv에 직접 넣을 수 없다. 정합성 있는 해법: S0에서 icosphere 42뷰 depth 렌더(S0 산출물 재사용) → 각 뷰를 **기하 채널만으로**(RGB branch 출력 0, rgb_valid_flag=0, domain_flag=0) 동일 인코더 통과 → 뷰별 per-pixel 특징을 3D로 역투영 → CAD 표면 포인트(Poisson disk 2048)에 최근접 집계(뷰 가중 평균) → `dense_fo`.
- 근거: ① cross-attention 매칭은 양측 특징이 동일 임베딩 공간에 있어야 하며, 가중치 공유가 이를 구조적으로 보장(SAM-6D도 템플릿을 동일 ViT에 통과 — `feature_extraction.py:170-181`). ② depth 렌더는 외형 무관(A1 유지). ③ 오프라인이므로 42뷰 forward 비용은 0(온라인 기준). ④ 장면 측이 2.5D 부분 관측 통계로 학습되는 것과 정합 — CAD 측도 2.5D 렌더 경유가 도메인 갭을 오히려 줄임. 설계 문서의 "CAD 포인트+노멀 직접 특징 추출"은 (b)/(c) 인코더에서만 문자 그대로 성립하며, (a)에서는 본 방식이 M2-C2 폴백("depth 렌더 템플릿 특징, 외형 무관성 유지")과 사실상 동일 — 이를 (a)의 기본 경로로 승격한다.
- 도메인 차이는 `domain_flag`(장면=1/CAD=0) 채널로 흡수. 분리 인코더는 파라미터 2배·특징 공간 정렬 리스크로 파일럿 비교군으로만(M2-B 공통 설계 승계).

**S0 캐시 명세 (`onboarding/<obj>/pem_cache.npz`, 모델 버전 해시 포함):**

| key | shape | dtype | 내용 |
|---|---|---|---|
| `dense_po` | (2048, 3) | fp32 | Poisson disk 포인트 (radius 정규화 전 원시 좌표, 단위 m) |
| `dense_no` | (2048, 3) | fp32 | 노멀 (인코더 (b)/(c) 및 S4용) |
| `dense_fo` | (2048, 256) | fp16 | 융합 후 특징 (§4 절차 산출) |
| `radius` | () | fp32 | max‖dense_po‖ — 정규화 스케일 (`feature_extraction.py:140` 대응) |
| `fps_idx_o` | (196,) | int32 | voxel 기반 coarse 인덱스 |
| `sparse_po` / `sparse_fo` | (196,3)/(196,256) | fp32/fp16 | coarse 토큰 |
| `geo_embedding_o` | (197, 197, H) | fp16 | bg_point(=100·1벡터, `pose_estimation_model.py:27`) 포함 RPE 임베딩. H=192 시 14.9MB |
| `model_pts` | (1024, 3) | fp32 | `compute_coarse_Rt`/`compute_fine_Rt`의 `end_points['model']` 대응 (`base.yaml n_sample_model_point: 1024`) |
| `pe_fo` | (2048, H) | fp16 | fine 입력 `in_proj(f2)+PE(p2)` 사전계산 (p2는 정적 — `fine_point_matching.py:49` 전체 캐시 가능) |
| `sym_group` | (S, 3, 3) | fp32 | 대칭군 (S4 소비) |
| `diameter`, `bbox` | (), (2,3) | fp32 | S2 게이팅용 |

캐시 총량: ~16MB/객체 @H=192 (geo_embedding_o 지배 — 추정: 197²×192×2B). 5종 캐시 ~80MB, Orin NX 예산 내.

---

## 5. 매칭 모듈 개조

### 5.1 hidden 256→192 축소와 RPE 차원 연동

`hidden_dim`을 참조하는 전 지점을 단일 파라미터 H로 연동(H=192 기본, E4에서 128 비교):
- `GeometricStructureEmbedding`(`transformer.py:286-349`): `SinusoidalPositionalEmbedding(d_model=H)` — H는 짝수면 됨(192, 128 모두 OK). `proj_d`/`proj_a`: Linear(H,H). **geo embedding 출력 차원 = RPE `proj_p`(`transformer.py:365`) 입력 차원이므로 반드시 동시 축소** — S0 캐시 `geo_embedding_o`도 H 차원으로 재생성(캐시에 모델 해시를 넣는 이유).
- heads 4 유지 → head dim 192/4=48, 128/4=32. 나눠떨어짐 확인(`transformer.py:96`).
- `in_proj`: Linear(256→H), `out_proj`: Linear(H→256) — 유사도 계산 공간(out_dim 256)은 유지해 distill 시 SAM-6D teacher 유사도행렬과 정렬 가능하게.
- nblock 3→2 (coarse·fine 공통, M2-D1 distill로 보전).

### 5.2 fine 2048→1024 + linear attention 유지

`fine_npoint=1024`는 §1의 N_f와 일치 — config 변경 + 입력 규약으로 자동 성립. `LinearAttention`(`transformer.py:518-564`)은 O(N) 구조라 1024에서 그대로 동작. `focusing_factor=3` 유지. 분기 `if i*j*(c+d) > c*d*(i+j)`(`transformer.py:556`)는 정적 shape에서 export 시 상수로 고정됨(1025 토큰, c=d=48 → kv 경로 선택).

**bg_token: 유지.** 근거: soft assignment의 "비대응" 배출구로서 `compute_coarse_Rt`의 `pred_label>0` 필터(`model_utils.py:207-213`)가 bg 열(index 0)을 전제로 동작. S1 마스크 오염(경계 침식·과분할)과 LiDAR-RGB 시차로 인한 배경 포인트 혼입이 있는 한 필수. 제거는 마스크 품질이 검증된 뒤에나 ablation.

### 5.3 compute_coarse_Rt: 2000/100 축소 + top-k(3) 출력 개조 (코드 수준)

축소는 config만: `nproposal1: 6000→2000`, `nproposal2: 300→100` (`coarse_point_matching.py:73`에서 소비).

top-k 개조 — `model_utils.py` `compute_coarse_Rt`의 마지막 블록(237~246행)이 유일한 변경 지점. 현재:

```python
# model_utils.py:242-246 (현행: best 1개)
scores = weights1.unsqueeze(1).sum(2) / ((dis * weights1.unsqueeze(1)).sum(2) + 1e-8)
idx = scores.max(1)[1]                                   # ← 여기가 best-1
pred_R = torch.gather(pred_rs, 1, idx.reshape(B,1,1,1).repeat(1,1,3,3)).squeeze(1)
pred_t = torch.gather(pred_ts, 1, idx.reshape(B,1,1,1).repeat(1,1,1,3)).squeeze(2).squeeze(1)
return pred_R, pred_t
```

개조안 (pose-NMS를 넣지 않으면 top-k가 사실상 동일 가설 3개가 되므로 필수):

```python
# 개조: top-k + 회전공간 NMS. n2=100이라 비용 무시 가능
def select_topk_hypotheses(scores, pred_rs, pred_ts, k=3,
                           rot_thresh_deg=30.0, trans_thresh=0.3):
    # scores (B,n2), pred_rs (B,n2,3,3), pred_ts (B,n2,1,3)
    order = torch.argsort(scores, dim=1, descending=True)      # (B,n2)
    sel = []                                                    # 그리디 NMS
    for b in range(B):                                          # 배치 루프 or 벡터화
        kept = [order[b,0]]
        for i in order[b,1:]:
            R_rel = pred_rs[b,i] @ pred_rs[b,kept].transpose(-1,-2)   # (m,3,3)
            ang = arccos(clamp((trace(R_rel)-1)/2, -1, 1))            # geodesic
            dt  = norm(pred_ts[b,i] - pred_ts[b,kept], dim=-1)
            if (ang > rot_thresh_deg·π/180).all() or (dt > trans_thresh).all():
                kept.append(i)
            if len(kept) == k: break
        while len(kept) < k: kept.append(kept[0])               # 패딩(고정 shape)
        sel.append(kept)
    idx = tensor(sel)                                           # (B,k)
    hypo_R = gather(pred_rs, idx)   # (B,k,3,3)
    hypo_t = gather(pred_ts, idx)   # (B,k,3)
    hypo_score = gather(scores, idx)
    return hypo_R, hypo_t, hypo_score
```

`coarse_point_matching.py:70-76` 연동 개조:
```python
hypo_R, hypo_t, hypo_score = compute_coarse_Rt(..., k=cfg.n_hypo)   # k=3
end_points['hypo_R'], end_points['hypo_t'], end_points['hypo_score'] = hypo_R, hypo_t, hypo_score
end_points['init_R'] = hypo_R[:, 0]     # fine은 best만 (기존 인터페이스 유지)
end_points['init_t'] = hypo_t[:, 0]
```

fine(`fine_point_matching.py:42-44`)은 `init_R/init_t`만 소비하므로 무개조로 best 가설만 통과. S4는 `hypo_R/t/score` k개 전부 + fine 결과(`pred_R/t/pose_score`)를 받아 잔차 재평가 — fine 결과가 가설 0을 대체하고 가설 1..k-1은 coarse 포즈 그대로 S4 잔차 스코어링에 참여.

NMS 임계는 대칭군 인식으로 확장 가능(등가 회전은 중복으로 간주) — S0 `sym_group` 소비, 2차 과제.

### 5.4 fine PositionalEncoding의 pointnet2 제거

`fine_point_matching.py:90-125`의 `QueryAndGroup(r, nsample)` 2스케일을 표준 op로 재구현:
```python
D = pairwise_distance(p, p)                      # (B,1024,1024) — 1M 원소, 허용
D = D.masked_fill(D > r², +inf)
idx = D.topk(nsample, largest=False)[1]          # (B,1024,ns)  ns: 32/64→16/32 축소
grouped = gather(p, idx) - p.unsqueeze(2)        # 상대좌표 (+use_xyz concat)
→ SharedMLP → max-pool (기존과 동일)
```
비고: ① 오브젝트 측 `PE(p2)`는 정적 → S0 캐시 `pe_fo`(§4). ② 장면 측 `PE(p1_)`의 이웃 인덱스는 강체변환 불변(거리 보존)이므로 p1에서 1회 계산해 가설 간 재사용 가능(현재는 best 1 가설만 fine 통과라 실익은 S4 확장 시). ③ nsample 축소(16/32)로 PE FLOPs를 절반 이하로(전체 fine 비용의 최대 항목이므로 — §7).

---

## 6. 전체 forward 의사코드 (텐서 shape 흐름)

### 추론 (B=1)

```python
# ---------- 전처리 (엔진 밖, CPU/경량 CUDA) ----------
mask, K, rgb_full, lidar_pts = S2_output
crop_box = square_bbox(mask); rgb = resize_crop(rgb_full, crop_box)     # (1,3,224,224)
P = lidar_pts[in_mask]                                                  # (P_m,3)  P_m: 수백~수천
pts, pixel_uv = fix_count(P, N_f=1024)          # 중복샘플+jitter        # (1,1024,3),(1,1024,2)
geo_maps = build_maps(P, crop_box)              # XYZ+normal+mask+z      # (1,8,224,224)
fps_idx_m = voxel_sample_idx(pts, 196)                                   # (1,196) int32
cache = load_S0(obj_id)   # dense_po(1,2048,3) dense_fo(1,2048,256) sparse_* geo_embedding_o pe_fo model_pts radius

# ---------- TRT 엔진 1: 인코더+융합 ----------
geo56 = GridGeoEncoder(cat(geo_maps, flags_bcast))                       # (1,256,56,56)
rgb56 = RGBBranch(rgb)                                                   # (1,64,56,56)
F_geo = grid_sample(geo56, pixel_uv)                                     # (1,1024,256)
F_rgb = grid_sample(rgb56, pixel_uv) * gate(rgb_valid_flag)              # (1,1024,64)
dense_fm = proj(cat(F_geo, F_rgb))                                       # (1,1024,256)
dense_pm = pts / cache.radius                                            # (1,1024,3)  feature_extraction.py:140 규약

# ---------- TRT 엔진 2: 매칭 ----------
sparse_pm = gather(dense_pm, fps_idx_m); sparse_fm = gather(dense_fm, fps_idx_m)  # (1,196,3/256)
geo_emb_m = GeoStructEmbedding(cat(bg_point, sparse_pm))                 # (1,197,197,192)
geo_emb_o = cache.geo_embedding_o                                        # (1,197,197,192) 캐시
# coarse: in_proj 256→192, bg_token 부착 → (1,197,192) 양측
f1,f2 = [GeometricTransformer(RPE self + cross)]×2                       # (1,197,192)
atten_c = cos_sim(out_proj(f1), out_proj(f2)) / 0.1                      # (1,197,197)
hypo_R, hypo_t, hypo_score = compute_coarse_Rt(atten_c, sparse_pm, sparse_po,
                                model_pts/radius, n1=2000, n2=100, k=3)  # (1,3,3,3),(1,3,3)
# fine: best 가설만
p1_ = (dense_pm - hypo_t[:,0]) @ hypo_R[:,0]                             # (1,1024,3)
f1 = in_proj(dense_fm) + PE(p1_); f1 = cat(bg_token, f1)                 # (1,1025,192)
f2 = cache.pe_fo (bg 포함 사전계산)                                       # (1,2049,192)
f1,f2 = [SparseToDenseTransformer(sparse RPE + dense linear attn)]×2
atten_f = cos_sim(out_proj(f1), out_proj(f2)) / 0.1                      # (1,1025,2049)
pred_R, pred_t, pose_score = compute_fine_Rt(atten_f, dense_pm, dense_po, model_pts/radius)
pred_t *= cache.radius                                                   # 역정규화 (fine_point_matching.py:80)

# ---------- S4로 ----------
return {pred_R (1,3,3), pred_t (1,3), pose_score (1,),
        hypo_R (1,3,3,3), hypo_t (1,3,3), hypo_score (1,3)}
```

### 학습 (B=32)

```python
# 오브젝트 측도 forward: 42뷰 중 랜덤 V=2뷰 depth 렌더를 동일 인코더 통과(§4, RGB=0)
#   → dense_po/fo (32,2048,3/256) — SAM-6D의 tem1/tem2 경로(feature_extraction.py:144-163) 구조 승계, RGB→depth 교체
# 증강: modality dropout p=0.3, 희소 LiDAR 패턴(M2-A2), 외형 랜덤화, 기하 범프
# coarse/fine 각각 compute_correspondence_loss (기존 loss_utils 승계, 대칭 GT 처리는 M2-A3 확인 후)
# init_R/t는 aug_pose_noise(gt) (coarse_point_matching.py:59-62 그대로)
# distill(M2-D): SAM-6D teacher의 atten_c/atten_f (out_dim 256 공간) MSE + feature hint
```

---

## 7. 파라미터 / FLOPs / 레이턴시 추정표 (SAM-6D PEM 대비)

전부 추정. 산출 근거: 파라미터·MACs는 레이어 스펙 합산, 레이턴시는 Orin NX FP16 실효 5~8 TFLOPS 가정(공칭 대비 실효 20~30%) + 메모리 바운드 항목 보정. 실측(M0-2 환산계수)으로 대체될 값.

| 블록 | SAM-6D PEM | Shape6D-PEM (H=192, 2블록, 1024pt) |
|---|---|---|
| 특징 인코더 | ViT-B 86M / 17.6 GFLOPs + 업스케일 Linear 12.6M / 4.9 GFLOPs | grid-conv 2.0M / 3.2 GFLOPs + RGB branch 0.4M / 0.5 GFLOPs |
| 오브젝트 측 (추론) | dense_po/fo 캐시 가능(코드 지원)·FPS/geo_emb는 매 프레임 재계산 | 전량 S0 캐시 — 온라인 0 |
| geo embedding | 197²·256, 매칭과 합산 | 197²·192, 장면 측만 온라인 (~0.06 GFLOPs) |
| coarse 매칭 | 3블록·256d ≈ 2.0M / ~1.4 GFLOPs | 2블록·192d ≈ 1.3M / ~0.7 GFLOPs |
| fine 매칭 | 3블록·256d·2048pt + PE(ns 32/64) ≈ 3.5M / ~8 GFLOPs | 2블록·192d·1024pt + PE(ns 16/32) ≈ 2.0M / ~2.5 GFLOPs |
| 가설 생성 | 6000 SVD + 300 검증 | 2000 SVD + 100 검증 + NMS top-3 (~1/3 비용) |
| **합계 (추론 온라인)** | **~104M / ~32 GFLOPs** | **~5.9M / ~7 GFLOPs (파라미터 1/18, FLOPs ~1/4.5)** |
| Orin NX TRT FP16 레이턴시 | (미변환 — 데스크톱 4.4s 전체 파이프라인의 일부) | 인코더 10~20ms + 매칭 40~70ms + 가설 15~30ms ≈ **S3 총 70~120ms** (예산 150~250ms 내, 추정) |

M2 게이트 "S3 FLOPs ≤ SAM-6D 1/3" 충족(~1/4.5). H=128·1블록 축소 여유분은 E4 곡선으로 결정.

---

## 8. TRT 변환 계획 — 문제 op 목록과 대체

| # | 문제 op | 위치 | 문제 | 대체 |
|---|---|---|---|---|
| 1 | `furthest_point_sample`, `gather_operation` | `model_utils.py:53-66`, `transformer.py:654` | 커스텀 CUDA, ONNX 불가 | voxel 다운샘플 인덱스를 **엔진 입력**으로 (전처리 계산), gather는 `torch.gather`(ONNX Gather)로 재작성 |
| 2 | `QueryAndGroup`/`SharedMLP`(pointnet2) | `fine_point_matching.py:93-94` | 커스텀 CUDA | 거리행렬 + masked TopK + Gather (§5.4) |
| 3 | `torch.svd` | `model_utils.py:344` (WeightedProcrustes) | ONNX에 SVD op 없음 | 3×3 전용: ① 1차 — 가설 생성/Procrustes를 엔진 밖 후처리(PyTorch CUDA)로 분리 실행, ② 2차 — Horn quaternion법(4×4 대칭행렬 최대고유벡터, 고정 반복 멱승법) 또는 고정 5-sweep Jacobi로 표준 op 구현("구현 필요") |
| 4 | `torch.searchsorted` + `torch.rand` | `model_utils.py:219` | ONNX 지원 불안정 + 내부 난수 | Gumbel-top-k 샘플링으로 등가 대체: `topk(log(pred_score) + G)`, 난수 G는 **엔진 입력 텐서**로 주입(결정론적 리플레이도 가능해짐) |
| 5 | `torch.einsum` (RPE `bhnc,bhnmc->bhnm` 등) | `transformer.py:390-391,555-561` | TRT Einsum 커버리지 제한적(특히 5차원) | matmul+reshape 조합으로 수동 전개 |
| 6 | `masked_fill(-inf)` + softmax | `transformer.py:138,398` | FP16 오버플로 | -1e4 상수로 대체 + softmax 레이어 FP32 고정(temp 0.1로 로짓 ×10 증폭되므로 필수) |
| 7 | `topk` (geo embedding kNN, NMS) | `transformer.py:319` | 지원되나 K 동적 금지 | K 상수 고정(angle_k=3, nsample, k=3) — 이미 상수 |
| 8 | 데이터 의존 분기 | `transformer.py:556` (linear attn), `feature_extraction.py:135` | trace 시 한 분기로 고정 | 정적 shape이므로 export 시 kv 경로로 상수 고정 — 의도 명시 주석 |
| 9 | 배치 루프형 NMS (§5.3) | 신규 | 루프는 ONNX 비친화 | B=1 추론 + n2=100이므로 엔진 밖 후처리(3×3 행렬 100개 — CPU µs급) |
| 10 | `grid_sample` | 신규 (§1.1) | — | ONNX opset16 / TRT 8.5+ 네이티브 지원, Orin JetPack TRT 버전 확인 필요("검증 필요") |

**엔진 분할 전략:** ① 인코더+융합 엔진(정적 224² 입력) ② 매칭 엔진(정적 196/1024 토큰) ③ 가설 생성·Procrustes·NMS는 1차 릴리스에서 엔진 밖 CUDA/CPU 후처리(전체의 ~20% 연산, §8-3의 SVD 대체 구현이 검증되면 매칭 엔진에 흡수). Partial conv·노멀 추정·voxel 샘플은 전처리 단계(표준 op 조합이므로 필요 시 엔진 ①에 흡수 가능).

**잔여 리스크:** grid_sample의 Jetson TRT 버전 호환(§8-10), linear attention의 `x**3`(pow) FP16 다이나믹레인지(스케일 파라미터가 학습되므로 캘리브레이션 확인), geo embedding sinusoid의 FP16 정밀도(coarse 197²는 FP32 고정 권장 — 비용 미미).

---

### 열린 결정 사항 (상위 문서 의존)
1. **D2-a(물체 위 유효 포인트 수)** — N_f=1024의 타당성과 인코더 (a)vs(b) 선정의 캐스팅보트. 수백 pt 하한이면 fine 512 변형도 E4 격자에 추가 권장.
2. CAD 측 42뷰 렌더 경유(§4)를 (a) 채택 시 기본 경로로 승격하는 안 — M2-C2 실험 항목의 재정의에 해당하므로 설계 문서 반영 필요.
3. pose-NMS 임계(30°/0.3)와 대칭군 인식 NMS 도입 시점 — E5(k 스윕)와 함께 T-LESS 대칭군에서 결정.