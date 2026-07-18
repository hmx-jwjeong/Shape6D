# 교차 검증 리포트 — 관점 2: 실현 가능성·수치 타당성 (v0.1 설계 5종)

검증 방법: 설계가 인용한 SAM-6D 코드(`model_utils.py`, `transformer.py`, `fine_point_matching.py`, `coarse_point_matching.py`, `pose_estimation_model.py`, `detector.py`, `base.yaml`, `training_dataset.py`, `run_inference_custom.py`)를 실제로 열어 라인 단위 대조했고, 수치는 전부 재계산했다. 코드 인용(라인 번호, bg_point=100·ones, LinearAttention 마스크 부재, searchsorted 복원추출, betas (0.5,0.999), dilate 50%, outlier 1.2× 등)은 **전부 사실로 확인**됨 — 아래는 그 위에서 발견한 문제만 나열한다.

---

## 심각도 상 (설계 성립 자체를 위협)

### [상-1] 대칭 자동검출 τ_sym이 샘플링 노이즈 바닥보다 낮다 → 대칭이 하나도 검출되지 않는다 (s0s1 §1.5)
재계산: N=4096 포인트, 표면적 A≈2D²(전형 물체)일 때 회전된 샘플 집합→원본 샘플 집합의 point-to-**point** 1-way chamfer 기대값은 최근접 간격 ≈ 0.5·√(A/N) = 0.5·D·√(2/4096) ≈ **0.011·D**. 완전 대칭 물체조차 이 바닥값이 나온다. 그런데 τ_sym = 0.004·D — **노이즈 바닥의 1/2.8**. 10cm 큐브(D=0.173m, A=0.06m²)로 구체화: 간격 √(0.06/4096)=3.8mm, 기대 chamfer ≈ 1.9mm vs τ=0.69mm. 결과: 원기둥·정다각 부품 전부 "비대칭" 판정 → S4 dedupe·등가 평가 미작동 → 설계 문서 스스로 경고한 "대칭 미등록 → S4 오reject"가 **기본 동작**이 된다.
**수정**: point-to-point가 아니라 point-to-**mesh** 거리(`trimesh.proximity.closest_point` — 오프라인이므로 허용) 사용, 또는 τ_sym = max(0.004·D, 0.7·√(A/N)) + 검증용 단위테스트(합성 원기둥/큐브에서 검출률 100% 확인)를 온보딩 CI에 포함.

### [상-2] S2 point-to-template 정합: 시선축(오프센터) 미보정 — in-plane 탐색 축 자체가 틀림 (s2s4 §1.2)
관측 포인트를 **카메라 z축** 주위로 R_z(θ) 회전시켜 탐색하는데, 템플릿은 물체 중심을 광축에 놓고 렌더된다. 1280×800·HFOV 70°에서 화면 좌우 끝 물체의 시선 방향은 카메라 z와 최대 **35°** 어긋난다(수평 반화각). 이 각도는 뷰 양자화 예산(아래 [상-3]) 전체보다 크다 — 화면 주변부 물체는 정상 후보도 S_depth가 붕괴한다.
**수정**: TDF 조회 전 관측 포인트를 "centroid 시선 → z" 정렬 회전(Rodrigues 1회, 무료)으로 뷰잉-레이 프레임에 옮길 것. 템플릿 렌더 규약과 정합됨.

### [상-3] S2 회전 양자화 오차 과소평가 — τ_trunc 대비 2배 (s2s4 §1.2)
재계산: 42뷰의 커버 반각 = arccos(1−2/42) = **17.8°** (문서의 "이웃 간격 ~37°"에서 나온 절반 ~18.5°와 유사하나, 문서는 결과 오차를 "~0.1D"로 축소 기재). in-plane 30° 간격 → 최대 15°. 합성 최악 ≈ √(17.8²+15²) ≈ 23°. 반경 0.5D 포인트의 변위 = 2·0.5D·sin(11.5°) = **0.20D = 2×τ_trunc(0.1D)**. 즉 정상 후보의 주변부 포인트 대부분이 절단 밖으로 나가 s≈0 기여 → 정답 S_depth가 θ_depth_min=0.3 근처까지 떨어질 수 있고 유사 형상 변별력이 소실된다.
추가로 **jitter ±1 voxel = ±0.025D**는 가림 시 가시부 median 편이(가림률 30~50%에서 0.1~0.2D)의 1/4~1/8 — 가림 후보 구제 불능.
**수정**: pass2에서 상위 뷰에 대해 in-plane 15° 간격 재탐색(K_ip=24 등가) + jitter 범위 ±2~3 voxel 확대, 또는 τ_trunc를 뷰 양자화에서 유도(0.2D)하되 판별력 손실을 E6에서 실측. M2 전 합성 스모크 테스트(정답 포즈 주입 → S_depth 분포 확인) 필수.

### [상-4] 패딩 vs 복원추출 — training과 s3pem의 정면 모순 (train/infer 분포 불일치)
s3pem §1.1은 "패딩+key_mask 기각 — `LinearAttention`(transformer.py:518-564, **마스크 경로 없음을 코드로 확인**)이 개조 필요"라며 **복원추출+jitter를 채택**했다. 그런데 training §2-①과 config(`n_sample_observed_point: 2048  # 패딩 방식 (복원추출 아님, n_valid 전달)`)은 **제로패딩+유효길이**를 채택했다. 제로패딩 포인트는 fine 매칭에서 원점 좌표의 실점처럼 attention·PE·geo embedding에 참여한다(마스크가 없으므로) — 학습이 이 오염 분포로 진행되고 추론(복원추출)과 분포가 어긋난다.
**수정**: 한쪽으로 통일(권장: s3pem의 복원추출 — SAM-6D 검증 경로, `run_inference_custom.py:223-226`에서 확인됨). training config·§2-① 문구 수정.

### [상-5] LiDAR 거리 정밀도 σ 전제가 문서 3곳에서 4배 이상 불일치
- sensor §1.1: 포인트당 1σ **5~20mm** (Avia급 스펙 ~2cm 계열)
- s2s4: τ_z=10mm, Huber δ=5mm = "σ의 ~3배" → **σ≈1.7~3mm 가정**
- training: σ ~ U[**2, 5mm**]
σ가 실제 10~20mm면: ① inlier 게이트 |r|<10mm가 정답 포즈 포인트의 ~50-70%만 통과 → inlier_ratio 특징 자체가 무의미 ② ICP 정밀도 σ/√N = 20mm/√300 ≈ 1.2mm (병진)로 아직 성립하나 ③ "희소·**고정밀**"이라는 D2 전제와 S2 TDF voxel(0.025D=2.5mm@10cm 물체)이 흔들린다.
**수정**: D2-a 실측 프로토콜에 **거리 노이즈 σ 계측 축을 명시적으로 추가**(현재는 포인트 수만 계측), τ_z·δ·training σ를 전부 "σ_lidar의 배수"로 파라미터화해 1곳에서 주입.

---

## 1. 레이턴시 합산 검증 (Orin NX, 1s 예산)

각 문서 추정치를 직렬 합산(설계상 파이프라인은 순차):

| 시나리오 | 합산 | 판정 |
|---|---|---|
| 공칭: T=200ms + FB 5 + S1 70 + S2 24 + S3 95 + S4 26 | **≈ 420ms** | 여유 있음 ✓ |
| 설계 내 최악: T=400 + FB 5 + S1 90 + S2 24 + **S3 120×3** + **S4 26×3** + 재시도 25 | **≈ 982ms** | 여유 <2% |
| + S1 grid fallback 발동 | +250~480ms | **명백 초과** |

발견된 문제:

- **[상-6] k_inst=3 곱셈 누락**: s2s4는 top-k_inst=3 인스턴스를 S3로 전달한다고 명시했는데, s3pem·s2s4의 레이턴시 표는 전부 **인스턴스 1개(B=1) 기준**이다. S3 70~120ms, S4 26ms가 ×3 되면 그것만 288~438ms — 어느 문서의 예산표에도 반영되어 있지 않다. 수정: k_inst=3을 배치 차원으로 묶어 엔진 1회 실행(B=3 정적 shape)하거나, 예산표를 ×k_inst로 갱신.
- **[중-1] S1 fallback 비용 미계상**: decoder 8프롬프트 = 8~15ms 추정인데 grid fallback은 **256프롬프트** → 비례 환산 256~480ms. LiDAR 공백 타일 프롬프트(§5.3a)도 동일 문제. fallback 발동 시의 예산 초과를 "허용된 degraded 모드(예: 1.5s)"로 명문화하거나 grid를 8×5=40 수준으로 제한할 것.
- **[중-2] Orin 환산 계수가 문서마다 다름**: s0s1 "1/10~1/20", s2s4 "5~10×", s3pem "실효 5~8 TFLOPS"(RTX PRO 6000 실효 200~400 TFLOPS 가정 대비 **25~80×**). 메모리 바운드 항목은 대역폭비(≈1.79TB/s vs 102GB/s ≈ **17.5×**)가 지배하므로 s2s4의 5~10×는 낙관 하한이다. M0-2 환산 계수를 "컴퓨트/메모리/런치 바운드 3종"으로 분리 정의해 전 문서가 공유할 것.
- **[중-3] 재적분·재촬영 트리거 vs 1s**: sensor의 "T 적응(포인트 부족 시 재적분)"과 S4 UNCERTAIN 재촬영은 각각 +200~400ms, +1사이클이다. A3(단발 견고성)와 1s 예산의 관계 — "1s는 정상 경로 SLA이고 재시도는 예외 경로"인지 — 를 상위 문서에서 확정해야 한다. 현재는 세 문서가 각자 트리거를 만들면서 예산은 아무도 합산하지 않는다.

## 2. VRAM / 학습 시간 산술 검증

검산 결과 맞는 것: fine 활성화 4098·256·16·2B·3 = **100.7MB/샘플** ✓, 175k iter@bs96 = 16.8M 샘플 = 600k@bs28 등가 ✓, 0.8 iter/s → 60.8h ✓, 파일럿 17.5k iter ≈ 6.1h ✓, TDF 9.29MB ✓, geo_embedding_o 14.9MB ✓, 대칭 라벨 coarse 텐서 96×16×2048×196 fp16 = 1.23GB(일시) — per-sample 표기와 정합 ✓.

- **[상-7] 오브젝트 측 2뷰 인코더 forward 활성화 누락**: s3pem 학습 의사코드는 "42뷰 중 랜덤 V=2뷰를 **동일 인코더에** 통과"(gradient 필요 — 가중치 공유)라고 명시했는데, training §5.1의 (a) 인코더 활성화 130~250MB는 **장면 측 1회 forward 분량**이다. 오브젝트 측 +2회 → per-sample ≈ 560~940MB → bs96 = **54~90GB**. "안전 bs ~180"은 약 2배 낙관이고 bs96도 상단에서 OOM 경계다. 수정: 인코더 activation checkpointing 또는 bs64+grad_accum을 기본 예비책으로 승격(필드는 이미 있음), §5.1 표에 오브젝트 측 행 추가.
- **[하-1] 파라미터 총량 모순**: s3pem "합계 ~5.9M" vs training "전체 ~30M 추정". 옵티마이저 메모리엔 영향 없지만 두 문서가 서로 다른 모델을 셈하고 있다는 신호 — full-size(256d·3블록) 기준으로 재합산해 통일할 것(재계산: SAM-6D 매칭부 5.5M + 인코더 2.4M ≈ 8M, 30M 근거 불명).
- 결론(질문 5 대응): **파일럿(3~4일) + 본학습 1회(5~7일) + distill(~5일) ≈ 13~16 GPU-일 구조는 96GB 1장에서 유지 가능**하다. 단 "본학습 2회전 여유"는 distill을 포함하면 캘린더상 빠듯하며, [상-7] 반영 시 bs96→64로 낮추면 벽시계 ×1.5(≈4일/런)까지 각오해야 한다 — §5.2 표를 이 조건부로 갱신 권장.

## 3. 존재 의심 API / 확인 필요 항목

- **[중-4] `scatter_min` / `scatter_argmin_z`** (s2s4 §2.1·2.2): torch 네이티브에 없다. `torch_scatter`는 커스텀 CUDA 확장이라 **A4 저촉**. 네이티브 대체는 `Tensor.scatter_reduce_(reduce='amin')`(torch≥1.12) + argmin은 2-pass 트릭 필요 — "구현 필요" 표기가 빠져 있다. TRT 측 대응(ONNX ScatterElements reduction)도 "검증 필요".
- **[하-2] `mesh.remove_degenerate_faces()`** (s0s1 §1.2): trimesh 4.x에서 제거됨(→ `mesh.update_faces(mesh.nondegenerate_faces())`). 버전 고정 또는 표기 수정.
- **[하-3] Gumbel-top-k "등가 대체" 주장** (s3pem §8-4): 원 코드(model_utils.py:219, 확인)는 searchsorted **복원**추출, Gumbel-top-k는 **비복원** — 분포가 다르다. 개선일 수는 있으나 "등가"가 아니므로 ablation 항목으로 강등할 것.
- **[하-4] `torch.svd`**: deprecated — 이식 시 `torch.linalg.svd`(U,S,Vh 규약 상이, V 전치 주의).
- 확인되어 문제 없는 것: `trimesh.sample.sample_surface_even`·`repair.fill_holes`·`fix_normals`, `open3d segment_plane`, `cv2.convexHull/fillConvexPoly`, pyrender EGL, `grid_sample`(opset16), DINOv2 ViT-S 384d/4.6GFLOPs, Livox 3기종 FOV·rate 수치(Ω 재계산 일치: Avia 1.4716sr, Mid-70 1.149sr, Mid-360 5.717sr, Tele-15 0.0711sr→3.37M pts/sr/s ✓), koide3 `direct_visual_lidar_calibration` 실존 ✓.

## 4. 알고리즘 결함

- point-to-template 회전 처리: **[상-2], [상-3]** 참조.
- **[중-5] Projective ICP association 창과 tau_assoc의 모순** (s2s4 §2.2): tau_assoc=20mm를 의도하지만, 관측점 투영 셀의 3×3 이웃(stride 2)의 측방 도달은 ±3px ≈ **±3.3mm@1m**(1px≈1.1mm). coarse 포즈의 측방 오차가 1cm(≈9px)면 대응이 아예 안 잡혀 ICP가 no-op으로 수렴 선언한다. 수정: win을 `ceil(tau_assoc·fx/(Z·stride))`에서 유도하거나 stride 8→2 coarse-to-fine 2단. (방향 선택 자체 — 희소 관측을 source, CAD를 target, CAD 노멀 사용 — 는 타당. 부호 규약도 재유도 결과 일관 ✓. `min_pool_3x3` 구멍 메움이 실루엣을 1셀 팽창시켜 경계에서 가짜 free_viol을 만들 수 있는 점만 소소하게 주의.)
- 대칭 검출: **[상-1]** 외 2건 — **[중-6]** 관성 고유값이 3중 축퇴(큐브·정사면체류)면 축 후보가 "평면 내 30° 스캔"으로 부족(전구면 탐색 필요), n∈{7,9,11} 미검사는 의도라면 명기할 것.
- **[중-7] LiDAR 포인트 수 모델의 균일 밀도 가정**: 공식(rate/Ω_FOV 균일)은 재계산상 자기일관 ✓이나, Livox 비반복 로제트는 **FOV 중심 밀도가 주변부 대비 수 배** 높다. 물체가 FOV 중심이냐 가장자리냐로 표값이 ×2~3 흔들리는데 η=0.5 스칼라로는 표현 불가. D2-a 프로토콜에 **"물체의 FOV 내 위치" 축이 누락** — {중심, 반경 50%, 가장자리} 3점 추가 권장.
- **[하-5] 템플릿 렌더 거리 근거 오류** (s0s1 §1.4): "tan(22.5°)·2.5r ≈ 1.04r > r이므로 잘림 없음"은 평판 근사. 구면 실루엣은 arcsin(r/d) = arcsin(0.4) = **23.6° > 22.5°** — 접선 조건은 d ≥ r/sin(22.5°) = **2.613r**이므로 2.5r에서는 바운딩 스피어를 채우는 물체가 잘린다. d=2.7r 권장. (부수: fx = 112/tan22.5° = **270.4**, 문서 270.9는 오기. "프레임 70% 차지"도 실제 ~97%로 과소 기재.)

## 5. 문서 간 정합성 (통합 시 즉시 충돌하는 것)

- **[중-8] S0 캐시 스키마 3원화**: `onboard_v1.npz`(s0s1: dense 4096, dino_cls **[12,384]**, tpl_pts 512) vs `cache.npz`(s2s4: dense **16384**, dino_cls **(42,384)**, tdf, tpl_center — s0s1 스키마에 tdf·tpl_center·16384pt가 없음) vs `pem_cache.npz`(s3pem: dense **2048**). dino_cls 12 vs 42는 직접 모순(S2 §1.3은 42뷰 cls를 요구). 단일 스키마 + 소비자별 필수 필드 매트릭스로 통합 필요.
- **[중-9] FrameBundle 이중 정의**: sensor(`shape6d/common/frame_bundle.py`, lidar_points/pix2pt/point_quality 포함)와 s0s1(`shape6d/types.py`, lidar_xyz/lidar_uv만). S1 의사코드는 sensor판의 quality flag를 사용하지 않는다(EDGE_MIXED 포인트가 프롬프트 대표점으로 뽑힐 수 있음).
- **[중-10] 포인트 수 하한 3원화**: sensor `N_min_reject=64` vs s2s4 `N_min=30`(게이트)·25(inlier) vs training `min_valid_obj_points=64`. 64에서 reject하는 스테이지와 30까지 받는 스테이지가 공존 — 하한 소유자를 1곳으로.
- **[하-6] coarse 가변 길이(sensor §3.3 "패딩 마스크 처리 가능") vs s3pem 고정 196 중복 인덱스** — [상-4]와 같은 뿌리. sensor 문서 문구 수정 필요.
- **[하-7] s2s4 내부 오참조**: §2.1·§1.4가 "§6.1/§6.2"를 가리키나 실제 실패모드는 §5.1/§5.2.
- **[하-8] s3pem "B=32 기준" vs training bs96** — 표기 통일.

---

**총평**: 코드 레벨 사실관계와 대부분의 산술은 정확하다(검산 통과 다수). 그러나 ① 대칭 검출과 S2 정합은 **현 수치 그대로면 동작하지 않는** 구체적 결함이 있고([상-1]~[상-3]), ② 레이턴시는 k_inst 곱셈·fallback·재시도를 합산하는 순간 여유가 0~음수가 되며([상-6], [중-1~3]), ③ 학습은 오브젝트 측 forward 누락으로 bs96이 경계선이고([상-7]), ④ 패딩/복원추출·σ 전제·캐시 스키마의 문서 간 모순([상-4], [상-5], [중-8~10])은 M2 착수 전에 단일 소스로 정리하지 않으면 통합 단계에서 반드시 터진다.