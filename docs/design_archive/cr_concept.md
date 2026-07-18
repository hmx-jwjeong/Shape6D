# Shape6D — "데이터를 다루는 방식"의 신규 컨셉 정리

대상 독자: 외부 엔지니어. 근거는 전부 리포 코드(`/Users/jaewoo/Documents/Shape6D/shape6d/`)와 docs 01~04, 대조군은 `/Users/jaewoo/Documents/SAM-6D/SAM-6D/` 원본 코드.

---

## 0. 한 문장 요약

Shape6D는 **"dense depth를 지어내지 않고, LiDAR 실측점이 있는 곳만 기하 증거로 취급하며, 오브젝트에 관한 모든 연산을 오프라인 캐시로 밀어내고, 판별 연산의 argmax를 그대로 포즈 초기값으로 재사용한 뒤, 최종 판정은 물리 잔차 기반 가설 간 증거 비교로 내리는"** 파이프라인이다. 합성 검증(04)에서 학습 0으로 요구 스펙(10mm/1°)을 평균 5배 여유(1.9mm/0.23°)로 통과했고, 실패 10%는 전량 자기 검출(REJECT)됐다.

---

## 1. 엔드투엔드 데이터 흐름

### 1.1 오프라인: CAD → `onboard_v1.npz`

단일 CAD에서 단일 npz(스키마 `onboarding/cache.py:_SCHEMA`, 검증 `validate()`)를 굽는다. 핵심 설계는 **"마스터 1개 + 이름 있는 서브셋 인덱스"** — 모든 소비자가 같은 Poisson disk 샘플에서 파생되어 표현 분열이 없다(03 §2.4, 검증 [H2]/[H3] 반영).

| 필드 | shape/dtype | 왜 존재하나 | 온라인 소비자 |
|---|---|---|---|
| `pts_master`/`nrm_master` | [16384,3] f16 | Poisson disk 표면 샘플+face 노멀. 16384는 ICP 타깃 밀도 실증치(4096pt에서 yaw 1.04° → 8192+에서 0.71°, `verify/icp.py:34` 주석) | S4 ICP target (`ProjectiveICP.refine`의 `X_m`,`N_m`) — **노멀은 CAD 측**(희소 관측에서 노멀 추정 불가, CAD 노멀은 정확·무료) |
| `idx_pem`/`idx_sparse`/`idx_model` | [2048]/[196]/[1024] i32 | S3(M2 Shape6D-PEM)용 FPS 서브셋. `idx_sparse ⊂ idx_pem` 포함 관계 보장 (`onboarding/sampling.py:make_subsets`) | M2 예정 |
| `idx_verify` | [2048] i32 | S4 스플랫 스코어용. 1024pt는 스플랫 홀필이 팔레트 포켓 개구를 오폐색(04 결함 6) — 2048 의무 | S4 `HypothesisScorer` |
| `tpl_depth`/`tpl_pose`/`tpl_K` | [42,224,224] u16(mm)/[42,4,4]/[3,3] | icosphere L2 42뷰(SAM-6D 뷰 규약 유지) depth 템플릿. **point-splat z-buffer 렌더**(`templates.py:splat_depth`) — GL 무의존, TDF의 τ_trunc=0.1D 관용이 스플랫 근사를 허용 | S2 정합·coarse 포즈 복원(`tpl_pose`는 센터링 흡수된 T_obj2cam) |
| `tpl_pts`/`tpl_center` | [42,512,3]/[42,3] | 뷰별 z-buffer 가시 포인트(모델계 저장 — S4 소비 규약)와 뷰계 median 센터 | S2 센터링 기준, coarse 병진 복원 |
| `tdf` | [42,48³] f16 | 뷰별 truncated distance field LUT([−0.6D,+0.6D]³, trunc 0.1D). **KD-tree를 온라인에서 못 쓰므로(A4) O(1) voxel gather로 거리 조회를 선계산**(`templates.py:build_tdf`, `tdf_lookup`) | S2 `PointToTemplateMatcher` |
| `sym_rots`/`sym_axes` | [S,3,3]/[A,3] | 대칭 자동검출(`symmetry.py:detect_symmetry`): point-to-mesh 거리 + 노이즈 바닥 임계 τ=max(0.004D, 0.7√(A/N)) — 초안 임계는 샘플링 노이즈 바닥의 1/2.8이라 대칭 전멸(검증 [상-1]) | S4 `SymmetryHandler`(dedupe·canonicalize·sym-aware 오차), M2 sym-aware loss |
| `dense_fo`/`geo_embedding_o`/`pe_fo` | (선택) | M2 기하 인코더 특징 — `encoder_hash` 종속 2-pass 굽기(manifest로 정합성 검증) | S3 (M2) |
| `diameter`/`bbox`/`radius` | 스칼라류 | 메트릭 크기 — RGB가 절대 못 주는 무료 판별 신호의 원천 | S2 SizeGate, TDF 스케일, ICP τ 스케줄 |

`manifest.json`에 `cad_sha256`, `unit`, `sym_summary`, `encoder_hash` 기록 (`cache.py:save`). 팔레트 캐시 실측 1.6MB (03 §11).

### 1.2 온라인: 원시 관측 → verdict

```
RGB[800,1280,3] + LiDAR 원시 포인트(LiDAR계)
  └▶ build_frame_bundle (common/frame_bundle.py, <5ms 예산)
       T_cam_lidar 적용 → project_points(uv, 후방/화면밖=NaN)
       → rasterize: z-buffer → sparse_depth[H,W], valid_mask, pix2pt[H,W]→포인트 인덱스, losers
       → flag_quality: point_quality[N] u8 (EDGE_MIXED|LOW_INTENSITY|MULTI_RETURN)
  = FrameBundle (전 스테이지 유일 입력 정본, 이후 좌표 변환 없음)

S1 (proposal/prompt_gen.py: LidarPromptGenerator)
  in-image 필터(NaN uv 제외) → 반복 RANSAC 지지평면 제거(_ransac_plane)
  → voxel-hash 26-연결 BFS 클러스터링(_voxel_clusters — kNN/KD-tree 불사용, A4)
  → 크기 상한 위반 시 주축 2-분할(_try_split), 실패 시 원본 통과(리콜 우선)
  = list[Cluster{point_indices, centroid(실측점 스냅), bbox_diag}]
  + PromptSet(대표점 = centroid 스냅점 + 주축 양끝 실측점 — 합성점 금지) [EViT-SAM은 M1]

S2 (identify/)
  ① SizeGate: d_obs = 5–95 percentile 트리밍 extent vs D_cad
     상한 (1+0.15)D 하드 리젝, 하한 β_occ·D(0.35/가림 0.25), N<30 탈락(유일 후보는 low_geo 구제)
  ② PointToTemplateMatcher.match(pts_cam[N,3]):
     median 센터링 → _ray_align_rotation(시선축→z, 화면 주변부 보정)
     → pass1: 활성 뷰(직립 프루닝 upright_view_mask → 25/42) × in-plane 12@30° TDF 점수
     → pass2: top-3 뷰 × 24@15° × 병진 jitter ±2voxel
  = MatchResult{s_depth, topk[MatchHypo{score, view, θ, δ}], R_align, centroid}

무학습 coarse (pose/template_init.py: coarse_poses_from_match)
  argmax(뷰,θ,δ) → 폐형식 복원: R_est = R_alignᵀ·R_z(θ)ᵀ·R_v,
  t_est = R_alignᵀ·R_zᵀ·(t_v − c_tpl − δ) + c_obs
  = list[PoseHypothesis] (k≤5, 점수순)

S4 (verify/)
  SymmetryHandler.dedupe(대칭 인식 병합) → 가설 전부 ICP 평가:
    ProjectiveICP.refine: projective association(스플랫 그리드 + win×win 3D 최근접,
      win = ⌈τ·fx/(Z·stride)⌉), Huber δ=1.5σ, τ 스케줄 0.25D→0.10D→0.05D→0.02D,
      trust region(회전≤5.7°/iter), 퇴화 판정 cond(H_p2pl)
    → 최종 p2pl 잔차 r_p2pl = 주 검증 신호
    HypothesisScorer(프레임 전체 유효 관측 대상): free_viol = (r > τ_eff) ∧ interior,
      τ_eff = 3σ + 셀 3×3 깊이범위(grazing 적응), 실루엣 1셀 침식, 홀 채움은 빈 셀만
  → 선택 = max(inlier_ratio × coverage − free_viol)   [그리디 first-ACCEPT 금지]
  → make_features 10-d → 로지스틱 p_conf → 하드가드(free_viol>0.15 REJECT,
     n_inlier<25 ∨ degenerate → ACCEPT 금지) → ACCEPT / UNCERTAIN / REJECT
  = VerifyResult{pose T_cam_obj[4,4], p_conf, verdict, diag}
```

`FrameBundle.object_points(mask)`가 마스크→포인트 추출의 표준 절차(erosion 2px 후 `pix2pt` 경유, `quality==0` 우선, 30pt 미만이면 EDGE_MIXED만 제외로 완화)로, S2·S4가 전부 이 경로를 쓴다.

---

## 2. 컨셉의 여섯 기둥

### ① "유효 기하 = LiDAR 포인트가 있는 곳만" — dense depth 비사용

- **결정**: 03 §1.3에서 (B) dense 보정, (C) depth completion을 명시 기각. 기각 논리가 이 프로젝트의 핵심 공리다: *completion은 RGB 외형에서 기하를 추론하므로 재도장되면 hallucinated depth도 바뀌고, 실패가 감지 불가 — S4 검증조차 통과할 수 있다.* 결측이 결측으로 나타나는 것(포인트 수를 셀 수 있음)이 LiDAR 채택의 본질적 이유.
- **각 스테이지 강제 효과**:
  - FrameBundle이 depth map이 아니라 **포인트 리스트 + 역인덱스(`pix2pt`)** 를 정본으로 삼음 — `sparse_depth`는 파생물(`frame_bundle.py:rasterize`).
  - S1: depth 이미지 분할이 아니라 3D 포인트 클러스터링(`prompt_gen.py:_voxel_clusters`), 프롬프트 대표점도 실측점만(`_representatives` 주석 "합성점 금지").
  - S2: dense 깊이 크롭 상관이 불가능하므로 **단방향(관측→템플릿) point-to-template TDF 정합**으로 설계(`depth_match.py`) — 단방향인 이유는 가림 불변, 파편 정합 부작용은 크기 게이팅이 차단(03 §5.2).
  - S4: 비교 지점은 "LiDAR 유효 픽셀뿐"(`scorer.py` 모듈 docstring), ICP는 source=희소 관측/target=CAD, 노멀은 CAD 측(`icp.py` docstring).
  - 포인트 수가 1급 시민: 하한 소유권 표(03 §2.5 — S1 15pt/S2 30pt/S4 inlier 25/재적분 트리거 64pt)가 스테이지별로 단 한 곳씩 자른다.
- **SAM-6D와 차이**: SAM-6D는 dense depth 전제(RGB crop + dense 역투영 포인트, ISM의 visible ratio도 dense depth 기반).
- **실측 근거(04)**: 물체 위 평균 ~763pt, 주사선 1/5 희소화, σ5mm 조건에서 9/10 ACCEPT 평균 1.9mm/0.23°.

### ② 외형 불신(A1)의 신호 가중 구현

- **SAM-6D의 문제(코드 확인)**: `Instance_Segmentation_Model/model/detector.py:384` — `final_score = (semantic_score + appe_scores + geometric_score*visible_ratio) / (1+1+visible_ratio)`. semantic·appearance 둘 다 DINOv2 RGB 특징 → **판별의 2/3가 외형 의존**.
- **Shape6D**: appearance 항 완전 제거, semantic은 "약한 prior"로 강등(가중 0.2 고정 + depth 결측 시 동적 캡 0.4, 03 §5.3). v0-geo 경로는 아예 RGB 0으로 동작. S4 검증 신호는 전부 GEO(잔차·inlier·coverage·free_viol — `confidence.py:FEATURE_NAMES` 10개 중 RGB 유래는 s2_sem 하나, 초기 가중 0.5로 최소). 오염이 형상을 바꾸는 경우(퇴적)는 부호 비대칭으로 처리 — `scorer.py` docstring: `r < −τ = 가림/오염(관대)`, `r > +τ = free-space 위반(강증거)`.
- **완전 제거가 아닌 강등인 이유**: 유사 형상 타 부품 혼입 시나리오에서만 RGB가 캐스팅보트(01 시나리오 매트릭스).
- **실측 근거**: 04 전체가 "A1의 극한형(RGB 0)" 검증 — 형상 신호만으로 스펙 통과.

### ③ 오프라인 전부 원칙 — 온라인 오브젝트 연산 0

- **SAM-6D의 문제(코드 확인)**: `Pose_Estimation_Model/model/pose_estimation_model.py:25~37` — 추론 forward 안에서 오브젝트 측 `sample_pts_feats`(FPS)와 `geo_embedding_o`를 **매 프레임 재계산**.
- **Shape6D**: 오브젝트 파생물 전량이 `onboard_v1.npz` (§1.1 표의 소비자 열이 곧 증명). FPS는 오프라인 전용으로 명시(`sampling.py:fps_indices` 주석 "O(N·n) numpy — 오프라인 전용, A4의 온라인 금지와 무관"). 온라인은 배포 금지 연산(kNN/KD-tree/커스텀 CUDA op)을 원천 회피: S1 voxel-hash BFS, S2 TDF O(1) gather, S4 projective association — 전부 표준 gather/scatter 구조(TRT 이식 전제, `depth_match.py` docstring).
- **부수 효과**: 온보딩이 GL 무의존(point-splat)이라 Mac에서도 완결, 모델 종속 필드는 `encoder_hash`로 2-pass 분리(캐시 무결성).

### ④ "판별의 argmax가 곧 coarse 포즈" — 검색과 초기화의 통합 (신규 컨셉의 핵심 착상)

- **착상**: S2 TDF 정합은 원래 "이 클러스터가 우리 물체인가"의 판별 점수 S_depth를 내는 연산인데, 그 최적화 변수 (뷰 v, in-plane θ, 병진 jitter δ)가 **이미 포즈의 파라미터화**다. `depth_match.py` docstring: "argmax (뷰, θ, δ)는 곧 무학습 coarse 포즈의 파라미터다". `template_init.py:coarse_poses_from_match`가 폐형식 유도(모듈 docstring에 유도 전문)로 T_cam_obj를 복원 — **별도 포즈 추정 모델 없이 판별 연산의 부산물로 초기 포즈를 얻는다**.
- **SAM-6D와 차이**: SAM-6D는 판별(ISM)과 포즈(PEM 104M 파라미터, 학습 필수)가 완전 분리된 2모델이고 특징도 공유하지 않는다. Shape6D v0-geo는 S3 없이 S2→S4 직결로 완결.
- **성립 조건**: coarse가 거칠어도(뷰 양자화 ≤17.8° + in-plane 7.5°) S4의 광역 τ 스케줄 ICP(0.25D 시작 — 양자화 오차의 수렴 반경 확보, `icp.py:refine` 주석)와 가설 k≤5 비교가 회복한다는 것.
- **실측 근거(04 §5)**: coarse 평균 235mm/39.8° → ICP 후 1.9mm/0.23°. "coarse는 거칠어도 된다"가 수치로 실증.

### ⑤ 검증 중심주의 — 판정은 물리 잔차와 증거 비교로

다섯 겹 구조, 전부 다중 시행 검증(04 §6)에서 결함으로 발견되어 확정된 것:

1. **주 잔차 = ICP p2pl(법선 투영)**: z-차이 잔차는 grazing 표면(팔레트 데크 상판)에서 픽셀당 수십 mm가 정상이라 정답 포즈도 대량 위반 판정(실측 free_viol 40%, 03 §11) → 스플랫 깊이 잔차는 free-space 검출 전용으로 축소(`verifier.py:_evaluate` 주석, `icp.py:refine` 말미 `r_p2pl`).
2. **frame-wide free-space**: 후보 포인트만 보면 near-대칭 오포즈(90°/플립)의 위반이 관측 밖이라 놓친다 — 프레임 전체 유효 관측 대상으로 검사(`verifier.py:_evaluate` 주석, `frame_obs` 인자). 적응 임계 τ_eff = 3σ + 3×3 깊이범위, 실루엣 1셀 침식, 홀 채움은 빈 셀만(`scorer.py`). 다공성 물체(팔레트 포켓)의 정당한 배경 관측 때문에 하드 REJECT가 아니라 **소프트 특징**(04 결함 5).
3. **가설 전부 평가 후 증거 비교**: 그리디 first-ACCEPT는 near-대칭 오포즈가 먼저 평가되면 수락되는 결함 실증(04 결함 2) → k≤5 전부 ICP 후 `inlier_ratio×coverage−free_viol` 최대 선택(`verifier.py:__call__`의 `_quality`). 플립은 coverage 0.42 vs 정답 0.99로 확실히 진다.
4. **sym-aware**: S0 대칭 정본(`sym_rots`/`sym_axes`)을 `SymmetryHandler`가 소비 — dedupe(등가 가설 병합, fine품 생존 우선), canonicalize(연속축 swing-twist 제거), sym_aware_error(ADD-S식 평가·라벨) (`symmetry_eval.py`). 대칭 물체 오REJECT 방지.
5. **정직한 REJECT**: 하드가드(free_viol>0.15 / inlier<25 / cond(H) 퇴화 → ACCEPT 금지) + 3단 verdict. 04에서 coarse top-5 전부 플립이었던 yaw 252° 1건을 REJECT — **오수락 0·오거부 0, verdict가 정오와 완전 일치**. 부수 증거: free-space 검증기가 합성 데이터의 물리 오류(가림 누락)까지 검출(04 결함 3·4).

SAM-6D 대비: SAM-6D의 판정은 매칭 신뢰도(pred_pose_score) 단일값이고 accept/reject 개념·대칭 등가 평가·free-space 검사가 없다.

### ⑥ 학습의 역할 재정의

- **04 결론**: 확정 운용 조건의 클린 합성에서 무학습 기하 파이프라인만으로 스펙을 평균 5배 여유로 달성, 실패는 전량 자기 검출. 따라서 M2 학습(Shape6D-PEM: 기하 주도 인코더 + SAM-6D 매칭 구조 승계)의 역할은 "성능 달성"이 아니라 **coarse 강건화(플립·근대칭 리콜 — 04의 유일 실패 모드)와 저품질 조건 확장**.
- **구조 반영**: `template_init.py` docstring — "v0-geo 경로의 S3 대체재이자, Shape6D-PEM(M2) 도입 후에도 비학습 폴백으로 유지". 학습 모델이 실패해도 시스템이 무학습 하한 성능으로 후퇴할 뿐 죽지 않는다.
- **SAM-6D와 차이**: SAM-6D는 학습된 PEM이 포즈의 유일 경로 — 학습 분포 밖에서 폴백이 없다.

---

## 3. 데이터 흐름 표 (한 장)

| 단계 | 입력 | 처리 | 출력 | 근거 파일 |
|---|---|---|---|---|
| S0 샘플링 | CAD mesh | Poisson disk 16384 + face 노멀, FPS 서브셋(2048⊃196, 1024, 2048) | `pts_master`,`nrm_master`,`idx_*` | `onboarding/sampling.py` |
| S0 템플릿 | 표면 샘플 [M,3] | icosphere 42뷰 point-splat depth, 가시 512pt, 뷰별 TDF 48³ | `tpl_depth/pose/K/pts/center`,`tdf` | `onboarding/templates.py` |
| S0 대칭 | 마스터 샘플(+mesh) | 축 후보(PCA·관성·축퇴시 전구면) → 스캔 → 군 폐포, 노이즈 바닥 임계 | `sym_rots`,`sym_axes` | `onboarding/symmetry.py` |
| S0 캐시 | 위 전부 | 스키마 검증 + manifest(sha256, encoder_hash) | `onboard_v1.npz` | `onboarding/cache.py` |
| 취득 | RGB, LiDAR 원시(LiDAR계) | 좌표 변환→투영→z-buffer 래스터→품질 플래그 | `FrameBundle` (pix2pt, point_quality 포함) | `common/frame_bundle.py:build_frame_bundle` |
| S1 | FrameBundle | in-image 필터→RANSAC 평면 제거→voxel-hash BFS 클러스터→분할/대표점 | `list[Cluster]`,`list[PromptSet]` | `proposal/prompt_gen.py` |
| S2-① | Candidate.pts, D_cad | 5–95 percentile extent 게이팅(상한 하드/하한 β_occ/N≥30) | 생존 Candidate + S_size | `identify/size_gate.py` |
| S2-② | pts_cam [N,3], tdf, tpl_center | median 센터링→ray→z 정렬→pass1(25뷰×12θ)→pass2(top3×24θ×jitter) | `MatchResult{s_depth, topk(v,θ,δ)}` | `identify/depth_match.py` |
| coarse | MatchResult, tpl_pose, tpl_center | 폐형식 복원 R=R_aᵀR_zᵀR_v, t=R_aᵀR_zᵀ(t_v−c−δ)+c_obs | `list[PoseHypothesis]` k≤5 | `pose/template_init.py` |
| S4 ICP | 가설, obs_pts, pts/nrm_master | projective p2pl, Huber 1.5σ, τ 0.25D→0.02D, trust region, cond(H) | 정련 R,t + `r_p2pl`, coverage, degenerate | `verify/icp.py` |
| S4 스코어 | R,t, idx_verify 2048pt, 프레임 전체 관측 | 스플랫 z-buffer→적응 임계→침식 내부 free_viol(소프트) | score, coverage, free_viol | `verify/scorer.py` |
| S4 판정 | 전 가설 평가 결과 | sym dedupe→전부 평가→inlier×cov−free_viol 선택→10-d 로지스틱→하드가드 | `VerifyResult{T, p_conf, verdict}` | `verify/verifier.py`, `confidence.py`, `symmetry_eval.py` |

---

## 4. 실측 근거 요약 (04, 전부 결정론적 재현 — 시드 고정)

| 항목 | 수치 |
|---|---|
| 조건 | T11 팔레트 근사 CAD, 2.5~3.5m, 주사선 1/5 희소화, 물체 위 평균 ~763pt, σ=5mm, 10시행 |
| coarse (템플릿 argmax) | 평균 235mm / 39.8° — ICP가 회복 가능한 수준이면 충분 |
| 최종 (ACCEPT 9건) | 평균 1.9mm / 0.23°, 최대 4.9mm / 0.71° (스펙 10mm/1°) |
| verdict 신뢰성 | 오수락 0 · 오거부 0, 플립 1건(yaw 252°) 정직 REJECT |
| 오포즈 분리 증거 | 플립 coverage 0.42 vs 정답 0.99 |
| 레이턴시 | numpy CPU 레퍼런스 ~1.1s/프레임(가설 5개 전체 ICP 포함) — TRT 이식(M1·M4) 대상 |

주의 표기: p_conf는 미보정 휴리스틱 가중(`confidence.py:_W0`) — verdict 일치는 하드가드+증거 비교의 결과이며 확률로서는 M3-3 캘리브레이션 전까지 무의미(04 §7). σ_lidar 8mm는 스펙시트 중간치 추정으로 D2-a 실측 대체 예정. 클린 합성 한계(실 센서 바이어스·mixed pixel·캘리브레이션 오차 미반영)는 04 §7에 명시되어 있다.