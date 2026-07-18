# 검증 관점 1: 스테이지 경계 일관성 — 위반/불일치 목록

코드 대조 확인: `/Users/jaewoo/Documents/SAM-6D/SAM-6D/Pose_Estimation_Model/config/base.yaml` (train `n_sample_model_point: 2048` / test `1024` — s3pem·training 양쪽 인용이 각각 test/train 설정을 가리켜 표면상 모순은 아님), `utils/model_utils.py` L237–246 (s3pem의 개조 지점 인용 정확).

---

## High

**H1. FrameBundle 이중 정의 — 필드명·타입·모듈 경로 전면 충돌** ([sensor] §5 vs [s0s1] §3)
- [sensor]는 `shape6d/common/frame_bundle.py`에 `lidar_points`/`lidar_pixels`/`K: CameraIntrinsics(dataclass)` + `sparse_depth`/`valid_mask`/`pix2pt`/`point_quality`를 정의. [s0s1]은 `shape6d/types.py`에 `lidar_xyz`/`lidar_uv`/`K: np.ndarray[3,3]`만 있는 별개 FrameBundle을 정의.
- 실질 피해: [sensor]의 스테이지 소비 규약("S1은 valid_mask를 전경 프롬프트로 소비", "S2는 pix2pt로 마스크→포인트 매핑, quality==0 우선")이 [s0s1]의 S1 구현 인터페이스에서는 **필드 자체가 없어 성립 불가**. [s2s4]도 `uv ∩ m_j` 역투영을 직접 기술해 pix2pt를 쓰지 않음.
- 수정: [sensor] 정의를 단일 정본으로 확정하고 [s0s1] §3 FrameBundle 삭제·참조 치환, [s2s4] §1.0의 포인트 추출을 `pix2pt` 경유로 재기술.

**H2. S0 캐시 스키마 3중 분열 + dense 포인트 수 3원화 (4096 vs 16384 vs 2048)**
- [s0s1] §1.6 `onboard_v1.npz`: `pts_dense [4096,3]` (2048의 2배 근거까지 제시). [s2s4] §3 `cache.npz`: `dense_pts (16384,3)` (ICP target), `verify_pts (2048,3)`. [s3pem] §4 `pem_cache.npz`: `dense_po (2048,3)`. [training] §1.3: `pts_dense [2048,3]`. **같은 Poisson disk 샘플이 설계마다 4096/16384/2048로 다르며 서로 인용 관계도 없음.**
- 누락 필드 열거 — S0 소유 설계([s0s1] §1.6)가 생산하지 않는 하류 소비 필드:
  - [s2s4] 소비: `tdf (42,48,48,48)`, `tpl_center (42,3)` (median 센터링 기준 — [s0s1]에는 캐시 항목 자체가 없음), `verify_pts`, `dense_pts/normals (16384)`.
  - [s3pem] 소비: `dense_fo`, `geo_embedding_o (197,197,H)`, `pe_fo`, `radius`, `fps_idx_o`, `model_pts`. ([s0s1]의 `geo_feat_dense [4096,C_g]`는 shape·의미 모두 불일치.)
- 수정: 단일 `onboard_v1.npz` 스키마로 통합하고 dense 포인트 수를 1개 값(+ 서브셋 인덱스)으로 확정. [s2s4]의 16384는 레이턴시 표(§4 "scatter 16k")까지 연동 재산정 필요.

**H3. DINOv2 템플릿 descriptor 12뷰 vs 42뷰 직접 모순**
- [s0s1] §1.4·§1.6: RGB 템플릿 12뷰만 렌더, `dino_cls [12,384]` ("A1에 따라 최소화"). [s2s4] §1.3·§3: `dino_cls (42,384)`, "S0에서 42뷰 클린 CAD 렌더의 cls", `S_sem = mean(top5(cos))` — top5/42와 top5/12는 통계 특성이 다름(12뷰면 avg_5가 상위 42% 집계가 됨). 어느 쪽이든 한쪽 설계·튜닝(θ_S2=0.45, w_s=0.2)이 무효화된다.

**H4. 희소 포인트 가변성 처리 메커니즘 — 학습·추론 정면 상충**
- [training] §2-①: "**복원추출 대신 제로패딩 + n_valid 전달** — 복원추출(SAM-6D 방식)은 중복점이 attention 통계를 왜곡", 인터페이스에 `valid_pad_mask[2048]` 포함. [s3pem] §1.1: **복원추출+jitter를 권장 채택**하고 "패딩+key_mask는 LinearAttention(`transformer.py:518-564`)에 마스크 경로가 없어 개조 필요 — 기각"이라고 명시. 두 설계가 서로의 채택안을 상대방의 기각 근거로 부정한다. 이대로면 학습은 패딩 분포, 추론은 중복점 분포 — 도메인 갭을 파이프라인이 스스로 만든다.
- 수정: 한 메커니즘으로 통일(어느 쪽이든 무방하나 [s3pem]의 TRT 정적 shape 논거가 더 구체적이므로 training 쪽을 복원추출로 맞추거나, LinearAttention 마스크 개조를 정식 과제로 승격).

**H5. modality dropout 구현 상충 — 0-치환 vs 학습 임베딩**
- [s3pem] §3: `F_rgb ← 0` + `rgb_valid_flag ← 0` (출력 0 치환). [training] §2-④: `f_rgb = m·f_rgb + (1−m)·e_norgb` — **학습되는 "RGB 없음" 임베딩**을 사용하고 "추론 시 RGB 결측 프레임에도 동일 임베딩 사용 — 학습·추론 일관"이라 명시. e_norgb ≠ 0이므로 두 설계의 추론 그래프가 다르다. 게이트(`g·F_rgb`, [s3pem])와 e_norgb의 상호작용도 미정의(게이트가 e_norgb까지 곱하는가?).

**H6. 최소 포인트 수 하한 3원화 (64 vs 30 vs 25) — 게이트 의미 모순**
- [sensor] §3.3: `N_min_reject=64` "미만이면 S4에서 reject 또는 재적분 트리거". [training] §1.1·config: `min_valid_obj_points=64` (미만 샘플은 학습에서 재추첨 = **학습 분포에 64 미만이 존재하지 않음**). 그러나 [s2s4] §1.1: `N_min=30`으로 S2 통과 허용 + §5.1 구제 경로는 **N_j<30조차** S_size+S_sem으로 통과시킴, S4는 `N_inlier≥25`로 별도 하한.
- 결과: 30~63pt 구간은 [sensor] 규약상 reject 대상이면서 [s2s4]에서는 정상 동작 구간이고, 학습은 이 구간을 본 적이 없다(희소화 하한 64 + N_obj~U[100,2000]이라 100 미만도 사실상 미학습). 하한 값과 "누가 어디서 자르는가"를 단일 표로 확정할 것.

**H7. coarse 196 처리 — 가변 길이(패딩 마스크) vs 정적 shape(중복 인덱스) 모순**
- [sensor] §3.3 `N_target_coarse`: "관측 포인트가 196 미만이면 다운샘플 없이 전 포인트 사용 (**매칭 transformer는 가변 길이 허용, 패딩 마스크로 처리**)". [s3pem] §1.1: `fps_idx_m`을 **중복 인덱스로 196 고정**(TRT 정적 shape, A4 논거), 패딩안 기각. [sensor]의 문구는 [s3pem]이 기각한 안을 확정 사실처럼 기술 — [sensor] 쪽 문구를 "중복 샘플로 196 고정"으로 정정해야 함.

## Med

**M1. 기하 인코더 입력 채널 수 불일치 (10ch vs 7ch)**
- [s3pem] §1.1·§2(a): `geo_maps` 8ch(XYZ+노멀+유효마스크+정규화 깊이) + flags 2ch = 10ch 입력. [training] config: `in_channels: 7 # XYZ(3)+normal(3)+valid_mask(1)`, 데이터셋 반환도 `grid_xyz[7,224,224]`. 정규화 깊이 채널과 flag 브로드캐스트 채널이 학습 데이터 경로에 없음 — 이대로 구현하면 학습된 인코더가 추론 입력 규약과 shape부터 안 맞는다.

**M2. point_quality 플래그·마스크 erosion이 하류 설계에 미반영**
- [sensor] §4.2는 "S2/S3 메트릭 연산은 flags==0 우선, S4 ICP는 EDGE_MIXED 제외 기본", §4.4는 "마스크 2~3px erosion 후 포인트 선별을 표준 절차로"를 규정하고, EDGE_MIXED 필터가 소형 물체 유효 포인트를 추가 20~40% 깎는다고 추정까지 했다. 그러나 [s2s4] `Candidate`에는 quality 필드가 없고 §1.0 포인트 추출·§2.2 ICP 어디에도 필터 단계가 없으며, [s3pem] 전처리(`P = lidar_pts[in_mask]`)도 erosion·quality 미언급. 특히 20~40% 감소분은 H6의 N_j 하한·[sensor] §3 포인트 수 예산 어디에도 반영되지 않았다(예: 실전 기대 ~160pt가 필터 후 ~100pt로 내려가 N_geo_ok=150 미달).

**M3. 카메라 FOV 밖 LiDAR 포인트(NaN uv) 처리 부재**
- [sensor]는 `lidar_pixels`의 이미지 밖 포인트를 NaN으로 정의. [s0s1] §2.2 클러스터링은 **전체 lidar_xyz**를 사용하므로(LiDAR FOV가 카메라 FOV보다 넓은 기종 다수 — [sensor] §3.2의 Avia 70°×77° vs 카메라 HFOV ~70°) 클러스터 대표점·hull 정점에 NaN uv가 유입될 수 있고, `reps_uv*0.4` SAM 프롬프트와 `cv2.fillConvexPoly`가 그대로 실패한다. 클러스터링 전 in-image 필터 또는 대표점 선정 시 NaN 배제 규칙 명시 필요.

**M4. 대칭 표현 포맷·연속축 규약 4분열**
- [s0s1]: `sym_rots [S,3,3]` + `sym_axes [A,3]` (npz, 연속축은 축 벡터). [s2s4]: `symmetry.json`(이산군+연속축) — `canonicalize`가 `cont_axes` swing-twist를 요구. [s3pem]: `sym_group (S,3,3)`**만** — 연속축 필드가 없어 무한대칭 물체에서 S4·대칭 NMS로 전달이 끊긴다. [training]: 연속축을 12분할 이산화해 `G≤16`에 포함. 포맷(npz vs json)·연속축 취급(벡터 유지 vs 12분할 흡수)을 단일 규약으로 확정하고, 학습 라벨(12분할)과 S4 등가 평가(연속 swing-twist)의 정합성을 명시할 것.

**M5. S3→S4 가설 품질 비대칭 미인지**
- [s3pem] §5.3: fine은 best 가설만 통과, S4로 가는 가설 1..k-1은 **coarse 포즈 그대로**. [s2s4] §2.1·§2.4는 k개 가설을 동질로 취급해 dedupe(geodesic<15° & Δt<0.05D) 후 최상만 ICP — fine 결과와 그 기원 coarse 가설이 15° 이내면 병합되어 실질 가설 수가 줄고, UNCERTAIN 재시도는 정련 안 된 coarse 포즈에서 ICP를 시작한다(수렴 반경 초과 위험). [s2s4]에 "가설 0=fine 정련품, 1..k-1=coarse"라는 입력 명세와 재시도 시 ICP iter 증가 등 대응 규칙 필요.

## Low

**L1. Proposal→Candidate 변환 계층 미정의**: [s0s1] `Proposal(lidar_idx, truncated 메타, source, cluster_id)` → [s2s4] `Candidate(pts, uv, flags={"border",...})` 매핑(truncated→border, lidar_idx→pts 역투영, source별 score 취급)을 어느 모듈이 수행하는지 양쪽 다 미기술.

**L2. 병진 dedupe/NMS 임계 이원화·단위 모호**: [s3pem] `select_topk_hypotheses`의 `trans_thresh=0.3`은 radius 정규화 좌표인지 미터인지 불명(정규화 공간이면 0.3·radius, 미터면 30cm — 20배 차이), [s2s4] dedupe는 `0.05·D_cad`로 별도 정의. 같은 목적의 임계가 두 곳에서 다른 단위 체계로 존재.

**L3. S2 semantic crop 도메인 불일치**: [s2s4]는 관측 crop의 마스크 밖을 ImageNet mean으로 치환하는데, 템플릿 cls([s0s1] 렌더)의 배경 색 규약이 미정의 — 배경 도메인이 다르면 cls cosine에 계통 편향. 템플릿도 동일 mean-fill로 렌더/마스킹하도록 명시 필요.

**L4. 필드 명명 오류**: [s0s1] `Proposal.mean_depth`가 실제로는 `median(z)` — 소비 측 혼동 방지를 위해 개명 권장.

## A1~A4 점검 결과
- 커스텀 CUDA op 잔존: 온라인 경로에서는 미발견([s0s1] S0의 FPS는 오프라인 허용 범위, [s3pem] §8이 pointnet2 제거를 코드 위치까지 특정). 단 [s2s4]의 `scatter_argmin_z`/`gather_nearest3d`는 표준 op 조합이라는 근거가 [s3pem] §8 수준으로 명시되어 있지 않음 — TRT/torch 표준 op 매핑 표에 편입 권장(Low).
- 물체별 학습 혼입·render-and-compare 반복 루프: 미발견(S4는 포인트 스플랫 잔차 + ICP 1회+재시도 1회로 A3 범위 내).
- A1 관련: [training] §4.1 SAM-6D RGB teacher의 KL distill은 "A1 위배 편향 주입 위험"을 스스로 명시하고 조건부 취급 — 위반 아님, 감시 항목으로 유지.