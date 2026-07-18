# Shape6D vs 참조 방법들 — 데이터 처리 방식 정면 비교

근거: SAM-6D는 코드 직접 확인(/Users/jaewoo/Documents/SAM-6D/SAM-6D/), Shape6D는 코드 직접 확인(/Users/jaewoo/Documents/Shape6D/). 그 외 방법은 논문 기반 지식 — 불확실 항목은 "확인 필요" 표기. 수치 82.1/75.9/73.4 등은 임무 지시문에 주어진 값을 그대로 사용.

---

## 1. 축별 비교표

### 표 1a — 데이터 표현·의존성 (① ~ ⑤)

| 방법 | ① 기하 데이터 표현 | ② 외형 의존도 | ③ 희소 depth 내성 | ④ 온보딩 비용·산출물 | ⑤ 온라인 오브젝트 측 연산 |
|---|---|---|---|---|---|
| **Shape6D (v0-geo)** | 희소 LiDAR 포인트 그대로 + 뷰별 TDF 48³ LUT(`templates.py:68-77`) + point-splat depth 템플릿(`templates.py:92-125`). dense depth 무가정 | **0** (v0-geo는 RGB 미사용; DINOv2 prior는 M1 옵션) | **설계 전제** — 물체당 수백 pt에서 검증됨(04: ~800pt·σ5mm에서 1.9mm/0.23°) | CPU 수 분, GL 무의존. onboard_v1.npz 단일 캐시(마스터 16384+서브셋 인덱스, 42뷰 depth·TDF·대칭, `sampling.py:6-10`, `cache.py:1-5`) | TDF gather + 산술뿐: 42뷰×12θ pass1 → top3뷰×24θ×125 jitter pass2 (`depth_match.py:83-106`). 신경망 forward 없음 |
| **SAM-6D** | dense depth→마스크 내 2048pt 리프팅(`run_inference_custom.py:184`, `data_utils.py:92-110`; `n_sample_observed_point: 2048`, base.yaml:62) + 픽셀 정렬 RGB 특징 | **높음**: ISM은 DINOv2 semantic+appearance 점수(`detector.py:260-308`), PEM은 ViT-MAE RGB 특징이 매칭의 주 신호(`feature_extraction.py:122-181`) | **낮음**: 마스크 픽셀↔depth 1:1 정렬 가정. 수백 pt에서는 choose 샘플링·패치 특징 정렬이 성립 안 함 | BlenderProc PBR 42뷰 렌더(Render/render_custom_templates.py) + DINOv2 descriptor 캐시. GPU 필요 | ViT 인코더 2회(관측+템플릿) + Geometric Transformer 3블록×2단(coarse 196pt/fine 2048pt, base.yaml:17-18) |
| **FreeZe (v2.1)** | dense depth 포인트 클라우드, GeDi 기하 디스크립터 + DINOv2 특징 융합 | 중간: DINOv2가 절반, GeDi 기하가 절반 — 융합이라 외형 열화에 부분 내성 | 낮음 추정: GeDi는 국소 패치 밀도를 요구 (수백 pt에서의 동작 보고 없음 — 확인 필요) | CAD 포인트에 GeDi+DINOv2(렌더 뷰) 특징 사전계산. GPU 필요 | 양측 디스크립터 추출 + 3D-3D 대응 + RANSAC 정합 — 24.9s/img의 주범 |
| **GigaPose** | RGB 템플릿 패치 특징 (기하는 간접 — 템플릿 뷰 이산화). depth는 병진 스케일 보정에 선택적 사용(확인 필요) | 높음: 학습된 패치 특징이 전부 RGB 기반 | 해당 없음(coarse가 RGB): depth 붕괴에는 강하나 기하 검증 수단도 없음 | 162뷰 템플릿 렌더+특징 캐시(뷰 수 확인 필요) | 패치 대응 2쌍→유사변환(스케일·in-plane·2D 병진) 폐형해 — coarse 자체는 수십 ms급 |
| **Co-op (CVPR25)** | RGB 렌더-쿼리 반밀집(semi-dense) 대응 + 확률적 flow. depth 사용 여부·정도 확인 필요 | 높음(RGB 대응 기반) | 해당 없음~낮음 (확인 필요) | 템플릿 렌더 + 학습 모델 (세부 확인 필요) | 대응 네트워크 + flow 정련 네트워크, 0.8s/img |
| **FoundationPose** | RGBD crop vs 신경 렌더 비교(모델 기반) 또는 neural object field(모델 프리) | 중간~높음: refiner/ranker 입력이 RGBD — RGB와 depth 공동 사용 | 낮음: 렌더-관측 비교가 dense depth crop 전제 | 모델 기반은 렌더 준비 수준, 모델 프리는 참조 이미지로 field 학습(분 단위, 확인 필요) | 가설 다수 렌더 → transformer refiner 반복 + 계층 랭킹 — 29.3s/img |
| **MegaPose** | RGB(-D) crop vs 다수 후보 포즈 렌더 | 높음(RGB 주도; RGBD 변형 존재) | 낮음 | 렌더러 준비만(사전 2M 합성 이미지로 1회 학습) | coarse 분류기: 후보 뷰 수십 장 렌더·스코어 + DeepIM식 반복 refiner — 렌더 횟수가 병목 |
| **PPF/Drost (HALCON)** | 씬 포인트+**노멀** 쌍 특징 (d, ∠n₁d, ∠n₂d, ∠n₁n₂) 해시 | **0** (순수 기하) | **낮음**: 씬 노멀 추정과 쌍 통계가 밀도 요구 — 수백 pt·격자 패턴에서 붕괴 위험 높음(추정) | CAD 쌍 특징 해시 테이블, CPU 수 초~분. 산업적으로 가장 가벼움 | 참조점별 Hough 투표 + 클러스터링 + ICP — CPU에서 성숙 |
| **OVE6D** | depth crop(마스크 필요) → 학습된 뷰포인트 latent 코드 | **0**(depth-only) — 단 마스크는 외부(RGB) 검출기 필요 | 중간: 인코더가 dense depth 이미지 학습 — 희소 스플랫 입력은 분포 밖(확인 필요) | 뷰포인트 코드북 1회 생성(수천 뷰 렌더·인코딩; ~4000뷰는 확인 필요). 인코더는 ShapeNet 1회 학습으로 전 물체 공용 | 인코딩 1회 + 코사인 검색 + in-plane 회귀 + 검증 헤드 — 매우 가벼움 |
| **CNOS/MUSE** (검출 전용) | 기하 없음 — RGB 프로포절 vs 렌더 템플릿 CLS 매칭 | 전부 | 해당 없음 | 42뷰 렌더 + DINOv2 descriptor | SAM 프로포절 전수 + descriptor 비교. MUSE 세부 확인 필요 |

### 표 1b — 파이프라인 구조 (⑥ ~ ⑩)

| 방법 | ⑥ coarse 획득 | ⑦ 정련 | ⑧ 자기 검증/신뢰도 | ⑨ 엣지 배포성 | ⑩ 제로샷 메커니즘 |
|---|---|---|---|---|---|
| **Shape6D** | **무학습**: S2 TDF argmax(뷰,θ,δ)를 그대로 포즈로 복원(`template_init.py:19-31`) + 직립 뷰 프루닝 25/42(`templates.py:43-47`) | projective p2pl ICP, CAD 노멀, τ 0.25D→0.02D, trust region(`icp.py:75-125`) | **가장 두터움**: frame-wide free-space(`scorer.py:1-5`), 전 가설 ICP 후 inlier×coverage−free_viol 비교 선택(`verifier.py:88-99`), cond(H) 퇴화 검출(`icp.py:132-136`), 10-d 로지스틱 + ACCEPT/UNCERTAIN/REJECT(`verifier.py:56-66`) | 설계 공리(A4): gather+GEMM만, KD-tree 불사용(`prompt_gen.py:1-4`), v0-geo는 GPU 없이도 동작 | 순수 기하 템플릿 — CAD만 있으면 학습·렌더팜 없이 온보딩 |
| **SAM-6D** | 학습된 soft 대응 → 6000→300 가설 weighted SVD(`coarse_point_matching.py:70-74`, base.yaml:41-42) | 학습된 fine matching(2048pt dense soft 대응 SVD, `fine_point_matching.py:75-81`) — ICP 없음 | pred_pose_score 1개(`fine_point_matching.py:81`) + ISM의 sem×appe×geo 점수(`detector.py:310-322`). 거부 개념 없음 | ViT 2개+pointnet2 커널 — Orin에서 무거움(추정) | 대규모 합성 학습의 일반화 + 템플릿 |
| **FreeZe** | 특징 대응 RANSAC 정합 | ICP + 대칭 인지 정련(SAR) | 명시적 신뢰도 출력 없음(확인 필요) | 24.9s — 엣지 부적합 | 사전학습 특징(GeDi+DINOv2)만으로 무학습 |
| **GigaPose** | 템플릿 검색(out-of-plane) + 패치 2쌍 폐형해(in-plane·스케일) | 자체 없음 — 외부 refiner(MegaPose/GenFlow) 결합 관행 | 매칭 점수 수준 | coarse는 가볍지만 refiner 포함 시 무거움 | 대규모 합성 학습 + 템플릿 |
| **Co-op** | 반밀집 대응 기반(세부 확인 필요) | 확률적 flow 반복 | flow 확실성(certainty) 추정 보유(확인 필요) | 0.8s는 데스크톱 GPU 기준 — Orin 수치 확인 필요 | 대규모 합성 학습 |
| **FoundationPose** | 병진 추정 후 회전 가설 전역 샘플 → 랭킹 | render&compare transformer 반복 | 계층 랭킹 점수(가설 상대 비교 — 절대 신뢰도 아님) | 29.3s — 부적합 | 대규모 합성 학습 + 통합 모델 |
| **MegaPose** | 후보 렌더 분류 | DeepIM식 render&compare 반복 | coarse 분류 점수 수준 | 렌더 다수 — 부적합 | 대규모 합성 학습 |
| **PPF/Drost** | Hough 투표 피크 | ICP | 투표 점수·클러스터 크기 (보정 안 됨) | CPU 성숙 — 산업 실적 최고 | 무학습 기하 — 본질적 제로샷 |
| **OVE6D** | 코드북 코사인 검색(뷰) + in-plane 회귀 | 선택적 ICP | 학습된 검증 헤드로 가설 랭킹 | 코드북·인코더 소형 — 양호 | 물체 무관 인코더 1회 학습 + 물체별 코드북 |
| **CNOS/MUSE** | (포즈 없음) | — | 매칭 점수 | SAM이 병목 | 템플릿 descriptor 매칭 |

---

## 2. 각 방법에서 가져온 것 / 버린 것 / 새로 만든 것

- **SAM-6D**: [가져옴] 42뷰 icosphere 템플릿 규약(`templates.py:19-40` 주석에 명시), 2단 coarse→fine 구조 관념, 마스터+서브셋 포인트 체계(N_PEM 2048 = PEM 승계 대비, `sampling.py:6-10`), M2의 매칭 구조 승계 계획. [버림] dense depth 리프팅 전제, RGB 특징 주도 매칭(A1 위배), 거부 없는 단일 점수. [신규] 템플릿을 RGB-D 렌더가 아닌 TDF LUT로 변환해 무학습 정합 대상으로 만든 것.
- **FreeZe**: [가져옴] "기하+의미 이원 신호" 구도(단, Shape6D는 의미를 M1 prior로 강등) 및 training-free 지향. [버림] GeDi식 국소 디스크립터(희소에서 성립 불가)와 초 단위 예산. [신규] 없음(방향 확인용 참조).
- **GigaPose**: [가져옴] "coarse = 검색 + 저차원 폐형해" 분해 사상 — Shape6D의 (뷰, θ, δ) argmax→포즈 복원(`template_init.py`)은 이 사상의 기하 버전. [버림] RGB 패치 대응 자체. [신규] 대응 없이 point-to-field 점수만으로 같은 분해를 달성.
- **Co-op**: [가져옴] "빠른 coarse + 확실성 인지 정련" 예산 철학. [버림] RGB 대응·flow(A1). 세부는 M2 설계 시 재검토 대상(확인 필요).
- **FoundationPose/MegaPose**: [가져옴] "가설 다수 → 렌더 비교 → 랭킹" 관념 — Shape6D의 전 가설 ICP 후 증거 비교(`verifier.py:88-99`)는 이를 렌더 없는 스플랫 잔차로 치환한 것. [버림] 온라인 신경 렌더링 전부(SLA·A4 위배). [신규] 랭킹 지표를 free-space 위반이라는 물리 증거로 구성(`scorer.py` 부호 규약 r>+τ=위반).
- **PPF/Drost**: [가져옴] "무학습 기하만으로 coarse가 가능하다"는 존재 증명과 ICP+검증의 산업 관행. [버림] 씬 노멀 의존 쌍 특징·Hough 투표(희소·격자 LiDAR에서 통계 붕괴). [신규] 투표 대신 뷰 이산화 + TDF 조회로 노멀 무요구화.
- **OVE6D**: [가져옴] 뷰포인트 검색 + in-plane 분해라는 골격, 물체별 경량 코드북 사상. [버림] 학습된 latent 인코더(dense depth 이미지 전제). [신규] latent 대신 명시적 TDF — 학습 0, 희소 입력 그대로 소비, δ(병진)까지 탐색 변수로 포함.
- **CNOS/MUSE**: [가져옴] proposal→템플릿 매칭 구도와 SAM 계열 활용 계획(M1 EViT-SAM). [버림] RGB descriptor 단독 판별(A1) — Shape6D S2는 크기 게이트(`size_gate.py`)+TDF가 1차 판별. [신규] LiDAR 클러스터를 프롬프트로 쓰는 S1(`prompt_gen.py`) — 검출 시드 자체를 기하에서 시작.

---

## 3. 정직한 열세 분석 — 그들이 Shape6D보다 잘하는 것

| 방법 | Shape6D 대비 우위 |
|---|---|
| SAM-6D | dense depth가 실제로 있을 때의 활용도: 2048pt 밀집 대응 + 학습 특징은 저텍스처가 아닌 일반 물체·잡동사니 씬에서 상위. BOP 전 도메인 일반화 실증. Shape6D의 42뷰×θ 이산화(양자화 ~20°)보다 연속적 coarse. |
| FreeZe | 정확도 상한(82.1 AR, BOP24 1위). 특징 대응 기반이라 뷰 이산화 오차 자체가 없음. 예산이 허락되면 오프라인 정답 생성기·감사 도구로 Shape6D가 역이용할 대상. |
| GigaPose | coarse 속도와 RGB-only 동작 — depth가 아예 없는 프레임에서도 산다. Shape6D는 LiDAR 결손 시 전면 불능. |
| Co-op | RGB 기반으로 75.9@0.8s — 속도·정확도 균형이 학습 기반 중 최상급. 확률적 flow의 서브픽셀 정련은 회전 정밀도에서 우위 가능(확인 필요). |
| FoundationPose | model-free 모드(CAD 없는 물체를 참조 이미지만으로 온보딩) — Shape6D의 A2(CAD 필수)가 못 가는 영역. 텍스처·반사 물체 강건성도 학습으로 흡수. |
| MegaPose | 생태계 성숙(refiner로 타 방법과 조합되는 표준 부품). |
| PPF/HALCON | 20년 산업 검증·인증·툴체인·지원 — 신뢰성 실적 자체가 기능. 클러터·부분 가림에서의 파라미터 튜닝 노하우 축적. 마스크·검출기 없이 씬 전체에서 동작. |
| OVE6D | 학습된 인코딩이라 뷰 유사도가 부드러움(코드북 수천 뷰 밀도) — Shape6D 42뷰보다 out-of-plane 해상도가 촘촘. 검증 헤드가 학습됨. |
| CNOS/MUSE | RGB만으로 인스턴스 분리 — 근접·접촉 물체 분리는 LiDAR 클러스터(S1)가 원리적으로 못 하는 것(M1에서 EViT-SAM으로 보완 예정인 이유). |
| 공통 | Shape6D는 아직 합성 검증(04, 팔레트 1종·10시행)뿐 — 실기 LiDAR·BOP류 벤치마크 실적 0. 모든 비교 우위 주장은 그 한계 안에서만 유효. |

---

## 4. 가장 가까운 이웃 판별

**결론: OVE6D가 컨셉상 최근접이다.** 구조가 동형이다 — (a) 오프라인: 구면 뷰 렌더→물체별 경량 표현, (b) 온라인: 관측 depth→뷰포인트 검색→in-plane 분해→검증. Shape6D의 S0 42뷰 템플릿+S2 뷰 검색+θ 탐색+S4 검증은 이 골격을 그대로 공유한다.

**결정적 차이 3가지:**
1. **표현이 latent가 아니라 명시적 거리장**: OVE6D는 ShapeNet 학습 인코더의 latent 코드 코사인 유사도로 뷰를 찾는다 — dense depth crop이 입력 분포다. Shape6D는 학습 0의 TDF LUT에 희소 포인트를 직접 넣는다(`depth_match.py:73-76`) — 수백 pt·격자 패턴이 곧 정의역이고, 신뢰도 판단의 물리적 해석(거리 mm)이 보존된다.
2. **병진이 검색 변수**: OVE6D는 병진을 마스크/depth에서 별도 추정하지만, Shape6D는 δ(±2 voxel jitter)를 argmax 변수에 포함하고(`depth_match.py:93-106`) 그 (뷰,θ,δ)를 곧바로 SE(3)로 복원한다(`template_init.py:19-31`) — coarse가 완결된 6-DoF 출력.
3. **검증의 성격**: OVE6D의 검증은 학습된 가설 랭커. Shape6D는 frame-wide free-space·coverage·cond(H)라는 비학습 물리 증거로 ACCEPT/UNCERTAIN/REJECT를 내린다(`verifier.py:56-99`) — A3(최초 사이클 견고성)의 요구가 만든 차이.

**PPF는 차점**: "무학습·기하만·CAD 온보딩·ICP·산업 지향"이라는 가치관은 가장 가깝지만, 데이터 처리 방식이 다르다 — PPF는 쌍 특징의 전역 투표(뷰포인트 개념 없음, 씬 노멀 필수, 밀도 의존 통계)이고 Shape6D는 뷰 이산화 + 점-대-장(point-to-field) 정합(노멀 불요, 점 하나하나가 독립 증거)이다. 희소 LiDAR라는 전제에서 이 차이가 생사를 가른다: 수백 pt에서 노멀·쌍 통계는 무너지지만 TDF 조회는 점 수에 선형으로 열화될 뿐이다. 한 줄 요약 — **Shape6D = OVE6D의 골격 × PPF의 가치관, 단 표현은 둘 다 버리고 TDF로 재구축.**