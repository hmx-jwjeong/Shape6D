# 사실 검증 결과 — 기존 방법 관련 주장 전수 점검

검증 방법: SAM-6D 주장은 `/Users/jaewoo/Documents/SAM-6D/SAM-6D/` 코드 직접 대조(파일:라인 실측). 타 논문은 학습된 지식 범위 내 판정, 불확실 시 UNCERTAIN. 환경에 torch/numpy 부재로 실행 검증은 불가(정적 대조만).

## 1. SAM-6D (코드 직접 검증)

| # | 주장 | 출처 | 판정 | 근거 |
|---|---|---|---|---|
| 1 | `final_score = (semantic_score + appe_scores + geometric_score*visible_ratio) / (1+1+visible_ratio)` (detector.py:384) | concept ②, compare | **CONFIRMED** | 코드 :384와 문자 그대로 일치 |
| 2 | semantic·appearance 둘 다 DINOv2 RGB 특징 → 판별의 2/3가 외형 의존 | concept ② | **CONFIRMED** | semantic=CLS token, appearance=masked patch, 둘 다 CustomDINOv2(dinov2.py:148-227, ISM_sam.yaml:11); 외형 항 가중 2/(2+vr) ≥ 2/3 (vr∈[0,1]) |
| 3 | ISM 점수 산식 위치 detector.py:260-308(semantic+appearance) | compare 1a② | **CONFIRMED** | compute_semantic_score :260-296, compute_appearance_score :298-308 |
| 4 | "ISM의 visible ratio도 dense depth 기반" | concept ① | **WRONG** | visible_ratio는 DINOv2 패치 특징 유사도로 계산(loss.py:64-76, detector.py:313) — depth 무관 |
| 5 | SAM-6D는 dense depth 전제(RGB crop + dense 역투영 포인트) | concept ①, compare | **CONFIRMED** | get_point_cloud_from_depth 전면 역투영(data_utils.py:92-110), mask∧depth>0(run_inference_custom.py:204), ISM 병진도 마스크 depth 평균(detector.py:234-246) |
| 6 | 마스크 내 2048pt 리프팅, `n_sample_observed_point: 2048`(base.yaml:62) | compare 1a① | **CONFIRMED** (라인 표기 1건 부정확) | 샘플링 run_inference_custom.py:223-228, base.yaml:62 일치. 단 인용된 :184는 CAD 모델 샘플링 라인 — 관측 리프팅은 :204-228 |
| 7 | 추론 forward 안에서 오브젝트 측 sample_pts_feats(FPS)·geo_embedding_o 매번 재계산(pose_estimation_model.py:25~37) | concept ③ | **CONFIRMED** | :34-37에서 재계산; sample_pts_feats=furthest_point_sample(model_utils.py:53-64, pointnet2 CUDA) |
| 8 | coarse 196pt / fine 2048pt(base.yaml:17-18), Geometric Transformer 3블록×2단 | compare 1a⑤ | **CONFIRMED** | coarse_npoint:196·fine_npoint:2048(:17-18), nblock:3(:33,:44), 블록당 self+cross(coarse_point_matching.py:28-35) |
| 9 | 학습된 soft 대응 → 6000→300 가설 weighted SVD(coarse_point_matching.py:70-74, base.yaml:41-42) | compare 1b⑥ | **CONFIRMED** | compute_coarse_Rt(model_utils.py:187-246): 3점 가설 6000 → top300 → 전점 스코어 선택, WeightedProcrustes |
| 10 | fine = 2048pt dense soft 대응 SVD(fine_point_matching.py:75-81), ICP 없음 | compare 1b⑦ | **CONFIRMED** | compute_fine_Rt(model_utils.py:250-283); PEM 전체 grep에 ICP 0건 |
| 11 | 판정은 pred_pose_score 단일값(fine_point_matching.py:81), 거부 개념·대칭 등가 평가·free-space 검사 없음 | concept ⑤, compare 1b⑧ | **CONFIRMED** | :81 출력, 최종 = pred_pose_score×det_score(run_inference_custom.py:291-293) 후 저장만; symmetry_flag는 로드만 되고(bop_object_utils.py:81-86) 추론 판정 소비처 0건; verdict/reject/free-space 부재 |
| 12 | ISM·PEM 완전 분리 2모델, 특징 비공유 | concept ④ | **CONFIRMED** | PEM은 ISM의 seg JSON만 소비(run_inference_custom.py:166-171); 인코더 상이(DINOv2 vs MAE ViT-base, feature_extraction.py:79-83) |
| 13 | PEM 104M 파라미터 | concept ④ | **UNCERTAIN** | 정적 합산 추정 ≈104~107M(ViT-base ~85.8M + 업스케일 12.6M + 매칭 모듈 ~6M) — 자릿수·근사값 부합하나 torch 부재로 실측 불가 |
| 14 | PEM 학습 필수(학습된 모델이 포즈의 유일 경로, 폴백 없음) | concept ④·⑥ | **CONFIRMED** | sam-6d-pem-base.pth 로드 필수(run_inference_custom.py:270-271); 비학습 대체 경로 부재 |
| 15 | 42뷰 icosphere 템플릿 규약, cam_poses_level0.npy 존재 | concept, compare §2, gaps-product 5-2j | **CONFIRMED** | cam_poses_level0.npy shape **(42,4,4)** 실측(npy 헤더); total_nView=42(run_inference_custom.py:155); n_template_view:42(base.yaml:91); 42=icosphere L1 정점 수 |
| 16 | 온보딩 = BlenderProc PBR 42뷰 렌더 + DINOv2 descriptor, GPU 필요 | compare 1a④ | **CONFIRMED** | render_custom_templates.py:1 `import blenderproc`, :19 cnos cam path→cam_poses_level0; ISM/PEM 코드 .cuda() 하드코딩 |
| 17 | 온라인 = ViT 인코더 2회(관측+템플릿) | compare 1a⑤ | **CONFIRMED** | 동일 rgb_net을 템플릿 1회(run_inference_custom.py:277)+관측 crop 1회(feature_extraction.py:128-133) 실행 |
| 18 | pointnet2 커널 의존 | compare 1b⑨ | **CONFIRMED** | model/pointnet2/ 존재, furthest_point_sample·QueryAndGroup 사용 |
| 19 | (학습 분포) PEM은 MegaPose-GSO + ShapeNetCore로 학습 | gaps-product 1-2 전제 | **CONFIRMED** | training_dataset.py:50-59 경로 하드코딩 |
| 20 | "ISM의 sem×appe×geo 점수(detector.py:310-322)" | compare 1b⑧ | **CONFIRMED** (표기 주의) | :310-322는 geometric score 함수뿐, 결합은 :384의 **가중 합**(곱 아님) — '×'는 오독 소지 |
| 21 | 수백 pt에서는 choose 샘플링·패치 특징 정렬이 성립 안 함 | compare 1a③ | **UNCERTAIN** | 코드상 32px 마스크만 넘으면 중복 복원추출로 동작은 함(run_inference_custom.py:205,223-224) — "성립 안 함"은 품질 열화 추정이지 코드 사실 아님 |

## 2. 타 방법 (지식 기반 판정)

| # | 주장 | 출처 | 판정 | 근거 |
|---|---|---|---|---|
| 22 | FreeZe: GeDi 기하 + DINOv2 시각 특징 융합, training-free | compare | **CONFIRMED** | FreeZe(ECCV 2024)는 동결된 기하(GeDi)·시각(DINOv2) 파운데이션 모델 융합의 무학습 방법 |
| 23 | FreeZe: ICP + 대칭 인지 정련(SAR) | compare | **CONFIRMED** | 논문이 symmetry-aware refinement 제안, ICP 계열 정합 포함 |
| 24 | FreeZe: 82.1 AR·BOP24 1위·24.9s/img·"v2.1" | compare | **UNCERTAIN** | 지시문 인용값 — FreeZe가 BOP 2024 model-based unseen 상위권인 것은 부합하나 정확 수치·순위·제출명은 리더보드 대조 불가 |
| 25 | GigaPose: 템플릿 검색(out-of-plane) + 패치 대응 **2쌍**→유사변환 폐형해 | compare | **WRONG** | GigaPose(CVPR 2024)의 핵심이 대응 **1쌍**(부제 "via One Correspondence") — 패치별 회귀된 스케일·in-plane 각으로 4-DoF 복원 |
| 26 | GigaPose: 162뷰 템플릿 | compare(확인 필요 표기) | **CONFIRMED** | 논문 162 뷰포인트(icosphere L1=162 정점, 리포의 cam_poses_level1.npy (162,4,4)와도 정합) |
| 27 | GigaPose: 자체 refiner 없음, MegaPose/GenFlow refiner 결합 관행 | compare | **CONFIRMED** | 논문·BOP 제출이 MegaPose refiner(및 GenFlow) 결합으로 보고 |
| 28 | GigaPose: coarse 수십 ms급, RGB-only 동작 | compare | **CONFIRMED** | coarse가 MegaPose 대비 ~38× 고속(수십 ms 오더) 주장, RGB 기반 방법 |
| 29 | GigaPose: depth를 병진 스케일 보정에 선택적 사용 | compare(확인 필요 표기) | **UNCERTAIN** | 세부 미확인 — 원문 하네스의 hedge 유지가 적절 |
| 30 | Co-op(CVPR25): 반밀집 대응+확률적 flow, 75.9 AR@0.8s, certainty 추정 보유 | compare | **UNCERTAIN** | 방법 개요는 GenFlow 계열 후속으로 그럴듯하나 학회·수치·세부 전부 미검증(지시문 인용값) |
| 31 | FoundationPose: model-based/model-free(neural object field) 통합 | compare | **CONFIRMED** | CVPR 2024 논문의 핵심 구성 |
| 32 | FoundationPose: 병진 추정 후 회전 가설 전역 샘플→transformer refiner 반복→계층 랭킹, 입력 RGBD | compare | **CONFIRMED** | 논문 파이프라인과 일치(전역 회전 샘플, render&compare 정련, hierarchical pose ranking) |
| 33 | FoundationPose: 29.3s/img, model-free field 학습 분 단위 | compare | **UNCERTAIN** | 지시문 인용값·hedge 항목 — 미검증 |
| 34 | MegaPose: 후보 포즈 렌더 분류 coarse + DeepIM식 render&compare refiner, RGB 주도(RGBD 변형 존재) | compare | **CONFIRMED** | MegaPose(CoRL 2022) 구조와 일치 |
| 35 | MegaPose: 2M 합성 이미지 1회 학습 | compare | **CONFIRMED** | 논문의 ~2M 합성 렌더 학습셋(GSO+ShapeNet) |
| 36 | GSO ≈ 1천 종 가정용품 스캔, 대부분 <30cm | gaps-product 1-2 | **CONFIRMED** | Google Scanned Objects ≈1,030종 가정용품 실스캔, 탁상 스케일 위주 |
| 37 | PPF/Drost: 쌍 특징 (‖d‖, ∠n₁d, ∠n₂d, ∠n₁n₂) 해시 + 참조점별 Hough류 투표 + 클러스터링 + ICP | compare | **CONFIRMED** | Drost et al. CVPR 2010 정의 그대로 |
| 38 | PPF: 씬 노멀 필수, 마스크·검출기 없이 씬 전체 동작, HALCON 채택 | compare | **CONFIRMED** | 쌍 특징이 씬 노멀 요구, 전역 투표 방식, HALCON surface-based 3D matching이 PPF 계열 |
| 39 | PPF/HALCON: "20년 산업 검증" | compare §3 | **UNCERTAIN** | Drost PPF 기준 ~16년(2010→2026), surflet-pair(2003) 기원으로 읽으면 ~23년, HALCON 툴체인 전체는 25년+ — 기준 명시 필요 |
| 40 | OVE6D: depth crop(외부 마스크 필요)→학습 뷰포인트 latent, 코사인 코드북 검색 + in-plane 회귀 + 검증 헤드 | compare | **CONFIRMED** | OVE6D(CVPR 2022) 구조와 일치 |
| 41 | OVE6D: ShapeNet 1회 학습으로 전 물체 공용 인코더 | compare | **CONFIRMED** | 논문 핵심 주장(재학습 없는 일반화) |
| 42 | OVE6D: 코드북 ~4000뷰 | compare(확인 필요 표기) | **UNCERTAIN** | 수천 뷰 규모는 맞으나 정확 수치 미확인 — hedge 유지 적절 |
| 43 | OVE6D: 병진은 마스크/depth에서 별도 추정(검색 변수 아님) | compare §4 | **CONFIRMED** | 뷰포인트 검색은 회전만 담당, 위치는 마스크·깊이에서 산출 |
| 44 | CNOS: SAM 프로포절 전수 + 42뷰 렌더 템플릿 DINOv2 CLS 매칭, SAM이 병목 | compare | **CONFIRMED** | 논문 구조 일치 + 코드 방증: render_custom_templates.py:19이 "cnos camera path"로 (42,4,4) 포즈 재사용; FastSAM 변형의 존재가 병목 방증 |
| 45 | MUSE: 세부 불명 | compare | **UNCERTAIN** | 원문도 "확인 필요" — 판정 유지(BOP 2024+ unseen 검출 계열이라는 것 이상 미확인) |
| 46 | BOP val(T-LESS·ITODD)은 밀집 structured-light depth 도메인 | gaps-algo 6-B, gaps-product 3-1·4-2 | **CONFIRMED**(T-LESS) / **UNCERTAIN**(ITODD 센서 방식) | T-LESS는 Primesense Carmine 1.09(structured light) 밀집 depth; ITODD도 산업용 3D 센서 밀집 depth이나 방식(SL 여부) 미확인 — "밀집"이라는 논지 자체는 성립 |

## 3. WRONG 수정문

**#4** (concept §2-①):
> 수정 전: "SAM-6D는 dense depth 전제(RGB crop + dense 역투영 포인트, ISM의 visible ratio도 dense depth 기반)."
> 수정 후: "SAM-6D는 dense depth 전제(RGB crop + dense 역투영 포인트; ISM의 geometric score는 마스크 내 depth 평균으로 병진을 추정하므로 dense depth 의존 — detector.py:221·234-246). 단 visible_ratio 자체는 DINOv2 패치 특징 유사도로 계산된다(loss.py:64-76)."

**#25** (compare 표 1a·1b GigaPose):
> 수정 전: "패치 대응 2쌍→유사변환(스케일·in-plane·2D 병진) 폐형해."
> 수정 후: "패치 대응 **1쌍**으로 4-DoF 복원 — 네트워크가 패치별 스케일·in-plane 각을 회귀하므로 대응 1쌍이면 유사변환이 결정된다(논문 부제 'via One Correspondence'). §2의 '분해 사상' 논지는 유지되나 '2쌍' 표기는 정정 필요."

## 4. 부기 (판정 외 정밀도 주의 2건)

- **#6**: `run_inference_custom.py:184`는 CAD 모델 샘플링(`n_sample_model_point`) 라인 — 관측 2048pt 리프팅의 정확 근거는 :204-228. 논지 영향 없음, 인용 라인만 교정 권고.
- **#20**: "sem×appe×geo"의 '×'는 실제 결합(가중 합, detector.py:384)과 다르게 읽힐 수 있음 — "sem+appe+geo·vr 가중 합"으로 표기 권고.

집계: 총 46건 — CONFIRMED 33, WRONG 2, UNCERTAIN 11 (지시문 인용 수치 82.1/75.9/73.4/24.9s/0.8s/29.3s는 전부 UNCERTAIN 처리 — 리더보드·논문 실시간 대조 불가).