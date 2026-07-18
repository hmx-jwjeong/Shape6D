# Shape6D 센서 융합 설계 — LiDAR 결정 장단 분석 및 FrameBundle 인터페이스 (v0.1 초안)

전제 문서: `01_stage_design.html`, `02_development_plan.html` (D2 확정 사항 "LiDAR로 depth 리파인, LiDAR 포인트 존재 지점만 사용"의 상세 설계 및 장단 분석). 본 문서의 모든 수치 중 "추정"으로 표기된 것은 실측 전 스펙시트/문헌 기반 대략치이며, D2-a 실측(§3.4)으로 대체되어야 한다.

---

## 1. LiDAR 선택의 장단 분석

### 1.1 대안 센서 비교표

| 항목 | LiDAR (Livox급 solid-state) | 산업 structured light (Photoneo/Zivid급) | ToF 카메라 (Basler blaze급) | Active stereo (RealSense D455) |
|---|---|---|---|---|
| 깊이 정밀도 (1σ, 물체 위 1점) | 5~20mm (추정: Livox 스펙 random error ~2cm@20m 계열, 근거리에서도 동급 오더) | 0.05~0.3mm@1m (추정: 벤더 스펙) — **압도적 최고** | 3~10mm (추정) | ~2%×거리 → 20mm@1m, 80mm@2m (추정: 벤더 스펙) |
| 유효 밀도 (물체 위) | **수백~수천 pt (희소)** — §3 모델 참조 | 물체 위 수만~수십만 pt (dense) | dense (VGA급) | dense (1280×720) |
| 조명 독립성 | ◎ 905nm 능동, 야외 직사광 강건 | △~○ 실내 설계, 강한 주변광에서 노출 증가 필요 (레이저 구조광 기종은 양호) | △ 주변광에 취약~중간 | △ 능동 IR 보조지만 강광에서 품질 저하 |
| 흑색/저반사 표면 | ○ 리턴 약화·드롭 발생하나 상대적 양호 (고감도 APD) | ○ HDR 다중 노출로 대응 가능 — 단 취득 시간 증가 | △ 리턴 부족 | △ 텍스처 소실 시 홀 |
| 경면(거울성) 표면 | ✕ 무리턴 또는 미러 반사 고스트 — **어떤 센서도 못 푸는 공통 약점**이나 LiDAR는 "결측"으로 나타나 오측정보다는 안전 | ✕ 무리턴/인터리플렉션 오측정 | ✕ multipath 오측정 (결측이 아니라 **틀린 값**이 나옴 — 더 위험) | ✕ 반사 텍스처 오매칭 |
| 다중 센서 간섭 (멀티 셀 배치) | ◎ 비반복 스캔·시간 코딩으로 간섭 확률 낮음 | ✕ 패턴 상호 간섭 심함 — 시분할 운용 필요 | △ 변조 주파수 분리로 회피 가능 | ○ 패턴 간섭 경미 |
| 모션 왜곡 | ✕ **스캔 적분(100~500ms) 중 왜곡** — 정적 씬 가정 필요 (§4.3) | ✕ 다중 패턴 투사(수백 ms) 중 모션 취약 | ◎ 프레임 단위 취득 | ◎ 프레임 단위 취득 |
| 근거리 최소거리 | △ 기종별 0.05~1m (Avia급 ~1m, Mid-70급 ~0.05m — **기종 선정 제약**, 추정) | ○ 0.4~1m 최적 거리로 설계됨 | ○ ~0.3m | ○ ~0.4m |
| 엣지 정확도 | ✕ 빔 발산(~0.28°→1m에서 풋프린트 ~5mm, 추정)에 의한 mixed pixel — 깊이 불연속 경계에서 중간값 오염 | ◎ 픽셀 단위 선명 | △ flying pixel | △ edge fattening |
| 비용 | ○ $0.8k~1.5k (추정) | ✕ $10k~25k (추정) | ○ $1k~3k (추정) | ◎ ~$0.5k |
| Jetson 연동 | ○ Ethernet + Livox SDK2, 호스트 부하 낮음 | △ GigE + 벤더 SDK, 포인트클라우드 생성 부하 큼 | ○ GigE/USB SDK | ○ librealsense (ARM 빌드 이슈 이력 있음) |

### 1.2 정직한 평가

**LiDAR가 이기는 축**: 비용, 조명 독립성, 다중 셀 간섭, 작업 거리 유연성, "틀린 값 대신 결측"이라는 실패 양상(파이프라인이 결측은 다룰 수 있지만 조용히 틀린 depth는 S4 검증까지 오염시킴).

**LiDAR가 지는 축 (숨기지 않음)**:
1. **희소성** — 물체 위 수백 pt 수준(§3). S3 fine 매칭의 설계 npoint 1024를 관측만으로 못 채우는 상황이 정상 동작 범위가 된다. 파이프라인 전체가 "가변·희소 포인트" 전제로 설계되어야 하며(이미 S2 단방향 point-to-template, S3 grid-conv 유효포인트 마스크 채널로 반영됨), 이는 되돌릴 수 없는 아키텍처 구속이다.
2. **포인트당 정밀도** — structured light의 0.1mm급 대비 LiDAR는 mm~cm급. 포즈 정밀도는 다점 평균(ICP에서 유효 오차 ≈ σ/√N)으로 회복하지만, N이 작으면 이 회복도 제한된다. 정밀 조립급(±0.1mm) 요구가 나오면 LiDAR 단독으로는 미달 — 요구 정밀도 스펙을 먼저 확정할 것.
3. **엣지 부정확** — 빔 풋프린트가 물체 실루엣 경계에 걸치면 전경/배경 중간 깊이가 나온다. 경계 인접 포인트 필터링(§4.2 quality flag)이 필수이며, 소형 물체일수록 "경계 인접" 비율이 높아 유효 포인트가 추가로 깎인다.
4. **캘리브레이션 부담** — 카메라와 물리적으로 분리된 센서라 외부 캘리브레이션(6DoF)이 정확도 바닥을 결정하고(§4.4), 진동·온도로 드리프트하면 재캘리브레이션 운용 절차가 필요하다. Structured light 일체형은 이 부담이 없다.
5. **근거리 제약** — 작업 거리 0.5m 요구 시 Avia급은 최소거리 미달 가능(추정 ~1m). 기종 선정이 작업 거리 스펙에 종속된다.

**결정의 본질**: 이 선택은 "S3 기하 주도 매칭이 dense·중정밀이 아니라 **희소·고신뢰 포인트로 성립한다**"는 베팅이다. 검증 게이트는 D2-a 실측(§3.4)과 M2-B 인코더 파일럿의 희소 패턴 평가다.

---

## 2. 운용 모드 3안 비교

| | (A) LiDAR 희소 포인트만 (현 계획) | (B) 별도 dense depth를 LiDAR로 스케일/바이어스 보정 | (C) LiDAR+RGB depth completion 네트워크로 densify |
|---|---|---|---|
| 구성 | LiDAR 유효 포인트 = 유일한 메트릭 기하 | 제2 depth 센서(stereo/ToF)의 dense depth에 대해 LiDAR 포인트로 z′=a·z+b (또는 공간 가변 보정) 피팅 | 희소 LiDAR + RGB → 경량 completion net (NLSPN·CompletionFormer 계열 존재, 경량판은 구현 필요) → dense depth |
| 정확도 | 포인트당 mm~cm, **전 포인트 실측치** | dense이나 정확도는 제2 센서에 종속. 전역 affine으로는 stereo의 공간 가변 계통 오차(edge fattening, 텍스처 의존 바이어스)를 못 잡음 | LiDAR 지점은 정확, **그 외 지점은 추론값** — 오차 상한 보장 없음 |
| 강건성 | 결측은 결측으로 정직하게 나타남 → 파이프라인이 인지 가능 | 제2 센서의 실패 표면(무텍스처·흑색 = 바로 우리 대상 부품)에서 홀/오측정 상속. 센서 2개분 캘리브레이션·간섭·유지보수 | **A1 정면 충돌**: completion은 RGB 외형에서 기하를 추론한다. 재도장·오염으로 RGB가 바뀌면 hallucinated depth도 바뀜 — "외형 불신" 원칙이 depth라는 뒷문으로 재침투. 깊이 불연속에서 매끈한 면을 지어내는 전형적 실패가 S3 대응·S4 잔차 검증을 **조용히** 오염 |
| 연산 (Orin, 추정) | ~0 (투영만, <5ms) | 피팅 ~수 ms + 제2 센서 파이프라인 | TRT FP16 기준 10~30ms 추정 + 학습·검증 부담 |
| 실패 모드 | 포인트 수 부족(소형·원거리·저반사) → S3 하한 붕괴. **실패가 감지 가능**(포인트 수는 셀 수 있음) | 두 센서 불일치 시 어느 쪽을 믿을지 모호. 비용·복잡도 2배 | 실패가 **감지 불가능** — 그럴듯한 가짜 depth는 신뢰도 검증(S4)조차 통과할 수 있음 |
| 판정 | **채택 (주 경로)** | 기각 — D2 확정("LiDAR 포인트 존재 지점만 사용")과 모순이고, 추가 센서 비용으로 사는 것이 '우리 대상 표면에서 가장 먼저 죽는 dense'임 | 기각 (메트릭 경로). 단서 조항: **S1 제안 생성에 한정**해서는 재평가 가능 — 마스크 리콜용 보조 신호는 메트릭 정확도가 필요 없어 hallucination 피해가 국소적. D2-a 실측 후 S1 리콜이 부족할 때만 트리거 (02 문서 부록 형식의 조건부 항목으로 등재) |

**권장**: A를 유일한 메트릭 기하 경로로 확정. C는 "S1 한정, 조건부 보류". 이 결정은 다운스트림 설계(S2 단방향 point-to-template 거리, S3 유효포인트 마스크 채널, M2-A2 희소 depth 증강)와 이미 정합한다. 추가로 A의 포인트 부족 리스크 완화는 densify가 아니라 **적분 시간 T의 파라미터화**(§3.3)로 푼다 — 실측치를 더 모으는 것이지 지어내는 것이 아니므로 A1과 충돌하지 않는다.

---

## 3. 물체 위 유효 포인트 수 모델

### 3.1 공식

```
pts_on_object ≈ rate × T × (Ω_obj / Ω_FOV) × η

  rate    : LiDAR 포인트 레이트 [pts/s] (single return)
  T       : 적분 시간 [s]
  Ω_obj   : 물체 입체각 ≈ A_proj / d²   (A_proj: 투영 면적, d: 거리)
  Ω_FOV   : LiDAR FOV 입체각 [sr]
            사각 FOV(θh×θv): Ω = 4·arcsin(sin(θh/2)·sin(θv/2))
            원형 FOV(θ):     Ω = 2π(1−cos(θ/2))
            360° 밴드(el1~el2): Ω = 2π(sin el2 − sin el1)
  η       : 현실 보정 계수 ≈ 0.5 추정
            (근거: 입사각 cos 감쇠 + 물체 형상이 정사각 투영보다 작음 + 저반사 드롭아웃을 합산한 대략치)
```

아래 표는 A_proj = s² (한 변 s인 정면 정사각, **상한 모델**), η 미적용 값. 실전 기대치는 표값 × 0.5 추정.

### 3.2 대표 기종별 표 (T = 100ms 기준, 전부 추정 — 스펙시트 대략치 기반)

밀도 상수 (rate/Ω_FOV, 추정):

| 기종 | rate | FOV | Ω_FOV | 밀도 [pts/sr/s] | 최소거리 | 비고 |
|---|---|---|---|---|---|---|
| Livox Avia급 | 240k pts/s | 70.4°×77.2° | 1.47 sr | ~163,000 | ~1m 추정 — **0.5m 열 무효 가능** | 비반복 로제트: T가 짧으면 커버리지 불균일 |
| Livox Mid-70급 | 100k pts/s | 원형 70.4° | 1.15 sr | ~87,000 | ~0.05m 추정 | 근거리 작업 적합. 단종 여부 확인 필요 |
| Livox Mid-360급 | 200k pts/s | 360°×(−7°~52°) | 5.72 sr | ~35,000 | ~0.1m 추정 | 360° 커버가 밀도를 희석 — 이 용도엔 비효율 |

pts_on_object (T=100ms, 상한 모델):

| 물체 크기 → / 거리 ↓ | | 5cm | 10cm | 20cm |
|---|---|---|---|---|
| **0.5m** | Avia급 | (163)* | (652)* | (2,610)* |
| | Mid-70급 | 87 | 349 | 1,395 |
| | Mid-360급 | 35 | 140 | 560 |
| **1m** | Avia급 | 41 | 163 | 652 |
| | Mid-70급 | 22 | 87 | 349 |
| | Mid-360급 | 9 | 35 | 140 |
| **2m** | Avia급 | 10 | 41 | 163 |
| | Mid-70급 | 5 | 22 | 87 |
| | Mid-360급 | 2 | 9 | 35 |

\* 최소거리 미달 추정 — 실효 0일 수 있음. T에 선형: T=500ms면 ×5.

**시사점**: (1) Mid-360급은 이 용도에 부적합 — FOV 낭비. (2) 좁은 FOV 기종(Tele-15급: 14.5°×16.2°, 밀도 ~3.4M pts/sr/s 추정 — 위 대비 20배)은 밀도 문제를 하드웨어로 해결하지만 최소거리(수 m 추정)가 근거리 작업과 상충 — FOV 집중도가 rate만큼 중요하다는 것이 기종 선정 제1 기준. (3) 1m·10cm 부품·Avia급·T=100ms에서 실전 기대 ~80pt(η=0.5) — **T=100ms는 부족, 200~400ms가 기본 운용점**.

### 3.3 S3 npoint 설계 권고

| 파라미터 | 값 | 근거 |
|---|---|---|
| `N_min_reject` | 64 | 이 미만이면 S4에서 low-confidence/reject 또는 재적분 트리거. coarse 대응이 유의미하려면 최소 수십 pt 필요 (추정 — M2-B 파일럿에서 하한 곡선 실측로 확정) |
| `N_target_coarse` | 150~196 | SAM-6D coarse 196pt 규약과 정합. 관측 포인트가 196 미만이면 다운샘플 없이 전 포인트 사용 (매칭 transformer는 가변 길이 허용, 패딩 마스크로 처리) |
| `N_fine_cap` | 1024 | fine npoint = min(N_valid_obj, 1024). 초과분만 voxel 다운샘플 |
| 학습 시 랜덤화 범위 | 64~2048 | M2-A2 희소 증강의 물체 위 포인트 수 분포 — 위 표의 실전 범위를 커버 |
| 적분 시간 T | 기본 200ms, 상한 400ms | Avia급·1m·10cm에서 실전 ~160~320pt 기대. 사이클 예산: 취득 200ms + 연산 ~310~560ms(01 문서 예산) = 1s 내. T는 포인트 수 피드백으로 적응 가능하게 파라미터화 |

### 3.4 D2-a 실측 프로토콜 (이 모델의 검증)

실기종 확보 시: 기준 물체(교정구 또는 대상 부품) × 거리 {0.5, 1, 2m} × 반사율 {백색, 회색, 무광흑} × T {100, 200, 500ms}로 물체 위 유효 포인트 수를 계측 → 위 표 대비 η 실측 → M2-A2 증강 분포·`N_min_reject` 갱신.

---

## 4. 캘리브레이션·동기화 설계

### 4.1 외부 캘리브레이션 (LiDAR–카메라, 타겟 기반)

- **타겟**: 체커보드(또는 ChArUco) 평면 보드 ~0.6×0.8m. 비반복 스캔 LiDAR는 2~5s 적분하면 보드 위 준밀집 포인트를 얻는다(희소 LiDAR 캘리브레이션의 이점).
- **절차**: ① 보드를 20+ 포즈로 배치, 각 포즈에서 {RGB 1장, LiDAR 2~5s 적분 클라우드} 수집 ② 카메라 측: PnP로 보드 평면 π_i^cam ③ LiDAR 측: 보드 영역 포인트 평면 피팅 π_i^lidar ④ T_cam_lidar = argmin Σ [point-to-plane 잔차 + (선택) 보드 실루엣 엣지–이미지 엣지 잔차], robust kernel 적용.
- **도구**: MATLAB Lidar-Camera Calibrator, Livox 공식 `livox_camera_lidar_calibration`, `direct_visual_lidar_calibration`(Koide, targetless) 등 공개 구현 존재 — 자체 파이프라인 통합은 구현 필요.
- **기대 정확도**: 회전 0.05~0.2°, 병진 1~5mm (추정: 해당 계열 문헌 일반 보고 수준). **목표 예산: 회전 ≤0.1°, 병진 ≤2mm** — §4.4 근거.
- **운용**: 잔차 통계를 calib.yaml에 기록, 주기 점검(보드 1포즈 스팟체크)으로 드리프트 감시.

### 4.2 투영 파이프라인 (LiDAR pts → 1280×800)

```
입력: P_lidar [N,3] (LiDAR 좌표), T_cam_lidar [4,4], K(fx,fy,cx,cy; RGB는 사전 undistort)
1. P_cam = R·P_lidar + t                      # [N,3], 카메라 좌표계로 통일
2. Z>z_near 필터, u = fx·X/Z+cx, v = fy·Y/Z+cy # 서브픽셀 float 유지 → lidar_pixels [N,2]
3. 이미지 내부(0≤u<1280, 0≤v<800) 필터
4. 래스터화: 픽셀당 z-buffer(최근접 유지) → sparse_depth [800,1280] f32 (0=무효)
                                          → valid_mask  [800,1280] bool
                                          → pix2pt      [800,1280] i32 (포인트 인덱스, −1=무효)
5. 품질 플래그 flags [N] u8 (비트 OR):
   EDGE_MIXED   =1  # 반경 r_e px 내 이웃 유효픽셀과 깊이차 > δ_edge (기본 3cm) — 빔 풋프린트 mixed pixel 의심
   LOW_INTENSITY=2  # 반사 강도 하위 임계 — 저반사 리턴, 거리 노이즈 증가 의심
   MULTI_RETURN =4  # 2nd return — 반투명/엣지 의심
   NEAR_MASK_BOUNDARY=8  # (S1 이후 갱신) 인스턴스 마스크 경계 2~3px 이내
S2/S3 메트릭 연산은 flags==0 포인트를 우선 사용, S4 ICP는 EDGE_MIXED 제외를 기본값으로.
```

빔 발산 근거: 발산각 ~0.28° 추정 → 1m에서 풋프린트 ~4.9mm. 10cm 부품 실루엣 경계 밴드가 전체 표면의 상당 비율이므로 EDGE_MIXED 필터는 소형 물체에서 유효 포인트를 추가로 20~40% 깎을 수 있다(추정) — §3 표 해석 시 반영할 것.

### 4.3 시간 동기

- **클럭 동기**: LiDAR는 PTP(gPTP)/PPS 동기 지원(Livox 계열, 기종별 확인 필요 — 추정), 카메라는 하드웨어 트리거. 카메라 노출을 LiDAR 적분 창의 중앙에 배치.
- **모션 왜곡**: 스캐닝 LiDAR는 T=200~400ms 적분 중 씬이 움직이면 포인트가 번진다. **v1 운용 가정: 취득 창 동안 정적 씬** (A3 최초 사이클·로봇 정지 후 촬영 시나리오와 정합 — 운용 제약으로 명문화). 포인트별 타임스탬프는 FrameBundle에 보존하여, 컨베이어 등 동적 씬 확장 시 deskew(등속 보정)를 후행 구현할 수 있게 함 (구현 필요, v1 범위 외).

### 4.4 캘리브레이션 오차 → 파이프라인 영향 정량화

가정: fx ≈ 914px (HFOV 70° 렌즈 가정 시 1280/(2·tan35°) — 렌즈 확정 전 추정). 환산 기준: **1px ≈ Z/fx ≈ 1.1mm @1m**.

| 오차원 | 픽셀 오차 @1m | 3D 오차 @1m | 파이프라인 영향 |
|---|---|---|---|
| 회전 δθ=0.1° | fx·δθ ≈ 1.6px | Z·δθ ≈ 1.7mm (클라우드 전체 강체 편이) | **최종 포즈에 그대로 바이어스로 전사** — S4 ICP는 LiDAR 포인트를 참으로 정합하므로 회복 불가. 포즈 정확도 바닥을 결정 |
| 병진 δt=2mm | fx·δt/Z ≈ 1.8px | 2mm | 상동 (강체 바이어스) |
| 합성 (예산 내) | ~2px | ~3mm | 포즈 정확도 목표가 ~5mm급이라면 캘리브레이션이 그 절반을 소모 — 예산 ≤0.1°/2mm의 근거 |

**S3 대응 오차로의 전파 경로 3가지**:
1. **RGB 보조 branch 특징 오배치**: LiDAR 포인트에 붙는 RGB 특징이 2px 어긋난 픽셀에서 샘플링됨. 특징 스트라이드(conv stride 4 또는 ViT patch 14) 대비 2px는 대체로 허용 — 단 물체 실루엣 경계에서는 배경 특징이 붙을 수 있음 → NEAR_MASK_BOUNDARY 플래그 포인트는 RGB 특징 신뢰도 하향.
2. **마스크–포인트 소속 오판**: S1 마스크와 valid_mask의 2px 어긋남 + 빔 풋프린트 5mm가 겹치면 경계 밴드에서 배경 포인트가 물체 크롭에 혼입(깊이 불연속이므로 cm~수십 cm 오차 포인트) → **마스크 2~3px erosion 후 포인트 선별**을 표준 절차로.
3. **강체 바이어스**: 위 표 — 학습·매칭으로 회복 불가능한 하한이므로, M5 실물 리그의 정밀도 리포트에서 캘리브레이션 잔차를 분리 계상할 것.

---

## 5. 인터페이스 정의 — FrameBundle

전 스테이지(S1~S4)가 소비하는 표준 입력. 위치: `shape6d/common/frame_bundle.py` (신규).

```python
# shape6d/common/frame_bundle.py
from dataclasses import dataclass, field
import numpy as np

# 품질 플래그 비트
EDGE_MIXED = 1; LOW_INTENSITY = 2; MULTI_RETURN = 4; NEAR_MASK_BOUNDARY = 8

@dataclass
class CameraIntrinsics:
    fx: float; fy: float; cx: float; cy: float
    width: int = 1280
    height: int = 800
    dist_coeffs: np.ndarray = None   # [5] f64; rgb를 사전 undistort하면 zeros

@dataclass
class FrameBundle:
    # --- 원시 관측 ---
    rgb: np.ndarray            # [800,1280,3] uint8, undistorted
    lidar_points: np.ndarray   # [N,3] f32, **카메라 좌표계** (T_cam_lidar 적용 완료)
    lidar_intensity: np.ndarray# [N]   f32, 반사 강도 (기종 스케일 그대로)
    lidar_t_offset: np.ndarray # [N]   f32, 적분 창 시작 대비 초 단위 (deskew 확장용)
    # --- 투영 산출물 (build 시 1회 계산, 전 스테이지 공용) ---
    lidar_pixels: np.ndarray   # [N,2] f32, 서브픽셀 (u,v); 이미지 밖 포인트는 NaN
    sparse_depth: np.ndarray   # [800,1280] f32, z-buffer 래스터, 0=무효
    valid_mask: np.ndarray     # [800,1280] bool
    pix2pt: np.ndarray         # [800,1280] i32, 픽셀→포인트 인덱스, -1=무효
    point_quality: np.ndarray  # [N] uint8, 플래그 비트 OR (0=클린)
    # --- 캘리브레이션·시각 ---
    K: CameraIntrinsics
    T_cam_lidar: np.ndarray    # [4,4] f64 (기록용; lidar_points는 이미 변환됨)
    t_rgb: float               # 카메라 노출 중심 (epoch sec)
    t_lidar: tuple             # (적분 시작, 끝)
    meta: dict = field(default_factory=dict)  # 기종, T, calib 버전/잔차, 파워모드 등

def build_frame_bundle(rgb, lidar_pts_lidar_frame, intensity, t_offsets,
                       calib) -> FrameBundle: ...        # §4.2 파이프라인 구현
def project_points(pts_cam, K) -> tuple: ...             # (uv[N,2], z[N], in_img[N] bool)
def rasterize(uv, z, hw) -> tuple: ...                   # (sparse_depth, valid_mask, pix2pt)
def flag_quality(pts_cam, uv, sparse_depth, intensity,
                 delta_edge=0.03, r_edge_px=3) -> np.ndarray  # [N] uint8

# 스테이지 소비 규약
# S1: rgb + valid_mask(전경 프롬프트) + lidar_points(클러스터링 병렬 경로)
# S2: 마스크별 pts = lidar_points[pix2pt[mask & valid_mask]] 중 quality==0 우선
#     (크기 게이팅·point-to-template 정합의 입력)
# S3: 크롭 그리드 채널 = [XYZ(카메라계), valid, quality] + rgb 크롭(보조 branch)
# S4: ICP 타깃 = quality에 EDGE_MIXED 없는 포인트, 잔차 검증은 sparse_depth 대비
```

**파일 포맷**:
- 로깅/재현용: `frame_%06d.npz` — FrameBundle 전 필드 (npz 키 = 필드명), meta는 JSON 문자열.
- 캘리브레이션: `calib.yaml` — `K{fx,fy,cx,cy,dist}`, `T_cam_lidar` (4×4 row-major), `calibrated_at`, `residual{rot_deg, trans_mm, n_poses}`, `lidar_model`.

**설계 노트**: (1) 노멀은 FrameBundle에 넣지 않는다 — 희소 포인트의 노멀은 크롭 스케일·이웃 정의에 의존하므로 S3/S4에서 투영 이웃 기반으로 지연 계산. (2) lidar_points를 카메라 좌표계로 통일해 두는 것이 이후 전 스테이지에서 좌표 변환을 제거한다(T_cam_lidar는 기록·디버그용). (3) 연산 예산: build_frame_bundle 전체 <5ms 추정(투영·래스터는 단순 산술, N~수만) — S1 예산에 흡수.

---

## 요약 (설계 문서 반영용 결론)

1. LiDAR 채택은 비용·조명 강건성·"결측으로 실패하는 정직함"을 사는 대신 밀도·엣지 정확도·캘리브레이션 부담을 지는 거래이며, 성립 조건은 §3의 포인트 수 하한(D2-a 실측으로 검증)이다.
2. 운용 모드는 A(희소 실측만) 확정. B 기각(비용·모순), C 기각(A1 정면 충돌·감지 불가 실패) — 단 C는 S1 리콜 한정 조건부 보류.
3. 포인트 수 모델: 밀도[pts/sr/s] × T × s²/d² × η(≈0.5). 기본 운용점 T=200ms, `N_min_reject`=64, coarse 150~196, fine cap 1024 (가변 길이 설계).
4. 캘리브레이션 예산 회전 ≤0.1°·병진 ≤2mm (= 포즈 바이어스 ~3mm@1m 하한), 마스크 2~3px erosion + EDGE_MIXED 필터 표준화, v1은 취득 창 정적 씬 가정(타임스탬프는 보존).
5. FrameBundle(카메라 좌표계 통일, 투영 산출물 1회 계산 공용)이 S1~S4 표준 입력.