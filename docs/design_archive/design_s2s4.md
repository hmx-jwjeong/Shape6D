# Shape6D — S2 후보 판별 · S4 정련·검증 상세 설계 (v0.1)

작성 기준: `/Users/jaewoo/Documents/SAM-6D/SAM-6D/Instance_Segmentation_Model/model/detector.py` 코드 분석 + `/Users/jaewoo/Documents/Shape6D/docs/01_stage_design.html`, `02_development_plan.html`의 확정 스택. 전 설계에 1280×800 RGB + 희소 LiDAR depth(물체 위 유효 포인트 수백~수천 pt, D2-a 미확정 → 파라미터화) 전제를 적용한다.

---

## 0. SAM-6D ISM 스코어링 분석 — 무엇을 대체하는가

`detector.py`의 최종 점수는 `test_step()` L384:

```
final_score = (semantic + appe + geometric·visible) / (1 + 1 + visible)
```

| SAM-6D 신호 | 코드 위치 | 내용 | Shape6D 처분 |
|---|---|---|---|
| semantic score | `compute_semantic_score` L260–296 | DINOv2 ViT-L cls token vs 42뷰 RGB 템플릿 descriptor cosine, `avg_5` 집계 → 물체 배정 + best template 선정 | **강등·축소 승계** → S2-③: ViT-L→ViT-S, 가중치 하향(약한 prior), 물체 배정 결정권 박탈(융합의 1/5 이하) |
| appearance score | `compute_appearance_score` L298–308 | best template의 masked patch feature vs 관측 crop patch 유사도(`MaskedPatch_MatrixSimilarity`) | **완전 제거** — 패치 단위 외형 비교는 오염·재도장에 가장 먼저 붕괴(A1 정면 위배) |
| geometric score | `compute_geometric_score` L310–322 + `project_template_to_image` L209–232 | best template 회전 + 마스크 depth 평균 병진(L234–246)으로 CAD 포인트 투영 → **bbox IoU**. visible_ratio는 patch 유사도 기반 | **교체** — bbox IoU는 형상 정합이 아니라 "크기·위치 일치"의 조악한 근사이고, visible_ratio는 RGB 패치 신호라 A1 위배. S2-①(메트릭 크기 게이팅) + S2-②(point-to-template 거리 정합)로 대체. "마스크 depth 평균 = 병진" 아이디어는 robust화(median)하여 승계 |

핵심 진단: SAM-6D 판별 점수의 분자 3항 중 2.5항(semantic, appearance, visible_ratio)이 RGB 신호다. 또한 semantic 임계(`confidence_thresh`)를 **descriptor forward 이후에** 적용하므로 모든 제안에 대해 ViT-L forward가 발생 — 레이턴시 주범. Shape6D는 무료 신호(크기 게이팅)를 맨 앞에 두어 descriptor forward 횟수 자체를 줄인다.

---

## 1. S2 후보 판별 — 3단 캐스케이드 상세

### 1.0 입출력과 표기

- 입력: S1 마스크 집합 `{m_j}` (M개, 각 1280×800 bool), LiDAR 유효 depth(유효 픽셀 좌표 `uv` + 깊이 `z`의 희소 리스트), 카메라 내참 K, S0 캐시.
- 마스크별 관측 포인트: `P_j = backproject(uv ∩ m_j, z, K)` → `(N_j, 3)` float32, 카메라 좌표(m 단위). **N_j는 수백~수천, 마스크에 따라 <50도 발생**(§6.1).
- 출력: 대상 물체로 판별된 인스턴스 top-k (기본 k_inst=3) + 융합 점수 + 진단 플래그.

### 1.1 ① 크기 게이팅 (size_gate) — 전 후보, 무료

**robust extent**: 마스크 내 LiDAR 포인트의 축별 5–95 percentile bbox 대각.

```
ext_a = percentile(P_j[:,a], 95) − percentile(P_j[:,a], 5),  a ∈ {x,y,z}
d_obs = sqrt(ext_x² + ext_y² + ext_z²)
```

percentile 트리밍이 마스크 경계 bleeding(배경 포인트 혼입)과 LiDAR 플라잉 포인트를 제거한다. S0 캐시의 CAD 직경 `D_cad`(최대 포인트쌍 거리)와 비교:

```
게이트 통과 ⇔ β_occ · D_cad ≤ d_obs ≤ (1 + τ_up) · D_cad   그리고   N_j ≥ N_min
```

| 파라미터 | 기본값 | 근거 |
|---|---|---|
| τ_up (상한 허용) | 0.15 | 관측 extent는 물체보다 커질 물리적 이유가 없음(노이즈·bleed만) → 타이트. E6 실험으로 ±10/20/30% 스윕 |
| β_occ (하한, 비가림) | 0.35 | 가림 시 가시 단편은 얼마든지 작아질 수 있으나 너무 작으면 판별력 자체가 없음 |
| β_occ (가림 정황) | 0.25 | **하한 완화 규칙**: (i) 마스크가 이미지 경계에 접함(절단) 또는 (ii) 마스크 bbox가 더 가까운 depth의 타 마스크와 IoU>0.1로 겹침(전방 가림) 이면 완화 |
| N_min | 30 | 6-DoF 검증 가능 최소치(§6.1). D2-a 실측 후 재조정 |

연속 점수(융합용): `S_size = clamp(d_obs / D_cad, 0, 1)` — 상한 위반은 hard reject이므로 점수는 "작은 쪽"만 벌점. 비용: 후보당 percentile 3회 = 산술 연산뿐.

### 1.2 ② point-to-template 정합 (depth_match) — 주 판별 신호

#### 정규화 (스케일·병진 — 회전은 미지)

- **스케일**: 정규화하지 않는다. LiDAR와 CAD 모두 메트릭이므로 스케일은 그 자체가 판별 신호다(①이 이미 소비). 스케일 정규화는 "크기만 다른 유사 형상" 오판을 유발하므로 금지.
- **병진**: 관측 포인트를 축별 **median**으로 센터링: `p̃_i = p_i − median(P_j)`. 템플릿 측도 뷰별 가시면 포인트의 median을 S0에서 캐시(`tpl_center[v]`, (42,3))하고 동일하게 센터링해 둔다. mean이 아닌 median인 이유: 부분 가림 시 centroid보다 이동량이 작다. 그래도 남는 센터 편향은 아래 jitter 탐색으로 흡수.
- **회전**: 미지. out-of-plane 회전은 **42뷰 각각과 비교**함으로써 탐색하고(icosphere L2, 이웃 뷰 간격 ~37°), **in-plane 회전**은 뷰 축(z) 주위 K_ip=12개(30° 간격) 표본으로 관측 포인트를 회전시켜 탐색한다. K_ip를 빼면 in-plane 미정합으로 정상 후보 점수가 붕괴한다(반경 r 포인트가 최대 2r·sin15°≈0.52r 이동).

#### 거리 계산 가속 — 권장안: 뷰별 truncated distance field (TDF) voxel LUT

| 후보 | 판정 |
|---|---|
| 템플릿당 KD-tree 오프라인 캐시 | 탈락. 쿼리당 O(log N)이지만 포인터 추적·분기 구조라 GPU 배치화·TRT 이식이 안 됨(A4 위배). CPU 폴백 구현으로만 유지 |
| **뷰별 TDF voxel LUT (권장)** | 채택. 조회가 O(1) gather 연산 하나 → M×42×12×P 전체를 단일 배치 gather로 처리. TRT/CUDA 직행 |

S0 온보딩 시 뷰 v마다: depth 템플릿을 역투영한 가시면 포인트(뷰 프레임, median 센터링)로부터 3D unsigned distance transform을 계산, `τ_trunc`에서 절단해 저장.

```
tdf[v] : (48, 48, 48) float16, 커버 범위 [−0.6·D_cad, +0.6·D_cad]³
voxel_size = 1.2·D_cad / 48 = 0.025·D_cad     (D=200mm → 5mm)
τ_trunc = 4 voxel = 0.1·D_cad
메모리: 42 × 48³ × 2B ≈ 9.3 MB/물체 (fp16) — 수용 가능
```

τ_trunc가 0.1·D로 후한 이유: 뷰 양자화(37°) + in-plane 양자화(30°)에 의한 정합 오차가 주변부에서 ~0.1·D까지 발생 — S2는 포즈가 아니라 **판별**이 목적이므로 이 관용이 옳다. 정밀 정합은 S3/S4의 일.

#### 점수 수식

후보 j, 뷰 v, in-plane 각 θ에 대해:

```
q_i = R_z(θ) · p̃_i                        # 관측 포인트 회전 (뷰 축 in-plane)
d_i = tdf[v][ voxel(q_i) ]                 # O(1) gather, 범위 밖은 d_i = τ_trunc
s(j, v, θ) = (1/N_j) · Σ_i max(0, 1 − d_i / τ_trunc)     ∈ [0, 1]

S_depth(j) = max over (v, θ, δ) of s     # 2-pass:
  pass 1: 전체 42뷰 × 12각, jitter 없음
  pass 2: pass1 상위 3뷰에 대해 병진 jitter δ ∈ {−1,0,+1}³ voxel (27개) 추가 탐색
          — median 센터 편향(가림 시) 흡수용
```

#### 가림 대응 — 관측→템플릿 단방향인 이유

- 관측→템플릿: "모든 관측 포인트는 물체 표면 위에 있어야 한다" — 가림으로 물체 일부가 안 보여도 성립하는 조건. 가림에 불변.
- 템플릿→관측(역방향): "템플릿의 모든 가시 포인트가 관측에서 발견돼야 한다" — 가림·LiDAR 희소성 때문에 정상 후보도 대량 벌점. 채택 불가.
- 단방향의 부작용(작은 파편이 큰 물체 일부에 정합해 고점)은 ①크기 게이팅(하한)과 N_min이 차단한다 — 캐스케이드 순서가 이 상호보완을 전제로 설계됨.

#### 복잡도

```
pass1: M' × 42 × 12 × P  gather   (M' = 게이팅 생존 후보 수)
  예: M'=8, P=1000 → 4.0M gather
pass2: M' × 3 × 12 × 27 × P → 7.8M gather
합계 ~12M fp16 gather + 동수 FMA → GPU에서 수 ms 이하 (§5)
```

### 1.3 ③ DINOv2 ViT-S 의미 점수 (semantic_prior) — 약한 prior

- **crop 규약**: 마스크 bbox의 정사각 확장(패딩 계수 1.25, 1280×800 경계에서 clamp) → 마스크 밖 픽셀을 ImageNet mean 색으로 치환(**마스크 적용함** — 배경 leakage가 약한 prior마저 오염시키는 것 방지, SAM-6D의 masked 규약 승계) → 224×224 리사이즈, ImageNet 정규화.
- **점수**: cls token(384-d, L2 정규화) vs S0 캐시 템플릿 descriptor `(42, 384)` cosine → SAM-6D의 `avg_5` 집계 승계: `S_sem = mean(top5(cos))`. 
- **캐시**: S0에서 42뷰 클린 CAD 렌더의 cls를 fp16으로 저장(42×384×2B = 32KB/물체). detector.py의 `descriptors.pth` 등가물이나 ViT-S·cls 전용으로 축소.
- **호출 대상**: ②의 `S_depth ≥ θ_depth_min`(기본 0.3) 생존 후보만. forward 횟수가 ISM 레이턴시 주범이었다는 M0 진단의 직접 반영.

### 1.4 ④ 융합·임계·top-k

```
S2(j) = w_d·S_depth + w_g·S_size + w_s·S_sem
기본 가중: w_d = 0.6, w_g = 0.2, w_s = 0.2      (M0-3 오염 프로토콜로 튜닝, E6)

동적 규칙 (01 문서의 모호한 문구를 다음으로 확정):
  N_j ≥ N_geo_ok(=150): 위 기본 가중 (기하 신뢰 → w_s 낮게 고정)
  N_min ≤ N_j < N_geo_ok: w_d를 N_j에 비례해 선형 하향, 잔여분을 w_s에 이전
      (상한 w_s ≤ 0.4 캡), 후보에 low_geo 플래그 부착
  N_j < N_min: ① 하드 게이트에서 이미 제외 — 단 §6.1의 구제 경로 참조

판정:  S2(j) ≥ θ_S2 (기본 0.45) 인 후보를 점수순 정렬, 인스턴스 NMS(마스크 IoU 0.5)
       후 top-k_inst(=3)를 S3로 전달. 전무하면 "미검출" 조기 종료.
```

시나리오 정합성: 재도장 → S_sem 붕괴해도 w_s=0.2뿐이라 판별 유지. 유사 형상 혼입 → S_depth·S_size 동률일 때 S_sem 0.2가 캐스팅보트 — ③을 제거하지 않고 강등하는 이유.

---

## 2. S4 정련·검증 상세

입력: S3 가설 `{(R_h, t_h, s3_h)}` (인스턴스당 k=3), 관측 포인트 `P_obs (N,3)` + 유효 픽셀 좌표, S0 캐시(dense 포인트+노멀 `X_m (16384,3)`, `N_m (16384,3)`, 대칭 메타), S2 점수.

### 2.1 ① top-k 가설 선별 — 투영 depth 잔차 + 가시율

**규약**: 풀 depth 렌더링을 하지 않는다. 관측이 희소하므로 비교 지점은 **LiDAR 유효 픽셀뿐** — S0 캐시 포인트를 포인트 스플랫 z-buffer로 뿌리고 유효 픽셀에서만 gather 한다. 메시 래스터라이저 불요.

```python
def score_hypothesis(R, t, X_m, obs_uv, obs_z, K,
                     stride=2, tau_z=0.010, gamma=1.0):
    # X_m: (Nm,3) S0 캐시 포인트(서브셋 2048 사용), obs_uv:(N,2) 유효 픽셀, obs_z:(N,)
    Xc  = X_m @ R.T + t                        # (Nm,3) 카메라 좌표
    uv  = project(Xc, K)                       # (Nm,2) float, 1280×800 좌표계
    g   = floor(uv / stride)                   # 640×400 그리드 셀
    zbuf = scatter_min(Xc[:,2], g)             # 셀별 최소 z (GPU scatter, 미점유=+inf)
    zbuf = min_pool_3x3(zbuf)                  # 스플랫 구멍 메움 (2048pt는 성기므로 필수)

    zm       = zbuf[floor(obs_uv / stride)]    # (N,) 관측 유효 픽셀에서만 gather
    covered  = isfinite(zm)                    # 모델 footprint가 관측 픽셀을 덮는가
    r        = obs_z - zm                      # depth 잔차 (관측 − 모델)
    inlier   = covered & (abs(r) < tau_z)

    v     = covered.float().mean()             # 가시율: 관측 설명률
    s_res = (relu(1 - abs(r[covered])/tau_z)).mean() if covered.any() else 0.
    free_viol = (covered & (r >  tau_z)).float().mean()  # 모델이 관측 앞에 떠 있음
    return s_res * v**gamma, dict(v=v, inlier=inlier, free_viol=free_viol)
```

- `tau_z=10mm` 기본(LiDAR 고정밀이므로 타이트하게 시작, MAD 적응은 §6.2).
- 잔차 부호 구분: `r > τ`(모델이 관측보다 앞) = free-space 위반 — 오포즈의 강한 증거. `r < −τ` = 관측이 모델 앞 = 가림/오염 퇴적 가능성 — 관대하게. `free_viol`을 별도 진단으로 유지.
- k개 가설을 대칭 등가 dedupe(§2.3) 후 점수순 정렬, **최상 가설만** ICP로(예산 규약, M2-C3와 일치). 최상 가설 ICP가 reject되면 차순위로 1회 재시도(§2.4).

### 2.2 ② projective point-to-plane ICP

**대응 방향 결정: source = 관측 LiDAR 포인트(희소), target = 포즈 적용 CAD(dense).** 근거:

1. 가림 강건성 — S2-②와 동일 논리. CAD→관측 방향은 가림부 CAD 포인트가 대응을 못 찾아 대량 아웃라이어 생성.
2. 관측은 수백~수천 pt뿐 → 정규방정식 조립 비용이 관측 수에 비례해 최소.
3. **노멀은 CAD 측(S0 캐시)을 쓴다** — point-to-plane의 plane을 관측 측에 세우려면 희소 LiDAR에서 노멀 추정이 필요한데, 수백 pt 희소 클라우드의 노멀은 신뢰 불가. CAD 노멀은 정확·무료.
4. 대응은 projective association — kNN/KD-tree 없이 투영 그리드 조회(A4 규약).

```python
class ProjectiveICP:
    def refine(self, R, t, P, X_m, N_m, K, iters=10, delta=0.005,
               tau_assoc=0.02, stride=2):
        # P:(N,3) 관측, X_m/N_m:(16384,3) CAD 포인트/노멀 (물체 좌표)
        for it in range(iters):
            Xc, Nc = X_m @ R.T + t, N_m @ R.T                # 모델을 카메라 프레임에
            grid   = scatter_argmin_z(project(Xc,K)//stride) # 셀→최전방 모델 idx
            # 대응: 관측점 투영 셀의 3×3 이웃 후보 중 3D 최근접 (kNN 없음)
            j      = gather_nearest3d(P, grid, Xc, win=3)    # (N,), 거리>tau_assoc→무효
            q, n   = Xc[j], Nc[j]
            r      = ((P - q) * n).sum(-1)                   # (N,) p2pl 잔차
            w      = torch.where(r.abs() < delta, 1.0, delta / r.abs())  # Huber
            J      = torch.cat([torch.cross(q, n), n], -1)   # (N,6): [ (q×n)ᵀ, nᵀ ]
            H      = (w[:,None,None] * J[:,:,None] @ J[:,None,:]).sum(0)  # 6×6
            b      = (w[:,None] * J * r[:,None]).sum(0)      # 6
            xi     = torch.linalg.solve(H + lam*I6, b)       # ω, ν  (lam=1e-9 안정화)
            R, t   = exp_so3(xi[:3]) @ R, exp_so3(xi[:3]) @ t + xi[3:]  # 좌측 갱신
            if xi[:3].norm() < 1e-4 and xi[3:].norm() < 5e-5: break     # 0.006°/0.05mm
            if it > 0 and abs(rmse_prev - rmse)/rmse_prev < 1e-3: break
        return R, t, dict(H=H, r=r, w=w, valid=(j>=0))
```

- 선형화 규약: 좌측 섭동 `T ← Exp(δξ)·T`, `r(δξ) ≈ r − J·δξ`, `J = [(q×n)ᵀ, nᵀ]` → `(Σ w JᵀJ) δξ = Σ w Jᵀ r`. 부호 규약은 합성 데이터 단위 테스트로 고정할 것(구현 필요).
- Huber δ = 5mm 기본(LiDAR 정밀도 σ의 ~3배 가정 — D2-a 확정 후 `max(3σ, 3mm)`로 재설정).
- 6×6 조립은 전부 배치 reduce(GPU), solve는 6×6이라 어디서 하든 무시 가능 비용.
- **퇴화 검출**: `cond = λ_min(H)/λ_max(H) < 1e-6`이면 관측 기하가 평면 조각 등 6-DoF 미구속 — `degenerate` 플래그, §2.4에서 accept 금지.
- ICP 후 §2.1 스코어러 재실행 → 최종 잔차 통계(inlier ratio, inlier RMSE, v, free_viol) 산출.

### 2.3 ③ 신뢰도 — 로지스틱 보정 + 대칭군 등가 평가

**특징 벡터** (ICP 후 재스코어 기준):

```
x = [ inlier_ratio,            # |r|<τ_z 비율 (covered 대비)
      rmse_inlier / τ_z,       # inlier 한정 RMSE (robust, §6.2)
      v,                       # 가시율(관측 설명률)
      free_viol,               # free-space 위반율
      S2_depth, S2_sem,        # S2 항목 점수
      s3_match,                # S3 매칭 점수
      log10(N_obs),            # 관측 포인트 수
      log10(cond_H),           # ICP 관측성
      border_flag ]            # 마스크 절단 여부
p_conf = sigmoid(wᵀx + b̂)      # 필요 시 isotonic 후처리
```

**캘리브레이션 데이터 구성**: 전체 파이프라인을 (a) BOP val — T-LESS·ITODD 우선 + YCB-V, (b) M0-3 오염 프로토콜(전면 도색 변경 / 녹·분진 텍스처 / 국소 기하 범프 합성) 렌더에 돌려 (x, label) 수집. label = **대칭 인식 오차** 기준 정답 여부(BOP MSSD/MSPD 임계 준용). 표본 ~1만 건이면 10-d 로지스틱에 충분 — 추정(파라미터 11개 대비 표본 수 근거). 오염 셋을 반드시 포함해야 "오염이지만 정답 포즈" 표본이 양성으로 학습됨(§6.2).

**대칭군 등가 평가** (S0 대칭 메타 소비):

```python
def canonicalize(R, sym):           # sym: 이산 회전군 {S_g} + 연속축 리스트
    for axis in sym.cont_axes:      # 연속 대칭축: swing-twist 분해로 twist 제거
        a  = R @ axis               # 카메라 프레임 축
        R  = remove_twist(R, axis)  # R = R_swing·R_twist(axis) → R_swing만 유지
    g_star = argmin_g geodesic_angle(R @ S_g, R_anchor)   # 이산군: 정준 대표 선택
    return R @ sym.S[g_star]        # R_anchor = I (고정 기준 프레임)

def sym_aware_error(R_est, t_est, R_gt, t_gt, X_m, sym):  # 캘리브레이션 라벨용
    return min over g of avg_i || (R_est·S_g)·x_i + t_est − (R_gt·x_i + t_gt) ||   # ADD-S/MSSD식

# 온라인 사용처 2곳:
# (1) 가설 dedupe: canonicalize(R_h) 후 geodesic < 15° & Δt < 0.05·D 이면 동일 가설로 병합
#     → ICP 예산을 실제로 다른 가설에만 사용
# (2) 캘리브레이션 라벨링: sym_aware_error로 정답 판정 → 대칭 물체 오reject 원천 차단
```

### 2.4 ④ accept/reject 판정 흐름

```
1. 가설 k개 → 대칭 dedupe → 잔차 스코어 → 최상 가설 ICP → p_conf 계산
2. 하드 가드 (로지스틱 이전, 무조건):
     N_inlier < 25            → accept 금지 (최대 UNCERTAIN)
     degenerate (cond_H)      → accept 금지
     free_viol > 0.05         → REJECT (관측 앞 공간을 모델이 침범 = 오포즈)
3. 3-구간 임계:
     p_conf ≥ θ_acc (기본 0.7)          → ACCEPT (최종 R,t + p_conf 출력)
     θ_rej ≤ p_conf < θ_acc (기본 0.3~) → UNCERTAIN: 차순위 가설로 ICP 1회 재시도
                                           → 그래도 미달 시 재촬영/재시도 신호 반환
     p_conf < θ_rej                      → REJECT
4. 임계 운용: θ_acc는 캘리브레이션 ROC에서 "오수락 <1% @ 리콜 손실 ≤5%" 지점으로
   초기 설정(M3-4), 배포 현장에서 공차 요구에 따라 조정 가능한 운영 파라미터로 노출.
   보정된 p_conf 덕에 임계의 의미가 "확률"로 고정 → 현장 튜닝이 예측 가능.
```

---## 3. 모듈 인터페이스

```python
# ---------- shape6d/identify/ ----------
@dataclass
class Candidate:
    mask: Tensor          # (800,1280) bool     (주의: H×W = 800×1280)
    pts:  Tensor          # (N_j,3) 마스크 내 LiDAR 유효 포인트, 카메라 좌표 [m]
    uv:   Tensor          # (N_j,2) 해당 유효 픽셀 좌표
    scores: dict          # {"size":…,"depth":…,"sem":…,"fused":…}
    flags:  set           # {"low_geo","border","occluded"}

class SizeGate:                                   # identify/size_gate.py
    def __init__(self, tau_up=0.15, beta_occ=0.35, beta_occ_relaxed=0.25, n_min=30): ...
    def __call__(self, cands: list[Candidate], D_cad: float) -> list[Candidate]: ...

class PointToTemplateMatcher:                     # identify/depth_match.py
    def __init__(self, cache: ObjectCache, k_inplane=12, tau_trunc_rel=0.1,
                 jitter_top_views=3): ...
    def score(self, cands: list[Candidate]) -> Tensor:   # (M',) S_depth
        """내부: (M',42,12,P) 배치 TDF gather → pass2 jitter. cache.tdf:(42,48,48,48) fp16"""

class SemanticPrior:                              # identify/semantic_prior.py
    def __init__(self, vits_engine, cache: ObjectCache, pad=1.25, topk_agg=5): ...
    def score(self, rgb: Tensor, cands: list[Candidate]) -> Tensor:  # (M'',) S_sem

class Identifier:                                 # identify/score_fusion.py
    def __init__(self, gate, matcher, prior, w=(0.6,0.2,0.2),
                 theta_depth_min=0.3, theta_s2=0.45, k_inst=3): ...
    def __call__(self, rgb, cands) -> list[Candidate]:   # top-k_inst, 점수·플래그 부착

# ---------- shape6d/verify/ ----------
class HypothesisScorer:                           # verify/render_residual.py
    def __init__(self, K: Tensor, stride=2, tau_z=0.010, gamma=1.0): ...
    def __call__(self, poses: Tensor, X_m: Tensor, cand: Candidate
                 ) -> tuple[Tensor, list[dict]]:  # poses:(k,4,4) → (k,) 점수 + 진단

class ProjectiveICP:                              # verify/icp.py
    def __init__(self, K, iters=10, huber_delta=0.005, tau_assoc=0.02,
                 stride=2, conv_rot=1e-4, conv_trans=5e-5): ...
    def refine(self, pose: Tensor, cand: Candidate, X_m, N_m
               ) -> tuple[Tensor, dict]:          # (4,4) 정련 포즈 + {H,r,w,inlier,...}

class SymmetryHandler:                            # verify/symmetry_eval.py
    def __init__(self, sym_meta: dict): ...       # S0 symmetry.json 소비
    def canonicalize(self, R: Tensor) -> Tensor
    def dedupe(self, poses: Tensor, D_cad, ang_th=15.0, t_th_rel=0.05) -> Tensor
    def sym_aware_error(self, pose_est, pose_gt, X_m) -> float   # 캘리브레이션용

class ConfidenceCalibrator:                       # verify/confidence.py
    def __init__(self, weights_path: str): ...    # 학습된 w,b (+isotonic) 로드
    def features(self, icp_diag, hyp_diag, cand, s3) -> Tensor   # (10,)
    def __call__(self, feats) -> float            # p_conf ∈ [0,1]
    @staticmethod
    def fit(samples_npz: str, out_path: str): ... # M3-3 오프라인 학습 CLI

class Verifier:                                   # verify/verifier.py (오케스트레이터)
    def __init__(self, scorer, icp, sym, calib, theta_acc=0.7, theta_rej=0.3,
                 n_inlier_min=25, free_viol_max=0.05, retry=1): ...
    def __call__(self, hyps, cand, cache) -> VerifyResult
        # VerifyResult: {pose:(4,4), p_conf, verdict: ACCEPT|UNCERTAIN|REJECT, diag}
```

**S0 캐시 파일 포맷 추가분** (`<obj_id>/cache.npz` 키):

| 키 | shape / dtype | 내용 |
|---|---|---|
| `dense_pts`, `dense_normals` | (16384,3) fp16 | Poisson disk 포인트·노멀 (S4 ICP target) |
| `verify_pts` | (2048,3) fp16 | 가설 스코어링용 서브셋 |
| `diameter` | scalar f32 | 최대 포인트쌍 거리 |
| `tdf` | (42,48,48,48) fp16 | 뷰별 truncated distance field, [−0.6D,0.6D]³ |
| `tpl_center` | (42,3) f32 | 뷰별 가시면 median (병진 정렬 기준) |
| `view_rots` | (42,3,3) f32 | icosphere L2 뷰 회전 |
| `dino_cls` | (42,384) fp16 | DINOv2 ViT-S cls 템플릿 descriptor |
| `symmetry.json` | — | 이산 회전군 행렬 리스트 + 연속축, 수동 오버라이드 병합 결과 |

---

## 4. 레이턴시 추정 (전부 추정 — M0-2 환산 계수로 재검증 대상)

Orin NX 환산 근거: TRT FP16 기준 데스크톱 최상급 대비 5~10× — 대형 커널은 연산량 격차(~10×), 소형 커널은 launch overhead 지배로 격차 축소(~3×)라는 공개 벤치 일반 경향 기반 추정.

| 항목 | RTX PRO 6000 | Orin NX (추정) | 산출 근거 (한 줄) |
|---|---|---|---|
| S2-① 크기 게이팅 (M=20) | 0.1 ms | 0.3 ms | percentile+산술뿐, 커널 launch가 지배 |
| S2-② TDF 정합 pass1+2 (M'=8, P=1000) | 1.5 ms | 8 ms | ~12M fp16 gather+FMA, 메모리 바운드 |
| S2-③ DINOv2 ViT-S (crop ≤6) | 2 ms | 15 ms | ViT-S 224² ≈ 4.6 GFLOPs/crop × 6, TRT FP16 |
| S2-④ 융합·NMS | <0.1 ms | 0.2 ms | 산술 |
| **S2 합계** | **~4 ms** | **~24 ms** | 01 문서 예산 50–100ms 내 |
| S4-① 가설 스코어 (k=3, 2048pt) | 1 ms | 5 ms | scatter-min+gather ×3, 소형 커널 |
| S4-② ICP 8 iter (N≤2000, 16k target) | 4 ms | 20 ms | iter당 scatter 16k + gather 2k×9 + 6×6 reduce, launch 지배 |
| S4-③④ 신뢰도·판정 | <0.1 ms | 0.2 ms | 10-d 로지스틱 |
| UNCERTAIN 재시도 1회 (발생 시) | +5 ms | +25 ms | ①+② 재실행 |
| **S4 합계 (재시도 미발생)** | **~5 ms** | **~26 ms** | 01 문서 예산 30–60ms 내 |

주의: Orin 수치는 실기 부재로 전부 추정치이며 M4 전제조건으로 실측 이월(D1). 소형 커널 다수 구조라 CUDA graph 캡처가 Orin에서 특히 유효할 것으로 추정.

---

## 5. 실패모드 분석

### 5.1 희소 포인트 극소 후보 (N_j < 50)

| 구간 | S2 규칙 | S4 규칙 |
|---|---|---|
| N_j < 30 (`N_min`) | ① 하드 게이트 탈락이 원칙. 단 **구제 경로**: S1이 후보를 1개만 냈고 그것이 이 구간이면, `S_size(완화 게이트) + S_sem`만으로 판별하되 `low_geo` 플래그 강제 | 검증 불능 — `N_inlier < 25` 하드 가드에 걸려 **ACCEPT 절대 불가**, 최대 UNCERTAIN(재촬영 신호). "모르면 모른다고 말한다"가 A4 신뢰도 원칙 |
| 30 ≤ N_j < 150 | §1.4 동적 가중(w_d 하향, w_s ≤0.4 캡), TDF 점수의 분산이 커지므로 θ_depth_min을 0.3→0.25로 완화 | ICP는 수행하되 cond(H) 퇴화 검사 필수(포인트가 적으면 평면 조각일 확률 상승), τ_assoc 2배 완화, p_conf 특징의 log10(N_obs)가 로지스틱에서 자동 벌점 |
| N_j ≥ 150 | 기본 동작 | 기본 동작 |

### 5.2 오염 잔차 상승 × reject 임계의 상호작용

문제: 퇴적·부착물 오염은 정답 포즈에서도 국소 잔차를 올린다. 순진한 "평균 잔차 < 임계" 판정은 오염 표본을 체계적으로 reject → 오염 강건이라는 프로젝트 목적 자체가 무너진다. 완화 장치 4겹:

1. **Huber + inlier 한정 통계**: RMSE는 inlier(|r|<τ_z)에서만 계산. 오염 패치는 아웃라이어로 분리되어 `inlier_ratio`만 낮추고 `rmse_inlier`는 유지 — "정답 포즈 + 국소 오염"(inlier_ratio↓, rmse_inl 낮음)과 "오포즈"(둘 다 나쁨)가 특징 공간에서 분리된다. 단일 잔차 임계가 아니라 로지스틱이 이 2차원을 함께 보는 것이 핵심.
2. **MAD 적응 임계**: `τ_z_eff = clamp(1.4826·MAD(r)·2.5, τ_z, 2τ_z)` — 잔차 분포가 전반적으로 부풀면(전면 얇은 퇴적) 임계를 상한 2τ_z까지만 따라 올림. 상한이 없으면 오포즈도 수용하므로 캡 필수.
3. **부호 비대칭**: 퇴적은 관측을 모델 **앞**으로 이동(r<0 방향) → 음의 잔차에 관대, free-space 위반(r>+τ)만 hard reject 신호로 유지. 오염은 물리적으로 free-space 위반을 만들지 않는다.
4. **캘리브레이션 셋에 오염 포함**(§2.3): "오염이지만 정답" 표본이 양성으로 학습되어 로지스틱 경계 자체가 오염 방향으로 이동. reject 임계를 손튜닝하지 않고 데이터로 흡수하는 구조.

### 5.3 기타

- **대칭 오reject**: 가설 dedupe와 라벨링 양쪽에 §2.3 canonicalize 적용으로 차단. T-LESS 대칭군에서 오reject 0이 M3 게이트.
- **유사 형상 혼입 + 재도장 동시 발생**: S2에서 ③이 붕괴하고 ①②는 동률 — S2가 오선별해도 S4 잔차가 형상 차이를 잡는 마지막 방어선(free_viol·inlier_ratio). 형상까지 동일한 타 부품은 정의상 기하 파이프라인으로 구분 불가 — 운영 문서(M5-4)에 한계로 명시할 것.
- **마스크 절단(이미지 경계)**: `border_flag`가 게이트 완화(§1.1)와 로지스틱 특징(§2.3) 양쪽에 전달되어 일관 처리.

---

## 6. 미확정 의존성

| 항목 | 영향 지점 | 처리 |
|---|---|---|
| D2-a: 물체 위 유효 포인트 수 | N_min/N_geo_ok 구간, τ_z·δ_huber(σ_lidar 함수), TDF 해상도 | 전부 config 파라미터로 노출, 실측 후 1회 재튜닝 |
| Orin NX 실기 | §4 전 수치 | M0-2 환산 계수 운용 → M4 실측 대체 |
| `MaskedPatch_MatrixSimilarity` 의존 코드 | 없음 — appearance 계열 완전 제거로 Shape6D는 이 모듈을 이식하지 않음 | — |

관련 파일: `/Users/jaewoo/Documents/SAM-6D/SAM-6D/Instance_Segmentation_Model/model/detector.py` (대체 대상 분석), `/Users/jaewoo/Documents/Shape6D/docs/01_stage_design.html` §S2·S4, `/Users/jaewoo/Documents/Shape6D/docs/02_development_plan.html` M1-4·M3.