# Shape6D — S0 온보딩 + S1 제안 생성 상세 설계 (v0.1)

기반 코드 확인: `/Users/jaewoo/Documents/SAM-6D/SAM-6D/Instance_Segmentation_Model/model/detector.py`, `configs/model/ISM_sam.yaml`, `ISM_fastsam.yaml`, `model/sam.py`, `Render/render_custom_templates.py`

**승계한 SAM-6D 관행 요약**: (1) 온보딩 산출물은 템플릿 디렉토리에 텐서 캐시(`descriptors.pth`, `pointcloud.pth`, `template_poses.npy`)로 저장하고 존재하면 재사용(`reset_descriptors` 플래그) — Shape6D는 이를 단일 npz 스키마로 통합. (2) 템플릿 42뷰는 `cam_poses_level0.npy`(icosphere 42뷰) 규약, 렌더 전 CAD를 `1/(2·r_bsphere)`로 정규화. (3) 세그멘터 입력은 aspect 유지 폭 640 리사이즈 후 `F.interpolate`로 원 해상도 복원. (4) 마스크 후처리: `min_box_size 0.05`(상대), `min_mask_size 3e-4`(상대), mask NMS `0.25`. (5) 모든 모듈은 config(_target_) 주입식 생성자 — Shape6D도 동일하게 hydra 호환 구조 유지.

---

## 1. S0 온보딩 파이프라인 (오프라인)

### 1.1 처리 흐름

```
CAD 메시 ──▶ [V] 입력 검증 ──▶ [S] Poisson disk 샘플+노멀 ──▶ [F] sparse 서브셋
                                        │
                                        ├──▶ [T] depth 템플릿 42뷰 렌더
                                        ├──▶ [Y] 대칭 자동검출
                                        └──▶ [C] 캐시 직렬화 (단일 .npz + manifest.json)
```

### 1.2 [V] CAD 입력 검증

| 검사 | 방법 | 실패 시 |
|---|---|---|
| 로드 가능성 | `trimesh.load(path, force='mesh')` | 에러 종료 |
| 단위 | `--unit {mm,m}` **필수 명시**(SAM-6D의 mm/1000 암묵 규약이 실패 모드였음). 로드 후 bbox 대각선 D가 `[0.005, 3.0] m` 밖이면 경고+`--force` 없으면 종료 | 경고/종료 |
| 워터타이트 | `mesh.is_watertight`. False면 `trimesh.repair.fill_holes()` 시도 → 여전히 False면 **샘플링·렌더는 진행하되** manifest에 `watertight: false` 기록(Poisson disk·depth 렌더는 비워터타이트도 동작; 대칭검출 신뢰도만 하락) | 경고 |
| 퇴화 면/중복 정점 | `mesh.remove_degenerate_faces()`, `merge_vertices()` | 자동 수리 |
| 노멀 일관성 | `trimesh.repair.fix_normals(mesh)` (워터타이트일 때만) | 자동 수리 |
| 스케일 메타 | 내부 표현은 **미터(m) 고정**. `diameter = max pairwise dist ≈ bbox 대각`, `r_bsphere`, `bbox_min/max` 산출 | — |

원점 규약: **모델 좌표계 그대로 유지**(BOP 관행). 중심 이동은 하지 않고 `center_offset = bbox 중심`을 메타로 저장 — S3가 CAD 좌표계 기준 R,t를 출력해야 하므로.

### 1.3 [S] Poisson disk 샘플링 + 노멀

- **N_dense = 4096 제안.** 근거: SAM-6D는 오브젝트 측 2048pt(`pointcloud_sample_num: 2048`, PEM fine 2048)를 쓰지만, Shape6D S2의 단방향 point-to-template 거리와 S3 오브젝트 측 특징 추출이 같은 캐시를 공유하므로, 장면 측 유효 LiDAR 포인트(수백~수천)보다 오브젝트 측이 항상 조밀해야 단방향 거리의 bias가 없다 → 2048의 2배인 4096. 메모리 비용은 4096×(3+3)×4B ≈ 96KB로 무시 가능. `--n-dense`로 파라미터화.
- 방법: `trimesh.sample.sample_surface_even(mesh, count=N_dense*2)` 후 반경 기반 rejection으로 정확히 N_dense개 유지(sample_surface_even은 개수 미보장 → 초과 샘플 후 voxel-hash rejection, **구현 필요**). 노멀은 샘플이 속한 face 노멀 사용(비워터타이트에서도 안정).
- **sparse 서브셋 N_sparse = 196** (SAM-6D `coarse_npoint 196` 승계). 오프라인이므로 FPS 사용 허용(A4의 커스텀 op 금지는 온라인 경로 제약) — numpy O(N·M) FPS로 충분. `sparse_idx [196] int32`를 dense에 대한 인덱스로 저장(중복 저장 방지 + dense↔sparse 대응 유지).

### 1.4 [T] depth 템플릿 42뷰 렌더 스펙

| 항목 | 값 | 근거 |
|---|---|---|
| 뷰포인트 | icosphere L2 42뷰 = SAM-6D `cam_poses_level0.npy` 회전 성분 재사용 | S2 템플릿 정합·S4 검증 공용, SAM-6D 규약 유지 |
| 해상도 | **224×224** | S2 판별의 crop 규약(224)과 일치, depth-only라 재렌더 비용 낮음 |
| 카메라 거리 | `d_cam = 2.5 × r_bsphere` (물체가 프레임의 약 70% 차지, FOV 45°: tan(22.5°)·2.5r ≈ 1.04r > r 이므로 잘림 없음) | SAM-6D의 "정규화 후 거리 2.0×(반경0.5 기준 4r)"보다 가깝게 잡아 depth 해상력 확보. **거리 정규화**: 물체 크기와 무관하게 화면 점유율이 일정 → 템플릿 간 스케일 정규화 불필요 |
| 렌더러 | `pyrender` OffscreenRenderer(EGL) — depth 채널만. BlenderProc(SAM-6D 방식)은 RGB PBR용이라 과함 | 온보딩 수 초 내 완료 |
| 저장 | depth `uint16`(mm 단위, 0=배경), 뷰당 K(고정 fx=fy=270.9=112/tan(22.5°), cx=cy=112) + `T_obj2cam [4,4]` | S4 잔차 비교 시 그대로 사용 |
| 템플릿 포인트 | 뷰별 depth→역투영한 유효 픽셀 중 **512pt 균일 서브샘플** `tpl_pts [42,512,3] (물체 좌표계)` | S2 단방향 거리를 depth 이미지가 아닌 포인트 집합으로 계산(희소 LiDAR와 대칭) |
| (보조) RGB 템플릿 | 동일 42뷰 중 12뷰만 RGB 렌더(DINOv2 cls용, 224×224) | S2의 약한 prior 전용, A1에 따라 최소화 |

### 1.5 [Y] 대칭 자동검출

대상: (i) 회전축 무한대칭(원통·원뿔류), (ii) 유한 차수 회전대칭(n=2..12), (iii) 반사면.

```
입력: P [4096,3] (dense, centroid 제거한 P̄), diameter D
τ_sym = 0.004 · D          # chamfer 임계, 추정: BOP 대칭 정의 관행(모델 직경 대비 잔차)에서 채택

def chamfer_1way(A, B):     # A→B 단방향 평균 거리. 오프라인이므로 KDTree(scipy) 허용
    return mean(min_dist(a, B) for a in A)

후보 축 생성:
  axes = PCA(P̄)의 3개 주축
       + 관성텐서 고유축 (PCA와 중복 시 제거, 각도 5° 이내 병합)
  # 고유값이 근사 중복(비율 < 1.02)이면 해당 평면 내 축 방향이 부정 →
  # 그 평면에서 30° 간격 6개 축 추가 스캔

각 축 a에 대해 (회전대칭):
  scores = []
  for θ in range(5°, 180°, 5°):                # 거친 스캔
      scores.append(chamfer_1way(R(a,θ)·P̄, P̄))
  if max(scores) < τ_sym:                       # 모든 각에서 정합
      → 무한대칭 축 (order = inf)
  else:
      for n in [2,3,4,5,6,8,10,12]:             # 유한 차수 검사
          if chamfer_1way(R(a, 360°/n)·P̄, P̄) < τ_sym:
              θ* = argmin 주변 ±4°를 0.5° 간격 미세 정렬   # 축 이산화 오차 보정
              order = max(order, n)

각 반사면 후보 (PCA 3평면):
  if chamfer_1way(Mirror(plane)·P̄, P̄) < τ_sym: → 반사대칭 기록

출력 정규화:
  유한 대칭 → 회전행렬 집합 sym_rots [S,3,3] (항등 포함, 군 폐포 계산)
  무한대칭 → sym_axes [A,3] + flag
  반사면   → S3/S4에서 사용하지 않고 메타만 기록 (포즈 등가 평가는 회전군만)
```

주의: 검출은 **제안**일 뿐이며 CLI가 요약을 출력하고 `--sym-override json`으로 사람이 정정 가능해야 함(설계 문서의 "대칭 미등록 → S4 오reject" 실패 모드 대응).

### 1.6 캐시 파일 스키마

`<out_dir>/<obj_id>/onboard_v1.npz` (단일 npz, SAM-6D의 .pth 산발 저장 관행을 통합) + `manifest.json`.

| 필드 | shape | dtype | 설명 |
|---|---|---|---|
| `pts_dense` | [4096,3] | float32 | Poisson disk 샘플, m, 모델좌표계 |
| `nrm_dense` | [4096,3] | float32 | 단위 노멀 |
| `sparse_idx` | [196] | int32 | dense에 대한 FPS 인덱스 |
| `tpl_depth` | [42,224,224] | uint16 | depth 템플릿, mm, 0=배경 |
| `tpl_K` | [3,3] | float32 | 템플릿 공통 intrinsic |
| `tpl_pose` | [42,4,4] | float32 | T_obj2cam |
| `tpl_pts` | [42,512,3] | float32 | 뷰별 가시 표면 포인트(모델좌표계) |
| `tpl_rgb` | [12,224,224,3] | uint8 | 보조 RGB 템플릿 |
| `tpl_rgb_view_idx` | [12] | int32 | 42뷰 중 어떤 뷰인지 |
| `dino_cls` | [12,384] | float32 | DINOv2 ViT-S cls (S2 prior, 온보딩 시 1회 계산) |
| `sym_rots` | [S,3,3] | float32 | 유한 회전대칭군(항등 포함, S≥1) |
| `sym_axes` | [A,3] | float32 | 무한대칭 축(없으면 A=0) |
| `diameter` | [] | float32 | m |
| `bbox` | [2,3] | float32 | min/max, m |
| `center_offset` | [3] | float32 | bbox 중심(모델좌표계) |
| `geo_feat_dense` | [4096,C_g] | float16 | S3 기하 인코더 오브젝트 측 특징 (인코더 확정 후 채움, v1에서는 옵션 필드) |

`manifest.json`: `{schema_version, obj_id, cad_path, cad_sha256, unit, watertight, sym_summary(사람용 문자열), created_at, tool_version}`.

### 1.7 온보딩 CLI

```bash
shape6d-onboard \
  --cad /path/model.ply --unit mm \
  --out /path/cache_dir --obj-id wheel_hub_a \
  [--n-dense 4096] [--n-sparse 196] [--n-views 42] \
  [--sym-override sym.json] [--no-sym-detect] [--force] [--overwrite]
```

```python
def onboard(cad_path: str, unit: str, out_dir: str, obj_id: str,
            n_dense: int = 4096, n_sparse: int = 196, n_views: int = 42,
            sym_override: dict | None = None, force: bool = False) -> "OnboardResult":
    """반환: OnboardResult(cache_path, manifest, warnings: list[str])"""
```

---

## 2. S1 제안 생성 상세 (온라인)

### 2.1 EfficientViT-SAM-L0 입력 규약 (1280×800)

- EfficientViT-SAM **L 시리즈의 공식 입력은 512×512**(XL 시리즈만 1024). 전처리는 SAM 규약(긴 변 리사이즈 + 우/하 zero-pad)을 따름. ※ L0 이미지 임베딩 grid 크기(64×64 유지 여부)는 공식 구현 확인 후 확정 — **구현 시 확인 필요**로 표기.
- **1280×800 경로**: 긴 변 1280→512 (scale s=0.4) → 512×320 → 하단 192px zero-pad → 512×512. LiDAR 픽셀 좌표는 `uv_512 = uv_full * 0.4` (pad는 우/하이므로 offset 불필요).
- **마스크 복원 경로**: decoder low-res 마스크 → 512×512로 upsample → `[:320, :512]` crop(pad 제거) → `F.interpolate(..., size=(800,1280), mode='bilinear')` → `> 0` threshold → bool. (SAM-6D `sam.py`의 `postprocess_resize`와 동일 패턴, box 좌표는 `/s` 스케일.)
- 참고: SAM-6D ISM은 폭 640 aspect 유지 리사이즈를 썼다(정사각 강제 없음, `segmentor_width_size: 640`). EfficientViT-SAM은 학습 해상도 512 고정이므로 letterbox 방식을 채택한다.
- 이미지 인코더는 **프레임당 1회** 실행하고 임베딩을 보관, 프롬프트 배치는 decoder만 반복 호출.

### 2.2 LiDAR 유효 포인트 → 포인트 프롬프트 (의사코드)

입력: `lidar_xyz [N_L,3]`(카메라 좌표계, m), `lidar_uv [N_L,2]`(1280×800 픽셀, 사전 캘리브 투영), N_L은 프레임 전체 수천~수만, 물체 위 수백~수천(D2-a 미확정 → 전 파라미터를 pt 밀도 무관하게 상대/미터 단위로 정의).

```
파라미터 (모두 config):
  ransac_dist = 0.008 m, ransac_iters = 200, max_planes = 2,
  plane_min_ratio = 0.15          # 전체의 15% 이상 차지하는 평면만 제거 (물체 평면 오제거 방지)
  voxel = 0.02 m                  # 클러스터 그리드
  min_cluster_pts = 15            # D2-a 하한(수백pt/물체)의 ~5%, 추정
  K_prompts = 3                   # 클러스터당 대표점
  size_gate = [0.2, 2.0] × max(diameter of onboarded objects)   # 느슨하게 (리콜 우선)

def make_prompts(lidar_xyz, lidar_uv):
    # 1) 지지 평면 제거 (반복 RANSAC, scipy/numpy 구현 — open3d segment_plane 사용 가능)
    keep = ones(N_L, bool)
    for _ in range(max_planes):
        plane, inliers = ransac_plane(lidar_xyz[keep], ransac_dist, ransac_iters)
        if len(inliers) < plane_min_ratio * N_L: break
        keep[inliers] = False

    # 2) voxel-hash 연결성분 클러스터링 (kNN 불필요 — A4 준수)
    key = floor(lidar_xyz[keep] / voxel)                  # [M,3] int
    vox = unique(key)                                     # 점유 복셀 집합 (hash set)
    clusters = connected_components(vox, 26-conn, hash lookup)   # BFS
    # 각 포인트 → 소속 복셀 → 클러스터 라벨

    # 3) 필터 + 대표점 선정
    prompts = []
    for c in clusters:
        if c.n_pts < min_cluster_pts: continue
        if not (size_gate[0] < c.bbox_diag < size_gate[1]):   # 크기 게이팅(느슨)
            if c.bbox_diag > size_gate[1]: c = try_split(c)   # §5.2, 실패 시 통과시킴
            else: continue
        reps3d = [c.centroid] + pca_extremes(c.pts, k=K_prompts-1)
                 # 제1주축 양 끝 근방의 실측점 2개 (합성점 아님 — 실제 LiDAR 점만)
        reps_uv = lidar_uv[index_of(reps3d)]              # 이미 캘리브된 대응 사용
        prompts.append(PromptSet(cluster_id=c.id, points_uv=reps_uv,   # [K,2]
                                 labels=ones(K),          # 전부 positive
                                 lidar_idx=c.point_indices))
    return prompts, clusters
```

클러스터당 SAM 호출: `points_uv*0.4`를 하나의 프롬프트 집합(K positive)으로 decoder 1회, `multimask_output=True`(3후보) → 3후보 모두 proposal 후보로 유지(병합 클러스터 대응, §5.2), score = predicted IoU.

### 2.3 병렬 경로: LiDAR 클러스터 → 투영 마스크

```
def lidar_mask(cluster, lidar_uv) -> bool[800,1280]:
    uv = lidar_uv[cluster.point_indices]        # [n,2]
    hull = cv2.convexHull(uv.astype(int32))     # 볼록껍질
    m = zeros(800,1280, uint8); cv2.fillConvexPoly(m, hull, 1)
    m = cv2.dilate(m, ellipse(9))               # 경계 보수(희소 샘플 보정), 9px ≈ 추정
    return m.astype(bool)
```

이 경로는 SAM이 실패(암부·저대비·오염 표면)해도 클러스터가 있으면 마스크를 보장하는 리콜 안전망. score는 고정 0.30(SAM 마스크보다 항상 후순위, NMS에서 SAM 마스크에 흡수됨).

### 2.4 합집합 + mask NMS

```
def merge_nms(sam_props, lidar_props, iou_thr=0.5) -> list[Proposal]:
    all = sam_props + lidar_props
    all = [p for p in all if p.area > 3e-4*W*H and box_ok(p, 0.02)]   # SAM-6D 후처리 승계(완화)
    all.sort(key=score, desc)                    # SAM predicted_iou > lidar 고정 0.30
    kept = []
    occ = []                                     # kept 마스크 비트팩(np.packbits) 캐시
    for p in all:
        if all(mask_iou(p, q) < iou_thr for q in kept):   # bbox IoU 프리필터 후 mask IoU
            kept.append(p)
    # 메타 부여
    for p in kept:
        p.lidar_idx = indices of lidar points whose uv falls inside p.mask
        p.n_lidar   = len(p.lidar_idx)
        p.area, p.bbox, p.mean_depth = ...       # mean_depth = median(z of p.lidar_idx), 없으면 NaN
    return kept
```

`iou_thr=0.5` (SAM-6D의 0.25보다 완화 — S1은 리콜 책임, 중복 제거는 S2가 object 판별 후 재NMS). 서로 다른 클러스터에서 나온 동일 물체 마스크는 여기서 자연 병합.

---

## 3. 모듈 인터페이스 (`shape6d/proposal/`)

```python
# shape6d/types.py
@dataclass
class FrameBundle:
    rgb: np.ndarray          # [800,1280,3] uint8
    lidar_xyz: np.ndarray    # [N_L,3] float32, 카메라좌표계 m
    lidar_uv: np.ndarray     # [N_L,2] float32, 1280×800 픽셀좌표 (캘리브 완료)
    K: np.ndarray            # [3,3] float32
    frame_id: str

@dataclass
class Proposal:
    mask: np.ndarray         # [800,1280] bool
    bbox: np.ndarray         # [4] float32, xyxy (full-res)
    score: float             # SAM predicted_iou 또는 0.30(lidar 경로)
    source: str              # "sam" | "lidar_hull"
    cluster_id: int          # -1 = 비-LiDAR 유래
    lidar_idx: np.ndarray    # [n] int32, FrameBundle.lidar_* 인덱스
    n_lidar: int
    area: int                # 픽셀 수
    mean_depth: float        # m, n_lidar==0이면 nan

# shape6d/proposal/prompt_gen.py
class LidarPromptGenerator:
    def __init__(self, voxel: float = 0.02, ransac_dist: float = 0.008,
                 max_planes: int = 2, min_cluster_pts: int = 15,
                 k_prompts: int = 3, size_gate: tuple[float,float] | None = None): ...
    def __call__(self, fb: FrameBundle) -> tuple[list[PromptSet], list[Cluster]]: ...

# shape6d/proposal/evit_sam.py
class EvitSamL0Segmenter:
    """EfficientViT-SAM-L0 래퍼. encode 1회/프레임, decode는 프롬프트 배치."""
    def __init__(self, checkpoint: str, device: str = "cuda",
                 input_size: int = 512, multimask: bool = True): ...
    def encode(self, rgb: np.ndarray) -> None: ...          # 임베딩 내부 보관
    def decode(self, prompts: list[PromptSet]) -> list[Proposal]: ...
    def decode_grid(self, side: int = 16) -> list[Proposal]: ...   # fallback 전용 (§5)

# shape6d/proposal/lidar_mask.py
class LidarHullMaskGenerator:
    def __init__(self, dilate_px: int = 9, fixed_score: float = 0.30): ...
    def __call__(self, clusters: list[Cluster], fb: FrameBundle) -> list[Proposal]: ...

# shape6d/proposal/merge_nms.py
class ProposalMerger:
    def __init__(self, iou_thr: float = 0.5, min_mask_ratio: float = 3e-4): ...
    def __call__(self, *groups: list[Proposal], fb: FrameBundle) -> list[Proposal]: ...

# shape6d/proposal/stage.py
class S1ProposalStage:
    def __init__(self, segmenter, prompt_gen, hull_gen, merger,
                 fallback_grid_side: int = 16): ...
    def __call__(self, fb: FrameBundle) -> list[Proposal]: ...   # score 내림차순
```

TensorRT 배포 시 `EvitSamL0Segmenter`만 encoder/decoder 2개 엔진으로 교체(인터페이스 동일).

---

## 4. 레이턴시 추정 (프레임당, 클러스터 ~8개 가정)

| 항목 | RTX PRO 6000 | Orin NX (TRT FP16, **추정**) | 근거(한 줄) |
|---|---|---|---|
| letterbox 전처리 | 0.5 ms | 2 ms | 1280×800 resize 1회, CPU/GPU 단순 연산 |
| EViT-SAM-L0 encoder(512²) | 2–4 ms | 25–45 ms | 추정: L0는 A100서 수 ms급으로 보고됨, Orin NX는 데스크톱 대비 유효 연산량 ~1/10–1/20 통례 적용 |
| decoder ×8 프롬프트(배치) | 1–2 ms | 8–15 ms | 추정: SAM decoder는 encoder 대비 ~1/20 연산 |
| 마스크 복원(24개, 800×1280) | 1 ms | 5 ms | bilinear interpolate, memcpy 바운드 |
| RANSAC 평면 + voxel 클러스터 (CPU) | 2–4 ms | 6–12 ms | 추정: N_L≈2만 pt, numpy 벡터화 O(N·iters) |
| convex hull 마스크 ×8 | 0.5 ms | 1.5 ms | cv2.fillConvexPoly, 수백 pt |
| merge NMS(≤30 마스크) | 1–2 ms | 3–6 ms | bbox 프리필터 후 mask IoU 수십 회 |
| **S1 합계** | **≈ 8–14 ms** | **≈ 50–90 ms (추정)** | 1s 예산 중 <10% — S3/S4 여유 확보 |
| S0 온보딩(오프라인 전체) | 1–3 min/객체 | — | 추정: 대칭 스캔(chamfer×수백 회, KDTree)이 지배 |

Orin NX 실기는 없으므로 위 수치는 전부 추정치이며 M1 마일스톤에서 실측 교체 필요.

---

## 5. 엣지 케이스

**5.1 프롬프트 0개 (평면 제거 후 전경 없음)**
정책: (a) 평면 제거를 1단계 완화(최대 평면 1개만 제거)하고 재클러스터 → (b) 그래도 0개면 `decode_grid(16×16=256pt)` 자동-그리드 fallback(마스크 상한 40개) 실행, 모든 proposal에 `n_lidar` 산출 → LiDAR 포인트 없는 마스크는 S2에서 기하 판별 불가로 자연 강등. (c) fallback도 0개면 빈 리스트 + `status="no_foreground"` 반환(파이프라인은 "미검출" 종료, 오탐 조작 금지 — A3의 단발 견고성은 정직한 실패 보고 포함).

**5.2 클러스터 병합 (접촉·적재 물체)**
voxel 연결성분은 접촉 물체를 하나로 묶는다. 3중 방어: (i) `multimask_output=True`로 SAM 3스케일 마스크를 전부 후보화 — 프롬프트가 병합 클러스터 위에 흩어져 있어도 개별 물체 스케일 마스크가 살아남음. (ii) 크기 게이팅 상한 위반 클러스터는 제1주축 기준 2-분할(`try_split`: 주축 투영 히스토그램 최소밀도 지점 절단, **구현 필요**) 후 각각 프롬프트 생성; 분할 실패 시 원본 그대로 통과(리콜 우선). (iii) hull 마스크는 병합 상태로 나가더라도 SAM 마스크가 NMS에서 우선하므로 실질 피해는 hull-only 상황으로 한정 — 이 경우 S2 크기 게이팅이 최종 방어선.

**5.3 대형 반사면/흡수면에서 LiDAR 공백**
검은 고무·경면 금속은 LiDAR 리턴이 없어 클러스터 자체가 안 생김. 대응: (a) 프레임 단위로 `LiDAR 커버리지 맵`(유효 uv를 32×20 타일 점유율로 집계)을 만들어 점유율 낮은 타일이 크게 연속되면 해당 영역에만 저밀도 그리드 프롬프트(타일당 1pt) 추가 — 전역 그리드보다 싸다. (b) 이렇게 생성된 마스크는 `n_lidar≈0` → S2는 DINOv2 prior + 크기 추정 불가 상태로 "저신뢰 후보" 태그를 붙여 통과시키고, S3에서 마스크 내 잔존 LiDAR 포인트(수십 개라도)로 포즈 시도, 부족하면 S4 reject 사유 `insufficient_geometry`로 종결. (c) hull 경로는 이 케이스에서 원천 무력(포인트 없음)이므로 문서에 한계로 명기하고, D2-a 센서 실측 후 반사면 비중이 크면 depth completion 보조를 후속 과제로 분리.

**5.4 기타 (요약)**
- 화면 경계 걸침 물체: bbox가 이미지 경계 접촉 시 `truncated=True` 메타(신규 필드) — S3 크기 게이팅 완화용.
- 프롬프트 대표점이 마스크 홀(투과 리턴)에 떨어짐: 대표점은 반드시 실측 LiDAR 점에서만 선택(합성 중심점 사용 금지)으로 이미 방지, centroid는 최근접 실측점으로 스냅.
- 42뷰/224 등 상수 변경 시 캐시 불일치: `manifest.schema_version` + 로드 시 shape 검증으로 즉시 실패.

---

**미확정/확인 필요 항목**: ① EfficientViT-SAM-L0 임베딩 grid 크기와 전처리 정확 규약(공식 repo 확인), ② D2-a(물체당 유효 LiDAR pt 수) 실측 → `min_cluster_pts`·`voxel` 재조정, ③ `geo_feat_dense` 채널 수 C_g는 S3 인코더 확정 후, ④ Orin NX 전 수치 실측 교체.