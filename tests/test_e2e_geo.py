"""v0-geo E2E 합성 테스트 — 무학습 기하 파이프라인 전체 사슬의 스펙 검증.

팔레트(1100×1100×150) CAD → S0 온보딩(샘플·템플릿·TDF·대칭) → 합성 LiDAR 프레임(3m, 희소)
→ S1 클러스터 → S2 게이팅+TDF 정합 → coarse 포즈(template_init) → S4 ICP·검증
→ sym-aware 오차: 위치 <10mm, 회전(yaw) <1° (확정 요구 스펙, 클린 합성 기준).
"""
import numpy as np
import pytest
import trimesh

from shape6d.common.frame_bundle import CameraIntrinsics, build_frame_bundle
from shape6d.common.types import Candidate, Proposal
from shape6d.identify.depth_match import PointToTemplateMatcher
from shape6d.identify.size_gate import SizeGate
from shape6d.onboarding.sampling import fps_indices
from shape6d.onboarding.symmetry import detect_symmetry
from shape6d.onboarding.templates import build_view_templates, splat_depth
from shape6d.pose.template_init import coarse_poses_from_match
from shape6d.proposal.prompt_gen import LidarPromptGenerator
from shape6d.verify.symmetry_eval import SymmetryHandler
from shape6d.verify.verifier import Verifier

K = CameraIntrinsics(fx=914.0, fy=914.0, cx=640.0, cy=400.0)
SIGMA = 0.005  # 합성 LiDAR 거리 노이즈 (D2-a 확정 전 보수값)


def _pallet_mesh():
    deck = trimesh.creation.box((1.1, 1.1, 0.03))
    deck.apply_translation([0, 0, 0.135])
    parts = [deck]
    for y in (-0.48, 0.0, 0.48):
        r = trimesh.creation.box((1.1, 0.14, 0.12))
        r.apply_translation([0, y, 0.06])
        parts.append(r)
    return trimesh.util.concatenate(parts)


@pytest.fixture(scope="module")
def onboarded():
    mesh = _pallet_mesh()
    surf, face_idx = trimesh.sample.sample_surface(mesh, 60000, seed=0)
    surf = np.asarray(surf)
    nrm = np.asarray(mesh.face_normals)[np.asarray(face_idx)]
    diam = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    r_bs = float(np.linalg.norm(surf - surf.mean(0), axis=1).max())

    # master는 균일 랜덤 8192 (sample_surface가 이미 균일 — ICP 접선항의 이산 스냅 바이어스 완화)
    master, master_n = surf[:8192], nrm[:8192]
    tpl = build_view_templates(surf.astype(np.float32), r_bs, diam)
    sym = detect_symmetry(master[:: max(1, len(master) // 4096)])
    return dict(mesh=mesh, surf=surf, master=master, master_n=master_n,
                diam=diam, tpl=tpl, sym=sym)


def _gt_pose():
    th = np.deg2rad(20.0)
    Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    R0 = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])  # 모델 z(up) → 카메라 −y
    return R0 @ Rz, np.array([0.1, 0.42, 3.0])


def _synthetic_frame(onb):
    rng = np.random.default_rng(0)
    R_gt, t_gt = _gt_pose()
    posed = onb["surf"] @ R_gt.T + t_gt

    depth = splat_depth(posed.astype(np.float32), K.fx, K.fy, K.cx, K.cy, (800, 1280))
    z = posed[:, 2]
    u = np.clip(np.round(K.fx * posed[:, 0] / z + K.cx).astype(np.int64), 0, 1279)
    v = np.clip(np.round(K.fy * posed[:, 1] / z + K.cy).astype(np.int64), 0, 799)
    vis = np.abs(depth[v, u] - z) < 0.01
    scan = (v % 5 == 0)  # 주사선 희소화
    sel = np.nonzero(vis & scan)[0]
    sel = rng.permutation(sel)[:1200]
    obj_pts = posed[sel]

    xs = np.linspace(-1.2, 1.2, 60)
    zs = np.linspace(2.0, 4.5, 50)
    Xf, Zf = np.meshgrid(xs, zs)
    floor = np.stack([Xf.ravel(), np.full(Xf.size, 0.57), Zf.ravel()], 1)
    distractor = rng.normal(0, 0.05, (150, 3)) + np.array([-0.9, 0.35, 2.4])

    pts = np.vstack([obj_pts, floor, distractor]).astype(np.float32)
    pts[:, 2] += rng.normal(0, SIGMA, len(pts)).astype(np.float32)
    fb = build_frame_bundle(np.zeros((800, 1280, 3), np.uint8), pts,
                            np.ones(len(pts), np.float32), np.zeros(len(pts), np.float32),
                            K, np.eye(4))
    return fb, R_gt, t_gt


def test_pallet_symmetry_detected(onboarded):
    assert len(onboarded["sym"].sym_rots) >= 2, \
        f"팔레트 C2 미검출: {onboarded['sym'].log}"


def test_e2e_pose_within_spec(onboarded):
    fb, R_gt, t_gt = _synthetic_frame(onboarded)

    # -- S1: 평면 제거 + 클러스터 --------------------------------------------
    gen = LidarPromptGenerator(voxel=0.06, min_cluster_pts=30)
    _, clusters = gen(fb)
    assert len(clusters) >= 2, f"클러스터 {len(clusters)}개 — 팔레트+방해물 기대"

    # -- S2: Candidate 변환 + 크기 게이팅 + TDF 정합 -------------------------
    cands = []
    for c in clusters:
        idx = c.point_indices
        cands.append(Candidate(
            proposal=Proposal(mask=None, bbox=np.zeros(4), score=0.5, source="lidar_hull",
                              cluster_id=c.id, lidar_idx=idx, n_lidar=len(idx)),
            pts=fb.lidar_points[idx], uv=fb.lidar_pixels[idx]))
    gated = SizeGate()(cands, onboarded["diam"])
    assert len(gated) == 1, f"게이팅 후 {len(gated)}개 — 방해물 미제거 또는 팔레트 탈락"
    cand = gated[0]

    tpl = onboarded["tpl"]
    matcher = PointToTemplateMatcher(tpl["tdf"], tpl["tpl_center"], onboarded["diam"])
    m = matcher.match(cand.pts, k=3)
    assert m.s_depth > 0.5, f"S2 정합 점수 {m.s_depth:.2f} — 정합 붕괴"
    cand.scores["depth"] = m.s_depth

    # -- coarse 포즈 (무학습 template_init) -----------------------------------
    hyps = coarse_poses_from_match(m, tpl["tpl_pose"], tpl["tpl_center"])
    assert len(hyps) >= 1

    # -- S4: ICP 정련 + 검증 ---------------------------------------------------
    sym = SymmetryHandler(onboarded["sym"].sym_rots, onboarded["sym"].sym_axes)
    ver = Verifier(K, sym, sigma_lidar=SIGMA)
    X_verify = onboarded["master"][fps_indices(onboarded["master"].astype(np.float32), 1024)]
    res = ver(hyps, cand.pts.astype(np.float64), cand.uv, onboarded["master"],
              onboarded["master_n"], X_verify, onboarded["diam"],
              s2_scores=cand.scores)

    R_est, t_est = res.pose[:3, :3], res.pose[:3, 3]
    e_pos, e_rot = sym.sym_aware_error(R_est, t_est, R_gt, t_gt, X_verify)
    assert e_pos < 0.010, f"위치 오차 {e_pos*1e3:.1f}mm ≥ 10mm (스펙 위반)"
    assert e_rot < 1.0, f"회전 오차 {e_rot:.2f}° ≥ 1° (스펙 위반)"
    assert res.verdict != "REJECT", f"정답 포즈 REJECT: p={res.p_conf:.2f} diag={res.diag['scorer']}"
