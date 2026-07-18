"""S4 검증 스택: projective ICP 수렴·퇴화 검출·free-space 위반·대칭 등가·신뢰도 IRLS."""
import numpy as np

from shape6d.common.frame_bundle import CameraIntrinsics
from shape6d.common.types import PoseHypothesis
from shape6d.verify.confidence import ConfidenceCalibrator
from shape6d.verify.icp import ProjectiveICP, _exp_so3
from shape6d.verify.scorer import HypothesisScorer
from shape6d.verify.symmetry_eval import SymmetryHandler

K = CameraIntrinsics(fx=914.0, fy=914.0, cx=640.0, cy=400.0)
RNG = np.random.default_rng(0)


def _box_master(lx=0.3, ly=0.2, lz=0.1, step=0.006):
    """박스 표면 포인트+노멀 (모델 프레임)."""
    pts, nrm = [], []
    grids = {
        "z": (np.arange(-lx/2, lx/2, step), np.arange(-ly/2, ly/2, step)),
        "y": (np.arange(-lx/2, lx/2, step), np.arange(-lz/2, lz/2, step)),
        "x": (np.arange(-ly/2, ly/2, step), np.arange(-lz/2, lz/2, step)),
    }
    for axis, (g1, g2) in grids.items():
        A, B = np.meshgrid(g1, g2)
        for sgn in (-1, 1):
            n = {"x": [sgn, 0, 0], "y": [0, sgn, 0], "z": [0, 0, sgn]}[axis]
            half = {"x": lx/2, "y": ly/2, "z": lz/2}[axis]
            if axis == "z":
                p = np.stack([A.ravel(), B.ravel(), np.full(A.size, sgn*half)], 1)
            elif axis == "y":
                p = np.stack([A.ravel(), np.full(A.size, sgn*half), B.ravel()], 1)
            else:
                p = np.stack([np.full(A.size, sgn*half), A.ravel(), B.ravel()], 1)
            pts.append(p); nrm.append(np.tile(n, (p.shape[0], 1)))
    return np.vstack(pts), np.vstack(nrm).astype(np.float64)


def _corner_view_obs(X_m, N_m, R, t, noise=0.003):
    """코너 뷰(3면 가시) 관측 시뮬레이션: 카메라를 향한 면의 포인트만."""
    Xc = X_m @ R.T + t
    Nc = N_m @ R.T
    vis = (Nc * (-Xc / np.linalg.norm(Xc, axis=1, keepdims=True))).sum(1) > 0.15
    P = Xc[vis][::3]
    return P + RNG.normal(0, noise, P.shape)


def _gt_pose():
    Rz = _exp_so3(np.array([0.0, 0.0, np.deg2rad(30)]))
    Ry = _exp_so3(np.array([0.0, np.deg2rad(35), 0.0]))
    Rx = _exp_so3(np.array([np.deg2rad(20), 0.0, 0.0]))
    return Rx @ Ry @ Rz, np.array([0.05, 0.03, 1.5])


def test_icp_converges_from_perturbation():
    X_m, N_m = _box_master()
    R_gt, t_gt = _gt_pose()
    P = _corner_view_obs(X_m, N_m, R_gt, t_gt)

    # 물체 중심 기준 섭동 (coarse 포즈 오차의 실제 구조 — template_init은 centroid를 맞춤)
    dR = _exp_so3(np.deg2rad(10) * np.array([0.3, 0.8, 0.52]))
    R0, t0 = dR @ R_gt, t_gt + np.array([0.03, -0.02, 0.03])

    icp = ProjectiveICP(K, huber_delta=0.009)
    R, t, diag = icp.refine(R0, t0, P, X_m, N_m, d_cad=0.374)
    ang = np.degrees(np.arccos(np.clip((np.trace(R @ R_gt.T) - 1) / 2, -1, 1)))
    assert ang < 0.6, f"회전 잔차 {ang:.2f}° (노이즈 3mm 기준)"
    assert np.linalg.norm(t - t_gt) < 0.004, f"병진 잔차 {np.linalg.norm(t-t_gt)*1e3:.1f}mm"
    assert not diag["degenerate"]


def test_icp_degenerate_plane():
    step = 0.006
    g = np.arange(-0.15, 0.15, step)
    A, B = np.meshgrid(g, g)
    X_m = np.stack([A.ravel(), B.ravel(), np.zeros(A.size)], 1)
    N_m = np.tile([0.0, 0.0, 1.0], (X_m.shape[0], 1))
    R_gt, t_gt = np.eye(3), np.array([0.0, 0.0, 1.5])
    R_gt = _exp_so3(np.array([np.deg2rad(30), 0, 0]))  # 카메라를 향하도록 기울임
    P = (X_m @ R_gt.T + t_gt) + RNG.normal(0, 0.002, X_m.shape)
    _, _, diag = ProjectiveICP(K).refine(R_gt, t_gt, P[::2], X_m, N_m, d_cad=0.42)
    assert diag["degenerate"], f"평면 퇴화 미검출: cond={diag['cond_H']:.2e}"


def test_scorer_free_space_violation():
    X_m, N_m = _box_master()
    R_gt, t_gt = _gt_pose()
    P = _corner_view_obs(X_m, N_m, R_gt, t_gt, noise=0.002)
    uv = np.stack([K.fx * P[:, 0] / P[:, 2] + K.cx, K.fy * P[:, 1] / P[:, 2] + K.cy], 1)
    scorer = HypothesisScorer(K, tau_z=0.024)

    s_good, d_good = scorer(R_gt, t_gt, X_m[::2], uv, P[:, 2])
    s_bad, d_bad = scorer(R_gt, t_gt - np.array([0, 0, 0.06]), X_m[::2], uv, P[:, 2])
    # 스플랫 스코어러의 역할 = free-space 위반 검출 (적응 임계) + 가설 상대 순위
    assert d_good["free_viol"] < 0.05, d_good
    # 적응 임계는 grazing 면에서 위반 일부를 흡수 — 하드가드(0.05) 대비 2배 이상 마진이면 기능
    assert d_bad["free_viol"] > 0.1, f"모델을 관측 앞으로 당겼는데 free_viol 미검출: {d_bad}"
    assert s_good > s_bad


def test_symmetry_dedupe_and_error():
    Rz180 = _exp_so3(np.array([0, 0, np.pi]))
    sym = SymmetryHandler(np.stack([np.eye(3), Rz180]), np.zeros((0, 3)))
    R_gt, t_gt = _gt_pose()
    h1 = PoseHypothesis(R=R_gt, t=t_gt, score=0.9)
    h2 = PoseHypothesis(R=R_gt @ Rz180, t=t_gt + 1e-4, score=0.8)   # 등가 포즈
    h3 = PoseHypothesis(R=_exp_so3(np.array([0, np.deg2rad(60), 0])) @ R_gt, t=t_gt, score=0.7)
    kept = sym.dedupe([h1, h2, h3], d_cad=0.374)
    assert len(kept) == 2, f"C2 등가 병합 실패: {len(kept)}"

    X_m, _ = _box_master(step=0.02)
    e_pos, e_rot = sym.sym_aware_error(R_gt @ Rz180, t_gt, R_gt, t_gt, X_m)
    assert e_pos < 1e-6 and e_rot < 1e-4, "등가 포즈의 sym-aware 오차가 0이 아님"


def test_symmetry_continuous_twist():
    sym = SymmetryHandler(np.eye(3)[None], np.array([[0.0, 0.0, 1.0]]))
    R_twist = _exp_so3(np.array([0, 0, np.deg2rad(73)]))    # 축 twist = 등가
    R_swing = _exp_so3(np.array([np.deg2rad(5), 0, 0]))     # swing = 실제 차이
    assert sym.sym_distance_deg(R_twist, np.eye(3)) < 0.5
    d = sym.sym_distance_deg(R_swing @ R_twist, np.eye(3))
    assert 4.0 < d < 6.0, f"swing 5° 기대, 실측 {d:.2f}°"


def test_confidence_irls():
    rng = np.random.default_rng(1)
    n = 400
    X = rng.normal(0, 1, (n, 10))
    w_true = rng.normal(0, 1, 10)
    y = (X @ w_true + rng.normal(0, 0.3, n) > 0).astype(float)
    calib = ConfidenceCalibrator.fit(X, y)
    pred = np.array([calib(x) for x in X]) > 0.5
    assert (pred == y.astype(bool)).mean() > 0.9
