"""S2: 크기 게이팅(§5.1) + TDF 정합(§5.2 — [상-2] 시선축 정렬 포함) 테스트."""
import numpy as np

from shape6d.common.types import Candidate, Proposal
from shape6d.identify.depth_match import PointToTemplateMatcher, _ray_align_rotation
from shape6d.identify.size_gate import SizeGate
from shape6d.onboarding.templates import build_tdf


def _cand(pts, truncated=False):
    prop = Proposal(mask=None, bbox=np.zeros(4), score=0.9, source="sam", truncated=truncated)
    return Candidate(proposal=prop, pts=pts, uv=np.zeros((len(pts), 2)))


def test_size_gate_pass_and_reject():
    rng = np.random.default_rng(0)
    d_cad = 0.3
    obj = rng.uniform(-0.1, 0.1, (200, 3)) * np.array([1.0, 1.0, 0.5])  # extent ~0.28
    giant = rng.uniform(-0.4, 0.4, (200, 3))                             # 상한 위반
    tiny = rng.uniform(-0.01, 0.01, (200, 3))                            # 하한 미달
    few = rng.uniform(-0.1, 0.1, (10, 3))                                # N < n_min
    out = SizeGate()( [_cand(obj), _cand(giant), _cand(tiny), _cand(few)], d_cad)
    assert len(out) == 1 and "size" in out[0].scores


def test_size_gate_sole_rescue():
    few = np.random.default_rng(1).uniform(-0.1, 0.1, (10, 3))
    out = SizeGate()([_cand(few)], 0.3)  # 유일 후보 구제 경로 (§2.5)
    assert len(out) == 1 and "low_geo" in out[0].flags


def _box_view_pts(lx=0.3, ly=0.2, lz=0.1, step=0.008):
    """전면(z=−lz/2 방향에서 보이는 3면 근사) 가시 포인트 — 뷰 프레임."""
    xs = np.arange(-lx / 2, lx / 2, step)
    ys = np.arange(-ly / 2, ly / 2, step)
    X, Y = np.meshgrid(xs, ys)
    front = np.stack([X.ravel(), Y.ravel(), np.full(X.size, -lz / 2)], 1)
    return front


def test_ray_align_rotation():
    cen = np.array([1.0, 0.5, 2.0])
    R = _ray_align_rotation(cen)
    aligned = R @ (cen / np.linalg.norm(cen))
    assert np.allclose(aligned, [0, 0, 1], atol=1e-6)


def test_tdf_match_same_vs_different():
    tpl = _box_view_pts()
    tpl_c = tpl - np.median(tpl, axis=0)
    D = float(np.linalg.norm(tpl.max(0) - tpl.min(0)))
    tdf = build_tdf(tpl_c, D)[None]
    m = PointToTemplateMatcher(tdf, np.zeros((1, 3), np.float32), D,
                               view_mask=np.array([True]))

    obs_same = tpl_c + np.array([0, 0, 1.5])          # 광축 위 1.5m — 정합 기대
    s_same, diag = m.score(obs_same)

    th = np.deg2rad(30.0)                              # pass1 격자 위 in-plane 회전
    Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    s_rot, _ = m.score(tpl_c @ Rz.T + np.array([0, 0, 1.5]))

    rng = np.random.default_rng(0)
    sphere = rng.normal(0, 1, (tpl.shape[0], 3))
    sphere = 0.5 * D * sphere / np.linalg.norm(sphere, axis=1, keepdims=True)
    s_diff, _ = m.score(sphere + np.array([0, 0, 1.5]))

    assert s_same > 0.7, f"동일 형상 점수 {s_same:.2f} — 정합 붕괴([상-2]/[상-3] 회귀)"
    assert s_rot > 0.6, f"in-plane 30° 회전 미복구: {s_rot:.2f}"
    assert s_same > s_diff + 0.2, f"판별력 부족: same={s_same:.2f} diff={s_diff:.2f}"


def test_tdf_match_offcenter():
    """화면 주변부(오프센터) 물체 — 시선축 정렬([상-2]) 없으면 붕괴하는 케이스."""
    tpl = _box_view_pts()
    tpl_c = tpl - np.median(tpl, axis=0)
    D = float(np.linalg.norm(tpl.max(0) - tpl.min(0)))
    m = PointToTemplateMatcher(build_tdf(tpl_c, D)[None], np.zeros((1, 3), np.float32), D)

    # 물체를 광축에서 30° 옆에 배치: 시선 방향으로 면을 돌려서(ray-align의 역) 배치
    off_dir = np.array([np.sin(np.deg2rad(30)), 0, np.cos(np.deg2rad(30))])
    R_align = _ray_align_rotation(off_dir * 1.5)
    obs = tpl_c @ R_align + off_dir * 1.5   # R_align.T의 역회전 적용 = 시선 프레임에서 정면
    s, _ = m.score(obs)
    assert s > 0.6, f"오프센터 정합 붕괴: {s:.2f} — [상-2] 회귀"
