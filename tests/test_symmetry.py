"""대칭 자동검출 회귀 테스트 — 교차 검증 [상-1] 수정의 검증 게이트 (03 §3.3).

합성 원기둥/박스에서 검출 100%, 비대칭 형상에서 오검출 0이 통과 조건.
"""
import numpy as np

from shape6d.onboarding.symmetry import detect_symmetry, discretize_for_training


def _cylinder(R=0.04, H=0.15, n_theta=96, n_z=36, n_cap=10):
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    z = np.linspace(-H / 2, H / 2, n_z)
    T, Z = np.meshgrid(th, z)
    side = np.stack([R * np.cos(T).ravel(), R * np.sin(T).ravel(), Z.ravel()], 1)
    caps = []
    for zc in (-H / 2, H / 2):
        for r in np.linspace(R / n_cap, R, n_cap):
            m = max(6, int(n_theta * r / R))
            t = np.linspace(0, 2 * np.pi, m, endpoint=False)
            caps.append(np.stack([r * np.cos(t), r * np.sin(t), np.full(m, zc)], 1))
    return np.vstack([side] + caps)


def _box(lx=1.2, ly=0.8, lz=0.3, step=0.03):
    faces = []
    xs = np.arange(-lx / 2, lx / 2 + 1e-9, step)
    ys = np.arange(-ly / 2, ly / 2 + 1e-9, step)
    zs = np.arange(-lz / 2, lz / 2 + 1e-9, step)
    X, Y = np.meshgrid(xs, ys)
    for z in (-lz / 2, lz / 2):
        faces.append(np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], 1))
    X, Z = np.meshgrid(xs, zs)
    for y in (-ly / 2, ly / 2):
        faces.append(np.stack([X.ravel(), np.full(X.size, y), Z.ravel()], 1))
    Y, Z = np.meshgrid(ys, zs)
    for x in (-lx / 2, lx / 2):
        faces.append(np.stack([np.full(Y.size, x), Y.ravel(), Z.ravel()], 1))
    return np.vstack(faces)


def test_cylinder_continuous_axis():
    res = detect_symmetry(_cylinder())
    assert len(res.sym_axes) >= 1, f"원기둥 연속축 미검출 — [상-1] 회귀. log={res.log}"
    ax = res.sym_axes[np.argmax(np.abs(res.sym_axes[:, 2]))]
    assert abs(ax[2]) > 0.99, f"연속축이 z와 불일치: {ax}"
    # 수직 C2(옆으로 뒤집기)도 잡혀야 함
    assert len(res.sym_rots) >= 2, "원기둥 수직 C2 미검출"


def test_box_c2_group():
    res = detect_symmetry(_box())
    assert len(res.sym_axes) == 0, "박스에 연속축 오검출"
    # D2 군: {I, Rx180, Ry180, Rz180} = 4원소
    assert len(res.sym_rots) == 4, f"박스 C2 군 크기 {len(res.sym_rots)} != 4. log={res.log}"
    for R in res.sym_rots:
        ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
        assert ang < 1.0 or abs(ang - 180.0) < 6.0, f"D2 외 원소: {ang}°"


def test_asymmetric_no_false_positive():
    box = _box(0.6, 0.4, 0.2, step=0.02)
    lump = _box(0.25, 0.2, 0.2, step=0.02) + np.array([0.4, 0.25, 0.15])
    res = detect_symmetry(np.vstack([box, lump]))
    assert len(res.sym_axes) == 0, "비대칭 형상에 연속축 오검출"
    assert len(res.sym_rots) == 1, f"비대칭 형상에 유한대칭 오검출: |S|={len(res.sym_rots)}, log={res.log}"


def test_discretize_for_training():
    res = detect_symmetry(_cylinder())
    g = discretize_for_training(res, n_cont=12, max_group=16)
    assert g.shape[1:] == (3, 3) and 2 <= g.shape[0] <= 16
