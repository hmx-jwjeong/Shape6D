"""S0 대칭 자동검출 (03 문서 §3.3 — 교차 검증 [상-1]/[중-6] 수정판).

핵심 수정:
- 거리는 가능하면 point-to-mesh (trimesh.proximity). 포인트 폴백 시에는
  샘플링 노이즈 바닥을 반영한 임계 하한: tau = max(0.004·D, 0.7·sqrt(A/N)).
- 관성 고유값 축퇴(큐브류) 시 icosphere 42방향 전구면 축 탐색.
- 검출은 '제안'일 뿐 — CLI에서 사람이 --sym-override로 정정 가능해야 한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FINITE_ORDERS = (2, 3, 4, 5, 6, 8, 10, 12)
MAX_GROUP = 24  # 군 폐포 상한 (03 §3.3)


@dataclass
class SymmetryResult:
    sym_rots: np.ndarray                 # [S,3,3] 유한 회전군 (항등 포함)
    sym_axes: np.ndarray                 # [A,3] 무한(연속) 대칭축
    reflections: list = field(default_factory=list)  # 메타만 (등가 평가 미사용)
    log: list = field(default_factory=list)

    @property
    def summary(self) -> str:
        parts = [f"finite group |S|={len(self.sym_rots)}"]
        if len(self.sym_axes):
            parts.append(f"continuous axes A={len(self.sym_axes)}")
        return ", ".join(parts)


def _rot(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _chamfer_1way(A: np.ndarray, ref_tree) -> float:
    d, _ = ref_tree.query(A, k=1)
    return float(np.mean(d))


def _make_dist_fn(points: np.ndarray, mesh=None):
    """반환: (dist_fn, tau). mesh가 있으면 point-to-mesh, 없으면 point-to-point 폴백."""
    D = float(np.linalg.norm(points.max(0) - points.min(0)))
    n = points.shape[0]
    if mesh is not None:
        try:
            from trimesh.proximity import ProximityQuery
            pq = ProximityQuery(mesh)
            area = float(mesh.area)
            tau = max(0.004 * D, 0.25 * np.sqrt(area / n))  # mesh 거리는 노이즈 바닥이 낮음
            return (lambda P: float(np.mean(np.abs(pq.signed_distance(P)))), tau)
        except Exception:
            pass
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    area_est = 2.0 * D * D  # 전형 물체 표면적 근사 (검증 [상-1] 유도)
    tau = max(0.004 * D, 0.7 * np.sqrt(area_est / n))
    return (lambda P: _chamfer_1way(P, tree)), tau


def _candidate_axes(P: np.ndarray) -> list[np.ndarray]:
    """PCA + 관성텐서 고유축, 축퇴 시 평면 스캔/전구면 추가 (03 §3.3)."""
    cov = np.cov(P.T)
    w, V = np.linalg.eigh(cov)
    axes = [V[:, i] for i in range(3)]

    order = np.argsort(w)[::-1]
    w_sorted = w[order]
    # 2축 축퇴: 해당 평면에서 30° 간격 스캔
    for i in range(2):
        if w_sorted[i + 1] > 1e-12 and w_sorted[i] / w_sorted[i + 1] < 1.02:
            a, b = V[:, order[i]], V[:, order[i + 1]]
            for th in np.arange(0, np.pi, np.pi / 6):
                axes.append(np.cos(th) * a + np.sin(th) * b)
    # 3축 축퇴 (큐브류): icosphere 42방향 전구면 탐색 (검증 [중-6])
    if w_sorted[2] > 1e-12 and w_sorted[0] / w_sorted[2] < 1.05:
        from .templates import icosphere42
        axes.extend(list(icosphere42()))

    out: list[np.ndarray] = []
    for a in axes:
        a = a / np.linalg.norm(a)
        if not any(min(np.linalg.norm(a - b), np.linalg.norm(a + b)) < np.sin(np.deg2rad(5.0)) for b in out):
            out.append(a)
    return out


def _refine_angle(dist_fn, P, axis, angle0, tau, half_window=np.deg2rad(4.0), step=np.deg2rad(0.5)):
    best_a, best_d = angle0, dist_fn(P @ _rot(axis, angle0).T)
    for a in np.arange(angle0 - half_window, angle0 + half_window + 1e-9, step):
        d = dist_fn(P @ _rot(axis, a).T)
        if d < best_d:
            best_a, best_d = a, d
    return best_a, best_d


def _group_closure(rots: list[np.ndarray], ang_tol_deg: float = 5.0) -> np.ndarray:
    """생성원 → 유한군 폐포 (상한 MAX_GROUP)."""
    def _is_dup(R, group):
        for G in group:
            c = (np.trace(R @ G.T) - 1.0) / 2.0
            if np.degrees(np.arccos(np.clip(c, -1, 1))) < ang_tol_deg:
                return True
        return False

    group = [np.eye(3)]
    for R in rots:
        if not _is_dup(R, group):
            group.append(R)
    changed = True
    while changed and len(group) < MAX_GROUP:
        changed = False
        for A in list(group):
            for B in list(group):
                C = A @ B
                if not _is_dup(C, group):
                    group.append(C)
                    changed = True
                    if len(group) >= MAX_GROUP:
                        break
            if len(group) >= MAX_GROUP:
                break
    return np.stack(group)


def detect_symmetry(points: np.ndarray, mesh=None) -> SymmetryResult:
    """points: [N,3] Poisson disk 샘플 (모델좌표계). mesh: trimesh(선택, 정밀 거리용)."""
    P = points - points.mean(axis=0)
    dist_fn, tau = _make_dist_fn(P, mesh)
    log = [f"tau={tau*1e3:.2f}mm (D={np.linalg.norm(points.max(0)-points.min(0))*1e3:.0f}mm)"]

    cont_axes: list[np.ndarray] = []
    generators: list[np.ndarray] = []

    for axis in _candidate_axes(P):
        # 거친 스캔: 무한대칭 판정 (전 각도 정합)
        scan = [dist_fn(P @ _rot(axis, a).T) for a in np.deg2rad(np.arange(5, 180, 5))]
        if max(scan) < tau:
            cont_axes.append(axis)
            log.append(f"axis {np.round(axis,3)}: continuous (max scan {max(scan)*1e3:.2f}mm)")
            continue
        # 유한 차수: 큰 n부터 (최고 차수 우선)
        for n in sorted(FINITE_ORDERS, reverse=True):
            ang = 2 * np.pi / n
            if dist_fn(P @ _rot(axis, ang).T) < tau:
                a_ref, d_ref = _refine_angle(dist_fn, P, axis, ang, tau)
                if d_ref < tau:
                    generators.append(_rot(axis, a_ref))
                    log.append(f"axis {np.round(axis,3)}: C{n} (d={d_ref*1e3:.2f}mm)")
                    break

    reflections = []
    cov_V = np.linalg.eigh(np.cov(P.T))[1]
    for i in range(3):
        nrm = cov_V[:, i]
        M = np.eye(3) - 2 * np.outer(nrm, nrm)
        if dist_fn(P @ M.T) < tau:
            reflections.append(nrm.copy())

    sym_rots = _group_closure(generators) if generators else np.eye(3)[None]
    return SymmetryResult(
        sym_rots=sym_rots.astype(np.float32),
        sym_axes=(np.stack(cont_axes).astype(np.float32) if cont_axes else np.zeros((0, 3), np.float32)),
        reflections=reflections,
        log=log,
    )


def discretize_for_training(res: SymmetryResult, n_cont: int = 12, max_group: int = 16) -> np.ndarray:
    """학습용 이산화 (03 §2.4): 연속축은 n_cont 분할, 상한 max_group, 항등 포함."""
    rots = [r for r in res.sym_rots]
    for axis in res.sym_axes:
        for k in range(1, n_cont):
            rots.append(_rot(axis, 2 * np.pi * k / n_cont))
    group = _group_closure(rots[1:], ang_tol_deg=3.0) if len(rots) > 1 else np.eye(3)[None]
    return group[:max_group].astype(np.float32)
