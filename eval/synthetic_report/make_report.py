"""v0-geo 합성 검증 보고서 자산 생성 (docs/04 보고서용 이미지 + 통계 JSON).

실행: uv run python eval/synthetic_report/make_report.py
산출: docs/assets_04/*.png, docs/assets_04/stats.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh

from shape6d.common.frame_bundle import CameraIntrinsics, build_frame_bundle
from shape6d.common.types import Candidate, Proposal
from shape6d.identify.depth_match import PointToTemplateMatcher
from shape6d.identify.size_gate import SizeGate
from shape6d.onboarding.sampling import fps_indices
from shape6d.onboarding.symmetry import detect_symmetry
from shape6d.onboarding.templates import build_view_templates, splat_depth, upright_view_mask
from shape6d.pose.template_init import coarse_poses_from_match
from shape6d.proposal.prompt_gen import LidarPromptGenerator
from shape6d.verify.symmetry_eval import SymmetryHandler
from shape6d.verify.verifier import Verifier

# ── 스타일 (dataviz 팔레트 — 카테고리 고정 순서) ──────────────────────────────
C_PALLET = "#2a78d6"    # blue    — 관측(팔레트)
C_REFINED = "#1baf7a"   # aqua    — ICP 정련 포즈
C_COARSE = "#eda100"    # yellow  — coarse 포즈
C_DISTR = "#eb6834"     # orange  — 방해물
C_FLOOR = "#9aa3af"     # gray    — 바닥(맥락)
C_BAD = "#e34948"       # red     — 스펙/실패 기준선
INK, INK2 = "#1a1d23", "#5a6270"

plt.rcParams.update({
    "font.family": "AppleGothic", "axes.unicode_minus": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#dde2ea", "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": "#eef1f5", "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titlecolor": INK,
})

K = CameraIntrinsics(fx=914.0, fy=914.0, cx=640.0, cy=400.0)
SIGMA = 0.005
OUT = Path(__file__).resolve().parents[2] / "docs" / "assets_04"
OUT.mkdir(parents=True, exist_ok=True)


# ── 합성 구성 (tests/test_e2e_geo.py와 동일 파라미터) ─────────────────────────
def pallet_mesh():
    deck = trimesh.creation.box((1.1, 1.1, 0.03)); deck.apply_translation([0, 0, 0.135])
    parts = [deck]
    for y in (-0.48, 0.0, 0.48):
        r = trimesh.creation.box((1.1, 0.14, 0.12)); r.apply_translation([0, y, 0.06])
        parts.append(r)
    return trimesh.util.concatenate(parts)


def onboard():
    mesh = pallet_mesh()
    surf, fidx = trimesh.sample.sample_surface(mesh, 60000, seed=0)
    surf = np.asarray(surf); nrm = np.asarray(mesh.face_normals)[np.asarray(fidx)]
    diam = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    r_bs = float(np.linalg.norm(surf - surf.mean(0), axis=1).max())
    tpl = build_view_templates(surf.astype(np.float32), r_bs, diam)
    sym = detect_symmetry(surf[:4096])
    return dict(surf=surf, master=surf[:8192], master_n=nrm[:8192], diam=diam,
                tpl=tpl, sym=sym)


def gt_pose(yaw_deg=20.0, dist=3.0, dx=0.1, dy=0.42):
    th = np.deg2rad(yaw_deg)
    Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    R0 = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])   # 모델 z(up) → 카메라 −y
    return R0 @ Rz, np.array([dx, dy, dist])


def synth_frame(onb, R_gt, t_gt, seed=0):
    """장면 전체 z-buffer 가시성 적용 — 구성 요소 간 가림을 물리적으로 일관되게.
    (초기 버전은 팔레트 뒤/밑 바닥 포인트가 잔존하는 물리 오류가 있었고, 이것이
    frame-wide free-space 검증에서 정답 포즈 22% 위반으로 정확히 검출됐다.)"""
    rng = np.random.default_rng(seed)
    posed = onb["surf"] @ R_gt.T + t_gt

    Xf, Zf = np.meshgrid(np.linspace(-1.2, 1.2, 120), np.linspace(2.0, 4.5, 100))
    floor_y = float(t_gt[1] + 0.15)
    floor = np.stack([Xf.ravel(), np.full(Xf.size, floor_y), Zf.ravel()], 1)
    distr = rng.normal(0, 0.05, (900, 3)) + np.array([-0.9, floor_y - 0.22, 2.4])

    all_pts = np.vstack([posed, floor, distr]).astype(np.float32)
    labels_all = np.concatenate([np.zeros(len(posed)), np.ones(len(floor)), 2 * np.ones(len(distr))])
    # 가시성 판정은 stride-2 격자에서: 표면 샘플 밀도(가시 ~2.4만pt)가 풀해상도
    # 픽셀(~8.5만)보다 성겨서, 풀해상도 z-buffer는 구멍으로 뒤쪽 표면이 샌다
    # (frame-wide free-space 검증이 이 오류를 free_viol 18%로 검출했음)
    s = 0.5
    depth2 = splat_depth(all_pts, K.fx * s, K.fy * s, K.cx * s, K.cy * s, (400, 640))
    z = all_pts[:, 2]
    u2 = np.clip(np.round(K.fx * s * all_pts[:, 0] / z + K.cx * s).astype(np.int64), 0, 639)
    v2 = np.clip(np.round(K.fy * s * all_pts[:, 1] / z + K.cy * s).astype(np.int64), 0, 399)
    vis = np.abs(depth2[v2, u2] - z) < 0.015
    u = np.clip(np.round(K.fx * all_pts[:, 0] / z + K.cx).astype(np.int64), 0, 1279)
    v = np.clip(np.round(K.fy * all_pts[:, 1] / z + K.cy).astype(np.int64), 0, 799)

    keep = np.zeros(len(all_pts), bool)
    obj_sel = np.nonzero(vis & (labels_all == 0) & (v % 5 == 0))[0]
    keep[rng.permutation(obj_sel)[:1200]] = True
    keep |= vis & (labels_all == 1) & (np.arange(len(all_pts)) % 4 == 0)   # 바닥 서브샘플
    keep |= vis & (labels_all == 2) & (np.arange(len(all_pts)) % 6 == 0)   # 방해물

    pts = all_pts[keep].copy()
    labels = labels_all[keep]
    pts[:, 2] += rng.normal(0, SIGMA, len(pts)).astype(np.float32)
    fb = build_frame_bundle(np.zeros((800, 1280, 3), np.uint8), pts,
                            np.ones(len(pts), np.float32), np.zeros(len(pts), np.float32),
                            K, np.eye(4))
    return fb, labels


def run_pipeline(onb, fb, sym_h, X_verify):
    t0 = time.perf_counter()
    gen = LidarPromptGenerator(voxel=0.06, min_cluster_pts=30)
    _, clusters = gen(fb)
    t1 = time.perf_counter()
    cands = [Candidate(
        proposal=Proposal(mask=None, bbox=np.zeros(4), score=.5, source="lidar_hull",
                          cluster_id=c.id, lidar_idx=c.point_indices, n_lidar=len(c.point_indices)),
        pts=fb.lidar_points[c.point_indices], uv=fb.lidar_pixels[c.point_indices])
        for c in clusters]
    gated = SizeGate()(cands, onb["diam"])
    if not gated:
        return None
    cand = max(gated, key=lambda c: c.pts.shape[0])
    matcher = PointToTemplateMatcher(onb["tpl"]["tdf"], onb["tpl"]["tpl_center"], onb["diam"],
                                     view_mask=upright_view_mask(),  # §1.4e 뷰 프루닝
                                     top_views_pass2=5)
    m = matcher.match(cand.pts, k=5)
    cand.scores["depth"] = m.s_depth
    t2 = time.perf_counter()
    hyps = coarse_poses_from_match(m, onb["tpl"]["tpl_pose"], onb["tpl"]["tpl_center"])
    ver = Verifier(K, sym_h, sigma_lidar=SIGMA)
    # free-space 검사용 프레임 전체 유효 관측 (≤2만 pt 서브샘플)
    vs, us = np.nonzero(fb.valid_mask)
    step = max(1, len(vs) // 20000)
    frame_obs = (np.stack([us[::step], vs[::step]], 1).astype(np.float64),
                 fb.sparse_depth[vs[::step], us[::step]].astype(np.float64))
    res = ver(hyps, cand.pts.astype(np.float64), cand.uv, onb["master"],
              onb["master_n"], X_verify, onb["diam"], s2_scores=cand.scores,
              frame_obs=frame_obs)
    t3 = time.perf_counter()
    return dict(clusters=clusters, cand=cand, match=m, hyps=hyps, res=res,
                ms=dict(s1=(t1 - t0) * 1e3, s2=(t2 - t1) * 1e3, s4=(t3 - t2) * 1e3))


def project(pts):
    return (K.fx * pts[:, 0] / pts[:, 2] + K.cx, K.fy * pts[:, 1] / pts[:, 2] + K.cy)


# ── 그림 ──────────────────────────────────────────────────────────────────────
def fig_onboarding(onb):
    fig = plt.figure(figsize=(11, 6.4))
    gs = fig.add_gridspec(2, 4, hspace=0.42, wspace=0.32)

    ax = fig.add_subplot(gs[0, 0])
    M = onb["master"]
    ax.scatter(M[:, 0], M[:, 1], s=0.5, c=C_PALLET, alpha=0.5, linewidths=0)
    ax.set_title("마스터 샘플 16384→8192 (상면 x·y)"); ax.set_aspect("equal")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")

    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(M[:, 0], M[:, 2], s=0.5, c=C_PALLET, alpha=0.5, linewidths=0)
    ax.set_title("측면 (x·z) — 데크+런너 3열"); ax.set_aspect("equal")
    ax.set_xlabel("x [m]"); ax.set_ylabel("z [m]")

    ax = fig.add_subplot(gs[0, 2:])
    axis = onb["sym"].sym_rots
    txt = (f"대칭 자동검출 (S0, 교차검증 [상-1] 수정판)\n\n"
           f"검출 결과: 유한군 |S| = {len(axis)}  (C2, 수직축 180°)\n"
           f"연속축: {len(onb['sym'].sym_axes)}개\n\n" + "\n".join(onb["sym"].log[:4]))
    ax.text(0.02, 0.95, txt, va="top", ha="left", fontsize=10, color=INK,
            family="AppleGothic", transform=ax.transAxes)
    ax.axis("off")

    for i, v in enumerate((0, 14, 29)):
        ax = fig.add_subplot(gs[1, i])
        d = onb["tpl"]["tpl_depth"][v].astype(np.float32) / 1000.0
        im = ax.imshow(np.where(d > 0, d, np.nan), cmap="Blues_r")
        ax.set_title(f"depth 템플릿 뷰 {v}/42"); ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, label="깊이 [m]")

    ax = fig.add_subplot(gs[1, 3])
    tdf = onb["tpl"]["tdf"][0].astype(np.float32)
    im = ax.imshow(tdf[:, :, 24].T, cmap="Blues_r", origin="lower")
    ax.set_title("TDF 슬라이스 (뷰 0, z 중앙)"); ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, label="절단거리 [m]")
    fig.suptitle("S0 온보딩 — CAD → 마스터 포인트 · 대칭 · 42뷰 depth 템플릿 · TDF (point-splat 렌더, GL 무의존)",
                 fontsize=12, color=INK, y=0.99)
    fig.savefig(OUT / "fig1_onboarding.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_scene(fb, labels, R_gt, t_gt):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    P = fb.lidar_points
    names = {0: ("팔레트 (1200pt)", C_PALLET), 1: ("바닥", C_FLOOR), 2: ("방해물", C_DISTR)}
    ax = axes[0]
    for k, (nm, c) in names.items():
        m = labels == k
        ax.scatter(P[m, 0], P[m, 2], s=1.2 if k == 0 else 0.8, c=c, label=nm,
                   alpha=0.8 if k == 0 else 0.45, linewidths=0)
    ax.set_title("합성 장면 상면도 (카메라 x·z)")
    ax.set_xlabel("x [m]"); ax.set_ylabel("z (깊이) [m]")
    ax.legend(loc="upper right", frameon=False, markerscale=6)

    ax = axes[1]
    for k, (nm, c) in names.items():
        m = labels == k
        u, v = project(P[m])
        ax.scatter(u, v, s=1.2 if k == 0 else 0.8, c=c, alpha=0.8 if k == 0 else 0.45, linewidths=0)
    ax.set_xlim(0, 1280); ax.set_ylim(800, 0); ax.set_aspect("equal")
    ax.set_title("카메라 투영 1280×800 — 주사선 희소화(행 1/5)·σ 5mm")
    ax.set_xlabel("u [px]"); ax.set_ylabel("v [px]")
    fig.suptitle("합성 LiDAR 프레임 — 팔레트 @3m·yaw 20° + 바닥 + 방해물", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_scene.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_s1s2(run, fb):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    cols = [C_PALLET, C_DISTR, "#4a3aa7", "#e87ba4"]
    for i, c in enumerate(run["clusters"]):
        P = fb.lidar_points[c.point_indices]
        ax.scatter(P[:, 0], P[:, 2], s=1.5, c=cols[i % 4], linewidths=0,
                   label=f"클러스터 {c.id} ({len(c.point_indices)}pt, 대각 {c.bbox_diag:.2f}m)")
    ax.set_title("S1: 평면 제거 + voxel 클러스터링")
    ax.set_xlabel("x [m]"); ax.set_ylabel("z [m]")
    ax.legend(loc="upper right", frameon=False, markerscale=5, fontsize=9)

    ax = axes[1]
    pv_all = run["match"].per_view
    active = np.nonzero(pv_all >= 0)[0]           # 수평 밴드 프루닝 후 활성 뷰만
    pv = pv_all[active]
    order = active[np.argsort(pv)[::-1]]
    bar_c = [C_PALLET if v == run["match"].best.view else "#c9d4e3" for v in order]
    ax.bar(range(len(order)), pv_all[order], color=bar_c, width=0.82)
    ax.set_title(f"S2: TDF 정합 — 활성 {len(order)}/42뷰 (직립 프루닝), "
                 f"S_depth={run['match'].s_depth:.2f}")
    ax.set_xlabel("활성 뷰 (점수순)"); ax.set_ylabel("pass1 점수")
    ax.axhline(0.3, color=C_BAD, lw=1, ls="--")
    ax.text(len(order) - 0.5, 0.315, "θ_depth_min", ha="right", fontsize=8.5, color=C_BAD)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_s1s2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_pose(run, onb, R_gt, t_gt, sym_h, X_verify):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))
    cand = run["cand"]
    M = onb["master"][::6]
    h0 = run["hyps"][0]
    R_e, t_e = run["res"].pose[:3, :3], run["res"].pose[:3, 3]

    for ax, (R, t, cc, nm) in zip(axes[:2], [
            (h0.R, h0.t, C_COARSE, f"coarse (템플릿 argmax)"),
            (R_e, t_e, C_REFINED, "ICP 정련 후")]):
        u, v = project(cand.pts)
        ax.scatter(u, v, s=2.0, c=C_PALLET, linewidths=0, label="관측 LiDAR", alpha=0.7)
        u, v = project(M @ R.T + t)
        ax.scatter(u, v, s=0.7, c=cc, linewidths=0, label=nm, alpha=0.55)
        ep, er = sym_h.sym_aware_error(R, t, R_gt, t_gt, X_verify)
        ax.set_title(f"{nm}\n오차: {ep*1e3:.1f}mm / {er:.2f}°")
        ax.set_xlim(200, 1100); ax.set_ylim(700, 300)
        ax.set_aspect("equal"); ax.legend(loc="lower right", frameon=False, markerscale=5, fontsize=8.5)
        ax.set_xlabel("u [px]"); ax.set_ylabel("v [px]")

    ax = axes[2]
    r = run["res"].diag["icp"]["r_p2pl"] * 1e3
    ax.hist(r, bins=41, range=(-20, 20), color=C_REFINED, edgecolor="white", linewidth=0.4)
    tau = 3 * SIGMA * 1e3
    for s in (-tau, tau):
        ax.axvline(s, color=C_BAD, lw=1, ls="--")
    ax.text(tau + 0.5, ax.get_ylim()[1] * 0.92, "±τ_z=3σ", fontsize=8.5, color=C_BAD)
    st = run["res"].diag["stats"]
    ax.set_title(f"ICP p2pl 잔차 (grazing 불변 주 신호)\ninlier {st['inlier_ratio']*100:.0f}% · "
                 f"RMSE {st['rmse_inlier']*1e3:.1f}mm · free_viol {st['free_viol']*100:.1f}%")
    ax.set_xlabel("잔차 [mm]"); ax.set_ylabel("포인트 수")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_pose.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_trials(trials):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
    n = len(trials)
    x = np.arange(n)
    xt = [f"{t['yaw']:.0f}°\n{t['dist']:.1f}m" for t in trials]
    cols = [C_PALLET if t["verdict"] == "ACCEPT" else C_FLOOR for t in trials]

    for ax, key, spec, unit, title in [
            (axes[0], "e_pos", 10.0, "[mm] (log)", "위치 오차 (sym-aware)"),
            (axes[1], "e_rot", 1.0, "[deg] (log)", "회전 오차 (sym-aware)")]:
        vals = [t[key] * (1e3 if key == "e_pos" else 1.0) for t in trials]
        ax.bar(x, vals, color=cols, width=0.65)
        ax.set_yscale("log")
        ticks = [0.1, 1, 10, 100] if key == "e_pos" else [0.01, 0.1, 1, 10, 100]
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{t:g}" for t in ticks])   # AppleGothic mathtext 글리프 회피
        ax.axhline(spec, color=C_BAD, lw=1.2, ls="--")
        ax.text(n - 0.5, spec * 1.12, f"스펙 {spec:g}", ha="right", fontsize=9, color=C_BAD)
        ax.set_title(title); ax.set_ylabel(unit)
        ax.set_xticks(x); ax.set_xticklabels(xt, fontsize=8)
        for xi, (v, t) in enumerate(zip(vals, trials)):
            if t["verdict"] != "ACCEPT":
                ax.text(xi, v * 1.15, "REJECT\n(정직 실패)", ha="center", fontsize=7,
                        color=INK2)
    fig.suptitle(f"다중 시행 {n}회 — yaw 0~324° · 거리 2.5~3.5m · 오프셋 랜덤 (클린 합성, σ 5mm) · "
                 f"파랑 = ACCEPT, 회색 = REJECT", fontsize=11.5, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_trials.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── 실행 ──────────────────────────────────────────────────────────────────────
def main():
    print("온보딩...")
    onb = onboard()
    sym_h = SymmetryHandler(onb["sym"].sym_rots, onb["sym"].sym_axes)
    # idx_verify 2048 (03 §2.4 규약): 1024로는 스플랫이 포켓 개구를 홀필로 오폐색
    # → 포켓 너머 실관측 바닥이 가짜 free_viol이 됨 (다중 시행 실증)
    X_verify = onb["master"][fps_indices(onb["master"].astype(np.float32), 2048)]

    # 대표 시행 (E2E와 동일: yaw 20°, 3m)
    R_gt, t_gt = gt_pose()
    fb, labels = synth_frame(onb, R_gt, t_gt, seed=0)
    run = run_pipeline(onb, fb, sym_h, X_verify)
    fig_onboarding(onb)
    fig_scene(fb, labels, R_gt, t_gt)
    fig_s1s2(run, fb)
    fig_pose(run, onb, R_gt, t_gt, sym_h, X_verify)

    # 다중 시행
    rng = np.random.default_rng(7)
    trials = []
    for i in range(10):
        yaw = i * 36.0
        dist = float(rng.uniform(2.5, 3.5))
        dx = float(rng.uniform(-0.3, 0.3))
        Rg, tg = gt_pose(yaw, dist, dx, dy=0.42)
        fbi, _ = synth_frame(onb, Rg, tg, seed=100 + i)
        r = run_pipeline(onb, fbi, sym_h, X_verify)
        if r is None:
            trials.append(dict(yaw=yaw, dist=dist, e_pos=np.nan, e_rot=np.nan, verdict="NO_DET"))
            continue
        R_e, t_e = r["res"].pose[:3, :3], r["res"].pose[:3, 3]
        ep, er = sym_h.sym_aware_error(R_e, t_e, Rg, tg, X_verify)
        ch = r["hyps"][0]
        cp, cr = sym_h.sym_aware_error(ch.R, ch.t, Rg, tg, X_verify)
        trials.append(dict(yaw=yaw, dist=dist, dx=dx, e_pos=float(ep), e_rot=float(er),
                           coarse_pos=float(cp), coarse_rot=float(cr),
                           n_obs=int(r["cand"].pts.shape[0]), s2=float(r["match"].s_depth),
                           verdict=r["res"].verdict, p_conf=float(r["res"].p_conf),
                           ms=r["ms"]))
        print(f"  yaw {yaw:5.1f}° d {dist:.2f}m: coarse {cp*1e3:5.1f}mm/{cr:5.1f}° → "
              f"final {ep*1e3:5.1f}mm/{er:4.2f}°  [{r['res'].verdict}]")
    fig_trials(trials)

    ok = [t for t in trials if np.isfinite(t.get("e_pos", np.nan))]
    stats = dict(
        n=len(trials), n_ok=len(ok),
        pos_mm=dict(mean=float(np.mean([t["e_pos"] for t in ok]) * 1e3),
                    max=float(np.max([t["e_pos"] for t in ok]) * 1e3)),
        rot_deg=dict(mean=float(np.mean([t["e_rot"] for t in ok])),
                     max=float(np.max([t["e_rot"] for t in ok]))),
        pass_pos=sum(t["e_pos"] < 0.010 for t in ok),
        pass_rot=sum(t["e_rot"] < 1.0 for t in ok),
        accept=sum(t["verdict"] == "ACCEPT" for t in ok),
        trials=trials,
    )
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in stats.items() if k != "trials"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
