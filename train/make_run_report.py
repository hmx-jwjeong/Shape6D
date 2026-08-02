"""런 종료 후 결과 리포트 폴더 생성 — 개선 추적의 정본 저장소.

규약(2026-07-31 사용자 지시): 학습이 끝나면 그 런의 테스트 결과·네트워크 내부
시각화를 한 폴더에 모아 보고서를 만들고, 런 간 비교로 무엇이 개선되고 있는지
추적한다. train_phase_a.py가 종료 시 자동 호출(수동: --tag).

산출: reports/{tag}/
  index.html      메트릭·Δ(직전 대비)·아래 그림 전부 링크
  curves.png      수렴 곡선 4종 (val rot_p50·≤30°·trans / train CE)
  err_dist.png    sym-aware 회전오차 히스토그램+CDF, 대칭 클래스별 분해
  qual_cases.png  케이스 패널 (best/median/worst): 관측·대응 예측 착색·bg·A행렬
  feat_pca.png    인코더 특징 PCA(3ch) — 장면 vs CAD 뷰 (도메인 정렬 육안 점검)
  metrics.json    수치 전량 (summary.html 집계용)
+ reports/summary.html  전 런 비교표 + ≤30° 오버레이 곡선
"""
from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

import train.pem_mini as PM  # noqa: E402
import train.train_phase_a as T  # noqa: E402
from train.encoders import build_encoder  # noqa: E402

REPORTS = T.DATA / "reports"
INK, SUB, LINE = "#1a1d23", "#5a6270", "#dde2ea"
C_RUN = {"a0": "#2a78d6", "a1": "#eb6834", "a1d": "#1baf7a", "a2": "#eda100"}
plt.rcParams.update({"font.family": ["Noto Sans CJK JP"], "axes.unicode_minus": False,
                     "figure.facecolor": "white", "axes.edgecolor": LINE,
                     "axes.grid": True, "grid.color": LINE, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "font.size": 10})


def _enc_kind(tag: str) -> str:
    return tag.split("_")[0]


@torch.no_grad()
def run_eval(tag: str, n_eval: int = 3000, bs: int = 64):
    """체크포인트 로드 → val 상세 평가 (per-sample 오차·예측·전시 소재)."""
    bank = T.ObjBank(_pref)
    _cfgf = T.DATA / "runs" / f"cfg_{tag}.json"
    _pref = (json.load(open(_cfgf)).get("data_prefix", "phase_a")
             if _cfgf.exists() else "phase_a")
    va = {k: torch.from_numpy(v) for k, v in
          np.load(T.DATA / f"{_pref}_val.npz").items()}
    enc = build_encoder(_enc_kind(tag)).to(T.DEV)
    mat = PM.MiniMatcher().to(T.DEV)
    ck = torch.load(T.DATA / "runs" / f"ckpt_{tag}.pt")
    enc.load_state_dict(ck["enc"])
    mat.load_state_dict(ck["matcher"])
    enc.eval(); mat.eval()
    g = torch.Generator(device=T.DEV); g.manual_seed(0)
    E, TR, OI = [], [], []
    keep = None  # 전시 소재 1배치 보관
    for s in range(0, min(n_eval, len(va["oi"])), bs):
        pts = va["pts"][s:s + bs].to(T.DEV).float()
        npt = va["npt"][s:s + bs].to(T.DEV)
        valid = (torch.arange(pts.shape[1], device=T.DEV)[None] < npt[:, None])
        oi = va["oi"][s:s + bs].to(T.DEV)
        Rg = va["R"][s:s + bs].to(T.DEV).view(-1, 3, 3)
        te = va["t"][s:s + bs].to(T.DEV) + torch.einsum("bij,bj->bi", Rg, bank.c[oi])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            F_s, P_s, _ = T.encode_scene(enc, pts, valid)
            P_o, F_o = T.encode_cad(enc, bank, oi, g, n_views=6)
            sim = mat(F_s.float(), P_s.float(), F_o.float(), P_o.float(), bank.diam[oi])
        R_pr, t_pr, A = PM.MiniMatcher.solve(sim.float(), P_s.float(), P_o.float())
        e = PM.sym_aware_rot_err_deg(R_pr, Rg, bank.G[oi], bank.gn[oi])
        E.append(e.cpu()); OI.append(oi.cpu())
        TR.append(((t_pr - te).norm(dim=-1) / bank.diam[oi]).cpu())
        if keep is None:
            gm = torch.einsum("bji,bkj->bki", Rg, P_s.float() - te.unsqueeze(1))
            keep = dict(P_s=P_s.float().cpu(), A=A.float().cpu(),
                        P_o=P_o.float().cpu(), err=e.cpu(),
                        gt_model=gm.cpu(), oi=oi.cpu(), diam=bank.diam[oi].cpu())
    per = dict(err=torch.cat(E).numpy(), trans=torch.cat(TR).numpy(),
               oi=torch.cat(OI).numpy())
    per["gn"] = bank.gn.cpu().numpy()[per["oi"]]
    # 특징 PCA 소재: 장면/CAD 특징맵 1장씩
    pts = va["pts"][:1].to(T.DEV).float(); npt = va["npt"][:1].to(T.DEV)
    valid = (torch.arange(pts.shape[1], device=T.DEV)[None] < npt[:, None])
    geo_s, _ = PM.make_geo_maps(pts, valid, 1.0)
    cadp, cadv, _, _ = bank.cad_views(va["oi"][:1].to(T.DEV), g)
    geo_o, _ = PM.make_geo_maps(cadp[:, 0], cadv[:, 0], 0.0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        f_s = enc(geo_s).float().cpu()[0]
        f_o = enc(geo_o).float().cpu()[0]
    return per, keep, dict(f_s=f_s, f_o=f_o, v_s=geo_s[0, 6].cpu(), v_o=geo_o[0, 6].cpu())


def fig_curves(tag: str, out: Path):
    hist = json.load(open(T.DATA / "runs" / f"hist_{tag}.json"))
    col = C_RUN.get(_enc_kind(tag), "#e87ba4")
    fig, ax = plt.subplots(2, 2, figsize=(11, 6.4))
    for a, key, ttl, ref in ((ax[0, 0], "rot_p50", "미학습 회전 p50 (deg) ↓", 30),
                             (ax[0, 1], "le30", "≤30° 진입률 ↑", None),
                             (ax[1, 0], "trans_rel_p50", "병진 p50 (×D) ↓", None),
                             (ax[1, 1], "ce", "train 대응 CE ↓", 4.53)):
        xs = [h["ep"] for h in hist if h.get(key) is not None]
        ys = [h[key] for h in hist if h.get(key) is not None]
        a.plot(xs, ys, "-o", color=col, ms=3.5, lw=2)
        if ref:
            a.axhline(ref, color=SUB, ls="--", lw=1)
        a.set_title(ttl, fontsize=11)
        a.set_xlabel("epoch")
    fig.suptitle(f"수렴 곡선 — {tag}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out / "curves.png", dpi=130); plt.close(fig)


def fig_err_dist(per: dict, tag: str, out: Path):
    e = per["err"]
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4))
    ax[0].hist(e, bins=np.arange(0, 185, 7.5), color=C_RUN.get(_enc_kind(tag)),
               edgecolor="white")
    ax[0].axvline(30, color=SUB, ls="--"); ax[0].set_title("회전오차 분포 (deg)")
    xs = np.sort(e); ax[1].plot(xs, np.arange(1, len(xs) + 1) / len(xs), lw=2,
                                color=C_RUN.get(_enc_kind(tag)))
    ax[1].axvline(30, color=SUB, ls="--"); ax[1].set_xlim(0, 180)
    ax[1].set_title("CDF"); ax[1].set_ylabel("비율")
    groups = [("비대칭 gn=1", per["gn"] == 1), ("저대칭 2–4", (per["gn"] > 1) & (per["gn"] <= 4)),
              ("고대칭 ≥8", per["gn"] >= 8)]
    names = [n for n, m in groups]
    vals = [float((e[m] <= 30).mean()) if m.any() else 0 for _, m in groups]
    ax[2].bar(names, vals, color=["#2a78d6", "#eda100", "#b4232a"])
    ax[2].set_title("대칭 클래스별 ≤30° 진입률")
    for i, v in enumerate(vals):
        ax[2].text(i, v + 0.01, f"{v*100:.1f}%", ha="center", fontsize=10)
    fig.suptitle(f"오차 분포 — {tag} (미학습 {len(e)}샘플)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out / "err_dist.png", dpi=130); plt.close(fig)


def _xyz_color(P, diam):
    c = (P / (0.7 * diam) + 0.5).clamp(0, 1)
    return c.numpy()


def fig_qual(keep: dict, tag: str, out: Path):
    """케이스 패널: 대응 착색(예측 vs GT)·bg 확률·A 행렬 — 네트워크 내부 상태."""
    order = keep["err"].argsort()
    picks = [int(order[0]), int(order[len(order) // 2]), int(order[-1])]
    labels = ["best", "median", "worst"]
    fig, axes = plt.subplots(3, 4, figsize=(13.5, 10))
    for r, (b, lab) in enumerate(zip(picks, labels)):
        P_s = keep["P_s"][b]; A = keep["A"][b]; P_o = keep["P_o"][b]
        d = float(keep["diam"][b]); err = float(keep["err"][b])
        w = 1 - A[:, -1]
        A_fg = A[:, :-1] / A[:, :-1].sum(-1, keepdim=True).clamp(min=1e-8)
        P_hat = A_fg @ P_o
        gm = keep["gt_model"][b]
        for c_, (ttl, C) in enumerate((
                (f"{lab} · {err:.0f}° — 예측 대응 착색", _xyz_color(P_hat, d)),
                ("GT 모델좌표 착색 (정답 무늬)", _xyz_color(gm, d)),
                ("bg 확률 (밝을수록 배경 판정)", plt.cm.magma(A[:, -1].numpy())[:, :3]))):
            a = axes[r, c_]
            a.scatter(P_s[:, 0], -P_s[:, 1], c=C, s=14)
            a.set_title(ttl, fontsize=9.5); a.set_aspect("equal")
            a.set_xticks([]); a.set_yticks([])
        a = axes[r, 3]
        a.imshow(A[:, :-1].numpy(), aspect="auto", cmap="viridis")
        a.set_title("soft 대응 행렬 A (196×196)", fontsize=9.5)
        a.set_xlabel("CAD 슬롯"); a.set_ylabel("장면 점")
    fig.suptitle(f"케이스 패널 — {tag} · 열1(예측)≈열2(GT)면 대응 성공", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(out / "qual_cases.png", dpi=125); plt.close(fig)


def fig_pca(fp: dict, tag: str, out: Path):
    def pca3(f, mask):
        C, H, W = f.shape
        X = f.permute(1, 2, 0).reshape(-1, C)
        m = mask.reshape(-1) > 0.5
        if m.sum() < 10:
            return np.zeros((H, W, 3))
        mu = X[m].mean(0)
        _, _, V = torch.linalg.svd((X[m] - mu), full_matrices=False)
        Y = (X - mu) @ V[:3].T
        lo, hi = Y[m].quantile(0.02, 0), Y[m].quantile(0.98, 0)
        Y = ((Y - lo) / (hi - lo + 1e-9)).clamp(0, 1).reshape(H, W, 3).numpy()
        Y[~m.reshape(H, W).numpy()] = 1.0
        return Y
    # 56² 특징맵 위 마스크: 입력 224 유효맵 4배 다운
    ms = torch.nn.functional.max_pool2d(fp["v_s"][None, None].float(), 4)[0, 0]
    mo = torch.nn.functional.max_pool2d(fp["v_o"][None, None].float(), 4)[0, 0]
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.4))
    ax[0].imshow(pca3(fp["f_s"], ms)); ax[0].set_title("장면 특징 PCA (희소 LiDAR)")
    ax[1].imshow(pca3(fp["f_o"], mo)); ax[1].set_title("CAD 뷰 특징 PCA (렌더)")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"인코더 특징 PCA — {tag} · 색 유사 = 도메인 정렬", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out / "feat_pca.png", dpi=130); plt.close(fig)


def build_summary():
    rows = []
    for d in sorted(REPORTS.glob("*/metrics.json")):
        m = json.load(open(d))
        rows.append(m)
    rows.sort(key=lambda m: m["ts"])
    # 오버레이 곡선
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for m in rows:
        h = json.load(open(T.DATA / "runs" / f"hist_{m['tag']}.json")) \
            if (T.DATA / "runs" / f"hist_{m['tag']}.json").exists() else []
        xs = [e["ep"] for e in h]; col = C_RUN.get(_enc_kind(m["tag"]), "#e87ba4")
        ax[0].plot(xs, [e.get("le30", np.nan) for e in h], lw=2, label=m["tag"])
        ax[1].plot(xs, [e.get("rot_p50", np.nan) for e in h], lw=2, label=m["tag"])
    ax[0].set_title("≤30° 진입률 ↑ (전 런)"); ax[1].set_title("미학습 rot p50 ↓ (전 런)")
    ax[0].legend(fontsize=8, frameon=False); ax[1].axhline(30, color=SUB, ls="--")
    fig.tight_layout(); fig.savefig(REPORTS / "overlay.png", dpi=130); plt.close(fig)
    tr = "".join(
        f"<tr><td><a href='{m['tag']}/index.html'>{m['tag']}</a></td>"
        f"<td>{time.strftime('%m-%d %H:%M', time.localtime(m['ts']))}</td>"
        f"<td>{m['epochs']}</td><td>{m['le30']*100:.1f}%</td>"
        f"<td>{m['rot_p50']:.1f}°</td><td>{m['trans_p50']:.3f}D</td>"
        f"<td>{m['le30_sym1']*100:.1f}%</td></tr>" for m in rows[::-1])
    best = max(rows, key=lambda m: m["le30"]) if rows else None
    (REPORTS / "summary.html").write_text(f"""<!doctype html><meta charset=utf-8>
<title>Phase A 런 비교</title><style>body{{font-family:Pretendard,sans-serif;max-width:1000px;
margin:24px auto;padding:0 16px}}table{{border-collapse:collapse;width:100%;font-size:13px}}
th{{background:#1a1d23;color:#fff;padding:6px 9px;text-align:left}}
td{{padding:6px 9px;border-bottom:1px solid #dde2ea}}img{{max-width:100%}}</style>
<h2>Phase A 런 비교 (개선 추적)</h2>
<p>현재 최고 ≤30°: <b>{best['tag'] if best else '-'} — {best['le30']*100:.1f}%</b></p>
<img src="overlay.png">
<table><tr><th>런</th><th>일시</th><th>ep</th><th>≤30°</th><th>rot p50</th>
<th>trans</th><th>≤30° (비대칭만)</th></tr>{tr}</table>""", encoding="utf-8")


def main(tag: str):
    out = REPORTS / tag
    out.mkdir(parents=True, exist_ok=True)
    per, keep, fp = run_eval(tag)
    fig_curves(tag, out)
    fig_err_dist(per, tag, out)
    fig_qual(keep, tag, out)
    fig_pca(fp, tag, out)
    cfgf = T.DATA / "runs" / f"cfg_{tag}.json"
    cfg = json.load(open(cfgf)) if cfgf.exists() else {}
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True,
                         cwd=Path(__file__).parents[1]).stdout.strip()
    e = per["err"]
    m = dict(tag=tag, ts=time.time(), git=git, epochs=cfg.get("epochs"),
             le30=float((e <= 30).mean()), rot_p50=float(np.median(e)),
             rot_mean=float(e.mean()), trans_p50=float(np.median(per["trans"])),
             le30_sym1=float((e[per["gn"] == 1] <= 30).mean()) if (per["gn"] == 1).any() else 0.0,
             n=len(e))
    json.dump(m, open(out / "metrics.json", "w"), indent=1)
    # 직전 리포트 대비 Δ
    others = sorted([json.load(open(f)) for f in REPORTS.glob("*/metrics.json")
                     if f.parent.name != tag], key=lambda x: x["ts"])
    prev = others[-1] if others else None
    delta = (f"직전({prev['tag']}) 대비 Δ≤30° <b>{(m['le30']-prev['le30'])*100:+.1f}%p</b> · "
             f"Δrot p50 <b>{m['rot_p50']-prev['rot_p50']:+.1f}°</b>") if prev else "첫 리포트"
    cfg_rows = "".join(f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>"
                       for k, v in cfg.items())
    (out / "index.html").write_text(f"""<!doctype html><meta charset=utf-8>
<title>리포트 {tag}</title><style>body{{font-family:Pretendard,sans-serif;max-width:1060px;
margin:24px auto;padding:0 16px}}table{{border-collapse:collapse;font-size:12.5px;margin:8px 0}}
th{{background:#1a1d23;color:#fff;padding:5px 9px;text-align:left}}
td{{padding:5px 9px;border-bottom:1px solid #dde2ea}}img{{max-width:100%;border:1px solid #dde2ea;
border-radius:8px;margin:10px 0}}</style>
<h2>런 리포트 — {tag} <small style="color:#5a6270">(git {git})</small></h2>
<p><b>≤30° 진입률 {m['le30']*100:.1f}%</b> · rot p50 {m['rot_p50']:.1f}° ·
trans {m['trans_p50']:.3f}D · 비대칭 물체만 ≤30° {m['le30_sym1']*100:.1f}% · {delta} ·
<a href="../summary.html">전 런 비교 →</a></p>
<img src="curves.png"><img src="err_dist.png"><img src="qual_cases.png"><img src="feat_pca.png">
<h3>학습 파라미터</h3><table>{cfg_rows}</table>""", encoding="utf-8")
    build_summary()
    print(f"[report] {out}/index.html", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    main(ap.parse_args().tag)
