"""docs/08 MR6D 평가 보고서 자산 생성 — mr6d_results/results.json + 오버레이 사용.

산출: docs/assets_08/{fig_scatter.png, fig_scene.png, fig_visib.png, ov_*.jpg}
실행: uv run python eval/external_test/make_mr6d_report_assets.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
RES = Path(__file__).parent / "mr6d_results"
OUT = ROOT / "docs" / "assets_08"
OUT.mkdir(parents=True, exist_ok=True)

rs = json.load(open(RES / "results.json"))
A = [r for r in rs if r["A"]]
ep = np.array([r["A"]["e_pos"] for r in A]) * 1e3
er = np.array([r["A"]["e_rot"] for r in A])
vis = np.array([r["visib"] for r in A])
acc = np.array([r["A"]["verdict"] == "ACCEPT" for r in A])
ok = ep < 145
scene = np.array([r["scene"] for r in A])

C_OK, C_BAD, C_ACC = "#0a7d43", "#b4232a", "#0b5fff"

# --- fig 1: 회전 오차 vs ADD (모호성 밴드 구조) ---
fig, ax = plt.subplots(figsize=(6.4, 4.2))
add_rel = ep / 1449.4
ax.scatter(add_rel[ok], er[ok], s=26, c=C_OK, alpha=0.75, label="success (ADD<0.1D)")
ax.scatter(add_rel[~ok & acc], er[~ok & acc], s=26, c=C_ACC, alpha=0.75,
           label="fail but ACCEPT (false accept)")
ax.scatter(add_rel[~ok & ~acc], er[~ok & ~acc], s=26, c=C_BAD, alpha=0.75,
           label="fail, REJECT")
for y in (90, 180):
    ax.axhline(y, color="#999", lw=0.7, ls="--")
ax.set_xlabel("ADD / D")
ax.set_ylabel("rotation error (deg, sym-aware)")
ax.set_title("MR6D mode A: rotation error vs ADD (n=150)")
ax.legend(fontsize=8, loc="center right")
ax.set_xlim(0, max(add_rel) * 1.05)
fig.tight_layout()
fig.savefig(OUT / "fig_scatter.png", dpi=130)

# --- fig 2: 장면별 성공/플립 ---
scenes = sorted(set(scene))
labels = {"000000": "000000\nstacked 2", "000001": "000001\nfloor",
          "000002": "000002\nfloor", "000005": "000005\nupright+occl",
          "000006": "000006\nLOADED"}
fig, ax = plt.subplots(figsize=(6.4, 3.6))
w = 0.38
xs = np.arange(len(scenes))
n_ok = [int(ok[scene == s].sum()) for s in scenes]
n_all = [int((scene == s).sum()) for s in scenes]
n_flip = [int(((~ok) & (er > 150))[scene == s].sum()) for s in scenes]
ax.bar(xs - w / 2, [o / n for o, n in zip(n_ok, n_all)], w, color=C_OK, label="success rate")
ax.bar(xs + w / 2, [f / n for f, n in zip(n_flip, n_all)], w, color=C_BAD, label="flip rate")
ax.set_xticks(xs, [labels[s] for s in scenes], fontsize=8)
ax.set_ylabel("fraction")
ax.set_title("per-scene success / flip (mode A)")
for x, o, n in zip(xs, n_ok, n_all):
    ax.text(x - w / 2, o / n + 0.02, f"{o}/{n}", ha="center", fontsize=8)
ax.legend(fontsize=8)
ax.set_ylim(0, 1.05)
fig.tight_layout()
fig.savefig(OUT / "fig_scene.png", dpi=130)

# --- fig 3: 가시성 vs 위치 오차 ---
fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.scatter(vis[ok], ep[ok], s=24, c=C_OK, alpha=0.75, label="success")
ax.scatter(vis[~ok], np.clip(ep[~ok], 0, 600), s=24, c=C_BAD, alpha=0.7, label="fail (clip 600)")
ax.axhline(145, color="#999", lw=0.7, ls="--")
ax.text(0.31, 152, "ADD 0.1D = 145mm", fontsize=7.5, color="#666")
ax.axvline(0.8, color=C_ACC, lw=0.9, ls=":")
ax.text(0.805, 500, "visib 0.8\n→ 82% success\n24mm/1.75°", fontsize=8, color=C_ACC)
ax.set_xlabel("GT visible fraction")
ax.set_ylabel("position error (mm)")
ax.set_title("visibility governs accuracy")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "fig_visib.png", dpi=130)

# --- 대표 오버레이 복사 (jpg 압축) ---
picks = {
    "ov_success.jpg": "overlay_A_000000_000005.png",      # 13mm/0.9°
    "ov_grazing.jpg": "overlay_A_000000_000000.png",      # 사선 슬랩 103mm
    "ov_loaded.jpg": "overlay_A_000006_000000.png",       # 적재 팔레트
    "ov_upright.jpg": "overlay_A_000005_000000.png",      # 세워진 팔레트
}
for dst, src in picks.items():
    img = Image.open(RES / src).convert("RGB")
    img.save(OUT / dst, quality=82)
print("assets →", OUT)
