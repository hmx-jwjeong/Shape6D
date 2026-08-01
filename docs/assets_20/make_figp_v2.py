"""Fig 1 v2 — 전체 입출력 라인 (docs/20 v1.1, C-9 S1 개정 반영).

원본 figP(assets_17)는 생성 스크립트 미보존 — 본 스크립트가 이후 정본 생성기.
색 규약: 학습(초록)/동결(파랑)/비학습(회색)/오프라인(보라)/적합(주황).
실행: python3 docs/assets_20/make_figp_v2.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

from matplotlib import font_manager
for _f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"):
    if Path(_f).exists():
        font_manager.fontManager.addfont(_f)
plt.rcParams["font.family"] = font_manager.FontProperties(
    fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc").get_name() \
    if Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc").exists() else "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

C = dict(train="#0a7d43", train_f="#e6f5ec", frozen="#0b5fff", frozen_f="#e8f0ff",
         nolearn="#5a6270", nolearn_f="#f2f4f8", off="#6d28d9", off_f="#f1eafd",
         fit="#c2410c", fit_f="#fdf0e3", ink="#1a1d23", warn="#a05a00")

fig, ax = plt.subplots(figsize=(17.2, 8.6), dpi=155)
ax.set_xlim(0, 172); ax.set_ylim(0, 86); ax.axis("off")

def box(x, y, w, h, title, lines, kind, lw=1.6, title_fs=8.3, fs=6.9, dash=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.55",
                       fc=C[kind + "_f"], ec=C[kind], lw=lw,
                       linestyle=(0, (4, 2.6)) if dash else "-")
    ax.add_patch(p)
    ax.text(x + w / 2, y + h - 2.3, title, ha="center", va="top",
            fontsize=title_fs, fontweight="bold", color=C["ink"])
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 5.6 - i * 3.0, ln, ha="center", va="top",
                fontsize=fs, color=C["ink"])

def arrow(x1, y1, x2, y2, color="#5a6270", lw=1.7, dash=False, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=13, color=color, lw=lw,
                 linestyle=(0, (4, 2.6)) if dash else "-",
                 connectionstyle=f"arc3,rad={rad}"))

# ── 제목/범례 ───────────────────────────────────────────────
ax.text(2, 84, "Fig 1 v2 — 전체 입출력 라인 (S1 v2: 삭제식→선택식 ROI, C-9 반영)",
        fontsize=11.5, fontweight="bold", color=C["ink"])
lg = [("학습", "train"), ("동결", "frozen"), ("비학습", "nolearn"), ("오프라인", "off"), ("적합", "fit")]
for i, (t, k) in enumerate(lg):
    x = 112 + i * 11.5
    ax.add_patch(FancyBboxPatch((x, 82.6), 3.0, 2.2, boxstyle="round,pad=0.25",
                                fc=C[k + "_f"], ec=C[k], lw=1.3))
    ax.text(x + 4.0, 83.7, t, fontsize=7.6, va="center", color=C["ink"])

# ── 입력 열 ─────────────────────────────────────────────────
box(2, 60, 17, 17, "입력 (온라인)", ["rgb 1280×800", "sparse_depth f32", "(유효 ~9.6%)", "K·외부보정"], "nolearn")
box(2, 8, 17, 12, "CAD 메시", ["(오프라인 등록)", "물체당 1회"], "off")

# ── S0 오프라인 레인 ────────────────────────────────────────
box(24, 5, 44, 15, "S0 온보딩·캐시 (오프라인)",
    ["표면 20만 샘플 · icosphere 42뷰 depth 렌더",
     "TDF · sym_rots(G≤16 자동 검출) · 특징 캐시 ~30MB/물체"], "off")
arrow(19, 14, 24, 13, color=C["off"])

# ── S1 v2 ──────────────────────────────────────────────────
box(24, 47, 24, 30, "S1 · ROI (v2, C-9)",
    ["거친 클러스터", "(평면 제거 폐지)", "→ 투영 프롬프트", "", "EfficientViT-SAM-L0 (동결)",
     "마스크 · 침식 δpx", "ROI = LiDAR ∩ 마스크"], "frozen")
ax.text(36, 43.9, "폴백: 유효 마스크 0 → 클러스터 ROI 직행", fontsize=6.6,
        ha="center", color=C["warn"], style="italic")
arrow(19, 70, 24, 68, )

# ── S2 ─────────────────────────────────────────────────────
box(53, 47, 25, 30, "S2 · 후보 식별",
    ["① metric 크기 게이트", "② TDF 정합 + coverage", "③ 기하 디스크립터 0.35M",
     "    (자체 InfoNCE·학습)", "④ DINOv2-S CLS (동결)", "    — CAD 텍스처 시만"], "nolearn")
arrow(48, 62, 53, 62)

# ── S3 PEM ─────────────────────────────────────────────────
box(83, 47, 36, 30, "S3 · Shape6D-PEM (학습 ~8M)",
    ["기하 인코더: 절단 ConvNeXt-T 1.33M (timm Apache)",
     "RGB 보조 MBConv 0.4M · dropout p0.3 · e_norgb",
     "coarse 매칭 2blk (197tok·H192·RPE) → SVD·top-3",
     "fine 매칭 2blk (1024pt·linear attn·bg)", "→ (R,t) 후보"], "train")
arrow(78, 62, 83, 62)
arrow(68, 15, 90, 47, color=C["off"], rad=-0.25)  # S0 캐시 → S3 (CAD측 특징)
arrow(60, 20, 60, 47, color=C["off"])             # S0 → S2 (TDF·마스터)

# ── S4 ─────────────────────────────────────────────────────
box(124, 47, 26, 30, "S4 · 정밀화·검증",
    ["projective p2pl ICP", "계층 스케줄 (yaml)", "스플랫 잔차 · free-space",
     "10-d 로지스틱 보정", "(합성+sBOP 적합만)"], "fit")
arrow(119, 62, 124, 62)

# ── 출력 ────────────────────────────────────────────────────
box(155, 52, 15, 20, "출력", ["(R,t) + 보정 신뢰도", "accept / reject /", "UNCERTAIN", "→ 실패 시 재관측"], "nolearn")
arrow(150, 62, 155, 62)

# ── 학습 흐름 주석 (하단) ────────────────────────────────────
box(83, 5, 36, 15, "학습 (Phase A→B, C-10: PEM 집중)",
    ["MegaPose-GSO 합성 · 희소화(ML-X 격자) · frame_correction",
     "g* 대칭 CE(+bg) + 0.5·포즈 손실 · 분할 스택 학습 금지"], "train", dash=True)
arrow(101, 20, 101, 47, color=C["train"], dash=True)
box(124, 5, 26, 15, "평가 전용 (C-4)", ["UAM 필드 로그 70샘플", "학습·보정·튜닝 금지 (17 §5.5)"], "nolearn", dash=True)
arrow(137, 20, 137, 47, color=C["nolearn"], dash=True)

out = Path(__file__).parent / "figP_full_line_v2.png"
fig.savefig(out, bbox_inches="tight", facecolor="white")
print(out)
