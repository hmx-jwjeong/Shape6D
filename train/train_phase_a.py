"""Phase A 학습 — 최소 매칭부 + 인코더 비교 (20 §5 / G-A).

[클린룸 규약 — 19 §3.2] SAM-6D 소스 미열람. pem_mini.py 참조.
학습 규약 (20 C-7): 후보별 lr(사전학습 3e-4 / 스크래치 1e-3) · 수렴 곡선 기록 ·
평가 = 미학습 34종 (물체 교집합 0) · 지표 = sym-aware 회전 p50 · ≤30° 진입률 · 병진.

실행: python3 train/train_phase_a.py --enc a1 --epochs 8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

sys.path.insert(0, str(Path(__file__).parents[1]))
from shape6d.onboarding.templates import TPL_FX, TPL_RES               # noqa: E402
from train.encoders import build_encoder                               # noqa: E402
from train.pem_mini import (                                           # noqa: E402
    MiniMatcher, fps_torch, make_geo_maps, phase_a_loss,
    sample_point_features, sym_aware_rot_err_deg,
)

DATA = Path("/mnt/samsung2tb/datasets/megapose/phase_a")
DEV = "cuda"
N_TOK = 196
VIEW_PX = 1536        # CAD 뷰당 특징 샘플 픽셀 수 (2뷰 합 3072 → FPS 196)
LR = {"a0": 1e-3, "a1": 3e-4, "a2": 1e-3}   # C-7 후보별 lr 정책


class ObjBank:
    """오브젝트 팩 전량 GPU 상주 (~168종, 수백 MB)."""

    def __init__(self):
        z = np.load(DATA / "phase_a_objs.npz")
        t = lambda k, dt: torch.from_numpy(z[k]).to(DEV).to(dt)
        self.master = t("master", torch.float32)          # [O,2048,3] centered
        self.c = t("c", torch.float32)                    # [O,3]
        self.diam = t("diam", torch.float32)
        self.G = t("g", torch.float32)                    # [O,16,3,3]
        self.gn = t("gn", torch.long)
        self.pose = t("pose", torch.float32)              # [O,42,4,4]
        self.u = t("u", torch.float32)
        self.v = t("v", torch.float32)
        self.z = t("z", torch.float32)
        self.npx = t("npx", torch.long)                   # [O,42]

    def cad_views(self, oi: torch.Tensor, rng: torch.Generator, n_views: int = 2):
        """샘플별 뷰 n개 → (view_pts [B,n,PX_CAP,3] 뷰계, valid, R_v, t_v)."""
        B = len(oi)
        vsel = torch.randint(0, self.pose.shape[1], (B, n_views), generator=rng, device=DEV)
        oi2 = oi.unsqueeze(1).expand(B, n_views)
        u = self.u[oi2, vsel]
        v = self.v[oi2, vsel]
        z = self.z[oi2, vsel]
        npx = self.npx[oi2, vsel]
        valid = (torch.arange(u.shape[-1], device=DEV)[None, None] < npx.unsqueeze(-1))
        x = (u - TPL_RES / 2) * z / TPL_FX
        y = (v - TPL_RES / 2) * z / TPL_FX
        pts = torch.stack([x, y, z], -1)
        Rt = self.pose[oi2, vsel]
        return pts, valid, Rt[..., :3, :3], Rt[..., :3, 3]


def encode_scene(enc, pts, valid):
    geo, uvn = make_geo_maps(pts, valid, domain_flag=1.0)
    f56 = enc(geo)
    sel = fps_torch(pts, valid, N_TOK)
    F_s = sample_point_features(f56, uvn.gather(1, sel.unsqueeze(-1).expand(-1, -1, 2)))
    P_s = pts.gather(1, sel.unsqueeze(-1).expand(-1, -1, 3))
    val_s = valid.gather(1, sel).float()
    return F_s, P_s, val_s


def encode_cad(enc, bank: ObjBank, oi, rng, n_views: int = 2):
    """뷰 n개 인코딩 → 모델계 (P_o[196,3], F_o[196,C]) (03 §6.4·§9).
    학습 n=2(gradient 비용), 평가 n=6(추론은 캐시 기반이라 커버리지 정당)."""
    pts, valid, R_v, t_v = bank.cad_views(oi, rng, n_views)
    B = len(oi)
    p2 = pts.reshape(B * n_views, -1, 3)
    v2 = valid.reshape(B * n_views, -1)
    geo, uvn = make_geo_maps(p2, v2, domain_flag=0.0)
    f56 = enc(geo)
    # 뷰당 VIEW_PX 픽셀 서브샘플 (유효 우선)
    r = torch.rand(v2.shape, device=DEV) + v2.float()
    sub = r.topk(VIEW_PX, dim=-1).indices
    F_px = sample_point_features(f56, uvn.gather(1, sub.unsqueeze(-1).expand(-1, -1, 2)))
    P_px = p2.gather(1, sub.unsqueeze(-1).expand(-1, -1, 3))
    ok = v2.gather(1, sub)
    # 뷰계 → centered 모델계: X_m = R_v^T (X_view − t_v)
    Pm = torch.einsum("bvji,bvkj->bvki", R_v,
                      P_px.reshape(B, n_views, VIEW_PX, 3) - t_v.unsqueeze(2))
    Pm = Pm.reshape(B, n_views * VIEW_PX, 3)
    Fm = F_px.reshape(B, n_views * VIEW_PX, -1)
    okm = ok.reshape(B, n_views * VIEW_PX)
    sel = fps_torch(Pm, okm, N_TOK)
    P_o = Pm.gather(1, sel.unsqueeze(-1).expand(-1, -1, 3))
    F_o = Fm.gather(1, sel.unsqueeze(-1).expand(-1, -1, Fm.shape[-1]))
    return P_o, F_o


@torch.no_grad()
def evaluate(enc, matcher, bank, va, bs=96, cap=4000):
    enc.eval(); matcher.eval()
    n = min(len(va["oi"]), cap)
    rot, tr, thirty = [], [], []
    g = torch.Generator(device=DEV); g.manual_seed(0)
    for s in range(0, n, bs):
        pts = va["pts"][s:s + bs].to(DEV).float()
        npt = va["npt"][s:s + bs].to(DEV)
        valid = (torch.arange(pts.shape[1], device=DEV)[None] < npt[:, None])
        oi = va["oi"][s:s + bs].to(DEV)
        R_gt = va["R"][s:s + bs].to(DEV).view(-1, 3, 3)
        t_eff = va["t"][s:s + bs].to(DEV) + torch.einsum(
            "bij,bj->bi", R_gt, bank.c[oi])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            F_s, P_s, val_s = encode_scene(enc, pts, valid)
            P_o, F_o = encode_cad(enc, bank, oi, g, n_views=6)
            sim = matcher(F_s.float(), P_s.float(), F_o.float(), P_o.float(),
                          bank.diam[oi])
        R_pr, t_pr, _ = MiniMatcher.solve(sim.float(), P_s.float(), P_o.float())
        e = sym_aware_rot_err_deg(R_pr, R_gt, bank.G[oi], bank.gn[oi])
        rot.append(e)
        tr.append((t_pr - t_eff).norm(dim=-1) / bank.diam[oi])
        thirty.append((e <= 30).float())
    rot = torch.cat(rot); tr = torch.cat(tr); th = torch.cat(thirty)
    enc.train(); matcher.train()
    return dict(rot_p50=float(rot.median()), rot_mean=float(rot.mean()),
                le30=float(th.mean()), trans_rel_p50=float(tr.median()), n=int(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", default="a1", choices=["a0", "a1", "a2"])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--opt", default="adamw", choices=["adamw", "muon"])
    ap.add_argument("--sched", default="onecycle", choices=["onecycle", "cosine"])
    ap.add_argument("--lr", type=float, default=None, help="미지정 시 LR[enc]·√(bs/48) 스케일")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(DATA / "runs"))
    ap.add_argument("--init-from", default=None,
                    help="체크포인트에서 이어 학습(추가학습) — 태그에 ext 접미사")
    ap.add_argument("--ext-name", default="ext",
                    help="이어학습 태그 접미사 (2차 연장은 ext2 등으로 충돌 방지)")
    ap.add_argument("--no-dash", action="store_true", help="대시보드 자동 기동 끄기")
    a = ap.parse_args()
    if not a.no_dash:
        from train.autodash import ensure_dashboard
        ensure_dashboard()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    lr = a.lr if a.lr is not None else LR[a.enc] * (a.bs / 48) ** 0.5
    parts = [a.enc]
    if a.opt != "adamw":
        parts.append(a.opt)
    if a.opt == "adamw" and a.sched != "onecycle":
        parts.append(a.sched)                  # 태그 충돌 방지 (a1_s0 cosine 덮어쓰기 사고)
    if a.bs != 48:
        parts.append(f"bs{a.bs}")
    if a.init_from:
        parts.append(a.ext_name)               # 이어학습 런 태그 분리 (덮어쓰기 방지)
    tag = "_".join(parts) + f"_s{a.seed}"

    bank = ObjBank()
    ld = lambda sp: {k: torch.from_numpy(v) for k, v in
                     np.load(DATA / f"phase_a_{sp}.npz").items()}
    tr, va = ld("train"), ld("val")
    enc = build_encoder(a.enc).to(DEV)
    matcher = MiniMatcher().to(DEV)
    if a.init_from:
        sd = torch.load(a.init_from, map_location=DEV)
        enc.load_state_dict(sd["enc"]); matcher.load_state_dict(sd["matcher"])
        print(f"[{tag}] init from {a.init_from}", flush=True)
    npar = sum(p.numel() for p in enc.parameters()) / 1e6
    npm = sum(p.numel() for p in matcher.parameters()) / 1e6
    print(f"[{tag}] enc {npar:.2f}M + matcher {npm:.2f}M · train {len(tr['oi'])} · "
          f"val(미학습 34종) {len(va['oi'])} · {a.opt}/{a.sched} lr {lr:g} bs {a.bs}", flush=True)
    # 대시보드용 설정 덤프 — 학습에 실제 사용되는 파라미터 전량
    json.dump({
        "encoder": a.enc, "epochs": a.epochs, "batch_size": a.bs, "seed": a.seed,
        "lr": lr, "optimizer": ("Muon(0.02·√s)+AdamW" if a.opt == "muon" else "AdamW") + "(wd=0.05)",
        "scheduler": ("const" if a.opt == "muon" else a.sched), "grad_clip": 5.0,
        "precision": "bf16 autocast + fp32(softmax/CE/SVD)",
        "enc_params_M": round(npar, 3), "matcher_params_M": round(npm, 3),
        "matcher": "cross-attn 2blk H=192 heads=4 + 거리RPE(16bin) + bg",
        "sim": "cosine / temp 0.1 · sim_dim 256",
        "loss": "g* 단일선택 CE(τ=0.10D) + 0.5·(rot_rad + 2·trans/D)",
        "n_tok": N_TOK, "cad_views_train": 2, "cad_views_eval": 6,
        "view_px": VIEW_PX, "data_train": len(tr["oi"]), "data_val": len(va["oi"]),
        "obj_train": int(len(torch.unique(tr["oi"]))),
        "obj_val_unseen": int(len(torch.unique(va["oi"]))),
        "sparsify": "ML-X 격자 σ3mm · npt U[256,4096] · frame_correction 적용",
        "init_from": a.init_from,
    }, open(out / f"cfg_{tag}.json", "w"), indent=1, ensure_ascii=False)

    import torch.nn as nn
    net = nn.ModuleDict({"enc": enc, "matcher": matcher})
    steps = len(tr["oi"]) // a.bs
    if a.opt == "muon":
        from train.muon import HybridMuon
        # lr_muon 0.004: 표준 0.02는 소형망(수백 채널)에서 스텝 폭주 실측 (skip 37% → ep2 전멸)
        opt = HybridMuon(net, lr_adam=lr, lr_muon=0.004 * (a.bs / 48) ** 0.5,
                         weight_decay=0.05)
        sched = None                                   # Muon은 상수 lr 관행 (+말기 수동 감쇠 가능)
    else:
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.05)
        if a.sched == "onecycle":
            sched = torch.optim.lr_scheduler.OneCycleLR(
                opt, lr, total_steps=a.epochs * steps, pct_start=0.15)
        else:                                          # cosine + 3% warmup
            warm = max(1, int(0.03 * a.epochs * steps))
            sched = torch.optim.lr_scheduler.SequentialLR(
                opt, [torch.optim.lr_scheduler.LinearLR(opt, 0.05, 1.0, warm),
                      torch.optim.lr_scheduler.CosineAnnealingLR(
                          opt, a.epochs * steps - warm, eta_min=lr * 0.05)], [warm])
    g = torch.Generator(device=DEV); g.manual_seed(a.seed)

    hist = []
    m0 = evaluate(enc, matcher, bank, va)
    print(f"[{tag}] ep0(초기) rot_p50={m0['rot_p50']:.1f}° ≤30°={m0['le30']:.3f}",
          flush=True)
    hist.append(dict(ep=0, **m0))
    t0 = time.time()
    for ep in range(a.epochs):
        perm = torch.randperm(len(tr["oi"]))
        agg = {}
        for s in range(steps):
            i = perm[s * a.bs:(s + 1) * a.bs]
            pts = tr["pts"][i].to(DEV, non_blocking=True).float()
            npt = tr["npt"][i].to(DEV)
            valid = (torch.arange(pts.shape[1], device=DEV)[None] < npt[:, None])
            oi = tr["oi"][i].to(DEV)
            R_gt = tr["R"][i].to(DEV).view(-1, 3, 3)
            t_eff = tr["t"][i].to(DEV) + torch.einsum("bij,bj->bi", R_gt, bank.c[oi])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                F_s, P_s, val_s = encode_scene(enc, pts, valid)
                P_o, F_o = encode_cad(enc, bank, oi, g)
                sim = matcher(F_s.float(), P_s.float(), F_o.float(), P_o.float(),
                              bank.diam[oi])
            loss, diag = phase_a_loss(sim.float(), P_s.float(), val_s, P_o.float(),
                                      R_gt, t_eff, bank.G[oi], bank.gn[oi],
                                      bank.diam[oi])
            if not torch.isfinite(loss):
                agg["skip"] = agg.get("skip", 0.0) + 1.0
                agg["_cskip"] = agg.get("_cskip", 0) + 1
                if agg["_cskip"] >= 300:
                    print(f"[{tag}] 연속 skip 300 — 파라미터 발산 판정, 조기 중단",
                          flush=True)
                    json.dump(hist, open(out / f"hist_{tag}.json", "w"), indent=1)
                    return
                opt.zero_grad(set_to_none=True)
                continue
            agg["_cskip"] = 0
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(enc.parameters()) + list(matcher.parameters()), 5.0)
            opt.step()
            if sched is not None:
                sched.step()
            for k, v in diag.items():
                agg[k] = agg.get(k, 0.0) + v
        m = evaluate(enc, matcher, bank, va)
        hist.append(dict(ep=ep + 1, min=round((time.time() - t0) / 60, 1), **m,
                         **{k: v / steps for k, v in agg.items()}))
        ok = max(1, steps - int(agg.get("skip", 0)))
        print(f"[{tag}] ep{ep+1}/{a.epochs} 미학습 rot_p50={m['rot_p50']:.1f}° "
              f"≤30°={m['le30']:.3f} trans={m['trans_rel_p50']:.3f}D | "
              f"train ce={agg.get('ce', float('nan'))/ok:.2f} "
              f"rot={agg.get('rot_deg', float('nan'))/ok:.0f}° "
              f"bg={agg.get('bg_rate', float('nan'))/ok:.2f} "
              f"g*≠I={agg.get('g_nonid', float('nan'))/ok:.2f} "
              f"skip={int(agg.get('skip', 0))} "
              f"({(time.time()-t0)/60:.0f}m)", flush=True)
        json.dump(hist, open(out / f"hist_{tag}.json", "w"), indent=1)
        torch.save(dict(enc=enc.state_dict(), matcher=matcher.state_dict()),
                   out / f"ckpt_{tag}.pt")
    print(f"[{tag}] 완료 {(time.time()-t0)/60:.1f}분", flush=True)
    # 결과 리포트 자동 생성 (규약 2026-07-31) — 실패해도 학습 결과는 보존
    try:
        import subprocess as sp
        sp.Popen([sys.executable, "-u", str(Path(__file__).parent / "make_run_report.py"),
                  "--tag", tag], start_new_session=True)
        print(f"[{tag}] 리포트 생성 시작 → reports/{tag}/index.html", flush=True)
    except Exception as e:
        print(f"[{tag}] 리포트 생성 실패(무시): {e}", flush=True)


if __name__ == "__main__":
    main()
