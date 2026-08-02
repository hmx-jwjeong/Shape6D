"""학습형 가설 랭커 (D-11 후속) — 잔차 argmin 선택을 학습 선택으로 대체.

배경(24 §2): 클린 체제 k=16 잔차 선택 66.5%, 비대칭 플립이 지배 잔여 결손
(오라클 여지 존재). 플립은 점거리 잔차로 판별 불가(23 §7) — 가설별 다차원
특징(잔차·free-space·대응 일치율·기하 문맥)을 로지스틱으로 결합해 선택한다.
검증기 보정(S4 10-d 로지스틱)과 같은 인프라 사상 — 학습은 train 물체만(제로샷).

사용:
  python3 train/hyp_ranker.py --gen --split train --n 24000   # 특징 덤프
  python3 train/hyp_ranker.py --gen --split val   --n 4000
  python3 train/hyp_ranker.py --fit                            # 학습+평가
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
import train.train_phase_a as T                                        # noqa: E402
from train.encoders import build_encoder                               # noqa: E402
from train.multihyp import gen_hypotheses, icp_select, freespace_viol  # noqa: E402
from train.pem_mini import MiniMatcher, sym_aware_rot_err_deg, geodesic_deg  # noqa: E402

FEATS = ["res", "res_margin", "viol", "agree05", "agree02", "rank",
         "is_top1", "geo_to_top1", "log_npt", "log_gn", "trans_z_rel"]


@torch.no_grad()
def hyp_features(Ra, ta, res, viol, P_s, val_s, P_o, sim, diam, npt, gn):
    """가설별 특징 [B,k,F] — 전부 라벨 무관(추론 시 계산 가능)."""
    B, k = Ra.shape[:2]
    _, _, A = MiniMatcher.solve(sim, P_s, P_o)
    qi = A[..., :-1].max(-1).indices
    Q = P_o.gather(1, qi.unsqueeze(-1).expand(-1, -1, 3))       # [B,K,3]
    pred = torch.einsum("bkij,bknj->bkni", Ra,
                        Q.unsqueeze(1).expand(B, k, -1, -1)) + ta.unsqueeze(2)
    r = (pred - P_s.unsqueeze(1)).norm(dim=-1) / diam.view(B, 1, 1)
    vmask = val_s.unsqueeze(1) > 0
    def frac(th):
        return ((r < th) & vmask).sum(-1).float() / vmask.sum(-1).clamp(min=1)
    agree05, agree02 = frac(0.05), frac(0.02)
    res_margin = res - res.min(-1, keepdim=True).values
    rank = torch.argsort(torch.argsort(res, -1), -1).float() / max(k - 1, 1)
    is_top1 = torch.zeros(B, k, device=Ra.device); is_top1[:, 0] = 1.0
    g1 = geodesic_deg(Ra.reshape(-1, 3, 3),
                      Ra[:, :1].expand(B, k, 3, 3).reshape(-1, 3, 3)
                      ).view(B, k) / 180.0
    ctx_npt = torch.log(npt.float().clamp(min=1)).view(B, 1).expand(B, k) / 10.0
    ctx_gn = torch.log(gn.float().clamp(min=1)).view(B, 1).expand(B, k) / 3.0
    tz = (ta[..., 2] / diam.view(B, 1)).clamp(-50, 50) / 20.0
    X = torch.stack([res * 10, res_margin * 10, viol, agree05, agree02, rank,
                     is_top1, g1, ctx_npt, ctx_gn, tz], -1)
    return X


@torch.no_grad()
def gen(a):
    tag = a.tag
    ck = torch.load(T.DATA / "runs" / f"ckpt_{tag}.pt", map_location=T.DEV)
    enc = build_encoder(tag.split("_")[0]).to(T.DEV)
    enc.load_state_dict(ck["enc"]); enc.eval()
    mat = MiniMatcher().to(T.DEV)
    mat.load_state_dict(ck["matcher"]); mat.eval()
    cfgf = T.DATA / "runs" / f"cfg_{tag}.json"
    pref = (json.load(open(cfgf)).get("data_prefix", "phase_a")
            if cfgf.exists() else "phase_a")
    bank = T.ObjBank(pref)
    z = np.load(T.DATA / f"{pref}_{a.split}.npz")
    va = {k: torch.from_numpy(z[k]) for k in z.files}
    g = torch.Generator(device=T.DEV); g.manual_seed(0)

    Xs, Ys, Es, OIs = [], [], [], []
    bs = 48
    n = min(a.n, len(va["oi"]))
    for s in range(0, n, bs):
        pts = va["pts"][s:s + bs].to(T.DEV).float()
        npt = va["npt"][s:s + bs].to(T.DEV)
        valid = (torch.arange(pts.shape[1], device=T.DEV)[None] < npt[:, None])
        oi = va["oi"][s:s + bs].to(T.DEV)
        Rg = va["R"][s:s + bs].to(T.DEV).view(-1, 3, 3)
        te = va["t"][s:s + bs].to(T.DEV) + torch.einsum("bij,bj->bi", Rg, bank.c[oi])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            F_s, P_s, val_s = T.encode_scene(enc, pts, valid)
            P_o, F_o = T.encode_cad(enc, bank, oi, g, n_views=6)
            sim = mat(F_s.float(), P_s.float(), F_o.float(), P_o.float(),
                      bank.diam[oi])
        sim = sim.float(); P_s = P_s.float(); P_o = P_o.float()
        Rh, th, _ = gen_hypotheses(sim, P_s, P_o, val_s, T=a.T, k=a.k)
        Ra, ta, res = icp_select(Rh, th, P_s, val_s, bank.master[oi],
                                 bank.diam[oi], return_all=True)
        viol = freespace_viol(Ra, ta, P_s, val_s, bank.master[oi], bank.diam[oi],
                              P_full=pts, val_full=valid.float())
        X = hyp_features(Ra, ta, res, viol, P_s, val_s, P_o, sim,
                         bank.diam[oi], npt, bank.gn[oi])
        B, k = Ra.shape[:2]
        eh = sym_aware_rot_err_deg(
            Ra.reshape(-1, 3, 3), Rg.unsqueeze(1).expand(B, k, 3, 3).reshape(-1, 3, 3),
            bank.G[oi].unsqueeze(1).expand(-1, k, -1, -1, -1).reshape(-1, 16, 3, 3),
            bank.gn[oi].unsqueeze(1).expand(-1, k).reshape(-1)).view(B, k)
        Xs.append(X.cpu()); Ys.append((eh <= 30).cpu()); Es.append(eh.cpu())
        OIs.append(oi.cpu())
        if (s // bs) % 20 == 0:
            print(f"  {s + B}/{n}", flush=True)
    out = T.DATA / f"hypfeat_{tag}_{a.split}.npz"
    np.savez(out, X=torch.cat(Xs).numpy(), y=torch.cat(Ys).numpy(),
             err=torch.cat(Es).numpy(), oi=torch.cat(OIs).numpy(),
             feats=np.array(FEATS))
    print(f"[gen] {out} — {len(torch.cat(Xs))}샘플 × k{a.k}")


def fit(a):
    tag = a.tag
    tr = np.load(T.DATA / f"hypfeat_{tag}_train.npz")
    va = np.load(T.DATA / f"hypfeat_{tag}_val.npz")
    Xt = torch.tensor(tr["X"], dtype=torch.float32)
    yt = torch.tensor(tr["y"], dtype=torch.float32)
    B, k, F = Xt.shape
    w = torch.zeros(F, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=200)

    def closure():
        opt.zero_grad()
        # 리스트와이즈: softmax over k — "정답 가설을 고르는" 목적에 직결
        logit = Xt @ w + b
        ok = yt.sum(-1) > 0
        tgt = yt / yt.sum(-1, keepdim=True).clamp(min=1e-6)
        loss = -(tgt[ok] * torch.log_softmax(logit[ok], -1)).sum(-1).mean() \
               + 1e-4 * w.pow(2).sum()
        loss.backward()
        return loss
    opt.step(closure)

    def sel_acc(z, name):
        X = torch.tensor(z["X"], dtype=torch.float32)
        err = torch.tensor(z["err"])
        res_pick = err.gather(1, torch.tensor(z["X"][:, :, 0]).argmin(-1, keepdim=True)).squeeze(1)
        rk_pick = err.gather(1, (X @ w + b).argmax(-1, keepdim=True)).squeeze(1)
        orc = err.min(-1).values
        print(f"[{name}] 잔차 선택 ≤30° {100*(res_pick<=30).float().mean():.1f}% · "
              f"랭커 선택 {100*(rk_pick<=30).float().mean():.1f}% · "
              f"오라클(k) {100*(orc<=30).float().mean():.1f}%")
        return rk_pick

    with torch.no_grad():
        sel_acc(tr, "train")
        sel_acc(va, "val")
        print("가중치:", {f: round(float(v), 3) for f, v in zip(tr["feats"], w)})
        np.savez(T.DATA / f"hyp_ranker_{tag}.npz", w=w.numpy(), b=b.detach().numpy(),
                 feats=tr["feats"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="a1_cosine_ext2_d2_s0")
    ap.add_argument("--gen", action="store_true")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=24000)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--T", type=int, default=256)
    a = ap.parse_args()
    if a.gen:
        gen(a)
    if a.fit:
        fit(a)
