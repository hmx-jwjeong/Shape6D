"""fine 결합 엔드투엔드 평가 — 다중 가설 + fine 정제 (+선택 실험).

경로: top-1 / top-1+fine / mh(k)+ICP선택 / mh+fine / mh+fine² / mh+fine²+ICP
finev3 실측(docs/27): mh+fine²가 대표 경로 — ICP 후처리는 fine 이후 유해
(≤5° 21.7→19.1%, trans 0.043→0.096D — ICP p2p 잔차 바닥이 fine보다 얕음).

--select-exp: 전 가설 fine 정제 후 선택 점수 비교 (ICP잔차/fine확신/혼합/오라클).
finev3 실측: fine확신 선택 기각(69.0 vs 72.6%) — 브랜치 정합 정제라 틀린
대칭 브랜치에서도 확신 높음. 정제 후 오라클 83.8%/≤5° 48.2% — 선택 갭 ~11%p.

실행: python3 train/eval_fine_pipeline.py --tag a1_cosine_finev3_b_s0 --k 16
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
from train.multihyp import gen_hypotheses, icp_select                  # noqa: E402
from train.pem_mini import (FineMatcher, MiniMatcher,                  # noqa: E402
                            sym_aware_rot_err_deg)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="a1_cosine_finev3_b_s0")
    ap.add_argument("--n", type=int, default=2880)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--fine-tok", type=int, default=1024)
    ap.add_argument("--select-exp", action="store_true",
                    help="전 가설 정제 + 선택 점수 비교 (비용 ~k×fine)")
    a = ap.parse_args()

    ck = torch.load(T.DATA / "runs" / f"ckpt_{a.tag}.pt", map_location=T.DEV)
    enc = build_encoder(a.tag.split("_")[0]).to(T.DEV)
    enc.load_state_dict(ck["enc"]); enc.eval()
    mat = MiniMatcher().to(T.DEV); mat.load_state_dict(ck["matcher"]); mat.eval()
    fine = FineMatcher().to(T.DEV); fine.load_state_dict(ck["fine"]); fine.eval()
    _cfgf = T.DATA / "runs" / f"cfg_{a.tag}.json"
    _pref = (json.load(open(_cfgf)).get("data_prefix", "phase_a")
             if _cfgf.exists() else "phase_a")
    bank = T.ObjBank(_pref)
    z = np.load(T.DATA / f"{_pref}_val.npz")
    va = {k: torch.from_numpy(z[k]) for k in z.files}
    g = torch.Generator(device=T.DEV); g.manual_seed(0)

    paths = (["sel_res", "sel_conf", "sel_res4conf", "oracle"] if a.select_exp
             else ["top1", "top1_f", "mh_icp", "mh_f", "mh_ff", "mh_ff_icp"])
    E = {p: [] for p in paths}
    TRs = {p: [] for p in paths}
    GN = []
    bs = 48
    for s in range(0, min(a.n, len(va["oi"])), bs):
        pts = va["pts"][s:s + bs].to(T.DEV).float()
        npt = va["npt"][s:s + bs].to(T.DEV)
        valid = (torch.arange(pts.shape[1], device=T.DEV)[None] < npt[:, None])
        oi = va["oi"][s:s + bs].to(T.DEV)
        Rg = va["R"][s:s + bs].to(T.DEV).view(-1, 3, 3)
        te = va["t"][s:s + bs].to(T.DEV) + torch.einsum(
            "bij,bj->bi", Rg, bank.c[oi])
        G_, gn_, M_, D_ = bank.G[oi], bank.gn[oi], bank.master[oi], bank.diam[oi]
        B = len(oi)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            F_s, P_s, val_s, (F_sf, P_sf, val_sf) = T.encode_scene(
                enc, pts, valid, fine_tok=a.fine_tok)
            P_o, F_o, (P_of, F_of) = T.encode_cad(
                enc, bank, oi, g, n_views=6, fine_tok=a.fine_tok)
            sim = mat(F_s.float(), P_s.float(), F_o.float(), P_o.float(), D_)
        sim = sim.float(); P_s = P_s.float(); P_o = P_o.float()

        def add(p, R, t):
            E[p].append(sym_aware_rot_err_deg(R, Rg, G_, gn_).cpu())
            TRs[p].append(((t - te).norm(dim=-1) / D_).cpu())

        def frefine(R0, t0, ret_sim=False):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                sf = fine(F_sf.float(), P_sf.float(), F_of.float(),
                          P_of.float(), R0, t0, D_)
            R2, t2, _ = MiniMatcher.solve(sf.float(), P_sf.float(),
                                          P_of.float(), hard=True)
            return (R2, t2, sf.float()) if ret_sim else (R2, t2)

        Rh, th, _ = gen_hypotheses(sim, P_s, P_o, val_s, T=a.T, k=a.k)
        Ra, ta, res = icp_select(Rh, th, P_s, val_s, M_, D_, iters=15,
                                 return_all=True)
        bi = torch.arange(B, device=T.DEV)

        if not a.select_exp:
            R1, t1, _ = MiniMatcher.solve(sim, P_s, P_o)
            add("top1", R1, t1)
            add("top1_f", *frefine(R1, t1))
            best = res.argmin(-1)
            Rs, ts = Ra[bi, best], ta[bi, best]
            add("mh_icp", Rs, ts)
            Rf1, tf1 = frefine(Rs, ts)
            add("mh_f", Rf1, tf1)
            Rf2, tf2 = frefine(Rf1, tf1)
            add("mh_ff", Rf2, tf2)
            Rfi, tfi, _ = icp_select(Rf2.unsqueeze(1), tf2.unsqueeze(1),
                                     P_s, val_s, M_, D_, iters=15)
            add("mh_ff_icp", Rfi, tfi)
        else:
            Rf = torch.empty_like(Ra); tf = torch.empty_like(ta)
            conf = torch.empty(B, a.k, device=T.DEV)
            for j in range(a.k):
                R2, t2, _ = frefine(Ra[:, j], ta[:, j], ret_sim=True)
                Rf[:, j], tf[:, j] = R2, t2
                _, _, sf2 = frefine(R2, t2, ret_sim=True)
                A = sf2.softmax(-1)
                mp = A[..., :-1].max(-1).values * (1.0 - A[..., -1])
                conf[:, j] = (mp * val_sf).sum(-1) / val_sf.sum(-1).clamp(min=1)
            eK = sym_aware_rot_err_deg(
                Rf.reshape(B * a.k, 3, 3), Rg.repeat_interleave(a.k, 0),
                G_.repeat_interleave(a.k, 0),
                gn_.repeat_interleave(a.k)).view(B, a.k)
            tK = ((tf - te.unsqueeze(1)).norm(dim=-1) / D_.view(-1, 1))
            for p, idx in [("sel_res", res.argmin(-1)),
                           ("sel_conf", conf.argmax(-1))]:
                E[p].append(eK[bi, idx].cpu()); TRs[p].append(tK[bi, idx].cpu())
            top4 = res.topk(4, largest=False).indices
            i4 = top4[bi, conf.gather(1, top4).argmax(-1)]
            E["sel_res4conf"].append(eK[bi, i4].cpu())
            TRs["sel_res4conf"].append(tK[bi, i4].cpu())
            E["oracle"].append(eK.min(-1).values.cpu())
            TRs["oracle"].append(tK[bi, eK.argmin(-1)].cpu())
        GN.append(gn_.cpu())

    E = {k: torch.cat(v).numpy() for k, v in E.items()}
    TRs = {k: torch.cat(v).numpy() for k, v in TRs.items()}
    gn = torch.cat(GN).numpy(); asym = gn == 1
    names = {"top1": "top-1 (기준)", "top1_f": "top-1 + fine",
             "mh_icp": f"다중가설 k={a.k}+ICP선택", "mh_f": "  → + fine",
             "mh_ff": "  → + fine²", "mh_ff_icp": "  → + fine² + ICP",
             "sel_res": "선택=ICP잔차 (기존)", "sel_conf": "선택=fine확신",
             "sel_res4conf": "선택=잔차top4→확신", "oracle": "오라클(정제후 최적)"}
    print(f"=== fine 결합 평가 — {a.tag} (n={len(E[paths[0]])}, k={a.k}"
          f"{', 선택 실험' if a.select_exp else ''}) ===")
    print(f"{'경로':<26} {'≤30°':>6} {'≤5°':>6} {'≤1°':>6} {'p50':>7} "
          f"{'trans':>7}  {'비대칭≤30°':>9}")
    for p in paths:
        e, t = E[p], TRs[p]
        print(f"{names[p]:<28} {100*(e<=30).mean():5.1f}% {100*(e<=5).mean():5.1f}% "
              f"{100*(e<=1).mean():5.1f}% {np.median(e):6.1f}° {np.median(t):6.3f}D "
              f"{100*(e[asym]<=30).mean():8.1f}%")


if __name__ == "__main__":
    main()
