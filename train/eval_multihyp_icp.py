"""D-11 엔드투엔드 평가 — 다중 가설 + ICP 선택 vs 기존 top-1 (+동일 ICP).

공정 비교: 양쪽 모두 ICP 후처리 — 차이는 가설 수(1 vs k)와 선택뿐.
성공 정의: ① 수렴반경 진입 sym-aware ≤30° ② 실질 수렴 ≤5° (ICP 후).

실행: python3 train/eval_multihyp_icp.py --tag a1_cosine_ext2_s0 --k 8
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
import train.train_phase_a as T                                        # noqa: E402
from train.encoders import build_encoder                               # noqa: E402
from train.multihyp import gen_hypotheses, icp_select, freespace_viol  # noqa: E402
from train.pem_mini import MiniMatcher, sym_aware_rot_err_deg          # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="a1_cosine_ext2_s0")
    ap.add_argument("--n", type=int, default=1440)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--icp_iters", type=int, default=15)
    a = ap.parse_args()

    ck = torch.load(T.DATA / "runs" / f"ckpt_{a.tag}.pt", map_location=T.DEV)
    enc = build_encoder(a.tag.split("_")[0]).to(T.DEV)
    enc.load_state_dict(ck["enc"]); enc.eval()
    mat = MiniMatcher().to(T.DEV)
    mat.load_state_dict(ck["matcher"]); mat.eval()
    bank = T.ObjBank()
    z = np.load(T.DATA / "phase_a_val.npz")
    va = {k: torch.from_numpy(z[k]) for k in z.files}
    g = torch.Generator(device=T.DEV); g.manual_seed(0)

    LAMS = (0.0, 0.3, 1.0, 3.0)
    E1, E1i, GN, ms = [], [], [], []
    EK = {lam: [] for lam in LAMS}
    bs = 48
    for s in range(0, min(a.n, len(va["oi"])), bs):
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
        G_, gn_, M_, D_ = bank.G[oi], bank.gn[oi], bank.master[oi], bank.diam[oi]

        # ── 기존 top-1 (ICP 없음 / 동일 ICP)
        R1, t1, _ = MiniMatcher.solve(sim, P_s, P_o)
        E1.append(sym_aware_rot_err_deg(R1, Rg, G_, gn_).cpu())
        R1i, t1i, _ = icp_select(R1.unsqueeze(1), t1.unsqueeze(1), P_s, val_s,
                                 M_, D_, iters=a.icp_iters)
        E1i.append(sym_aware_rot_err_deg(R1i, Rg, G_, gn_).cpu())

        # ── 다중 가설 k → ICP → 잔차 선택 (+free-space 결합 선택 λ 스윕)
        t0 = time.time()
        Rh, th, _ = gen_hypotheses(sim, P_s, P_o, val_s, T=a.T, k=a.k)
        Ra, ta, res = icp_select(Rh, th, P_s, val_s, M_, D_, iters=a.icp_iters,
                                 return_all=True)
        viol = freespace_viol(Ra, ta, P_s, val_s, M_, D_,
                              P_full=pts, val_full=valid.float())
        ms.append((time.time() - t0) * 1e3 / len(oi))
        bi = torch.arange(len(oi), device=res.device)
        for lam in LAMS:
            best = (res + lam * viol).argmin(-1)
            EK[lam].append(sym_aware_rot_err_deg(
                Ra[bi, best], Rg, G_, gn_).cpu())
        GN.append(gn_.cpu())
        if (s // bs) % 10 == 0:
            print(f"  {s + len(oi)}/{a.n}", flush=True)

    e1 = torch.cat(E1).numpy(); e1i = torch.cat(E1i).numpy()
    gn = torch.cat(GN).numpy()
    asym = gn == 1

    def row(name, e):
        print(f"{name:26s} ≤30° {100*(e<=30).mean():5.1f}% · ≤5° {100*(e<=5).mean():5.1f}% "
              f"· p50 {np.median(e):5.1f}° │ 비대칭 ≤30° {100*(e[asym]<=30).mean():5.1f}%"
              f" · ≤5° {100*(e[asym]<=5).mean():5.1f}%")

    print(f"\n=== D-11 평가 — {a.tag} (n={len(e1)}, k={a.k}, T={a.T}, "
          f"ICP {a.icp_iters}회) ===")
    row("top-1 (ICP 없음)", e1)
    row("top-1 + ICP", e1i)
    for lam in LAMS:
        row(f"k={a.k} 선택 λ_fs={lam}", torch.cat(EK[lam]).numpy())
    print(f"다중 가설 경로 추가 비용: {np.mean(ms):.1f} ms/샘플 (RTX PRO, 학습 병행 중)")


if __name__ == "__main__":
    main()
