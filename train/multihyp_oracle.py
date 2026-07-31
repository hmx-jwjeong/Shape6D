"""다중 가설 오라클 측정 — any-of-k ≤30° 상한 (학습 없음, 판정용).

배경(법의학 실측): 비대칭 실패의 40%가 150°+(플립) — 모드 집중 실패.
질문: soft 대응 A에서 SAM-6D식 삼중쌍 가설 T개를 뽑아 k개를 검증기로 넘기면
      현 top-1 43% 대비 얼마나 회수되는가.
방법: 하드 대응(argmax A) → 신뢰도 가중 삼중쌍 샘플 → 3점 Kabsch → 정렬 점수
      순위 → 회전 15° 중복 제거 → any-of-k(sym-aware ≤30°) + 오라클 상한.

실행: python3 train/multihyp_oracle.py --tag a1_cosine_ext2_s0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
import train.train_phase_a as T                                        # noqa: E402
from train.pem_mini import MiniMatcher, sym_aware_rot_err_deg          # noqa: E402
from train.encoders import build_encoder                               # noqa: E402


def kabsch3(P, Q):
    """[N,3,3]×[N,3,3] 3점 강체 정합 (fp32 SVD)."""
    cp, cq = P.mean(1, keepdim=True), Q.mean(1, keepdim=True)
    H = (P - cp).transpose(1, 2) @ (Q - cq)
    U, S, Vt = torch.linalg.svd(H.float())
    d = torch.det(Vt.transpose(1, 2) @ U.transpose(1, 2))
    D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1))
    R = Vt.transpose(1, 2) @ D @ U.transpose(1, 2)
    t = cq.squeeze(1) - (R @ cp.transpose(1, 2)).squeeze(-1)
    return R, t


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="a1_cosine_ext2_s0")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--T", type=int, default=256, help="삼중쌍 가설 수")
    ap.add_argument("--sep_deg", type=float, default=15.0, help="가설 중복 제거 각")
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

    KS = (1, 4, 8, 16)
    hit = {k: [] for k in KS}
    hit_oracle, hit_top1_old, gn_all = [], [], []
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
            sim = mat(F_s.float(), P_s.float(), F_o.float(), P_o.float(), bank.diam[oi])
        sim = sim.float(); P_s = P_s.float(); P_o = P_o.float()
        B, K, _ = P_s.shape

        # 기존 top-1 (soft P_hat 경로)
        R1_, t1_, A = MiniMatcher.solve(sim, P_s, P_o)
        e1 = sym_aware_rot_err_deg(R1_, Rg, bank.G[oi], bank.gn[oi])
        hit_top1_old.append((e1 <= 30).float().cpu())

        # 하드 대응 + 신뢰도
        A_fg = A[..., :-1]
        conf, qi = A_fg.max(-1)                                   # [B,K]
        w = conf * val_s * (1.0 - A[..., -1])
        w = torch.where(w.sum(-1, keepdim=True) > 1e-6, w,
                        val_s + 1e-6)                             # 퇴화 가드
        Q = P_o.gather(1, qi.unsqueeze(-1).expand(-1, -1, 3))     # [B,K,3]

        # 삼중쌍 샘플 → 3점 Kabsch
        idx = torch.multinomial(w, 3 * a.T, replacement=True).view(B, a.T, 3)
        Ps3 = P_s.gather(1, idx.view(B, -1, 1).expand(-1, -1, 3)).view(B, a.T, 3, 3)
        Qo3 = Q.gather(1, idx.view(B, -1, 1).expand(-1, -1, 3)).view(B, a.T, 3, 3)
        Rh, th = kabsch3(Qo3.reshape(-1, 3, 3), Ps3.reshape(-1, 3, 3))  # 모델계→장면계
        Rh = Rh.view(B, a.T, 3, 3); th = th.view(B, a.T, 3)

        # 정렬 점수: 가중 대응 잔차 (SAM-6D Eq.7 정신의 경량판)
        pred = torch.einsum("btij,bkj->btki", Rh, Q) + th.unsqueeze(2)
        r = (pred - P_s.unsqueeze(1)).norm(dim=-1)                # [B,T,K]
        score = -(r * w.unsqueeze(1)).sum(-1) / w.sum(-1, keepdim=True)

        # 전 가설 sym-aware 오차
        eh = sym_aware_rot_err_deg(
            Rh.reshape(-1, 3, 3),
            Rg.unsqueeze(1).expand(B, a.T, 3, 3).reshape(-1, 3, 3),
            bank.G[oi].unsqueeze(1).expand(-1, a.T, -1, -1, -1).reshape(-1, 16, 3, 3),
            bank.gn[oi].unsqueeze(1).expand(-1, a.T).reshape(-1)).view(B, a.T)
        hit_oracle.append(((eh <= 30).any(-1)).float().cpu())

        # 점수순 + 각도 중복 제거 → any-of-k
        order = score.argsort(-1, descending=True)
        for b in range(B):
            sel, errs = [], []
            for j in order[b].tolist():
                Rj = Rh[b, j]
                if any(sym_aware_rot_err_deg(
                        Rj[None], Rk[None],
                        torch.eye(3, device=Rj.device).view(1, 1, 3, 3).expand(1, 16, 3, 3),
                        torch.ones(1, dtype=torch.long, device=Rj.device)) < a.sep_deg
                        for Rk in sel):
                    continue
                sel.append(Rj); errs.append(float(eh[b, j]))
                if len(sel) >= max(KS):
                    break
            for k in KS:
                hit[k].append(1.0 if any(e <= 30 for e in errs[:k]) else 0.0)
        gn_all.append(bank.gn[oi].cpu())
        if (s // bs) % 10 == 0:
            print(f"  {s + B}/{a.n}", flush=True)

    gn = torch.cat(gn_all).numpy()
    top1 = torch.cat(hit_top1_old).numpy()
    orc = torch.cat(hit_oracle).numpy()
    print(f"\n=== 다중 가설 오라클 — {a.tag} (n={len(top1)}, T={a.T}) ===")
    print(f"기존 top-1(soft): {top1.mean():.3f}")
    for k in KS:
        h = np.array(hit[k])
        print(f"any-of-{k:2d} (점수 선택): 전체 {h.mean():.3f} · "
              f"비대칭 {h[gn == 1].mean():.3f} · 저대칭 {h[(gn >= 2) & (gn <= 4)].mean():.3f} · "
              f"고대칭 {h[gn >= 8].mean():.3f}")
    print(f"오라클 상한(T={a.T} 전부): 전체 {orc.mean():.3f} · 비대칭 {orc[gn == 1].mean():.3f}")


if __name__ == "__main__":
    main()
