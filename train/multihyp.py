"""다중 가설 생성 + ICP 선택 (D-11, 20 v1.2 · 23 §4).

발견(23): 대응 행렬 내 정답 존재율 94.9% — 후보 1개 추출이 병목.
경로: 소프트 대응 A → 하드 대응·신뢰도 → 삼중쌍 Kabsch 가설 T개
     → 정렬 점수 순위 → 회전 sep_deg 중복 제거 → 상위 k
     → (선택) trimmed point-to-point ICP → 잔차 최소 가설 채택.
문헌 관례: SAM-6D 6000→300 가설·MegaPose 520·FoundationPose 504 (22 §4).
"""
from __future__ import annotations

import torch

from .pem_mini import MiniMatcher, geodesic_deg


def kabsch3(P: torch.Tensor, Q: torch.Tensor):
    """[N,3,3]→[N,3,3] 3점 강체 정합 (fp32 SVD). Q ≈ R·P + t."""
    cp, cq = P.mean(1, keepdim=True), Q.mean(1, keepdim=True)
    H = (P - cp).transpose(1, 2) @ (Q - cq)
    U, S, Vt = torch.linalg.svd(H.float())
    d = torch.det(Vt.transpose(1, 2) @ U.transpose(1, 2))
    D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1))
    R = Vt.transpose(1, 2) @ D @ U.transpose(1, 2)
    t = cq.squeeze(1) - (R @ cp.transpose(1, 2)).squeeze(-1)
    return R, t


@torch.no_grad()
def gen_hypotheses(sim, P_s, P_o, val_s, T: int = 256, k: int = 8,
                   sep_deg: float = 15.0):
    """소프트 대응에서 포즈 가설 상위 k개 (모델계→장면계).

    반환: R_h [B,k,3,3], t_h [B,k,3], score [B,k] (부족분은 top-1 복제 패딩).
    """
    sim = sim.float(); P_s = P_s.float(); P_o = P_o.float()
    B, K, _ = P_s.shape
    R1, t1, A = MiniMatcher.solve(sim, P_s, P_o)          # 기존 top-1 (0번 슬롯 보장)
    A_fg = A[..., :-1]
    conf, qi = A_fg.max(-1)
    w = conf * val_s * (1.0 - A[..., -1])
    w = torch.where(w.sum(-1, keepdim=True) > 1e-6, w, val_s + 1e-6)
    Q = P_o.gather(1, qi.unsqueeze(-1).expand(-1, -1, 3))

    idx = torch.multinomial(w, 3 * T, replacement=True).view(B, T, 3)
    Ps3 = P_s.gather(1, idx.view(B, -1, 1).expand(-1, -1, 3)).view(B, T, 3, 3)
    Qo3 = Q.gather(1, idx.view(B, -1, 1).expand(-1, -1, 3)).view(B, T, 3, 3)
    Rh, th = kabsch3(Qo3.reshape(-1, 3, 3), Ps3.reshape(-1, 3, 3))
    Rh = Rh.view(B, T, 3, 3); th = th.view(B, T, 3)

    pred = torch.einsum("btij,bkj->btki", Rh, Q) + th.unsqueeze(2)
    r = (pred - P_s.unsqueeze(1)).norm(dim=-1)
    score = -(r * w.unsqueeze(1)).sum(-1) / w.sum(-1, keepdim=True)

    order = score.argsort(-1, descending=True)
    R_out = R1.unsqueeze(1).repeat(1, k, 1, 1)
    t_out = t1.unsqueeze(1).repeat(1, k, 1)
    s_out = torch.full((B, k), float("-inf"), device=sim.device)
    for b in range(B):
        sel_R, n = [R1[b]], 1
        s_out[b, 0] = 0.0                                  # top-1 슬롯
        for j in order[b].tolist():
            if n >= k:
                break
            Rj = Rh[b, j]
            d = geodesic_deg(Rj.unsqueeze(0).expand(len(sel_R), 3, 3),
                             torch.stack(sel_R))
            if float(d.min()) < sep_deg:
                continue
            sel_R.append(Rj)
            R_out[b, n] = Rj; t_out[b, n] = th[b, j]; s_out[b, n] = score[b, j]
            n += 1
    return R_out, t_out, s_out


@torch.no_grad()
def icp_select(R_h, t_h, P_s, val_s, master, diam, iters: int = 15,
               trim: float = 0.7, chunk: int = 512):
    """가설별 trimmed point-to-point ICP(장면→마스터) 후 잔차 최소 가설 선택.

    반환: R_sel [B,3,3], t_sel [B,3], res_sel [B] (÷직경 정규화 trimmed RMS).
    좌표 규약: 장면점 p ≈ R·m + t (m: 마스터/모델계) — 역변환으로 모델계 정렬.
    """
    B, k = R_h.shape[:2]
    dev = R_h.device
    Rf = R_h.reshape(B * k, 3, 3).clone()
    tf = t_h.reshape(B * k, 3).clone()
    P = P_s.unsqueeze(1).expand(B, k, -1, -1).reshape(B * k, -1, 3)
    V = val_s.unsqueeze(1).expand(B, k, -1).reshape(B * k, -1).bool()
    M = master.unsqueeze(1).expand(B, k, -1, -1).reshape(B * k, -1, 3)
    D = diam.unsqueeze(1).expand(B, k).reshape(B * k)
    n_keep = max(int(trim * P.shape[1]), 8)

    res = torch.full((B * k,), float("inf"), device=dev)
    for s in range(0, B * k, chunk):
        e = slice(s, min(s + chunk, B * k))
        Rc, tc = Rf[e], tf[e]
        Pc, Vc, Mc = P[e], V[e], M[e]
        for _ in range(iters):
            pm = torch.einsum("nji,nkj->nki", Rc, Pc - tc.unsqueeze(1))  # 장면→모델계
            d = torch.cdist(pm, Mc)
            nn_d, nn_i = d.min(-1)
            nn_d = torch.where(Vc, nn_d, torch.full_like(nn_d, float("inf")))
            keep = nn_d.topk(n_keep, largest=False).indices
            src = Pc.gather(1, keep.unsqueeze(-1).expand(-1, -1, 3))
            tgt_i = nn_i.gather(1, keep)
            tgt = Mc.gather(1, tgt_i.unsqueeze(-1).expand(-1, -1, 3))
            # tgt(모델계) → src(장면계) 강체 재적합
            cs, ct = src.mean(1, keepdim=True), tgt.mean(1, keepdim=True)
            H = (tgt - ct).transpose(1, 2) @ (src - cs)
            U, _, Vt = torch.linalg.svd(H.float())
            dd = torch.det(U @ Vt)
            Dm = torch.diag_embed(torch.stack(
                [torch.ones_like(dd), torch.ones_like(dd), dd], -1))
            Rc = (U @ Dm @ Vt).transpose(1, 2)             # = V·Dm·U^T, 모델→장면
            tc = cs.squeeze(1) - torch.einsum("nij,nj->ni", Rc, ct.squeeze(1))
        pm = torch.einsum("nji,nkj->nki", Rc, Pc - tc.unsqueeze(1))
        nn_d = torch.cdist(pm, Mc).min(-1).values
        nn_d = torch.where(Vc, nn_d, torch.full_like(nn_d, float("inf")))
        tr = nn_d.topk(n_keep, largest=False).values
        res[e] = (tr.pow(2).mean(-1).sqrt()) / D[e].clamp(min=1e-3)
        Rf[e], tf[e] = Rc, tc

    res = res.view(B, k)
    best = res.argmin(-1)
    bi = torch.arange(B, device=dev)
    return (Rf.view(B, k, 3, 3)[bi, best], tf.view(B, k, 3)[bi, best],
            res[bi, best])
