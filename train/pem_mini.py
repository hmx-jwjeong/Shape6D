"""Phase A 최소 매칭부 — 대응 정식화 파일럿 (20 §5 Phase A).

[클린룸 규약 — 19 §3.2] SAM-6D 소스 미열람 서약. 참조: CVPR2024 논문(공개 문헌),
docs/03 §6.5(자체 언어 재기술), docs/15 §6(최소 요건). 코드 표현은 전부 신규 작성.

구조 (03 §6.5의 최소 부분집합):
  장면: pts(카메라계) → geo_maps(10ch) → 공유 인코더 → grid_sample @ FPS196 → F_s
  CAD : 뷰 2개 샘플 → geo_maps(뷰계) → 동일 인코더(gradient 포함 — 03 §9 검증[상-7])
        → 유효픽셀 역투영(모델계) → FPS196 → (P_o, F_o)
  매칭: 1블록 양방향 cross-attention(H=192) → 유사도(196×197, +bg) → soft 대응
        → 가중 Kabsch(SVD, fp32) → (R, t)
손실 (03 §9.3): 대칭군 g* 샘플당 단일 선택(coarse 해상도) → 대응 CE(+bg) + 포즈 손실.
  포인트별 min 금지 — 강체 비일관 라벨.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

RES = 224
N_TOK = 196          # coarse 토큰 수 (03 §6.5)


# ── 기하 텐서 유틸 ────────────────────────────────────────────────────────────

def fps_torch(pts: torch.Tensor, valid: torch.Tensor, n: int) -> torch.Tensor:
    """[B,N,3] + 유효 [B,N] → FPS 인덱스 [B,n]. 패딩점은 거리 -inf로 배제."""
    B, N, _ = pts.shape
    idx = torch.zeros(B, n, dtype=torch.long, device=pts.device)
    d = torch.where(valid, torch.full((B, N), torch.inf, device=pts.device),
                    torch.full((B, N), -torch.inf, device=pts.device))
    cur = valid.float().argmax(-1)                      # 첫 유효점에서 시작
    for i in range(n):
        idx[:, i] = cur
        diff = pts - pts.gather(1, cur.view(B, 1, 1).expand(B, 1, 3))
        d = torch.minimum(d, diff.pow(2).sum(-1))
        d = torch.where(valid, d, torch.full_like(d, -torch.inf))
        cur = d.argmax(-1)
    return idx


def make_geo_maps(pts: torch.Tensor, valid: torch.Tensor, domain_flag: float
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """포인트(임의 카메라형 프레임, z>0, 미터) → geo_maps (B,10,224,224) + uv_norm [B,N,2].

    crop 규약(03 §6.1 상당): 각도 좌표(x/z, y/z) bbox 정사각(+15%) → 224 격자
    z-buffer 스플랫. 채널 = XYZ(중심화·반경 정규화 3) · 노멀 근사(3) · 유효(1) ·
    정규화 깊이(1) · flags(rgb_valid=0, domain)(2).
    """
    B, N, _ = pts.shape
    dev, dt = pts.device, pts.dtype
    z = pts[..., 2].clamp(min=1e-6)
    ax, ay = pts[..., 0] / z, pts[..., 1] / z
    inf = torch.finfo(dt).max
    axm = torch.where(valid, ax, torch.full_like(ax, inf)).amin(1, keepdim=True)
    axM = torch.where(valid, ax, torch.full_like(ax, -inf)).amax(1, keepdim=True)
    aym = torch.where(valid, ay, torch.full_like(ay, inf)).amin(1, keepdim=True)
    ayM = torch.where(valid, ay, torch.full_like(ay, -inf)).amax(1, keepdim=True)
    cx, cy = (axm + axM) / 2, (aym + ayM) / 2
    s = torch.maximum(axM - axm, ayM - aym).clamp(min=1e-4) * 1.15

    u = ((ax - cx) / s + 0.5) * (RES - 1)
    v = ((ay - cy) / s + 0.5) * (RES - 1)
    lin = (v.round().long().clamp(0, RES - 1) * RES
           + u.round().long().clamp(0, RES - 1))
    zbuf = torch.full((B, RES * RES), 1e4, device=dev, dtype=dt)
    zsrc = torch.where(valid, pts[..., 2], torch.full_like(z, 1e4))
    zbuf.scatter_reduce_(1, lin, zsrc, "amin", include_self=True)
    d = torch.where(zbuf >= 1e4 - 1, torch.zeros_like(zbuf), zbuf).view(B, 1, RES, RES)
    vmap = (d > 0).to(dt)

    ys, xs = torch.meshgrid(torch.arange(RES, device=dev, dtype=dt),
                            torch.arange(RES, device=dev, dtype=dt), indexing="ij")
    AX = ((xs / (RES - 1) - 0.5).view(1, 1, RES, RES) * s.view(B, 1, 1, 1)
          + cx.view(B, 1, 1, 1))
    AY = ((ys / (RES - 1) - 0.5).view(1, 1, RES, RES) * s.view(B, 1, 1, 1)
          + cy.view(B, 1, 1, 1))
    P = torch.cat([AX * d, AY * d, d], 1) * vmap
    cnt = vmap.sum((2, 3), keepdim=True).clamp(min=1)
    Pc = (P - P.sum((2, 3), keepdim=True) / cnt) * vmap
    rad = (Pc.pow(2).sum(1, keepdim=True).sqrt() * vmap
           ).amax((2, 3), keepdim=True).clamp(min=1e-3)
    Pn = Pc / rad
    gx = F.pad(Pc[:, :, :, 2:] - Pc[:, :, :, :-2], (1, 1, 0, 0))
    gy = F.pad(Pc[:, :, 2:, :] - Pc[:, :, :-2, :], (0, 0, 1, 1))
    nr = torch.cross(gx, gy, dim=1)
    nr = nr / nr.norm(dim=1, keepdim=True).clamp(min=1e-6) * vmap
    dn = (d - d.sum((2, 3), keepdim=True) / cnt) * vmap
    dn = dn / dn.abs().amax((2, 3), keepdim=True).clamp(min=1e-6)
    flags = torch.zeros(B, 2, RES, RES, device=dev, dtype=dt)
    flags[:, 1] = domain_flag
    geo = torch.cat([Pn, nr, vmap, dn, flags], 1)
    uvn = torch.stack([u / (RES - 1) * 2 - 1, v / (RES - 1) * 2 - 1], -1)
    return geo, uvn


def sample_point_features(feat56: torch.Tensor, uvn: torch.Tensor) -> torch.Tensor:
    """(B,C,56,56) 특징맵에서 uv_norm [B,K,2] 위치 특징 [B,K,C] 추출."""
    g = uvn.unsqueeze(1)                                # (B,1,K,2)
    f = F.grid_sample(feat56, g, align_corners=True, mode="bilinear")
    return f.squeeze(2).transpose(1, 2)


def weighted_kabsch(P_s: torch.Tensor, P_hat: torch.Tensor, w: torch.Tensor
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """soft 대응 (P_hat: 모델계 예측 대응점) → (R, t): P_s ≈ R·P_hat + t.

    가중 SVD — fp32 고정 (03 §6.7 정밀도 규약), 미분 가능.
    """
    w = (w / w.sum(-1, keepdim=True).clamp(min=1e-8)).unsqueeze(-1).float()
    Ps, Ph = P_s.float(), P_hat.float()
    mu_s = (w * Ps).sum(1, keepdim=True)
    mu_h = (w * Ph).sum(1, keepdim=True)
    H = ((Ph - mu_h) * w).transpose(1, 2) @ (Ps - mu_s)
    U, S, Vh = torch.linalg.svd(H)
    det = torch.det(Vh.transpose(1, 2) @ U.transpose(1, 2))
    D = torch.diag_embed(torch.stack(
        [torch.ones_like(det), torch.ones_like(det), det], -1))
    R = Vh.transpose(1, 2) @ D @ U.transpose(1, 2)
    t = mu_s.squeeze(1) - (R @ mu_h.transpose(1, 2)).squeeze(-1)
    return R, t


def geodesic_deg(Ra: torch.Tensor, Rb: torch.Tensor) -> torch.Tensor:
    tr = torch.einsum("bij,bij->b", Ra, Rb)
    return torch.rad2deg(torch.arccos(((tr - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6)))


# ── 매칭 디코더 (클린룸) ─────────────────────────────────────────────────────

class _DistBias(nn.Module):
    """쌍거리(직경 정규화, 포즈 불변) → per-head attn 바이어스 — 03 §6.5 RPE 최소판."""

    def __init__(self, heads: int, n_bins: int = 16, d_max_rel: float = 1.0):
        super().__init__()
        self.emb = nn.Embedding(n_bins, heads)
        self.n_bins = n_bins
        self.d_max = d_max_rel
        nn.init.zeros_(self.emb.weight)

    def forward(self, P: torch.Tensor, diam: torch.Tensor) -> torch.Tensor:
        d = torch.cdist(P, P) / diam.view(-1, 1, 1).clamp(min=1e-3)
        b = (d / self.d_max * (self.n_bins - 1)).clamp(0, self.n_bins - 1).long()
        return self.emb(b).permute(0, 3, 1, 2)          # [B,heads,K,K]


class _Block(nn.Module):
    """self(+거리 RPE) + cross attention + FFN (pre-LN). H=192, heads 4 (03 §6.5)."""

    def __init__(self, h: int = 192, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.ns1 = nn.LayerNorm(h)
        self.qkv = nn.Linear(h, 3 * h)
        self.so = nn.Linear(h, h)
        self.rpe = _DistBias(heads)
        self.ns2 = nn.LayerNorm(h)
        self.cross_attn = nn.MultiheadAttention(h, heads, batch_first=True)
        self.ns3 = nn.LayerNorm(h)
        self.ffn = nn.Sequential(nn.Linear(h, 2 * h), nn.GELU(), nn.Linear(2 * h, h))

    def _self(self, x, P, diam):
        B, K, h = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        sh = (B, K, self.heads, h // self.heads)
        q, k, v = (t.view(sh).transpose(1, 2) for t in (q, k, v))
        att = (q @ k.transpose(-1, -2)) / (h // self.heads) ** 0.5
        att = att + self.rpe(P, diam)                    # 기하 구조 바이어스
        y = (att.softmax(-1) @ v).transpose(1, 2).reshape(B, K, h)
        return self.so(y)

    def forward(self, x, ctx, P, diam):
        x = x + self._self(self.ns1(x), P, diam)
        x = x + self.cross_attn(self.ns2(x), ctx, ctx, need_weights=False)[0]
        return x + self.ffn(self.ns3(x))


class MiniMatcher(nn.Module):
    """장면 196 ↔ CAD 196(+bg) 최소 매칭 디코더 — coarse 1블록 양방향.

    유사도 공간은 256 유지 (교사 유사도행렬 distill 정렬용 — 03 §6.5).
    """

    def __init__(self, c_in: int = 256, h: int = 192, sim_dim: int = 256,
                 temp: float = 0.1):
        super().__init__()
        self.in_s = nn.Linear(c_in + 3, h)     # 특징 ⊕ 좌표(스케일 정규화)
        self.in_o = nn.Linear(c_in + 3, h)
        self.blk_s = nn.ModuleList([_Block(h) for _ in range(2)])   # 03 §6.5: 2블록
        self.blk_o = nn.ModuleList([_Block(h) for _ in range(2)])
        self.out_s = nn.Linear(h, sim_dim)
        self.out_o = nn.Linear(h, sim_dim)
        self.bg = nn.Parameter(torch.zeros(1, 1, sim_dim))
        self.temp = temp

    def forward(self, F_s, P_s, F_o, P_o, diam):
        d = diam.view(-1, 1, 1).clamp(min=1e-3)
        xs = self.in_s(torch.cat([F_s, (P_s - P_s.mean(1, keepdim=True)) / d], -1))
        xo = self.in_o(torch.cat([F_o, P_o / d], -1))
        ys, yo = xs, xo
        for bs_, bo_ in zip(self.blk_s, self.blk_o):
            ys2 = bs_(ys, yo, P_s, diam)
            yo2 = bo_(yo, ys, P_o, diam)
            ys, yo = ys2, yo2
        # 안정화: 코사인 유사도(정규화 임베딩) — 로짓 ∈ [−1/τ, 1/τ] 유계.
        # (비정규화 내적 × τ=0.1은 로짓 폭주 → softmax/CE nan — 파일럿 1차 붕괴 원인)
        es = F.normalize(self.out_s(ys).float(), dim=-1)
        eo = torch.cat([self.out_o(yo).float(), self.bg.expand(len(es), 1, -1).float()], 1)
        eo = F.normalize(eo, dim=-1)
        sim = torch.einsum("bkc,bmc->bkm", es, eo) / self.temp                # fp32
        return sim                                                            # (B,196,197)

    @staticmethod
    def solve(sim: torch.Tensor, P_s: torch.Tensor, P_o: torch.Tensor):
        A = sim.softmax(-1)                          # fp32 softmax (03 §6.7)
        w = 1.0 - A[..., -1]
        # 전경 조건부 분포로 재정규화 → P_hat은 P_o의 볼록결합(항상 유계).
        # (1−bg 나눗셈은 bg→1에서 폭주 — 파일럿 1차 붕괴 원인 ②)
        A_fg = A[..., :-1]
        A_fg = A_fg / A_fg.sum(-1, keepdim=True).clamp(min=1e-8)
        P_hat = torch.einsum("bkm,bmc->bkc", A_fg, P_o.float())
        return weighted_kabsch(P_s, P_hat, w) + (A,)


# ── 손실 (03 §9.3) ───────────────────────────────────────────────────────────

def select_g_star(P_s, valid_s, R_gt, t_gt, G, gn, P_o):
    """샘플당 대칭군 g* 단일 선택 — coarse 해상도, 그래디언트 없음.

    기준: 장면점을 모델계로 되돌렸을 때 CAD 196점과의 평균 최근접 거리 최소.
    (포인트별 min 라벨 금지 — g* 하나로 전 라벨 일관)
    """
    with torch.no_grad():
        B, K, _ = P_s.shape
        Gmax = G.shape[1]
        pm = torch.einsum("bji,bkj->bki", R_gt, P_s - t_gt.unsqueeze(1))  # R^T(p−t)
        pg = torch.einsum("bgij,bkj->bgki", G.transpose(-1, -2), pm)      # g^T·모델계
        dist = torch.cdist(pg.reshape(B * Gmax, K, 3),
                           P_o.unsqueeze(1).expand(B, Gmax, -1, 3)
                           .reshape(B * Gmax, -1, 3)).min(-1).values
        dist = dist.view(B, Gmax, K)
        m = (torch.arange(Gmax, device=P_s.device)[None] < gn[:, None])
        score = (dist * valid_s.unsqueeze(1)).sum(-1) / valid_s.sum(-1, keepdim=True)
        score = torch.where(m, score, torch.full_like(score, torch.inf))
        gi = score.argmin(1)
        g = G.gather(1, gi.view(B, 1, 1, 1).expand(B, 1, 3, 3)).squeeze(1)
    return g                                                              # [B,3,3]


def phase_a_loss(sim, P_s, valid_s, P_o, R_gt, t_gt, G, gn, diam,
                 tau_rel: float = 0.10, w_pose: float = 0.5, P_ref=None):
    """대응 CE(+bg) + 포즈 손실. 반환 (loss, 진단 dict).

    P_ref: g* 선택 기준점(모델계). 03 §9.3 정본은 마스터(고정점) — 미지정 시
    P_o(학습 뷰 코어셋) 폴백. 뷰 기반 g*는 뷰 조합 따라 라벨이 요동하는
    스펙 일탈(마스터 대비 일치율 0.73 실측, C-4 → R1에서 마스터로 전환).
    CE 타깃은 그대로 P_o 최근접 — SAM-6D Eq.(13)과 동형 유지.
    """
    B, K, _ = P_s.shape
    g = select_g_star(P_s, valid_s, R_gt, t_gt, G, gn,
                      P_o if P_ref is None else P_ref)
    R_eff = R_gt @ g                                   # 정답 포즈의 g* 표현
    pm = torch.einsum("bji,bkj->bki", R_eff, P_s - t_gt.unsqueeze(1))
    with torch.no_grad():
        dmat = torch.cdist(pm.float(), P_o.float())
        nn_d, nn_i = dmat.min(-1)
        tau = (tau_rel * diam).view(-1, 1)
        target = torch.where(nn_d < tau, nn_i, torch.full_like(nn_i, P_o.shape[1]))
        target = torch.where(valid_s.bool(), target,
                             torch.full_like(target, -100))               # 패딩 무시
    ce = F.cross_entropy(sim.reshape(B * K, -1), target.reshape(B * K),
                         ignore_index=-100)
    R_pr, t_pr, A = MiniMatcher.solve(sim, P_s, P_o)
    rot = torch.deg2rad(geodesic_deg(R_pr, R_eff)).mean()
    trans = ((t_pr - t_gt).norm(dim=-1) / diam.clamp(min=1e-3)).mean()
    loss = ce + w_pose * (rot + 2.0 * trans)
    with torch.no_grad():
        diag = dict(ce=float(ce), rot_deg=float(torch.rad2deg(rot)),
                    trans_rel=float(trans),
                    bg_rate=float(A[..., -1][valid_s.bool()].mean()),
                    g_nonid=float((geodesic_deg(
                        g, torch.eye(3, device=g.device).expand_as(g)) > 1).float().mean()))
    return loss, diag


@torch.no_grad()
def sym_aware_rot_err_deg(R_pr, R_gt, G, gn):
    """평가용: min_g geodesic(R_pr, R_gt·g)."""
    B, Gmax = G.shape[:2]
    errs = geodesic_deg(
        R_pr.unsqueeze(1).expand(B, Gmax, 3, 3).reshape(-1, 3, 3),
        (R_gt.unsqueeze(1) @ G).reshape(-1, 3, 3)).view(B, Gmax)
    m = (torch.arange(Gmax, device=R_pr.device)[None] < gn[:, None])
    return torch.where(m, errs, torch.full_like(errs, torch.inf)).min(1).values
