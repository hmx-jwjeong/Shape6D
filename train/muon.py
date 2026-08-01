"""Muon 옵티마이저 — 2D 은닉 가중치용 (모멘텀 직교화, Newton-Schulz).

알고리즘 재구현 (공개 문헌 기반: K. Jordan 2024 블로그 정식화 + Kimi K2 기술보고서
arXiv:2507.20534). 2025-26 대규모 학습 채택 대세 — AdamW 대비 ~2× 샘플 효율 보고.

규약: 2D로 볼 수 있는 '은닉 행렬'(Linear.weight, conv를 [out, in·k²]로 평탄화)에만
적용. 바이어스·norm·임베딩·bg 토큰 등 1D/소형은 AdamW로 (하이브리드 — 표준 관행).
"""
from __future__ import annotations

import torch


@torch.no_grad()
def _newton_schulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """근사 직교화: G → UV^T (SVD의 부호 성분). bf16 5회 반복 표준 계수."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(torch.bfloat16)
    transposed = X.shape[-2] > X.shape[-1]
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        X = a * X + (b * A + c * (A @ A)) @ X
    if transposed:
        X = X.mT
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, weight_decay: float = 0.0, ns_steps: int = 5):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      weight_decay=weight_decay, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if "mom" not in st:
                    st["mom"] = torch.zeros_like(g)
                buf = st["mom"]
                buf.mul_(group["momentum"]).add_(g)
                u = g.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf
                W = p.view(p.shape[0], -1)                    # conv → [out, in·k²]
                O = _newton_schulz(u.view_as(W), group["ns_steps"])
                # 행렬 크기 보정 (행/열 비율 스케일 — 표준 정식화)
                scale = max(1.0, W.shape[0] / W.shape[1]) ** 0.5
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(O.view_as(p), alpha=-group["lr"] * scale)
        return loss


def split_params(model: torch.nn.Module):
    """(muon 대상: ≥2D 은닉 가중치, adamw 대상: 나머지) 분리."""
    muon_p, adam_p = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (muon_p if p.ndim >= 2 else adam_p).append(p)
    return muon_p, adam_p


class HybridMuon:
    """Muon(행렬) + AdamW(그 외) 하이브리드 — 단일 옵티마이저 인터페이스."""

    def __init__(self, model, lr_adam: float = 3e-4, lr_muon: float = 0.02,
                 weight_decay: float = 0.05, momentum: float = 0.95):
        mp, ap = split_params(model)
        self.muon = Muon(mp, lr=lr_muon, momentum=momentum, weight_decay=weight_decay)
        self.adam = torch.optim.AdamW(ap, lr=lr_adam, weight_decay=weight_decay)
        self.param_groups = self.muon.param_groups + self.adam.param_groups

    def zero_grad(self, set_to_none: bool = True):
        self.muon.zero_grad(set_to_none)
        self.adam.zero_grad(set_to_none)

    def step(self):
        self.muon.step()
        self.adam.step()

    def state_dict(self):
        return {"muon": self.muon.state_dict(), "adam": self.adam.state_dict()}
