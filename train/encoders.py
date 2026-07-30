"""Phase A 기하 인코더 후보 (20 §2 확정표).

[클린룸 규약 — 19 §3.2 / 커밋 규약 조치 2]
본 파일과 pem_mini.py는 SAM-6D 소스를 열람하지 않고 작성됐다. 참조는
CVPR2024 논문(공개 문헌)과 자체 설계 문서(docs/03 §6)뿐이다.

라이선스 (19 §3.1): a1 초기화는 timm 자체 학습 체크포인트
`convnext_tiny.in12k_ft_in1k` (Apache-2.0) 태그를 **명시 고정**한다.
FB 배포 가중치(`convnext_tiny.fb_in1k` 등 `fb_*` 태그)는 CC-BY-NC — 사용 금지.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# 19 §3.1 — Apache-2.0 확인된 태그. fb_* 금지.
TIMM_A1_TAG = "convnext_tiny.in12k_ft_in1k"


class PartialConv2d(nn.Conv2d):
    """유효마스크 정규화 conv: y = conv(x⊙m)·k²/(conv_ones(m)+ε) (03 §6.2)."""

    def forward(self, x, mask):  # type: ignore[override]
        with torch.no_grad():
            ones = torch.ones(1, 1, *self.kernel_size, device=x.device, dtype=x.dtype)
            cnt = nn.functional.conv2d(mask, ones, stride=self.stride, padding=self.padding)
            k = self.kernel_size[0] * self.kernel_size[1]
            scale = k / (cnt + 1e-6)
            new_mask = (cnt > 0).to(x.dtype)
        y = super().forward(x * mask)
        if self.bias is not None:
            b = self.bias.view(1, -1, 1, 1)
            y = (y - b) * scale + b
        else:
            y = y * scale
        return y * new_mask, new_mask


class PartialStem(nn.Module):
    """중첩 3×3 s2 partial conv ×2 → stride 4 (13 §3: 비중첩 patchify는 희소에서 붕괴)."""

    def __init__(self, cin: int, cout: int, mid: int = 24):
        super().__init__()
        self.c1 = PartialConv2d(cin, mid, 3, 2, 1)
        self.c2 = PartialConv2d(mid, cout, 3, 2, 1)
        self.n1 = nn.GroupNorm(1, mid)
        self.n2 = nn.GroupNorm(1, cout)
        self.act = nn.GELU()

    def forward(self, x, mask):
        x, mask = self.c1(x, mask)
        x = self.act(self.n1(x))
        x, mask = self.c2(x, mask)
        x = self.act(self.n2(x))
        return x, mask


class LightFPN(nn.Module):
    """1×1 lateral + upsample-add + depthwise 3×3 (13 §2: full 3×3은 예산 2배 초과)."""

    def __init__(self, chs: list[int], out: int = 256):
        super().__init__()
        self.lat = nn.ModuleList([nn.Conv2d(c, out, 1) for c in chs])
        self.smooth = nn.Conv2d(out, out, 3, 1, 1, groups=out)

    def forward(self, feats):
        p = self.lat[-1](feats[-1])
        for i in range(len(feats) - 2, -1, -1):
            p = self.lat[i](feats[i]) + nn.functional.interpolate(
                p, size=feats[i].shape[-2:], mode="nearest")
        return self.smooth(p)


class _BasicBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, 1, 1, bias=False)
        self.n1 = nn.BatchNorm2d(c)
        self.c2 = nn.Conv2d(c, c, 3, 1, 1, bias=False)
        self.n2 = nn.BatchNorm2d(c)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.act(self.n1(self.c1(x)))
        return self.act(x + self.n2(self.c2(y)))


class ResNet18LiteEncoder(nn.Module):
    """후보 a0 — 03 §6.2 (a) 재구성 (w=80: 절단 ConvNeXt와 용량 정합, 13 §2)."""

    def __init__(self, cin: int = 10, w: int = 80, cout: int = 256):
        super().__init__()
        self.stem = PartialStem(cin, w)
        self.layer1 = nn.Sequential(_BasicBlock(w), _BasicBlock(w))
        self.down = nn.Sequential(nn.Conv2d(w, 2 * w, 3, 2, 1, bias=False),
                                  nn.BatchNorm2d(2 * w), nn.ReLU(True))
        self.layer2 = nn.Sequential(_BasicBlock(2 * w), _BasicBlock(2 * w))
        self.fpn = LightFPN([w, 2 * w], cout)

    def forward(self, x, mask=None):
        if mask is None:
            mask = (x[:, 6:7] > 0.5).to(x.dtype)   # geo_maps ch6 = 유효마스크
        x, _ = self.stem(x, mask)
        f1 = self.layer1(x)
        f2 = self.layer2(self.down(f1))
        return self.fpn([f1, f2])                   # (B,256,56,56)


class TruncConvNeXtEncoder(nn.Module):
    """후보 a1 — 절단 ConvNeXt-T stem+stage1~2 (12 §7.1 / 13 §2 실측 1.33M/1.73G).

    pretrained=True 시 timm Apache 태그 고정 로드. stem은 partial conv로 교체
    (사전학습 stem 5k 파라미터는 폐기 — 유지분의 0.4%).
    """

    def __init__(self, cin: int = 10, cout: int = 256, pretrained: bool = False):
        super().__init__()
        import timm
        name = TIMM_A1_TAG if pretrained else "convnext_tiny"
        self.backbone = timm.create_model(name, pretrained=pretrained,
                                          features_only=True, out_indices=(0, 1))
        if pretrained:
            got = getattr(self.backbone, "pretrained_cfg", {}) or {}
            tag = str(got.get("tag", ""))
            assert not tag.startswith("fb_"), f"금지 태그 로드됨: {tag} (CC-BY-NC)"
        self.backbone.stem_0 = nn.Identity()
        self.backbone.stem_1 = nn.Identity()
        self.stem = PartialStem(cin, 96, mid=24)    # 예산: mid24 (13 §2)
        self.fpn = LightFPN([96, 192], cout)

    def forward(self, x, mask=None):
        if mask is None:
            mask = (x[:, 6:7] > 0.5).to(x.dtype)
        x, _ = self.stem(x, mask)
        feats = self.backbone(x)
        return self.fpn(feats)                      # (B,256,56,56)


def build_encoder(kind: str) -> nn.Module:
    """kind ∈ {a0, a1, a2}. a2 = a1 구조·스크래치 (사전학습 순수 기여 분리용)."""
    if kind == "a0":
        return ResNet18LiteEncoder()
    if kind == "a1":
        return TruncConvNeXtEncoder(pretrained=True)
    if kind == "a2":
        return TruncConvNeXtEncoder(pretrained=False)
    raise ValueError(f"미지원 인코더: {kind}")
