"""실표면 free-space 위반율 — 실제 픽셀 레이(uv/K) 기반 (26 로드맵 #2).

구 경량판(multihyp.freespace_viol, 각도 격자+3×3 스플랫)의 기각 원인:
스플랫이 모델의 실제 틈(격자·슬랫)을 메워 정답 포즈가 "관측이 모델을
뚫었다"로 오판됨 (D-9 동근원). 본판은 관측 레이를 실제 픽셀로 지정하고
모델 투영을 무팽창 셀 z-버퍼로 비교 — 모델에 구멍이 있으면 그 레이는
모델과 매칭 자체가 안 되어 위반이 될 수 없다.

위반 정의: 셀(bin_px 픽셀)에서 모델 최전면 z < 관측 최전면 z − margin
(그 레이는 z_obs까지 비어 있었음이 실측인데 가설이 그 앞을 막음).
정규화: 위반 셀 수 / 모델-관측 공존 셀 수 (관측 없는 셀은 증거 없음).

기각 실측 (2026-08-06, phase_b val n=1440, finev3e 정제 후 k=16 선택):
uv/K 정합 서브픽셀 검증 완료 상태에서도 정답/오답 가설 위반 분포가 겹침
(기본: p50 0.088/0.218 — 실루엣·포즈오차 오탐 / 내부+depth+margin0.08D:
p90 0.031/0.051 — 신호 소멸). 모든 λ에서 ICP 잔차 선택보다 후퇴, 비대칭
악화. 원인: 틀린 유역이 대부분 중심 회전 플립이라 점유 부피가 거의 동일 —
기하 전용 free-space는 k-way 선택 판별력이 없음 (선택 트릭 4연속 기각,
docs/27 §5). 잔여 용도: 실로그 S4 단일 포즈 검증(총돌출 거부)은 미실측.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def freespace_viol_real(R_h, t_h, uv, K, pts, val, master, diam,
                        bin_px: int = 4, margin_rel: float = 0.03,
                        margin_abs: float = 0.01, min_match: int = 20,
                        interior: bool = True, depth_score: bool = False):
    """가설별 위반율 [B,k].

    R_h [B,k,3,3] · t_h [B,k,3] · uv [B,N,2](int, 원영상 px) · K [B,4](fx,fy,cx,cy)
    pts [B,N,3](카메라계) · val [B,N] · master [B,M,3](centered 모델) · diam [B]
    interior: 모델 실루엣 내부 셀만 채점 (경계 셀은 배경 레이와 겹쳐 본질적 오탐)
    depth_score: 위반율 대신 평균 침범 깊이 (z_obs−z_m−margin)+/D — 총돌출 가중
    """
    B, k = R_h.shape[:2]
    dev = R_h.device
    Mpts = master.float()
    n_m = Mpts.shape[1]
    viol = torch.zeros(B, k, device=dev)
    uv = uv.long()
    W = 1 << 15                                       # 셀 해시 폭 (uv int16 범위)
    NBR = torch.tensor([1, -1, W, -W, W + 1, W - 1, -W + 1, -W - 1], device=dev)
    for b in range(B):
        v_b = val[b].bool()
        if v_b.sum() < min_match:
            continue
        zo = pts[b, v_b, 2].float()
        co = (uv[b, v_b, 0] // bin_px) * W + (uv[b, v_b, 1] // bin_px)
        uc, inv = torch.unique(co, return_inverse=True)
        zmin_o = torch.full((len(uc),), torch.inf, device=dev)
        zmin_o.scatter_reduce_(0, inv, zo, reduce="amin")
        fx, fy, cx, cy = K[b]
        margin = max(margin_abs, margin_rel * float(diam[b]))
        Mc = torch.einsum("kij,mj->kmi", R_h[b].float(), Mpts[b]) \
            + t_h[b].float().unsqueeze(1)                       # [k,M,3]
        zm = Mc[..., 2].clamp(min=1e-6)
        um = (fx * Mc[..., 0] / zm + cx).long() // bin_px
        vm = (fy * Mc[..., 1] / zm + cy).long() // bin_px
        cm = (um * W + vm).view(k, n_m)
        pos = torch.searchsorted(uc, cm.reshape(-1).contiguous()).clamp(max=len(uc) - 1)
        match = (uc[pos] == cm.reshape(-1)).view(k, n_m)
        # 셀별 모델 최전면: 매칭 실패 점은 inf로 밀어 배제
        pos = pos.view(k, n_m)
        zm_eff = torch.where(match, zm, torch.full_like(zm, torch.inf))
        zmin_m = torch.full((k, len(uc)), torch.inf, device=dev)
        zmin_m.scatter_reduce_(1, pos, zm_eff, reduce="amin")
        present = torch.isfinite(zmin_m)                        # 모델-관측 공존 셀
        if interior:
            # uc 셀이 모델 투영 셀 집합(가설별)의 8이웃을 모두 가져야 내부
            mcell = torch.sort(cm, dim=-1).values               # [k,M]
            inn = present.clone()
            for d in NBR:
                q = torch.searchsorted(mcell, (uc + d).unsqueeze(0).expand(k, -1)
                                       .contiguous()).clamp(max=n_m - 1)
                inn &= mcell.gather(1, q) == (uc + d).unsqueeze(0)
            present = inn
        depth = (zmin_o.unsqueeze(0) - zmin_m - margin).clamp(min=0)
        depth = torch.where(present, depth, torch.zeros_like(depth))
        n_pres = present.sum(-1).clamp(min=1)
        if depth_score:
            r = depth.sum(-1) / (n_pres.float() * float(diam[b]))
        else:
            r = (depth > 0).sum(-1).float() / n_pres.float()
        r = torch.where(present.sum(-1) >= min_match, r, torch.zeros_like(r))
        viol[b] = r
    return viol
