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
    FineMatcher, MiniMatcher, fps_torch, make_geo_maps, perturb_pose,
    phase_a_loss, sample_point_features, select_g_star, sym_aware_rot_err_deg,
)

DATA = Path("/mnt/samsung2tb/datasets/megapose/phase_a")
DEV = "cuda"
N_TOK = 196
VIEW_PX = 1536        # CAD 뷰당 특징 샘플 픽셀 수 (2뷰 합 3072 → FPS 196)
LR = {"a0": 1e-3, "a1": 3e-4, "a2": 1e-3}   # C-7 후보별 lr 정책


class ObjBank:
    """오브젝트 팩 전량 GPU 상주 (~168종, 수백 MB)."""

    def __init__(self, prefix: str = "phase_a"):
        z = np.load(DATA / f"{prefix}_objs.npz")
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


def encode_scene(enc, pts, valid, fine_tok: int | None = None):
    geo, uvn = make_geo_maps(pts, valid, domain_flag=1.0)
    f56 = enc(geo)

    def take(n):
        sel = fps_torch(pts, valid, n)
        F_ = sample_point_features(f56, uvn.gather(1, sel.unsqueeze(-1).expand(-1, -1, 2)))
        P_ = pts.gather(1, sel.unsqueeze(-1).expand(-1, -1, 3))
        return F_, P_, valid.gather(1, sel).float()

    F_s, P_s, val_s = take(N_TOK)
    if fine_tok is None:
        return F_s, P_s, val_s
    return F_s, P_s, val_s, take(fine_tok)


def encode_cad(enc, bank: ObjBank, oi, rng, n_views: int = 2, fine_tok: int | None = None):
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
    def take(n):
        sel = fps_torch(Pm, okm, n)
        return (Pm.gather(1, sel.unsqueeze(-1).expand(-1, -1, 3)),
                Fm.gather(1, sel.unsqueeze(-1).expand(-1, -1, Fm.shape[-1])))

    P_o, F_o = take(N_TOK)
    if fine_tok is None:
        return P_o, F_o
    return P_o, F_o, take(fine_tok)


@torch.no_grad()
def evaluate(enc, matcher, bank, va, bs=96, cap=4000, fine=None, fine_tok=1024,
             fine2=None):
    enc.eval(); matcher.eval()
    if fine is not None:
        fine.eval()
    if fine2 is not None:
        fine2.eval()
    n = min(len(va["oi"]), cap)
    rot, tr, thirty, rot_f, five_f = [], [], [], [], []
    rot_f2, five_f2, one_f2 = [], [], []
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
            if fine is None:
                F_s, P_s, val_s = encode_scene(enc, pts, valid)
                P_o, F_o = encode_cad(enc, bank, oi, g, n_views=6)
            else:
                F_s, P_s, val_s, (F_sf, P_sf, val_sf) = encode_scene(
                    enc, pts, valid, fine_tok=fine_tok)
                P_o, F_o, (P_of, F_of) = encode_cad(
                    enc, bank, oi, g, n_views=6, fine_tok=fine_tok)
            sim = matcher(F_s.float(), P_s.float(), F_o.float(), P_o.float(),
                          bank.diam[oi])
        R_pr, t_pr, _ = MiniMatcher.solve(sim.float(), P_s.float(), P_o.float())
        e = sym_aware_rot_err_deg(R_pr, R_gt, bank.G[oi], bank.gn[oi])
        rot.append(e)
        tr.append((t_pr - t_eff).norm(dim=-1) / bank.diam[oi])
        thirty.append((e <= 30).float())
        if fine is not None:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                sim_f = fine(F_sf.float(), P_sf.float(), F_of.float(),
                             P_of.float(), R_pr, t_pr, bank.diam[oi])
            R2, t2, _ = MiniMatcher.solve(sim_f.float(), P_sf.float(), P_of.float(),
                                          hard=True)
            ef = sym_aware_rot_err_deg(R2, R_gt, bank.G[oi], bank.gn[oi])
            rot_f.append(ef); five_f.append((ef <= 5).float())
            if fine2 is not None:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    sim_f2 = fine2(F_sf.float(), P_sf.float(), F_of.float(),
                                   P_of.float(), R2, t2, bank.diam[oi])
                R3, t3, _ = MiniMatcher.solve(sim_f2.float(), P_sf.float(),
                                              P_of.float(), hard=True)
                ef2 = sym_aware_rot_err_deg(R3, R_gt, bank.G[oi], bank.gn[oi])
                rot_f2.append(ef2)
                five_f2.append((ef2 <= 5).float())
                one_f2.append((ef2 <= 1).float())
    rot = torch.cat(rot); tr = torch.cat(tr); th = torch.cat(thirty)
    enc.train(); matcher.train()
    if fine is not None:
        fine.train()
    if fine2 is not None:
        fine2.train()
    m = dict(rot_p50=float(rot.median()), rot_mean=float(rot.mean()),
             le30=float(th.mean()), trans_rel_p50=float(tr.median()), n=int(n))
    if rot_f:
        rf = torch.cat(rot_f)
        m.update(rot_p50_fine=float(rf.median()),
                 le30_fine=float((rf <= 30).float().mean()),
                 le5_fine=float(torch.cat(five_f).mean()))
    if rot_f2:
        rf2 = torch.cat(rot_f2)
        m.update(rot_p50_fine2=float(rf2.median()),
                 le5_fine2=float(torch.cat(five_f2).mean()),
                 le1_fine2=float(torch.cat(one_f2).mean()))
    return m


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
    ap.add_argument("--data-prefix", default="phase_a",
                    help="데이터 파일 접두 (phase_a2 = v2 보정 재구축본)")
    ap.add_argument("--data-chunks", default=None,
                    help="콤마 구분 train 청크 접두 목록 — 에폭마다 교대 로딩(RAM 상수). "
                         "val은 항상 --data-prefix 것 사용. 예: phase_b,phase_b_c01,…")
    ap.add_argument("--gstar", choices=["view", "master"], default="view",
                    help="g* 기준점 — R1 실측: master 18.8%% vs view 27.0%% (기각), 기본 view")
    ap.add_argument("--views", type=int, default=2, help="학습 CAD 뷰 수 (R2: 6)")
    ap.add_argument("--compile", action="store_true",
                    help="인코더 torch.compile+channels_last (실측 1.58× — 수치 비트 비교 불가 주의)")
    ap.add_argument("--fine", action="store_true", help="fine 매칭 단계(1024pt) 결합 학습")
    ap.add_argument("--fine-tok", type=int, default=1024)
    ap.add_argument("--fine2", action="store_true",
                    help="fine2 캐스케이드(τ0.02D·U(0–15°)) — --fine 필요, 토큰 공유")
    ap.add_argument("--suffix", default="", help="태그 접미사 (예: gm — 코드 변형 런 구분)")
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
    if a.suffix:
        parts.append(a.suffix)
    tag = "_".join(parts) + f"_s{a.seed}"

    bank = ObjBank(a.data_prefix)
    ld = lambda sp, pfx=None: {k: torch.from_numpy(v) for k, v in
                               np.load(DATA / f"{pfx or a.data_prefix}_{sp}.npz").items()}
    chunks = a.data_chunks.split(",") if a.data_chunks else None
    cur_chunk = chunks[0] if chunks else a.data_prefix
    tr, va = ld("train", cur_chunk), ld("val")
    enc = build_encoder(a.enc).to(DEV)
    if a.compile:
        enc = enc.to(memory_format=torch.channels_last)
        enc = torch.compile(enc)
        print(f"[compile] 인코더 컴파일 활성 (channels_last)", flush=True)
    matcher = MiniMatcher().to(DEV)
    if a.fine2 and not a.fine:
        ap.error("--fine2는 --fine 필요")
    fine = FineMatcher().to(DEV) if a.fine else None
    fine2 = FineMatcher().to(DEV) if a.fine2 else None
    enc_raw = getattr(enc, "_orig_mod", enc)      # compile 래핑 시 원본
    if a.init_from:
        sd = torch.load(a.init_from, map_location=DEV)
        enc_raw.load_state_dict(sd["enc"]); matcher.load_state_dict(sd["matcher"])
        if fine is not None and "fine" in sd:
            try:
                fine.load_state_dict(sd["fine"])
            except RuntimeError:
                print(f"[{tag}] fine 헤드 구조 변경 — 재초기화, enc/matcher만 승계",
                      flush=True)
        if fine2 is not None and "fine2" in sd:
            try:
                fine2.load_state_dict(sd["fine2"])
            except RuntimeError:
                print(f"[{tag}] fine2 헤드 구조 변경 — 재초기화", flush=True)
        print(f"[{tag}] init from {a.init_from}", flush=True)
    npar = sum(p.numel() for p in enc_raw.parameters()) / 1e6
    npm = sum(p.numel() for p in matcher.parameters()) / 1e6
    print(f"[{tag}] enc {npar:.2f}M + matcher {npm:.2f}M · train {len(tr['oi'])} · "
          f"val(미학습 34종) {len(va['oi'])} · {a.opt}/{a.sched} lr {lr:g} bs {a.bs}", flush=True)
    # 대시보드용 설정 덤프 — 학습에 실제 사용되는 파라미터 전량
    json.dump({
        "encoder": a.enc, "epochs": a.epochs, "batch_size": a.bs, "seed": a.seed,
        "data_prefix": a.data_prefix, "data_chunks": a.data_chunks,
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
        "fine": (f"v3 2blk linear-attn {a.fine_tok}pt τ0.05D · PE8밴드 · "
                 f"좌표게이트(λ학습,init30) · g*정합 초기=R_gt·g*+U(0–30°/0–0.10D) · "
                 f"eval hard-solve"
                 if a.fine else None),
        "fine2": ("캐스케이드 2단 τ0.02D · g*정합 초기=U(0–15°/0–0.05D) · "
                  "토큰 공유 · eval 체인 coarse→fine→fine2"
                  if a.fine2 else None),
        "init_from": a.init_from,
    }, open(out / f"cfg_{tag}.json", "w"), indent=1, ensure_ascii=False)

    import torch.nn as nn
    net = nn.ModuleDict({"enc": enc, "matcher": matcher})
    if fine is not None:
        net["fine"] = fine
    if fine2 is not None:
        net["fine2"] = fine2
    steps = len(tr["oi"]) // a.bs
    if chunks:
        # 청크 교대 시 스케줄 총량 = 청크별 실제 스텝 합 (dfull 1차 사고:
        # 첫 청크 기준 총량 → 실행 42%에서 감쇠 미완 종료, 폴리시 소실)
        _sz = [int(np.load(DATA / f"{c}_train.npz")["oi"].shape[0]) for c in chunks]
        total_steps = sum(_sz[e % len(chunks)] // a.bs for e in range(a.epochs))
        print(f"[{tag}] 청크 {len(chunks)}개 · 스케줄 총 {total_steps} 스텝 "
              f"(고유 {sum(_sz)})", flush=True)
    else:
        total_steps = a.epochs * steps
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
                opt, lr, total_steps=total_steps, pct_start=0.15)
        else:                                          # cosine + 3% warmup
            warm = max(1, int(0.03 * total_steps))
            sched = torch.optim.lr_scheduler.SequentialLR(
                opt, [torch.optim.lr_scheduler.LinearLR(opt, 0.05, 1.0, warm),
                      torch.optim.lr_scheduler.CosineAnnealingLR(
                          opt, total_steps - warm, eta_min=lr * 0.05)], [warm])
    g = torch.Generator(device=DEV); g.manual_seed(a.seed)

    hist = []
    m0 = evaluate(enc, matcher, bank, va, fine=fine, fine_tok=a.fine_tok,
                  fine2=fine2)
    print(f"[{tag}] ep0(초기) rot_p50={m0['rot_p50']:.1f}° ≤30°={m0['le30']:.3f}",
          flush=True)
    hist.append(dict(ep=0, **m0))
    t0 = time.time()
    for ep in range(a.epochs):
        if chunks and chunks[ep % len(chunks)] != cur_chunk:
            cur_chunk = chunks[ep % len(chunks)]
            tr = ld("train", cur_chunk)
            print(f"[{tag}] 청크 교대 → {cur_chunk} ({len(tr['oi'])})", flush=True)
        ep_steps = len(tr["oi"]) // a.bs
        perm = torch.randperm(len(tr["oi"]))
        agg = {}
        for s in range(ep_steps):
            i = perm[s * a.bs:(s + 1) * a.bs]
            pts = tr["pts"][i].to(DEV, non_blocking=True).float()
            npt = tr["npt"][i].to(DEV)
            valid = (torch.arange(pts.shape[1], device=DEV)[None] < npt[:, None])
            oi = tr["oi"][i].to(DEV)
            R_gt = tr["R"][i].to(DEV).view(-1, 3, 3)
            t_eff = tr["t"][i].to(DEV) + torch.einsum("bij,bj->bi", R_gt, bank.c[oi])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if fine is None:
                    F_s, P_s, val_s = encode_scene(enc, pts, valid)
                    P_o, F_o = encode_cad(enc, bank, oi, g, n_views=a.views)
                else:
                    F_s, P_s, val_s, (F_sf, P_sf, val_sf) = encode_scene(
                        enc, pts, valid, fine_tok=a.fine_tok)
                    P_o, F_o, (P_of, F_of) = encode_cad(
                        enc, bank, oi, g, n_views=a.views, fine_tok=a.fine_tok)
                sim = matcher(F_s.float(), P_s.float(), F_o.float(), P_o.float(),
                              bank.diam[oi])
                if fine is not None:
                    # g*-정합 조건화: CE 라벨(R_eff=R_gt·g*)과 좌표 프라이어 일치.
                    # v1/v2는 R_gt 기준 — g*≠I 85%에서 좌표가 라벨과 대칭회전만큼
                    # 어긋나 헤드가 좌표를 무시하도록 학습됨 (finev2 진단).
                    gs_f = select_g_star(P_sf.float(), val_sf.float(), R_gt, t_eff,
                                         bank.G[oi], bank.gn[oi], P_of.float())
                    R0, t0_ = perturb_pose(R_gt @ gs_f, t_eff, bank.diam[oi],
                                           rot_deg=(0.0, 30.0), trans_rel=(0.0, 0.10))
                    sim_f = fine(F_sf.float(), P_sf.float(), F_of.float(),
                                 P_of.float(), R0, t0_, bank.diam[oi])
                    if fine2 is not None:
                        R02, t02 = perturb_pose(R_gt @ gs_f, t_eff, bank.diam[oi],
                                                rot_deg=(0.0, 15.0),
                                                trans_rel=(0.0, 0.05))
                        sim_f2 = fine2(F_sf.float(), P_sf.float(), F_of.float(),
                                       P_of.float(), R02, t02, bank.diam[oi])
            loss, diag = phase_a_loss(sim.float(), P_s.float(), val_s, P_o.float(),
                                      R_gt, t_eff, bank.G[oi], bank.gn[oi],
                                      bank.diam[oi],
                                      P_ref=(bank.master[oi, :512]
                                             if a.gstar == "master" else None))
            if fine is not None:
                loss_f, diag_f = phase_a_loss(
                    sim_f.float(), P_sf.float(), val_sf, P_of.float(),
                    R_gt, t_eff, bank.G[oi], bank.gn[oi], bank.diam[oi],
                    tau_rel=0.05, w_pose=0.0)
                loss = loss + loss_f
                diag["ce_f"] = diag_f["ce"]
                diag["rot_f"] = diag_f["rot_deg"]
                if fine2 is not None:
                    loss_f2, diag_f2 = phase_a_loss(
                        sim_f2.float(), P_sf.float(), val_sf, P_of.float(),
                        R_gt, t_eff, bank.G[oi], bank.gn[oi], bank.diam[oi],
                        tau_rel=0.02, w_pose=0.0)
                    loss = loss + loss_f2
                    diag["ce_f2"] = diag_f2["ce"]
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
        m = evaluate(enc, matcher, bank, va, fine=fine, fine_tok=a.fine_tok,
                     fine2=fine2)
        hist.append(dict(ep=ep + 1, min=round((time.time() - t0) / 60, 1), **m,
                         **{k: v / ep_steps for k, v in agg.items()}))
        ok = max(1, ep_steps - int(agg.get("skip", 0)))
        print(f"[{tag}] ep{ep+1}/{a.epochs} 미학습 rot_p50={m['rot_p50']:.1f}° "
              f"≤30°={m['le30']:.3f} trans={m['trans_rel_p50']:.3f}D | "
              f"train ce={agg.get('ce', float('nan'))/ok:.2f} "
              f"rot={agg.get('rot_deg', float('nan'))/ok:.0f}° "
              f"bg={agg.get('bg_rate', float('nan'))/ok:.2f} "
              f"g*≠I={agg.get('g_nonid', float('nan'))/ok:.2f} "
              f"skip={int(agg.get('skip', 0))} "
              + (f"| fine p50={m['rot_p50_fine']:.1f}° ≤5°={m['le5_fine']:.3f} "
                 f"ce_f={agg.get('ce_f', float('nan'))/ok:.2f} "
                 if 'rot_p50_fine' in m else "")
              + (f"| f2 p50={m['rot_p50_fine2']:.1f}° ≤5°={m['le5_fine2']:.3f} "
                 f"≤1°={m['le1_fine2']:.3f} "
                 f"ce_f2={agg.get('ce_f2', float('nan'))/ok:.2f} "
                 if 'rot_p50_fine2' in m else "")
              + f"({(time.time()-t0)/60:.0f}m)", flush=True)
        json.dump(hist, open(out / f"hist_{tag}.json", "w"), indent=1)
        ck = dict(enc=enc_raw.state_dict(), matcher=matcher.state_dict())
        if fine is not None:
            ck["fine"] = fine.state_dict()
        if fine2 is not None:
            ck["fine2"] = fine2.state_dict()
        torch.save(ck, out / f"ckpt_{tag}.pt")
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
