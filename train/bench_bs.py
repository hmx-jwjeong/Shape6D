"""배치 크기 스윕 — step 시간·처리량·피크 메모리 (실데이터, fwd+bwd 전체 경로)."""
import sys, time, torch, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
import train.train_phase_a as T
from train.encoders import build_encoder
from train.pem_mini import MiniMatcher, phase_a_loss

def bench(bs, enc_kind="a1", steps=12):
    torch.manual_seed(0)
    bank = T.ObjBank()
    tr = {k: torch.from_numpy(v) for k, v in np.load(T.DATA / "phase_a_train.npz").items()}
    enc = build_encoder(enc_kind).to(T.DEV)
    matcher = MiniMatcher().to(T.DEV)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(matcher.parameters()), lr=1e-4)
    g = torch.Generator(device=T.DEV); g.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    ts = []
    for s in range(steps):
        i = torch.randperm(len(tr["oi"]))[:bs]
        pts = tr["pts"][i].to(T.DEV).float(); npt = tr["npt"][i].to(T.DEV)
        valid = (torch.arange(pts.shape[1], device=T.DEV)[None] < npt[:, None])
        oi = tr["oi"][i].to(T.DEV)
        R_gt = tr["R"][i].to(T.DEV).view(-1, 3, 3)
        t_eff = tr["t"][i].to(T.DEV) + torch.einsum("bij,bj->bi", R_gt, bank.c[oi])
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            F_s, P_s, val_s = T.encode_scene(enc, pts, valid)
            P_o, F_o = T.encode_cad(enc, bank, oi, g)
            sim = matcher(F_s.float(), P_s.float(), F_o.float(), P_o.float(), bank.diam[oi])
        loss, _ = phase_a_loss(sim.float(), P_s.float(), val_s, P_o.float(),
                               R_gt, t_eff, bank.G[oi], bank.gn[oi], bank.diam[oi])
        opt.zero_grad(); loss.backward(); opt.step()
        torch.cuda.synchronize()
        if s >= 2: ts.append(time.perf_counter() - t0)
    dt = float(np.median(ts))
    mem = torch.cuda.max_memory_allocated() / 1e9
    del enc, matcher, opt; torch.cuda.empty_cache()
    return dt, bs / dt, mem

if __name__ == "__main__":
    print(f"{chr(39)}bs{chr(39):>4} {chr(39)}ms/step{chr(39):>9} {chr(39)}samples/s{chr(39):>10} {chr(39)}peak GB{chr(39):>8}")
    rows = []
    for bs in (48, 96, 192, 320, 512):
        try:
            dt, th, mem = bench(bs)
            rows.append((bs, dt * 1e3, th, mem))
            print(f"{bs:4d} {dt*1e3:9.0f} {th:10.1f} {mem:8.1f}", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"{bs:4d}  OOM", flush=True); break
    import json; json.dump(rows, open(Path(__file__).parent / "bench_bs.json", "w"))
