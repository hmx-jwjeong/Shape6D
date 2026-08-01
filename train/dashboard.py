"""Phase A 학습 모니터 대시보드 — stdlib 전용, 실행 중 프로세스 무간섭.

데이터 소스 (읽기 전용):
  - runs/hist_{tag}.json : 에폭별 지표 (train_phase_a.py가 에폭마다 갱신)
  - --log 로그 파일      : "[tag] epN/M ... (Xm)" 라인 → 총 에폭·경과분·ETA
  - /proc 스캔           : train_phase_a.py 프로세스 존재 → 실행 중 표시

실행:  python3 train/dashboard.py            # http://127.0.0.1:8035
       python3 train/dashboard.py --port 8035 --log <학습로그경로>
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

RUNS = Path("/mnt/samsung2tb/datasets/megapose/phase_a/runs")
LOG_RE = re.compile(r"\[(\w+)\] ep(\d+)/(\d+) .*\((\d+)m\)")
HDR_RE = re.compile(r"\[(\w+)\] enc ([\d.]+)M \+ matcher ([\d.]+)M · train (\d+) · "
                    r"val\(미학습 (\d+)종\) (\d+) · lr ([\deE.-]+)")

# dataviz 검증 팔레트 (CVD ΔE 통과) — 후보 태그별 고정 배정
COLORS = {"a0": "#2a78d6", "a1": "#eb6834", "a1d": "#1baf7a", "a2": "#eda100"}


def _shade(hex_color: str, t: float) -> str:
    """base 색을 흰색 쪽으로 t(0~1)만큼 블렌드 — 같은 그룹 내 TRY 시간순 그라데이션."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    mix = lambda c: round(c + (255 - c) * t)
    return f"#{mix(r):02x}{mix(g):02x}{mix(b):02x}"

# 구 런(cfg 덤프 이전)용 정적 상수 — train_phase_a.py/pem_mini.py 현행 값과 동기
STATIC_CFG = {
    "batch_size": 48, "optimizer": "AdamW(wd=0.05)",
    "scheduler": "OneCycleLR(pct_start=0.15)", "grad_clip": 5.0,
    "precision": "bf16 autocast + fp32(softmax/CE/SVD)",
    "matcher": "cross-attn 2blk H=192 heads=4 + 거리RPE(16bin) + bg",
    "sim": "cosine / temp 0.1 · sim_dim 256",
    "loss": "g* 단일선택 CE(τ=0.10D) + 0.5·(rot_rad + 2·trans/D)",
    "n_tok": 196, "cad_views_train": 2, "cad_views_eval": 6, "view_px": 1536,
    "sparsify": "ML-X 격자 σ3mm · npt U[256,4096] · frame_correction 적용",
}


def _running() -> bool:
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().decode(errors="ignore")
        except OSError:
            continue
        if "train_phase_a.py" in cmd:
            return True
    return False


def collect(log_path: Path | None) -> dict:
    runs = {}
    for f in sorted(RUNS.glob("hist_*.json")):
        tag = f.stem[5:]
        try:
            hist = json.load(open(f))
        except json.JSONDecodeError:
            continue
        runs[tag] = {"hist": hist, "mtime": f.stat().st_mtime,
                     "total_ep": None, "elapsed_m": None}
    hdrs = {}
    if log_path and log_path.exists():
        txt = log_path.read_text(errors="ignore")
        for m in LOG_RE.finditer(txt):
            tag, ep, tot, mins = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
            if tag in runs:
                runs[tag]["total_ep"] = tot
                runs[tag]["elapsed_m"] = mins
        for m in HDR_RE.finditer(txt):
            hdrs[m.group(1)] = {"enc_params_M": float(m.group(2)),
                                "matcher_params_M": float(m.group(3)),
                                "data_train": int(m.group(4)),
                                "obj_val_unseen": int(m.group(5)),
                                "data_val": int(m.group(6)), "lr": float(m.group(7))}
    now = time.time()
    alive = _running()
    out = []
    for tag, r in runs.items():
        h = r["hist"]
        last = h[-1] if h else {}
        ep = last.get("ep", 0)
        tot = r["total_ep"]
        # ETA: ① hist 자체 타이밍(min) + cfg.epochs (신규 런) ② 로그 파싱 (구 런)
        cfgf = RUNS / f"cfg_{tag}.json"
        cfg0 = json.load(open(cfgf)) if cfgf.exists() else {}
        tot = tot or cfg0.get("epochs")
        el = r["elapsed_m"]
        if last.get("min"):
            el = last["min"]
        eta = None
        if tot and ep and el and ep < tot:
            eta = round(el / ep * (tot - ep))
        r["elapsed_m"] = el
        live = alive and (now - r["mtime"] < 300)
        # 설정: cfg_{tag}.json(신규 런) > 로그 헤더 파싱(구 런) > 정적 상수
        cfg = dict(cfg0)
        for k, v in (hdrs.get(tag) or {}).items():
            cfg.setdefault(k, v)
        for k, v in STATIC_CFG.items():
            cfg.setdefault(k, v)
        cfg.setdefault("encoder", tag.split("_")[0])
        # 수렴 지표: 최근 5에폭 기울기 (에폭당 변화량)
        def slope(key, n=5):
            xs = [e for e in h if e.get(key) is not None][-n:]
            if len(xs) < 2:
                return None
            return round((xs[-1][key] - xs[0][key]) / (xs[-1]["ep"] - xs[0]["ep"]), 4)
        conv = {"ce": slope("ce"), "rot_p50": slope("rot_p50"), "le30": slope("le30")}
        # 시작 시각: cfg 파일 생성 시각(런 시작 시 덤프) > hist 최종 갱신 시각
        start_ts = cfgf.stat().st_mtime if cfgf.exists() else r["mtime"]
        out.append({"tag": tag, "hist": h, "ep": ep, "total_ep": tot,
                    "elapsed_m": r["elapsed_m"], "eta_m": eta, "live": live,
                    "cfg": cfg, "conv": conv, "start_ts": start_ts})
    # 색상: 그룹(a0/a1/…) 기본색은 유지, 같은 그룹 내 TRY는 시간순 그라데이션
    # (오래된 런일수록 연하게, 최신 런이 원색)
    groups: dict[str, list] = {}
    for r in out:
        groups.setdefault(r["tag"].split("_")[0], []).append(r)
    for base_tag, members in groups.items():
        base = COLORS.get(base_tag, "#e87ba4")
        members.sort(key=lambda x: x["start_ts"])
        n = len(members)
        for i, m in enumerate(members):
            t = 0.55 * (n - 1 - i) / (n - 1) if n > 1 else 0.0
            m["color"] = _shade(base, t)
    out.sort(key=lambda x: -x["start_ts"])  # 최신 런이 위
    return {"runs": out, "proc_alive": alive, "ts": now}


PAGE = """<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8">
<title>Phase A 학습 모니터</title>
<style>
:root{color-scheme:light dark;
  --bg:#fff;--ink:#1a1d23;--sub:#5a6270;--line:#dde2ea;--panel:#f6f8fb;
  --good:#0a7d43;--good-soft:#e6f5ec;--bad:#b4232a;}
@media (prefers-color-scheme:dark){:root{
  --bg:#14161b;--ink:#e8eaf0;--sub:#9aa3b2;--line:#2c3240;--panel:#1c2028;
  --good:#4cc38a;--good-soft:#15291f;--bad:#ef6b74;}}
*{box-sizing:border-box}
body{margin:0;font-family:"Pretendard","Noto Sans KR",sans-serif;background:var(--bg);
     color:var(--ink);font-size:14px;line-height:1.5}
.wrap{max-width:1240px;margin:0 auto;padding:22px 24px 60px}
header{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;
       border-bottom:2px solid var(--ink);padding-bottom:10px;margin-bottom:14px}
h1{font-size:19px;margin:0}
#meta{color:var(--sub);font-size:12px;margin-left:auto}
.pill{font-size:11px;font-weight:700;padding:2px 10px;border-radius:20px}
.pill.live{background:var(--good-soft);color:var(--good)}
.pill.done{background:var(--panel);color:var(--sub);border:1px solid var(--line)}
.runs{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}
.runcard{border:1px solid var(--line);border-radius:10px;background:var(--panel);
         padding:9px 13px;min-width:230px}
.runcard b{font-size:15px}
.dot{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:7px}
.ptry{margin:4px 0;border:1px solid var(--line);border-radius:10px;background:var(--panel)}
.ptry summary{cursor:pointer;padding:7px 12px;font-weight:700}
.ptry .psub{color:var(--sub);font-weight:400;font-size:11px;margin-left:8px;
            font-variant-numeric:tabular-nums}
.ptry .pt{border:none;margin:0;padding-top:0}
.runcard .kv{font-size:12px;color:var(--sub);margin-top:3px;
             font-variant-numeric:tabular-nums}
.conv{font-size:11px;margin-top:3px}
.conv b{font-variant-numeric:tabular-nums}
.pt{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:4px 18px;
    font-size:12px;background:var(--panel);border:1px solid var(--line);
    border-radius:10px;padding:10px 14px;margin:4px 0 10px}
.pt div{display:flex;justify-content:space-between;gap:10px;border-bottom:1px dashed var(--line);padding:2px 0}
.pt span{color:var(--sub)} .pt b{font-variant-numeric:tabular-nums;text-align:right}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:6px}
@media(max-width:900px){.grid{grid-template-columns:1fr}}
.card{border:1px solid var(--line);border-radius:10px;padding:10px 12px 6px;background:var(--bg)}
.card h3{margin:0 0 2px;font-size:13px}
.card .sub{font-size:11px;color:var(--sub);margin-bottom:4px}
svg{width:100%;height:220px;display:block}
.axis{stroke:var(--line);stroke-width:1}
.tick{fill:var(--sub);font-size:10px;font-variant-numeric:tabular-nums}
.tt{position:fixed;pointer-events:none;background:var(--ink);color:var(--bg);
    font-size:11px;padding:4px 8px;border-radius:6px;display:none;white-space:pre;
    font-variant-numeric:tabular-nums;z-index:9}
table{width:100%;border-collapse:collapse;font-size:12px;margin-top:16px}
th{background:var(--ink);color:var(--bg);text-align:left;padding:5px 8px}
td{padding:5px 8px;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}
</style></head><body><div class="wrap">
<header><h1>Phase A 학습 모니터</h1>
<span id="proc" class="pill done">확인 중</span>
<span id="meta"></span></header>
<div class="runs" id="runs"></div>
<div style="font-weight:700;margin:6px 0">학습 파라미터 (TRY별 토글 · 시간순)</div>
<div id="params"></div>
<div class="grid" id="charts"></div>
<table id="tbl"></table>
<div class="tt" id="tt"></div>
<script>
const CHARTS=[
 {key:"rot_p50",  title:"미학습 34종 · 회전 오차 p50 (deg) ↓", ref:{v:30,label:"ICP 수렴반경 30°"}},
 {key:"le30",    title:"미학습 34종 · ≤30° 진입률 ↑", pct:true},
 {key:"trans_rel_p50", title:"미학습 · 병진 오차 p50 (×D) ↓"},
 {key:"ce",      title:"train 대응 CE (수렴곡선) ↓", ref:{v:4.53,label:"전경 균등 바닥"}},
 {key:"rot_deg", title:"train 회전 오차 (deg) ↓"},
 {key:"bg_rate", title:"train bg 예측률 (라벨 ~14%)", pct:true},
];
const tt=document.getElementById('tt');
function lerp(a,b,t){return a+(b-a)*t}
function chart(id,runs,cfg){
 const W=560,H=220,L=44,R=8,T=10,B=22;
 let xs=[],ys=[];
 runs.forEach(r=>r.hist.forEach(h=>{if(h[cfg.key]!=null){xs.push(h.ep);ys.push(h[cfg.key]);}}));
 if(!xs.length)return '';
 let x0=0,x1=Math.max(...xs,1),y0=Math.min(...ys),y1=Math.max(...ys);
 if(cfg.ref){y0=Math.min(y0,cfg.ref.v);y1=Math.max(y1,cfg.ref.v);}
 if(y0===y1){y0-=1;y1+=1} const pad=(y1-y0)*0.08;y0-=pad;y1+=pad;
 const X=e=>L+(e-x0)/(x1-x0)*(W-L-R), Y=v=>T+(1-(v-y0)/(y1-y0))*(H-T-B);
 let s=`<svg viewBox="0 0 ${W} ${H}" data-c="${id}">`;
 for(let i=0;i<=4;i++){const v=lerp(y0,y1,i/4),y=Y(v);
   s+=`<line class="axis" x1="${L}" y1="${y}" x2="${W-R}" y2="${y}"/>`+
      `<text class="tick" x="${L-5}" y="${y+3}" text-anchor="end">${cfg.pct?(v*100).toFixed(0)+'%':v.toFixed(v<10?2:0)}</text>`;}
 for(let e=x0;e<=x1;e+=Math.max(1,Math.round((x1-x0)/6))){
   s+=`<text class="tick" x="${X(e)}" y="${H-6}" text-anchor="middle">${e}</text>`;}
 if(cfg.ref){const y=Y(cfg.ref.v);
   s+=`<line x1="${L}" y1="${y}" x2="${W-R}" y2="${y}" stroke="var(--sub)" stroke-dasharray="4 3"/>`+
      `<text class="tick" x="${W-R}" y="${y-4}" text-anchor="end">${cfg.ref.label}</text>`;}
 runs.slice().reverse().forEach(r=>{  // 최신 런이 맨 위에 그려지도록 (배열은 최신순)
   const pts=r.hist.filter(h=>h[cfg.key]!=null).map(h=>[X(h.ep),Y(h[cfg.key]),h.ep,h[cfg.key]]);
   if(!pts.length)return;
   s+=`<polyline fill="none" stroke="${r.color}" stroke-width="2" points="${pts.map(p=>p[0]+','+p[1]).join(' ')}"/>`;
   const lp=pts[pts.length-1];
   s+=`<circle cx="${lp[0]}" cy="${lp[1]}" r="3.5" fill="${r.color}"/>`+
      `<text x="${Math.min(lp[0]+6,W-90)}" y="${lp[1]-6}" fill="${r.color}" font-size="10" font-weight="700">${r.tag.replace(/_s[0-9]+$/,'')}</text>`;
   pts.forEach(p=>{s+=`<circle cx="${p[0]}" cy="${p[1]}" r="7" fill="transparent" `+
     `data-tt="${r.tag} · ep${p[2]}&#10;${cfg.title.split('·')[1]||cfg.key}: ${cfg.pct?(p[3]*100).toFixed(1)+'%':p[3].toFixed(3)}"/>`});
 });
 return s+'</svg>';
}
function fmt(v,pct){if(v===null||v===undefined)return '–';
 const s=pct?(v*100).toFixed(1)+'%p':v.toFixed(3);return (v>0?'+':'')+s}
const PKEYS=[["encoder","인코더"],["lr","learning rate"],["batch_size","batch"],
 ["epochs","epochs"],["seed","seed"],["optimizer","optimizer"],["scheduler","scheduler"],
 ["grad_clip","grad clip"],["precision","정밀도"],["enc_params_M","인코더 파라미터(M)"],
 ["matcher_params_M","매처 파라미터(M)"],["matcher","매처 구조"],["sim","유사도"],
 ["loss","손실"],["n_tok","코어스 토큰"],["cad_views_train","CAD 뷰(학습)"],
 ["cad_views_eval","CAD 뷰(평가)"],["view_px","뷰당 픽셀"],["data_train","학습 샘플"],
 ["data_val","검증 샘플"],["obj_val_unseen","미학습 물체 종"],["sparsify","희소화"]];
async function refresh(){
 const d=await (await fetch('/api/runs')).json();
 document.getElementById('proc').className='pill '+(d.proc_alive?'live':'done');
 document.getElementById('proc').textContent=d.proc_alive?'● 학습 프로세스 실행 중':'프로세스 없음';
 document.getElementById('meta').textContent='갱신 '+new Date(d.ts*1000).toLocaleTimeString()+' · 60s 자동';
 const stamp=ts=>{const d=new Date(ts*1000);
  return `${d.getMonth()+1}/${d.getDate()} ${String(d.getHours()).padStart(2,'0')}:${String(d.getMinutes()).padStart(2,'0')}`};
 document.getElementById('runs').innerHTML=d.runs.map(r=>{
  const last=r.hist[r.hist.length-1]||{};
  return `<div class="runcard"><span class="dot" style="background:${r.color}"></span>`+
   `<b>${r.tag}</b> <span class="pill ${r.live?'live':'done'}">${r.live?'LIVE':'완료/대기'}</span>`+
   `<div class="kv">시작 ${stamp(r.start_ts)} · ep ${r.ep}${r.total_ep?'/'+r.total_ep:''}`+
   (r.eta_m?` · ETA ~${r.eta_m}분`:'')+(r.elapsed_m?` · 경과 ${r.elapsed_m}분`:'')+`</div>`+
   `<div class="kv"><a href="/reports/${r.tag}/index.html" target="_blank">리포트 →</a></div>`+
   `<div class="kv">rot_p50 <b>${(last.rot_p50??0).toFixed(1)}°</b> · ≤30° <b>${((last.le30??0)*100).toFixed(1)}%</b> · CE ${(last.ce??0).toFixed(2)}</div>`+
   `<div class="conv">수렴(최근5ep/ep): CE <b>${fmt(r.conv.ce)}</b> · rot <b>${fmt(r.conv.rot_p50)}°</b> · ≤30° <b>${fmt(r.conv.le30,true)}</b>`+
   ` ${(r.conv.ce!==null&&Math.abs(r.conv.ce)<0.01)?'<span class="pill done">포화 근접</span>':''}</div></div>`;
 }).join('');
 const open=new Set([...document.querySelectorAll('.ptry[open]')].map(e=>e.dataset.tag));
 document.getElementById('params').innerHTML=d.runs.map(r=>
  `<details class="ptry" data-tag="${r.tag}"${open.has(r.tag)?' open':''}>`+
  `<summary><span class="dot" style="background:${r.color}"></span>${r.tag}`+
  `<span class="psub">시작 ${stamp(r.start_ts)} · ep ${r.ep}${r.total_ep?'/'+r.total_ep:''}`+
  `${r.live?' · LIVE':''}</span></summary>`+
  `<div class="pt">`+PKEYS.filter(([k])=>r.cfg[k]!==undefined).map(([k,l])=>
    `<div><span>${l}</span><b>${r.cfg[k]}</b></div>`).join('')+`</div></details>`).join('');
 document.getElementById('charts').innerHTML=CHARTS.map((c,i)=>
  `<div class="card"><h3>${c.title}</h3><div class="sub">x = epoch</div>${chart(i,d.runs,c)}</div>`).join('');
 const rows=d.runs.map(r=>{const l=r.hist[r.hist.length-1]||{};
  return `<tr><td><span class="dot" style="background:${r.color}"></span>${r.tag}</td>`+
  `<td>${r.ep}${r.total_ep?'/'+r.total_ep:''}</td><td>${(l.rot_p50??0).toFixed(1)}°</td>`+
  `<td>${((l.le30??0)*100).toFixed(1)}%</td><td>${(l.trans_rel_p50??0).toFixed(3)}D</td>`+
  `<td>${(l.ce??0).toFixed(2)}</td><td>${(l.rot_deg??0).toFixed(0)}°</td><td>${((l.bg_rate??0)*100).toFixed(0)}%</td>`+
  `<td>${((l.g_nonid??0)*100).toFixed(0)}%</td></tr>`;}).join('');
 document.getElementById('tbl').innerHTML=
  '<tr><th>런</th><th>epoch</th><th>미학습 rot_p50</th><th>≤30°</th><th>병진</th><th>train CE</th><th>train rot</th><th>bg</th><th>g*≠I</th></tr>'+rows;
 document.querySelectorAll('[data-tt]').forEach(el=>{
  el.onmousemove=e=>{tt.style.display='block';tt.textContent=el.dataset.tt;
    tt.style.left=(e.clientX+12)+'px';tt.style.top=(e.clientY-10)+'px';};
  el.onmouseleave=()=>tt.style.display='none';});
}
refresh();setInterval(refresh,60000);
</script></div></body></html>"""


class H(BaseHTTPRequestHandler):
    log_path: Path | None = None

    def log_message(self, *a):  # 콘솔 소음 억제
        pass

    def _send(self, body: bytes, ctype: str):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/reports/"):
            # 런 리포트 정적 서빙 (reports/{tag}/index.html·png)
            import mimetypes
            rel = self.path[len("/reports/"):].split("?")[0]
            f = (RUNS.parent / "reports" / rel).resolve()
            base = (RUNS.parent / "reports").resolve()
            if str(f).startswith(str(base)) and f.is_file():
                self._send(f.read_bytes(),
                           mimetypes.guess_type(str(f))[0] or "application/octet-stream")
            else:
                self.send_response(404); self.end_headers()
            return
        if self.path.startswith("/api/runs"):
            self._send(json.dumps(collect(self.log_path)).encode(),
                       "application/json; charset=utf-8")
        else:
            self._send(PAGE.encode(), "text/html; charset=utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8035)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--log", default=None, help="학습 stdout 로그 (ETA 계산용)")
    a = ap.parse_args()
    H.log_path = Path(a.log) if a.log else None
    srv = ThreadingHTTPServer((a.host, a.port), H)
    print(f"대시보드: http://{a.host}:{a.port}  (runs={RUNS}, log={a.log})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
