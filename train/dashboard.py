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
        # LIVE 판정: hist는 에폭 종료 시에만 갱신되므로 고정 5분 창이면
        # 에폭이 긴 런은 에폭 사이마다 죽은 것처럼 보임 → 에폭 소요시간 비례 창
        per_ep_s = el / ep * 60 if (el and ep) else None
        win = max(300.0, 2.0 * per_ep_s + 180) if per_ep_s else 300.0
        unfinished = tot is None or ep < tot
        live = alive and unfinished and (now - r["mtime"] < win)
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
    # 색상: 그룹(a0/a1/…) 기본색만 내려주고, 시간순 그라데이션은 클라이언트가
    # '표시 중인 런' 기준으로 계산 (런이 수십 개면 전체 기준 그라데이션은 구분 불가)
    for r in out:
        r["base"] = COLORS.get(r["tag"].split("_")[0], "#e87ba4")
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
.runs{display:flex;gap:10px;flex-wrap:wrap;margin:8px 0 12px}
.runcard{border:1px solid var(--line);border-radius:10px;background:var(--panel);
         padding:9px 13px;min-width:230px;cursor:pointer;user-select:none}
.runcard.off{opacity:.38;filter:saturate(.25)}
.hint{font-size:11px;color:var(--sub);margin-top:10px}
.hint a{cursor:pointer;text-decoration:underline}
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
.card{border:1px solid var(--line);border-radius:10px;padding:10px 12px 6px;background:var(--bg);
      position:relative}
.card h3{margin:0 0 2px;font-size:13px;cursor:help;display:inline-block}
.qi{color:var(--sub);font-size:11px;font-weight:400}
.desc{position:absolute;inset:44px 10px 10px;background:var(--panel);
      border:1px solid var(--line);border-radius:8px;padding:12px 14px;
      font-size:12px;line-height:1.7;color:var(--ink);
      opacity:0;pointer-events:none;transition:opacity .25s;z-index:5}
.desc b{display:block;margin-bottom:4px}
.card.showdesc .desc{opacity:.97}
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
<div class="hint" id="selhint"></div>
<div class="runs" id="runs"></div>
<div style="font-weight:700;margin:6px 0">학습 파라미터 (TRY별 토글 · 시간순)</div>
<div id="params"></div>
<div class="grid" id="charts"></div>
<table id="tbl"></table>
<div class="tt" id="tt"></div>
<script>
const CHARTS=[
 {key:"rot_p50",  title:"미학습 34종 · 회전 오차 p50 (deg) ↓", ref:{v:30,label:"ICP 수렴반경 30°"},
  desc:"학습에 쓰지 않은 물체 종(검증셋)에서 예측 회전과 정답 회전의 각도 차이 중앙값. "+
   "일반화 성능의 핵심 지표로, 낮을수록 좋음. 점선(30°)은 ICP 정밀 정렬이 수렴할 수 있는 "+
   "대략적 반경 — 이 아래로 들어와야 후단 refinement로 회복 가능."},
 {key:"le30",    title:"미학습 34종 · ≤30° 진입률 ↑", pct:true,
  desc:"검증 샘플 중 회전 오차가 30° 이하인 비율. \\\"ICP로 마무리 가능한 샘플이 몇 %인가\\\"로 "+
   "해석하는 실전 지표. p50과 달리 분포 꼬리에 둔감해서, 어려운 샘플을 포기하고도 "+
   "쉬운 샘플을 확실히 맞추는 개선을 잘 드러냄. 높을수록 좋음."},
 {key:"trans_rel_p50", title:"미학습 · 병진 오차 p50 (×D) ↓",
  desc:"예측 위치와 정답 위치의 거리 중앙값을 물체 지름 D로 나눈 상대값 (검증셋). "+
   "0.10D = 지름의 10%만큼 어긋남. 물체 크기와 무관하게 비교할 수 있도록 정규화. 낮을수록 좋음."},
 {key:"ce",      title:"train 대응 CE (수렴곡선) ↓", ref:{v:4.53,label:"전경 균등 바닥"},
  desc:"학습 배치에서 쿼리 토큰↔CAD 토큰 대응 분류의 cross-entropy. 순수 수렴 곡선. "+
   "점선(전경 균등 바닥)은 전경 토큰들을 균등 확률로 찍었을 때의 값 — 이 밑으로 내려가야 "+
   "대응 관계를 실제로 학습하고 있다는 뜻. 기울기가 0에 가까워지면 포화."},
 {key:"rot_deg", title:"train 회전 오차 (deg) ↓",
  desc:"학습 배치(이미 본 물체)에서의 회전 오차. 모델 용량·수렴 상태 참고용. "+
   "미학습 rot_p50과의 격차가 크게 벌어지면 과적합 신호 — 격차가 곧 일반화 갭."},
 {key:"bg_rate", title:"train bg 예측률 (라벨 ~14%)", pct:true,
  desc:"쿼리 토큰을 배경(bg)으로 예측한 비율. 정답 라벨 분포는 약 14%이므로 그 근처면 정상. "+
   "이 값이 치솟으면 모델이 대응을 포기하고 전부 배경으로 덤핑하는 붕괴 징후이고, "+
   "0에 붙으면 배경을 전경에 억지로 매칭하고 있다는 뜻."},
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
// 런 표시 선택: 기본은 최신 5개, 카드 클릭으로 개별 전환 (localStorage에 오버라이드 저장)
const SHOW_N=5;
let LAST=null, OVR=JSON.parse(localStorage.getItem('runOvr')||'{}');
const visOf=(tag,i)=>OVR[tag]!==undefined?OVR[tag]:i<SHOW_N;
function toggleRun(tag){
 const i=LAST.runs.findIndex(r=>r.tag===tag);
 const next=!visOf(tag,i);
 if(next===(i<SHOW_N))delete OVR[tag];else OVR[tag]=next;
 localStorage.setItem('runOvr',JSON.stringify(OVR));render();}
function resetSel(){OVR={};localStorage.setItem('runOvr','{}');render();}
async function refresh(){
 LAST=await (await fetch('/api/runs')).json();render();}
// 그룹 내 시간순 그라데이션: base 색을 흰색 쪽으로 t만큼 블렌드 (최신=원색)
const shade=(hex,t)=>{const c=parseInt(hex.slice(1),16),m=v=>Math.round(v+(255-v)*t);
 return '#'+[(c>>16)&255,(c>>8)&255,c&255].map(v=>m(v).toString(16).padStart(2,'0')).join('')};
function grad(list){
 const g={};list.forEach(r=>{(g[r.tag.split('_')[0]]??=[]).push(r)});
 Object.values(g).forEach(m=>{m.sort((a,b)=>a.start_ts-b.start_ts);
  const n=m.length;m.forEach((r,i)=>r.color=shade(r.base,n>1?0.55*(n-1-i)/(n-1):0))});}
function render(){
 const d=LAST;if(!d)return;
 const vis=d.runs.filter((r,i)=>visOf(r.tag,i));
 grad(d.runs);grad(vis); // 전체 기준으로 깔고, 표시 중 런끼리는 최대 대비로 재배정
 document.getElementById('selhint').innerHTML=
  `기본 최신 ${SHOW_N}개 표시 · 카드 클릭으로 표시/숨김 전환 (${vis.length}/${d.runs.length} 표시 중)`+
  (Object.keys(OVR).length?` · <a onclick="resetSel()">기본값 복원</a>`:'');
 document.getElementById('proc').className='pill '+(d.proc_alive?'live':'done');
 document.getElementById('proc').textContent=d.proc_alive?'● 학습 프로세스 실행 중':'프로세스 없음';
 document.getElementById('meta').textContent='갱신 '+new Date(d.ts*1000).toLocaleTimeString()+' · 60s 자동';
 const stamp=ts=>{const d=new Date(ts*1000);
  return `${d.getMonth()+1}/${d.getDate()} ${String(d.getHours()).padStart(2,'0')}:${String(d.getMinutes()).padStart(2,'0')}`};
 document.getElementById('runs').innerHTML=d.runs.map((r,i)=>{
  const last=r.hist[r.hist.length-1]||{};
  return `<div class="runcard${visOf(r.tag,i)?'':' off'}" onclick="toggleRun('${r.tag}')">`+
   `<span class="dot" style="background:${r.color}"></span>`+
   `<b>${r.tag}</b> <span class="pill ${r.live?'live':'done'}">${r.live?'LIVE':'완료/대기'}</span>`+
   `<div class="kv">시작 ${stamp(r.start_ts)} · ep ${r.ep}${r.total_ep?'/'+r.total_ep:''}`+
   (r.eta_m?` · ETA ~${r.eta_m}분`:'')+(r.elapsed_m?` · 경과 ${r.elapsed_m}분`:'')+`</div>`+
   `<div class="kv"><a href="/reports/${r.tag}/index.html" target="_blank" onclick="event.stopPropagation()">리포트 →</a></div>`+
   `<div class="kv">rot_p50 <b>${(last.rot_p50??0).toFixed(1)}°</b> · ≤30° <b>${((last.le30??0)*100).toFixed(1)}%</b> · CE ${(last.ce??0).toFixed(2)}</div>`+
   `<div class="conv">수렴(최근5ep/ep): CE <b>${fmt(r.conv.ce)}</b> · rot <b>${fmt(r.conv.rot_p50)}°</b> · ≤30° <b>${fmt(r.conv.le30,true)}</b>`+
   ` ${(r.conv.ce!==null&&Math.abs(r.conv.ce)<0.01)?'<span class="pill done">포화 근접</span>':''}</div></div>`;
 }).join('');
 const open=new Set([...document.querySelectorAll('.ptry[open]')].map(e=>e.dataset.tag));
 document.getElementById('params').innerHTML=vis.map(r=>
  `<details class="ptry" data-tag="${r.tag}"${open.has(r.tag)?' open':''}>`+
  `<summary><span class="dot" style="background:${r.color}"></span>${r.tag}`+
  `<span class="psub">시작 ${stamp(r.start_ts)} · ep ${r.ep}${r.total_ep?'/'+r.total_ep:''}`+
  `${r.live?' · LIVE':''}</span></summary>`+
  `<div class="pt">`+PKEYS.filter(([k])=>r.cfg[k]!==undefined).map(([k,l])=>
    `<div><span>${l}</span><b>${r.cfg[k]}</b></div>`).join('')+`</div></details>`).join('');
 document.getElementById('charts').innerHTML=CHARTS.map((c,i)=>
  `<div class="card"><h3 title="">${c.title} <span class="qi">ⓘ</span></h3>`+
  `<div class="sub">x = epoch</div>${chart(i,vis,c)}`+
  `<div class="desc"><b>${c.title}</b>${c.desc}</div></div>`).join('');
 // 제목에 호버할 때만 설명 오버레이 표시 (차트 영역 호버는 데이터 툴팁 전용)
 document.querySelectorAll('#charts .card h3').forEach(el=>{
  let t=null;
  el.onmouseenter=()=>{t=setTimeout(()=>el.parentElement.classList.add('showdesc'),250)};
  el.onmouseleave=()=>{clearTimeout(t);el.parentElement.classList.remove('showdesc')};
 });
 const rows=vis.map(r=>{const l=r.hist[r.hist.length-1]||{};
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
