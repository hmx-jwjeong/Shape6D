"""학습 시작 시 모니터 대시보드 자동 기동 + 브라우저 오픈.

프로젝트 규약(2026-07-31): 모든 학습 엔트리포인트는 시작 직후
`ensure_dashboard()`를 호출한다 — 서버가 없으면 백그라운드로 띄우고,
GUI 세션(DISPLAY/WAYLAND)이면 기본 브라우저 탭을 연다. 멱등:
이미 떠 있으면 재기동하지 않는다. 끄기: --no-dash 또는 env DASH_OFF=1.

향후 학습 스크립트(Phase B 등)도 동일하게 한 줄 호출로 연결할 것:
    from train.autodash import ensure_dashboard; ensure_dashboard()
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import webbrowser
from pathlib import Path

PORT = 8035
HERE = Path(__file__).parent


def _listening(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def ensure_dashboard(port: int = PORT, log: str | None = None,
                     open_browser: bool = True) -> str:
    """대시보드 보장 기동. 반환: URL. 실패해도 학습을 막지 않는다(경고만)."""
    url = f"http://127.0.0.1:{port}"
    if os.environ.get("DASH_OFF") == "1":
        return url
    try:
        if not _listening(port):
            cmd = [sys.executable, "-u", str(HERE / "dashboard.py"), "--port", str(port)]
            log = log or os.environ.get("TRAIN_LOG")
            if log:
                cmd += ["--log", log]
            out = open(HERE.parent / ".dashboard.out", "ab")
            subprocess.Popen(cmd, stdout=out, stderr=out,
                             start_new_session=True, cwd=HERE.parent)
            print(f"[autodash] 대시보드 기동: {url}", flush=True)
        else:
            print(f"[autodash] 대시보드 이미 실행 중: {url}", flush=True)
        if open_browser and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            # 브라우저 오픈 실패는 무해 — SSH/headless에선 자동 생략
            webbrowser.open(url, new=2)
    except Exception as e:  # 모니터 실패가 학습을 죽이면 안 됨
        print(f"[autodash] 경고: {type(e).__name__}: {e}", flush=True)
    return url
