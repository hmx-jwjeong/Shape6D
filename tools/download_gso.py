#!/usr/bin/env python3
"""GSO(Google Scanned Objects) 잔여 메시 다운로드 — Gazebo Fuel.

Phase B 데이터 선행 작업: megapose/gso_raw 에 없는 GSO 물체의 메시 zip을
Gazebo Fuel(fuel.gazebosim.org, owner=GoogleResearch)에서 내려받는다.
전 모델 라이선스: Creative Commons Attribution 4.0 International (CC-BY 4.0,
Fuel API license_name/license_url 메타데이터로 확인).

- 대상: /mnt/samsung2tb/datasets/gso_meshes_raw/{name}.zip
- 재개 가능: 이미 존재하는 무결한 zip 은 스킵 (.part 로 받고 검증 후 rename)
- 보유 스킵: megapose/gso_raw/{name}/meshes/model.obj 가 있으면 스킵
- 동시 5개, 모델당 3회 재시도(백오프), 진행 로그 download.log

실행(백그라운드 권장):
  setsid nohup python3 tools/download_gso.py >/dev/null 2>&1 &
재개: 같은 명령 재실행 — 받은 것은 건너뛴다.
진행 확인:
  tail -f /mnt/samsung2tb/datasets/gso_meshes_raw/download.log
  ls /mnt/samsung2tb/datasets/gso_meshes_raw/*.zip | wc -l
"""
from __future__ import annotations

import json
import logging
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DEST = Path("/mnt/samsung2tb/datasets/gso_meshes_raw")
LOG = DEST / "download.log"
NAMES_FILE = DEST / "all_names.txt"  # Fuel 전체 모델 이름 목록(1행 1이름)
HELD_DIR = Path("/mnt/samsung2tb/datasets/megapose/gso_raw")  # 읽기 전용!

API = "https://fuel.gazebosim.org/1.0/GoogleResearch/models"
ZIP_URL = API + "/{name}.zip"
WORKERS = 5
RETRIES = 3
TIMEOUT = 300  # 초/요청
UA = {"User-Agent": "Shape6D-GSO-fetch/1.0 (research; CC-BY-4.0 dataset)"}

log = logging.getLogger("gso")


def fetch_all_names() -> list[str]:
    """Fuel API 페이지네이션으로 전체 모델 이름 수집 (NAMES_FILE 없을 때만)."""
    names: set[str] = set()
    page = 1
    while True:
        url = f"{API}?page={page}&per_page=100"
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.load(r)
        if not isinstance(data, list) or not data:
            break
        names.update(e["name"] for e in data if isinstance(e, dict))
        if len(data) < 100:
            break
        page += 1
        time.sleep(0.5)
    return sorted(names)


def zip_ok(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as z:
            if z.testzip() is not None:
                return False
            return "meshes/model.obj" in z.namelist()
    except (zipfile.BadZipFile, OSError):
        return False


def download_one(name: str) -> str:
    final = DEST / f"{name}.zip"
    if final.exists() and zip_ok(final):
        return "skip"
    part = DEST / f"{name}.zip.part"
    url = ZIP_URL.format(name=urllib.parse.quote(name))
    for attempt in range(1, RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r, open(part, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            if not zip_ok(part):
                raise ValueError("zip 무결성/model.obj 검증 실패")
            part.rename(final)
            log.info("OK   %-60s %6.1f MB", name, final.stat().st_size / 2**20)
            return "ok"
        except Exception as e:  # noqa: BLE001
            log.warning("RETRY %d/%d %s: %s", attempt, RETRIES, name, e)
            part.unlink(missing_ok=True)
            time.sleep(10 * attempt)
    log.error("FAIL %s", name)
    return "fail"


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(LOG), logging.StreamHandler(sys.stdout)],
    )

    if NAMES_FILE.exists():
        names = NAMES_FILE.read_text().split()
    else:
        log.info("전체 목록 수집 중 (Fuel API)...")
        names = fetch_all_names()
        NAMES_FILE.write_text("\n".join(names) + "\n")
    log.info("Fuel 전체 모델: %d종", len(names))

    held = {
        d.name for d in HELD_DIR.iterdir()
        if (d / "meshes" / "model.obj").is_file()
    } if HELD_DIR.is_dir() else set()
    done = {p.stem for p in DEST.glob("*.zip")}
    todo = [n for n in names if n not in held and n not in done]
    log.info("보유(gso_raw): %d / 기수신 zip: %d / 잔여: %d", len(held), len(done), len(todo))

    counts = {"ok": 0, "skip": 0, "fail": 0}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(download_one, n): n for n in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            counts[fut.result()] += 1
            if i % 25 == 0:
                log.info("진행 %d/%d (ok=%d fail=%d)", i, len(todo), counts["ok"], counts["fail"])
    log.info("완료: ok=%d skip=%d fail=%d", counts["ok"], counts["skip"], counts["fail"])
    if counts["fail"]:
        log.info("실패분은 같은 명령 재실행으로 재시도 가능")
    return 1 if counts["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
