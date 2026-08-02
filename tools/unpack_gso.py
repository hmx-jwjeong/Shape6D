"""Phase B 데이터 준비 1단계 — GSO raw 메시 zip 일괄 언팩.

/mnt/samsung2tb/datasets/gso_meshes_raw/*.zip (내부 구조: 루트에 meshes/model.obj
+ materials/textures/ — gso_raw/{name}/ 규약과 동일, 이름 접두 없음)을
unpacked/{name}/ 으로 해제한다.

- zip 무결성 검사(testzip) 실패 / 해제 후 meshes/model.obj 부재 → 실패 목록 보고
- 이미 unpacked/{name}/meshes/model.obj 가 있으면 건너뜀 (멱등)
- 기존 megapose/ 트리는 건드리지 않음 (읽기 전용)

실행: python3 tools/unpack_gso.py [--workers 12]
"""
from __future__ import annotations

import argparse
import json
import zipfile
from multiprocessing import Pool
from pathlib import Path

SRC = Path("/mnt/samsung2tb/datasets/gso_meshes_raw")
DST = SRC / "unpacked"


def unpack_one(zp: Path) -> tuple[str, str]:
    """(name, status) — status ∈ ok | skip | bad_zip:... | no_model_obj | error:..."""
    name = zp.stem
    out = DST / name
    if (out / "meshes" / "model.obj").exists():
        return name, "skip"
    try:
        with zipfile.ZipFile(zp) as z:
            bad = z.testzip()
            if bad is not None:
                return name, f"bad_zip:{bad}"
            # 경로 탈출 방지 (절대경로 · ..)
            for n in z.namelist():
                p = Path(n)
                if p.is_absolute() or ".." in p.parts:
                    return name, f"bad_path:{n}"
            out.mkdir(parents=True, exist_ok=True)
            z.extractall(out)
    except zipfile.BadZipFile as e:
        return name, f"bad_zip:{e}"
    except Exception as e:
        return name, f"error:{type(e).__name__}:{e}"
    if not (out / "meshes" / "model.obj").exists():
        return name, "no_model_obj"
    return name, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    zips = sorted(SRC.glob("*.zip"))
    print(f"zip {len(zips)}개 → {DST}", flush=True)
    DST.mkdir(exist_ok=True)
    with Pool(a.workers) as p:
        res = p.map(unpack_one, zips)
    ok = [n for n, s in res if s == "ok"]
    skip = [n for n, s in res if s == "skip"]
    fail = [(n, s) for n, s in res if s not in ("ok", "skip")]
    print(f"성공 {len(ok)} · 기해제 건너뜀 {len(skip)} · 실패 {len(fail)}")
    for n, s in fail:
        print(f"  FAIL {n}: {s}")
    json.dump({"ok": ok, "skip": skip, "fail": fail},
              open(SRC / "unpack_report.json", "w"), indent=1)
    print(f"리포트: {SRC / 'unpack_report.json'}", flush=True)


if __name__ == "__main__":
    main()
