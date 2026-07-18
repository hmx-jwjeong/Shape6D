"""shape6d-onboard CLI (03 문서 §3.4). 렌더/특징 단계는 M1/M2에서 채워진다."""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description="Shape6D S0 온보딩: CAD → onboard_v1.npz")
    ap.add_argument("--cad", required=True)
    ap.add_argument("--unit", required=True, choices=["mm", "m"],
                    help="CAD 단위 — 필수 명시 (03 §3.1)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--obj-id", required=True)
    ap.add_argument("--n-master", type=int, default=16384)
    ap.add_argument("--sym-override", default=None, help="대칭 수동 정정 JSON")
    ap.add_argument("--no-sym-detect", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    import trimesh

    from . import cache, sampling, symmetry

    mesh = trimesh.load(args.cad, force="mesh")
    if args.unit == "mm":
        mesh.apply_scale(1e-3)

    # 입력 검증 (03 §3.1)
    bbox = mesh.bounds
    diag = float(np.linalg.norm(bbox[1] - bbox[0]))
    if not (0.005 <= diag <= 3.0) and not args.force:
        sys.exit(f"[중단] bbox 대각 {diag:.3f}m ∉ [0.005, 3.0] — 단위 확인. 강행: --force")
    watertight = bool(mesh.is_watertight)
    if not watertight:
        try:
            trimesh.repair.fill_holes(mesh)
            watertight = bool(mesh.is_watertight)
        except Exception:
            pass
    mesh.update_faces(mesh.nondegenerate_faces())  # trimesh 4.x API (검증 [하-2])
    mesh.merge_vertices()
    if watertight:
        trimesh.repair.fix_normals(mesh)

    pts, nrm = sampling.sample_master(mesh, args.n_master)
    subsets = sampling.make_subsets(pts)

    if args.no_sym_detect:
        sym = symmetry.SymmetryResult(sym_rots=np.eye(3, dtype=np.float32)[None],
                                      sym_axes=np.zeros((0, 3), np.float32))
    else:
        sym = symmetry.detect_symmetry(pts.astype(np.float64), mesh=mesh if watertight else None)
    if args.sym_override:
        ov = json.loads(open(args.sym_override).read())
        if "sym_rots" in ov:
            sym.sym_rots = np.asarray(ov["sym_rots"], np.float32)
        if "sym_axes" in ov:
            sym.sym_axes = np.asarray(ov["sym_axes"], np.float32).reshape(-1, 3)

    center = pts.mean(axis=0)
    arrays = {
        "pts_master": pts.astype(np.float16),
        "nrm_master": nrm.astype(np.float16),
        **subsets,
        "sym_rots": sym.sym_rots,
        "sym_axes": sym.sym_axes,
        "diameter": np.float32(diag),
        "bbox": bbox.astype(np.float32),
        "center_offset": ((bbox[0] + bbox[1]) / 2).astype(np.float32),
        "radius": np.float32(np.linalg.norm(pts - center, axis=1).max()),
    }
    path = cache.save(args.out, args.obj_id, arrays, cad_path=args.cad, unit=args.unit,
                      watertight=watertight, sym_summary=sym.summary)
    print(f"[완료] {path}")
    print(f"  대칭: {sym.summary}")
    for line in sym.log:
        print(f"    {line}")
    print("  [잔여] 템플릿 렌더(tpl_*, tdf, dino_cls)는 M1, dense_fo/geo_embedding_o는 M2에서 --refresh-features")


if __name__ == "__main__":
    main()
