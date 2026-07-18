"""S0 통합 캐시 — onboard_v1.npz 단일 스키마 (03 문서 §2.4 정본).

검증 [H2]/[중-8]/[H3] 반영: 마스터 포인트 1개 + 이름 있는 서브셋 인덱스,
dino_cls는 42뷰, dense_fo/geo_embedding_o는 encoder_hash 종속(2-pass 굽기).
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1

# 필드명 → (shape 검증 함수, dtype, 필수 여부). None 차원은 가변.
_SCHEMA: dict[str, tuple[tuple, type, bool]] = {
    "pts_master":     ((16384, 3), np.float16, True),
    "nrm_master":     ((16384, 3), np.float16, True),
    "idx_pem":        ((2048,), np.int32, True),
    "idx_sparse":     ((196,), np.int32, True),
    "idx_model":      ((1024,), np.int32, True),
    "idx_verify":     ((2048,), np.int32, True),
    "tpl_depth":      ((42, 224, 224), np.uint16, False),   # mm, 0=배경 (렌더는 M1)
    "tpl_pose":       ((42, 4, 4), np.float32, False),
    "tpl_K":          ((3, 3), np.float32, False),
    "tpl_pts":        ((42, 512, 3), np.float32, False),
    "tpl_center":     ((42, 3), np.float32, False),
    "tdf":            ((42, 48, 48, 48), np.float16, False),
    "dino_cls":       ((42, 384), np.float16, False),
    "sym_rots":       ((None, 3, 3), np.float32, True),
    "sym_axes":       ((None, 3), np.float32, True),
    "dense_fo":       ((2048, 256), np.float16, False),     # encoder_hash 종속 (M2)
    "sparse_fo":      ((196, 256), np.float16, False),
    "geo_embedding_o": ((197, 197, None), np.float16, False),
    "pe_fo":          ((None, None), np.float16, False),
    "diameter":       ((), np.float32, True),
    "bbox":           ((2, 3), np.float32, True),
    "center_offset":  ((3,), np.float32, True),
    "radius":         ((), np.float32, True),
}


@dataclass
class OnboardCache:
    arrays: dict[str, np.ndarray]
    manifest: dict = field(default_factory=dict)

    def __getattr__(self, name):
        arrays = object.__getattribute__(self, "arrays")
        if name in arrays:
            return arrays[name]
        raise AttributeError(name)


def _check_shape(name: str, arr: np.ndarray, spec: tuple) -> None:
    if len(spec) != arr.ndim:
        raise ValueError(f"{name}: ndim {arr.ndim} != {len(spec)}")
    for want, got in zip(spec, arr.shape):
        if want is not None and want != got:
            raise ValueError(f"{name}: shape {arr.shape} != {spec}")


def validate(arrays: dict[str, np.ndarray]) -> None:
    for name, (shape, dtype, required) in _SCHEMA.items():
        if name not in arrays:
            if required:
                raise ValueError(f"필수 필드 누락: {name}")
            continue
        _check_shape(name, arrays[name], shape)
    unknown = set(arrays) - set(_SCHEMA)
    if unknown:
        raise ValueError(f"스키마 외 필드: {unknown}")


def save(out_dir: str | Path, obj_id: str, arrays: dict[str, np.ndarray],
         cad_path: str = "", unit: str = "m", watertight: bool = True,
         sym_summary: str = "", encoder_hash: str = "") -> Path:
    validate(arrays)
    d = Path(out_dir) / obj_id
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "onboard_v1.npz", **arrays)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "obj_id": obj_id,
        "cad_path": str(cad_path),
        "cad_sha256": _file_sha256(cad_path) if cad_path and Path(cad_path).exists() else "",
        "unit": unit,
        "watertight": watertight,
        "sym_summary": sym_summary,
        "encoder_hash": encoder_hash,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return d / "onboard_v1.npz"


def load(cache_dir: str | Path, obj_id: str) -> OnboardCache:
    d = Path(cache_dir) / obj_id
    data = dict(np.load(d / "onboard_v1.npz"))
    validate(data)
    manifest = json.loads((d / "manifest.json").read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version 불일치: {manifest.get('schema_version')} != {SCHEMA_VERSION}")
    return OnboardCache(arrays=data, manifest=manifest)


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
