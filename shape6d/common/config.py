"""파이프라인 설정 로더 (10 권고 4 / A-1).

`shape6d/config/pipeline.yaml`이 임계값의 단일 출처다. 이 모듈이 생기기 전에는
yaml을 읽는 프로덕션 코드가 0개였고 함수 기본인자와 3곳이 어긋나 있었다
(free_viol_max 0.05/0.15, s4.stride 2/4, icp 스케줄). 해소 방향: yaml을 현행
실측 거동에 맞춰 갱신하고(공표된 04/07/09 수치 보호), 03 원안과 다른 값은
yaml 주석에 결정 대기로 표기.

사용: 조립 지점(Verifier.from_config 등)에서 명시적으로 주입 — 전역 암묵 로드 금지.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "config" / "pipeline.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """pipeline.yaml → dict. path=None이면 패키지 동봉 정본."""
    p = Path(path) if path is not None else DEFAULT_PATH
    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"설정 파일이 매핑이 아님: {p}")
    return cfg


def cfg_get(cfg: dict, dotted: str, default: Any = ...) -> Any:
    """점 표기 키 조회: cfg_get(cfg, 's4.free_viol_max').

    default 미지정 시 누락 키는 KeyError — 침묵 폴백으로 다시 어긋나는 것을 방지.
    """
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            if default is ...:
                raise KeyError(f"설정 키 없음: {dotted}")
            return default
        node = node[part]
    return node
