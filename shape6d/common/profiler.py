"""스테이지 레이턴시 프로파일러 (M0-4 회귀 게이트의 기반)."""
from __future__ import annotations

import json
import time
from contextlib import contextmanager


class StageTimer:
    def __init__(self):
        self.records: dict[str, list[float]] = {}

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.records.setdefault(name, []).append(time.perf_counter() - t0)

    def summary(self) -> dict:
        out = {}
        for k, v in self.records.items():
            arr = sorted(v)
            n = len(arr)
            out[k] = {
                "n": n,
                "mean_ms": 1e3 * sum(arr) / n,
                "p50_ms": 1e3 * arr[n // 2],
                "p95_ms": 1e3 * arr[min(n - 1, int(0.95 * n))],
            }
        return out

    def report(self, path: str | None = None) -> str:
        s = json.dumps(self.summary(), indent=2, ensure_ascii=False)
        if path:
            with open(path, "w") as f:
                f.write(s)
        return s
