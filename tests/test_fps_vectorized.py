"""fps_torch CUDA-graph 경로 정확성 — eager 기준 구현과 인덱스 비트단위 동일 검증.

경계 케이스: 전량 유효 / 행별 상이한 npt 패딩 / 유효점 < n(중복 선택 구간) /
유효점 1개 / bf16(encode_cad 경로 dtype) / 반복 호출(그래프 재사용) / 형상 변경.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
from train.pem_mini import _fps_eager, fps_torch  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA 필요")
DEV = "cuda"


def _mk(B, N, npts, seed, dtype=torch.float32):
    g = torch.Generator(device=DEV).manual_seed(seed)
    pts = torch.randn(B, N, 3, generator=g, device=DEV, dtype=torch.float32)
    npt = torch.as_tensor(npts, device=DEV)
    valid = torch.arange(N, device=DEV)[None] < npt[:, None]
    pts = torch.where(valid.unsqueeze(-1), pts, torch.zeros_like(pts))  # 패딩=0
    return pts.to(dtype), valid


def _check(pts, valid, n):
    ref = _fps_eager(pts, valid, n)
    out = fps_torch(pts, valid, n)
    assert out.shape == ref.shape and out.dtype == ref.dtype
    assert torch.equal(out, ref), (
        f"불일치 {int((out != ref).sum())}/{ref.numel()} "
        f"(B={pts.shape[0]}, N={pts.shape[1]}, dtype={pts.dtype})")


def test_full_valid():
    pts, valid = _mk(8, 4096, [4096] * 8, seed=0)
    _check(pts, valid, 196)


def test_mixed_npt_padding():
    npts = [4096, 1, 196, 195, 197, 1000, 50, 2]        # 경계 전후 + 극소
    pts, valid = _mk(8, 4096, npts, seed=1)
    _check(pts, valid, 196)


def test_fewer_valid_than_n():
    pts, valid = _mk(6, 512, [10, 100, 195, 196, 3, 1], seed=2)
    _check(pts, valid, 196)


def test_bf16_cad_path_shape():
    # encode_cad: autocast 하 einsum 산출 → bf16, N = n_views*VIEW_PX
    pts, valid = _mk(4, 3072, [3072, 3000, 1536, 7], seed=3, dtype=torch.bfloat16)
    _check(pts, valid, 196)


def test_graph_reuse_multiple_inputs():
    # 같은 형상으로 반복 호출 — 그래프 재사용 시 static 버퍼 오염 없는지
    for seed in range(5):
        pts, valid = _mk(8, 4096, [4096, 300, 4096, 196, 50, 2048, 1, 999], seed=10 + seed)
        _check(pts, valid, 196)


def test_shape_switch():
    # 형상 교차 호출 — 캐시 키 분리 검증
    for B, N in [(4, 1024), (7, 4096), (4, 1024), (2, 3072)]:
        g = torch.Generator(device=DEV).manual_seed(B * 1000 + N)
        npt = torch.randint(1, N + 1, (B,), generator=g, device=DEV)
        pts, valid = _mk(B, N, npt.tolist(), seed=B + N)
        _check(pts, valid, 196)


def test_output_independent_of_buffer():
    # 반환값이 clone인지 — 다음 호출이 이전 결과를 덮어쓰면 안 됨
    pts1, valid1 = _mk(8, 4096, [4096] * 8, seed=20)
    out1 = fps_torch(pts1, valid1, 196)
    keep = out1.clone()
    pts2, valid2 = _mk(8, 4096, [100] * 8, seed=21)
    fps_torch(pts2, valid2, 196)
    assert torch.equal(out1, keep)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
