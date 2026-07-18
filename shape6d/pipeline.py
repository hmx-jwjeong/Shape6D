"""S1→S4 오케스트레이션 + 스테이지 타이머 (03 문서 §8 SLA 규약).

M0 시점에는 S1(EViT-SAM)·S3(PEM)·S4(ICP)가 미구현 스텁 — 스테이지 단위로 교체된다
("항상 동작하는 파이프라인 유지" 원칙, 02 문서).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .common.frame_bundle import FrameBundle
from .common.profiler import StageTimer


class NotReady(NotImplementedError):
    pass


@dataclass
class PipelineResult:
    status: str                     # "ok" | "no_foreground" | "no_detection" | "not_ready"
    detections: list = field(default_factory=list)   # list[VerifyResult]
    degraded: bool = False          # 예외 경로(재적분·fallback) 발동 여부 — SLA 1.5s 트랙
    timing: dict = field(default_factory=dict)


class Shape6DPipeline:
    def __init__(self, s1=None, s2=None, s3=None, s4=None):
        self.s1, self.s2, self.s3, self.s4 = s1, s2, s3, s4
        self.timer = StageTimer()

    def __call__(self, fb: FrameBundle) -> PipelineResult:
        try:
            with self.timer.stage("S1_proposal"):
                if self.s1 is None:
                    raise NotReady("S1: EViT-SAM 래퍼는 M1 (proposal/evit_sam.py)")
                proposals = self.s1(fb)
            if not proposals:
                return PipelineResult("no_foreground", timing=self.timer.summary())

            with self.timer.stage("S2_identify"):
                if self.s2 is None:
                    raise NotReady("S2: Identifier 조립은 M1 (identify/score_fusion.py)")
                candidates = self.s2(fb, proposals)
            if not candidates:
                return PipelineResult("no_detection", timing=self.timer.summary())

            with self.timer.stage("S3_pose"):
                if self.s3 is None:
                    raise NotReady("S3: Shape6D-PEM은 M2 (pose/)")
                hypotheses = self.s3(fb, candidates)

            with self.timer.stage("S4_verify"):
                if self.s4 is None:
                    raise NotReady("S4: Verifier는 M3 (verify/)")
                results = self.s4(fb, candidates, hypotheses)

            return PipelineResult("ok", detections=results, timing=self.timer.summary())
        except NotReady as e:
            return PipelineResult("not_ready", timing=self.timer.summary(),
                                  detections=[], degraded=False) if False else \
                   PipelineResult(f"not_ready: {e}", timing=self.timer.summary())
