# Shape6D

제로샷·형상 우선(shape-primary) 산업용 6D 포즈 추정 — 희소 LiDAR + RGB 1280×800, 타깃 Jetson Orin NX < 1s.

SAM-6D의 2단 포인트 매칭 구조를 승계하되, 특징을 RGB 주도 → 기하 주도로 역전한 재설계 구현.

## 설계 문서 (정본)

| 문서 | 내용 |
|---|---|
| [docs/01_stage_design.html](docs/01_stage_design.html) | 스테이지 정의(S0~S4) + 후보 알고리즘 선정 근거 |
| [docs/02_development_plan.html](docs/02_development_plan.html) | M0~M5 개발 계획, 게이트, 일정 |
| [docs/03_implementation_design.html](docs/03_implementation_design.html) | **구현 상세 정본** — FrameBundle/캐시 스키마/알고리즘/임계값. 코드와 불일치 시 이 문서 기준 |
| [docs/04_synthetic_validation_report.html](docs/04_synthetic_validation_report.html) | v0-geo 합성 검증 보고서 (10시행, 이미지·통계) |
| [docs/05_concept_review.html](docs/05_concept_review.html) | 컨셉 6기둥 정리 · 교차 검증된 간과점 · 기존 방식 비교 |
| [docs/06_external_eval_visual_report.html](docs/06_external_eval_visual_report.html) | MegaPose 외부 평가 시각 자료집 (케이스 패널·3D·집계) |
| [docs/07_loaded_wrapped_pallet_survey.html](docs/07_loaded_wrapped_pallet_survey.html) | 적재·랩핑 팔레트 인식 방법 서베이 (딥리서치, 검증 24건) + 하부 밴드 모드 설계 시사점 |
| docs/design_archive/ | 영역별 설계 원문 5건 + 교차 검증 리포트 2건 |

## 확정 전제 (2026-07-18)

- 센서: SOS Lab **ML-X(80)** 후보 (RFQ 회신 대기), 상시 3m·커버 2~5m, 요구 위치 10mm·회전 ≤1°
- 대상: 팔레트급 대칭 물체 포함. **대칭 등가 포즈는 전부 정답 취급** (기본값 — 특정 면 구분 필요 시 재론)
- depth = LiDAR 유효 포인트만 (희소·고정밀), 외형(도장·오염)은 신뢰하지 않음 (A1)
- 학습: RTX PRO 6000 Blackwell 96GB ×1, Ubuntu (16코어/64GB/NVMe 2TB)

## 설치 / 테스트

```bash
uv venv && uv pip install -e ".[dev]"
uv run pytest
```

## 구조

```
shape6d/common/      FrameBundle(정본 §2.2), 타입, 프로파일러
shape6d/onboarding/  S0: 샘플링·대칭검출(§3.3)·템플릿·캐시(§2.4)·CLI
shape6d/proposal/    S1: LiDAR 프롬프트/클러스터 (EViT-SAM 래퍼는 M1)
shape6d/identify/    S2: 크기 게이팅·TDF 정합 (DINOv2는 M1)
shape6d/pose/        S3: Shape6D-PEM (M2에서 포팅·학습)
shape6d/verify/      S4: ICP·잔차·신뢰도 (M3)
shape6d/pipeline.py  S1→S4 오케스트레이션 + 스테이지 타이머
tests/               기반 모듈 단위테스트 (대칭 검출 회귀 포함)
```

## v0-geo — 무학습 기하 파이프라인 (동작 확인됨)

S2 TDF 정합의 argmax(뷰, in-plane 각, jitter)가 곧 coarse 포즈라는 점을 이용한
**학습 없는 전체 사슬**: 온보딩 → 클러스터 → 크기 게이팅 → TDF 정합 → coarse 포즈
(`pose/template_init.py`) → projective ICP → 신뢰도 판정.

E2E 합성 검증 (`tests/test_e2e_geo.py`, 팔레트 1100×1100@3m·희소 LiDAR 1200pt·σ5mm):
**위치 4.9mm · 회전 0.71° — 요구 스펙(10mm/1°) 통과.** PEM(M2) 학습 전 베이스라인이자 영구 비학습 폴백.

## 마일스톤 현황

- [x] M0-5 리포 부트스트랩 (기반 모듈 + 테스트 32개)
- [x] v0-geo 파이프라인 (S0 온보딩 CLI 완결 포함 — point-splat 렌더로 GL 무의존)
- [ ] M0-1 SAM-6D BOP 재현 (Ubuntu PC 필요)
- [ ] M0-3 오염·재도장 프로토콜 (Ubuntu PC)
- [ ] M1 무재학습 v0 (EViT-SAM·DINOv2 통합 — torch 필요)
