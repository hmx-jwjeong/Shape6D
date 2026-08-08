# Shape6D

제로샷·형상 우선(shape-primary) 산업용 6D 포즈 추정 — 희소 LiDAR + RGB 1280×800, 타깃 Jetson Orin NX < 1s.

SAM-6D의 2단 포인트 매칭 구조를 승계하되, 특징을 RGB 주도 → 기하 주도로 역전한 재설계 구현.

## 설계 문서 (정본)

| 문서 | 내용 |
|---|---|
| [docs/20_final_design_plan.html](docs/20_final_design_plan.html) | **★ 최종 확정 설계·실행 계획 (정본 진입점)** — 슬롯 확정표·네트워크/데이터 명세·Phase 0~C·확정/미결 대장. 충돌 시 이 문서 우선 |
| [docs/01_stage_design.html](docs/01_stage_design.html) | 스테이지 정의(S0~S4) + 후보 알고리즘 선정 근거 |
| [docs/02_development_plan.html](docs/02_development_plan.html) | M0~M5 개발 계획, 게이트, 일정 |
| [docs/03_implementation_design.html](docs/03_implementation_design.html) | **구현 상세 정본** — FrameBundle/캐시 스키마/알고리즘/임계값. 코드와 불일치 시 이 문서 기준 |
| [docs/04_synthetic_validation_report.html](docs/04_synthetic_validation_report.html) | v0-geo 합성 검증 보고서 (10시행, 이미지·통계) |
| [docs/05_concept_review.html](docs/05_concept_review.html) | 컨셉 6기둥 정리 · 교차 검증된 간과점 · 기존 방식 비교 |
| [docs/06_external_eval_visual_report.html](docs/06_external_eval_visual_report.html) | MegaPose 외부 평가 시각 자료집 (케이스 패널·3D·집계) |
| [docs/07_uam_field_log_eval_report.html](docs/07_uam_field_log_eval_report.html) | UAM 전시장 로봇 로그 학습 전(v0-geo) 평가 — 70샘플, 판정 역전 분석 |
| [docs/08_training_plan.html](docs/08_training_plan.html) | **학습 진행 계획** — Track 0/1/2, 증강 5종, 검증기 재설계, 일정 10주. 02의 M2·M3 개정판 |
| [docs/09_track0_calib_eval_report.html](docs/09_track0_calib_eval_report.html) | Track 0 + 보정 1차 학습 결과 — 시험세션 정선택 95%·오수락 0, free_viol 부호 반전 실증 |
| [docs/10_network_design_and_critique.html](docs/10_network_design_and_critique.html) | **Shape6D-PEM 설계 해설 + 2026 SOTA 대조 + 비판** — BOP 2024/25 리더보드 대조, 구현 실사(신경망 0줄), 권고 10건 |
| [docs/11_sota_approach_comparison.html](docs/11_sota_approach_comparison.html) | **Co-op·FreeZe 대조** — 논문 원문 실사. 문제 정식화(재투영 vs 3D 잔차)·네트워크·손실·신뢰도 축별 비교 + 05·10 정정 5건 |
| [docs/12_backbone_survey_dino.html](docs/12_backbone_survey_dino.html) | **백본 선정 서베이 (DINO 계열)** — 5개 자리 분해, DINOv2/v3·ConvNeXt·RADIO·Sonata/dGeDi 후보표, 감점요인 7건, 권고 + E-BB 실험 설계 |
| [docs/13_backbone_pilot_report.html](docs/13_backbone_pilot_report.html) | **E-BB 백본 파일럿 실측** — 예산 검증(오차 2%), stem 점유율 실측(이항 가정 11배 오차), a0/a1/a2+DINOv3 통제 비교 — DINOv3 절단 사용은 정본과 동률(−0.0007±0.0012), ImageNet보다 −0.025 |
| [docs/14_encoder_pilot_paper.html](docs/14_encoder_pilot_paper.html) | **인코더 통제 비교 (논문형)** — 데이터·과제·통제 설계 전문, 그림 6종·표 4종. 운용 밀도에서는 포화, **163pt에서 DINOv3가 오답률 절반**. 13의 수치를 대체 |
| [docs/15_network_and_training_scope.html](docs/15_network_and_training_scope.html) | **PEM 구조도 + 현재 학습 범위 + 영향 예측** — 8M 중 1.33M(17%)만 갱신, 매칭부 5.5M 미구현. 미학습 물체 top-1 0.99→0.15 실증 |
| [docs/16_pretraining_comparison.html](docs/16_pretraining_comparison.html) | **사전학습 대조** — SAM 2·MegaPose·FoundationPose·Co-op·SAM-6D vs 우리. 전부 "비교기"를 학습한다는 공통 구조와 그 함의 |
| [docs/17_full_line_redesign_and_training_plan.html](docs/17_full_line_redesign_and_training_plan.html) | **전체 입출력 라인 재구성 v3** — SAM3/DINOv3 제안의 3라운드 비판 검토, SAM-6D 승계 확정, 네트워크 명세 + Phase 0/A/B/C 학습 계획(~10주) |
| [docs/18_module_networks_and_pretraining_plan.html](docs/18_module_networks_and_pretraining_plan.html) | **슬롯별 네트워크·사전학습 결정** — 13개 슬롯 조달표, 학습 5슬롯, 데이터셋 인벤토리·라이선스 매트릭스, Phase별 데이터 흐름 (UAM 비사용) |
| [docs/19_license_assessment.html](docs/19_license_assessment.html) | **라이선스 실사** — 18 조달안 상업 판정. 빨강 5건 교체(FB가중치 NC→timm Apache, SAM-6D·GeDi 무라이선스→클린룸·자체학습, CroCo·EdgeSAM NC 제외), 교체 후 기본 경로 전부 초록 |
| [docs/21_phase_a_retry_and_optimization.html](docs/21_phase_a_retry_and_optimization.html) | **Phase A 재시도 보고** — 5회 안정화 이력(MegaPose 스케일 발견 포함), 첫 제로샷 신호(≤30° 10→25.9%), 배치/Muon 최적화 체인, 0.1°/1mm 타당성 [D-10] |
| [docs/07_loaded_wrapped_pallet_survey.html](docs/07_loaded_wrapped_pallet_survey.html) | 적재·랩핑 팔레트 인식 방법 서베이 (딥리서치, 검증 24건) + 하부 밴드 모드 설계 시사점 |
| [docs/08_mr6d_eval_report.html](docs/08_mr6d_eval_report.html) | MR6D 실측 유로팔레트 평가 (150프레임 — 이전성 확인, 적재 장면 붕괴 정량화) |
| [docs/09_industrial_robustness_plan.html](docs/09_industrial_robustness_plan.html) | 산업 강인화 개선 계획 — 34제안 적대 검증 종합, 4단계(정식 채택·밴드 모드·레이턴시·데이터) 로드맵 |
| docs/design_archive/ | 영역별 설계 원문 5건 + 교차 검증 리포트 2건 |

※ 번호 07–09는 두 작업 라인(학습 라인 / 팔레트 실측 라인)에서 중복 사용됨 — 파일명이 달라 충돌은 없으나 재번호 논의 필요.

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
