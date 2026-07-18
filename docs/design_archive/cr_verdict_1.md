# 간과점 실재성 검증 결과 — 코드·문서 대조 (Shape6D 리포 전수 확인)

검증 방법: shape6d/ 전 모듈, tests/ 전건, eval/, config/pipeline.yaml, docs 01~04(HTML→텍스트 변환 후 전문 확인), docs/assets_04/stats.json, docs/design_archive/design_training.md를 실제로 열어 대조. 산술은 재계산.

## A. 임무 지정 4개 초점

### A-1. 멀티 인스턴스 — "S2 k_inst=3 설계 존재 여부"
**설계는 존재한다. "사실상 미설계"(gaps-algo §1 표제)는 과장.**
- k_inst=3 설계 실재 근거: `shape6d/config/pipeline.yaml:38` `s2.k_inst: 3`, 03 §5.3 "인스턴스 NMS(IoU 0.5) → top-3 인스턴스를 S3로", 03 §6.6 "엔진을 B=3 정적 배치로 빌드"(검증 [상-6] 반영), `pipeline.yaml:58` `instance_batch: 3`, design_archive/verdict_1.md [상-6]에서 ×3 곱셈이 예산에 반영된 이력까지 있음. 접촉 병합도 03 §4.3에 "3중 방어(multimask 3후보 / try_split / 원본 통과→S2 최종 방어)"로 설계돼 있음.
- 그러나 **1-A의 구체 메커니즘(병합→크기 상한 하드 리젝→전량 소실)은 v0-geo에서 REAL**: ① try_split은 `prompt_gen.py:107` `if self.size_gate and ...`로 size_gate 주입 시에만 활성인데 기본값 None(`prompt_gen.py:17`)이고, `tests/test_e2e_geo.py:100`·`eval/synthetic_report/make_report.py:127` 모두 미주입, **yaml s1 섹션에도 size_gate 키 자체가 없음**(03 §4.2는 [0.2,2.0]×D 주입을 규정). ② 3중 방어 중 multimask는 M1 미구현, "원본 통과→S2 최종 방어"는 실제로는 `size_gate.py:38` 상한 하드 리젝이라 방어가 아니라 소실. ③ 접촉 병합 테스트 0건 맞음 — 단 "다중 인스턴스 테스트가 하나도 없다"는 부정확: `test_prompt_gen.py:26-31`이 분리된 2물체 클러스터링은 검증(접촉/E2E 멀티 인스턴스는 0건). ④ 04 §7이 "접촉·적재 병합 시나리오 미검증"을 이미 자인.
- **판정: 1-A REAL(단 '미설계'→'설계 존재·미배선·미검증'으로 정정). 심각도: 적재/나란히 운용 포함 시 상, 현 검증 범위(빈 팔레트 단독) 기준 중.** 1-B(물체 종 O 스케일링 예산 부재) REAL — 03 §8·§5.2 복잡도에 O 차원 없음 확인, 중 유지(1종 운용이면 하). 1-C(종 판별 미검증) **EXAGGERATED** — score_fusion은 M1 계획으로 명시(`pipeline.py:42`)이고 04 §7이 "단일 물체 종·유사 형상 혼입 미검증(M0-3 대상)"으로 이미 등재. 잔여 실체는 "M0-3에 이종 판별 매트릭스 구체 정의 부재"(하~중).

### A-2. 가설 전부 ICP 레이턴시 — 03 §8 수치로 재계산
**REAL, 심각도 상 유지.**
- §8 정본: S4 공칭 40~60ms/인스턴스(Orin), 최악 80(+재시도 25), 합계 450~500ms. 이 수치의 전제는 구판 §7.1(가설 3개 스플랫 스코어 + **best 1개만** ICP iter 5~10(§7.2) + UNCERTAIN 시 1회 재시도).
- v0-geo 실코드: `verifier.py:96-99` 가설 전부 평가, `icp.py:75-82` 스케줄 (6+4+4+4)=최대 18 iter + 최종 association 1회. 일량 비율 = (5가설×18iter)/(1가설×5~10iter) ≈ **9~18×** (원 분석의 "5~9×"보다 오히려 큼). 1단계 τ=0.25D에서 win 미캡 값 29.8→캡 15, iter당 (2·15+1)²=961 오프셋 루프(`icp.py:57`) — launch-bound 경고(§8 자체 계수)와 정합.
- 실측 재대조: stats.json s4 = **757~1,589ms, 평균 1,031ms**(원 분석의 "813~1,193ms"는 범위 오기 — 방향은 동일). §8 40~60ms를 일량비로 단순 스케일해도 Orin 인스턴스당 수백 ms → 인스턴스 3개면 S4만으로 적분 T 포함 1s SLA 불성립.
- 04 §5가 "~1.1s, TRT 이식 대상"으로 인지는 했으나 **§8 예산표를 (k, iter, win) 기준으로 재작성한 문서가 없고, 03 §11의 "02 문서 반영 필요 델타" 목록에도 없음**. → 예산 문서-구현 정합성 결함으로 REAL 상.

### A-3. BOP 평가 프로토콜 문제
**REAL, 심각도 상 유지.**
- `eval/bop/README.md`·`eval/contamination/README.md`·`eval/latency/README.md` = 각 1줄, 합계 3줄 스텁(정확).
- 02 목표 게이트 "BOP-Classic-Core AR ≥67 / 재도장 낙폭 ≤3pt / M2-C AR ≥68" 실재 확인. BOP depth를 희소화해 넣는 프로토콜(sparsified-BOP) 정의는 01~04·design_archive 어디에도 없음(03 §9.4 "10k BOP 미니"도 밀집 기준). 직립 뷰 마스크 의무화(03 §11 델타 ③)와 BOP 임의 자세의 충돌, v0-geo 역할 재정의(04 §9) 후에도 M2 게이트 수치 미개정 — 전부 사실.
- 단서: M0-1(SAM-6D 재현)은 원래 계획된 미착수 작업이므로 "스텁" 자체는 일정 문제이고, 간과의 실체는 **"Shape6D를 BOP에서 어떤 입력 규약으로 평가할지 정의가 0"**이라는 점. 이 부분은 정확한 지적.

### A-4. 문서-코드 불일치 주장(gaps-product 5-2 표)의 정확성 — 전건 대조

| # | 주장 | 대조 결과 | 판정 |
|---|---|---|---|
| a | free_viol 0.05/0.15 삼중 불일치 | yaml:69=0.05, `verifier.py:22`=0.15, `verifier.py:3` docstring=0.05, 03 §7.3 본문 0.05 vs §11 델타 0.15 — 전부 확인 | REAL, 단 심각도 **상→중** (03 §11이 "코드가 정본"을 선언했고 런타임 동작은 정본과 일치 — 위생 문제) |
| b | stride/X_verify | yaml:66 stride 2 vs `scorer.py:82` stride 4, `scorer.py:84` docstring "~1024pt" vs 04 결함 6 "2048 미만 금지" — 확인. 03 §11 내부에도 "stride 4(X_verify 1024pt)"와 "2048 미만 금지"가 공존 | REAL 중 |
| c | ICP 파라미터 | yaml icp_iters 10/tau 0.02 구판 vs `icp.py:75-82` 스케줄 — 확인, §11 델타에는 반영됨 | REAL 하 |
| d | UNCERTAIN 재시도 미구현 | `verifier.py:4` docstring이 미구현 기능 서술, icp_iters_retry 소비처 grep 0건 — 확인 | REAL 중 |
| e | floor_prior 미구현 | grep floor → np.floor뿐 — 확인 | REAL 중 |
| f | TDF voxel σ 하한 미반영 | `templates.py:13-14` 48³ 고정, voxel=1.2D/47≈0.0255D, §2.6 max(0.025D, σ) 규약의 σ항 부재 — 확인. D<0.31m(σ8mm)에서 위반 잠복 | REAL 중(팔레트는 무영향) |
| g | σ 0.008 vs 0.005 | `test_e2e_geo.py:24` 확인 | REAL 중 |
| h | k 3 vs 5 | `make_report.py:142` k=5, yaml topk 3, §11 "k≤5" 문서화 확인 | REAL 하 |
| i | S3 자리 | §11 델타 반영 완료 — 주장대로 위생 | 정확 |
| j | 42뷰 출처 | `templates.py:19-40` 자체 icosphere 생성, 03 §3.2는 "SAM-6D npy 재사용" — 확인 | REAL 하 |

**5-2 표의 파일:라인 인용은 전건 정확. 5-1("죽은 설정 파일")은 REAL이나 한 곳 오류: "yaml을 읽는 코드 0줄"은 부정확 — `tests/test_pipeline.py:20-21`이 로드함(단 값 존재 assert 전용, 프로덕션 주입 0은 맞음). 심각도는 상→중 권고(gaps-algo #7 스스로도 중으로 매김 — 두 분석 간 자기모순).**

## B. gaps-algo 잔여 항목 판정

| # | 항목 | 판정 | 근거 / 재평가 |
|---|---|---|---|
| 3-A | 적재·랩핑 미대응 | **REAL(부분 정정)** | 랩핑/수축필름 언급 0회 확인. 단 "적재 언급 0회"는 오류 — 04 §7 "접촉·적재 병합 시나리오 미검증" 자인 존재. 운용 요구('적재 팔레트 인식 여부')가 스펙 문서에 미확정인 것은 사실 → **상 유지** (자인은 있으나 대응 설계·결정 항목 등재가 없음) |
| 3-B | C2 합성 vs C4 실물 | REAL | `test_e2e_geo.py:31` 런너 3열=C2, 03 §1.4e "T11 ≈ C4" 자인, C4 하네스 부재 — **중** |
| 4-A | θ 축퇴 | **REAL(심각도 분해)** | `depth_match.py:98-99` 뷰당 θ argmax 1개 → topk에 θ 다양성 0 구조 확인. 04 §7이 "in-plane 후보 다양화" 필요를 자인하나 03 수정 대장 미등재 확인. **리콜/재촬영 축은 상(플립 실증), 90° 오수락 축은 미실증 추정 → 종합 중~상** |
| 4-B | grazing·TDF voxel | REAL | voxel 39.9mm > 데크 30mm 산술 확인 — **하~중** (coarse 전용, 10시행 s2 0.56~0.80으로 실증상 무해) |
| 4-C | 30~100pt argmax | REAL | stats n_obs 488~1,177만 검증, n_min=30과 갭 — **중** |
| 5 | 진동·간섭·직사광 | REAL | `frame_bundle.py:40` deskew v1 미사용, 정지 판정 게이트 부재, RFQ ①~⑥(03 §1.4b)에 간섭·조도 없음 — **중** |
| 6-A | 오포즈 카탈로그 | REAL | `test_verify.py:92` z−6cm 1종뿐 확인 — **중** (횡 스냅·90°는 미실증 추정) |
| 6-B | 보정셋 도메인 | REAL | 03 §7.3 "BOP val(T-LESS·ITODD)" 명시, 희소 보정셋 출처 미정의, v0-geo s2_sem 항상 0(`verifier.py:53`, cand.scores에 sem 없음) — **상 유지** |
| 6-C | 재시도 공허 | REAL | 5-2d와 동일 사안 — **중** |
| 8 | CI 규약 위반 | REAL | `test_e2e_geo.py:117` view_mask 미전달·`:129` X_verify=1024 — 04 결함 1·6 의무화 위반 확인. 규약 적용은 make_report(:140, :337)뿐 — **중** |
| 9 | 재적분 자기모순 | REAL | yaml:21 vs 03 §1.4b "T↑→노이즈만 감소" — 트리거(포인트 부족)를 조치(T 증액)가 해소 못함 확인 — **중** |
| 10 | RANSAC 상판 오제거 | REAL | `prompt_gen.py:41` 방어가 15% 임계뿐, 법선·높이 사전 없음 — **중** (시나리오는 미실증 추정) |
| 11 | 5m pitch/roll | REAL | 03 §1.4e 0.9°@3m/1.5°@5m 자인 + floor prior 미구현 — **중** |
| 12 | CAD-실물 공차 | REAL | §2.6 σ_lidar 단일 주입뿐, σ_model 부재 확인 — **중** |
| 13 | 대칭 과검출 무방비 | REAL | 폴백 τ=24.2mm 산술 확인(`symmetry.py:60-61`, 04 fig1 캡션 일치), dedupe가 등가 병합으로 정답 삭제 가능(`symmetry_eval.py:74-87`), MAX_GROUP=24 무경고(`symmetry.py:115`), 승인 게이트 규정 없음 — **중** |
| 15 win캡 | ICP win 캡 | **EXAGGERATED** | 산술은 정확(도달 0.197m < coarse 평균 235.2mm — 재계산 일치)하나 **stats.json이 반례**: coarse 병진 570~589mm인 3시행 전부 mm급 수렴·ACCEPT. 다단 반복으로 도달이 누적되므로 "여유가 없다"는 과장 — 하 |
| 15 NMS/SizeGate/σ | 중복 검출·구제 편향·σ 5vs8 | REAL | 각각 코드 확인 — 하 / 하 / (2-1 통합) |

## C. gaps-product 잔여 항목 판정

| # | 항목 | 판정 | 근거 / 재평가 |
|---|---|---|---|
| 1-1 | fixed_grid 증강 부재 | REAL | design_training.md:466-468 pattern_probs 3종 확인(fixed_grid 없음), 03 §9.2 동일, `test_e2e_geo.py:73` 행 1/5 확인 — **중** (M2-A2 착수 전 문서 수정으로 해소 가능; 방치 시 상) |
| 1-2 | 팔레트 학습셋 부재 | REAL | design_training.md:30-31 GSO/ShapeNet 확인, §1.4e는 **평가** 셋만 규정 — **중** (M2 역할이 '팔레트 coarse 강건화'로 재정의된 만큼 M2-B 전 반영 필요) |
| 1-3 | M2 게이트 미갱신 | REAL | 04 §9 재정의 vs 02 M2-C AR≥68 불변, 03 §11 델타 목록에 게이트 항목 없음 — **중** |
| 1-4 | 대칭 라벨 오염 | REAL | 03 §9.1 "~5만 종 일괄", 검증은 §3.3 합성 3종뿐, 신뢰도 스코어 캐시 필드 없음(`cache.py:_SCHEMA`) — **중** |
| 1-5 | ShapeNet 라이선스 | **EXAGGERATED** | D4에 "M2-C 본학습 전 확정" 기한 명시돼 있음. 잔여는 "다운로드(M2-A1) 전으로 앞당기라"는 개선 제안 수준 — 하 |
| 2-1 | σ 3중 불일치 | REAL | 0.005/0.008/15mm 전건 확인(§1.4d "σ_eff=30/√4=15mm"). 04 결론이 최낙관치 기반 + CI가 5mm 유지 — **상 유지** (σ 스윕 부재가 실질 리스크; 04 §7 한계 명시로 완화되나 감도 데이터 0) |
| 2-2 | 실측→파라미터 매핑 절반 | REAL | §2.6 σ 매핑만 존재, 거리 바이어스 보정 절차·포맷/delta_edge/드롭률 갱신 규칙 부재 확인 — **중** |
| 2-3 | FPN 플랜B 부재 | REAL(소폭 조정) | 리스크 자체는 RFQ ④로 등재됨 — 부재한 것은 2성분 노이즈 감도 분석과 플랜 B — **중** |
| 2-4 | frame_obs 품질 필터 | REAL | `verifier.py:41` 필터 없음, `make_report.py:148-151` valid_mask만 사용, §2.2 규약에 free-space 관측 필터 미정의 확인. 부기: v0-geo 하네스는 ICP source(클러스터 pts)에도 EDGE_MIXED 제외 미적용 — **중** |
| 2-5 | 캘리브 0.05° 검증 수단 | REAL | §2.3 합격 기준 부재 + **추가 발견**: §2.3 예산표는 여전히 "회전 ≤0.1°"로 §1.4c(0.05°)·yaml과 불일치 — **중** |
| 3-1 | 보정 도메인+현장 루프 | REAL | 6-B와 동일 사안 + 현장 약라벨 루프 문서 부재 확인, s3_match 의미 변동(`verifier.py:53` h.score 주입) 확인 — **상 유지** |
| 3-2 | view_mask 소유자 부재 | REAL | `cache.py:_SCHEMA`에 view_mask 없음, manifest에 pose_prior 없음, `cli.py`에 --upright 없음, CI 미적용(#8)이 휘발성의 실증 — **중** |
| 3-3 | 사이클타임 회계 | REAL(소폭 조정) | §8이 예외 경로 1.5s 트랙·degraded 로깅은 정의 — 부재한 것은 p95 합산 지표·재시도 상태기계 — **하~중** |
| 3-4 | 버전 배선 부재 | REAL | VerifyResult.diag에 config_hash/calibrator_version 스탬프 없음 확인 — **중** |
| 3-5 | 온보딩 UX | REAL | `cli.py:64-69` json.loads 직행, watertight 정량 경고 없음 등 전건 확인 — **하** |
| 4-1 | eval 스텁 | REAL | 각 1줄 확인. "최장 경로 = M0 하네스" 판단도 M2-B 의존 구조상 타당 — **상 유지** |
| 4-2 | 희소 도메인 벤치 부재 | REAL | A-3과 동일 — **상** |
| 4-3 | 통계 검정력 | REAL | 0/10의 95% CI 상한 25.9% 재계산 일치, CI 게이트 대표 1시행 확인 — **중** |
| 4-4 | 미커버 축 목록 | REAL | try_split 테스트 부재 확인(test_prompt_gen은 분리 2물체만) 등 — **중** |
| 4-5 | sym-aware 자기참조 | REAL | make_report가 검출 결과(sym_h)로 채점 확인, GT 대칭군 별도 명시 없음 — **하** |

## D. 원 분석 자체의 사실 오류(교정 목록)
1. **"S4 실측 813~1,193ms"** → 실제 757~1,589ms(평균 1,031ms). 방향 불변.
2. **"pipeline.yaml 로드 코드 0건/0줄"** → `tests/test_pipeline.py:20-21`이 로드(검증 전용). "프로덕션 주입 0"으로 정정.
3. **"멀티 인스턴스 테스트 리포에 하나도 없다"** → 분리 2물체 클러스터링 테스트는 존재(`test_prompt_gen.py:26`). 접촉 병합·멀티 인스턴스 E2E는 0건 맞음.
4. **"01~04에 적재 언급 0회"** → 04 §7 "접촉·적재 병합 시나리오 미검증" 존재. 랩핑은 0회 맞음.
5. **"멀티 인스턴스 사실상 미설계"** → k_inst=3·B=3·접촉 3중 방어 설계 실재. 정확한 상태는 "설계 존재, v0-geo 미배선(try_split)·M1 미구현(multimask)·미검증".
6. **win 캡 "여유가 없다"** → stats.json 반례(coarse 570~589mm 3건 전부 회복)로 과장.
7. gaps-product 5-1(상) vs gaps-algo #7(중)의 **심각도 자기모순** — 중이 타당.

## 종합
- **FALSE_GAP(완전 오류): 0건.** 파일:라인 인용 정확도가 매우 높음.
- **EXAGGERATED: 3건** — 1-C(종 판별: 문서 자인+M1 계획 존재), 1-5(라이선스: 기한 존재), 15-win캡(실측 반례).
- **나머지 전건 REAL.** 심각도 '상' 유지 항목: 가설 전부 ICP vs §8 예산(A-2), BOP/희소 벤치 프로토콜 부재(A-3, 4-1·4-2), 적재·랩핑 운용 공백(3-A), 보정셋 도메인 불일치(6-B/3-1), σ 3중 불일치(2-1), 접촉 병합 소실(1-A, 운용 조건부). '상→중' 하향: free_viol 삼중 불일치(5-2a), yaml 미주입(5-1), fixed_grid 증강(1-1, 착수 전 수정 가능 기준), 4-A 오수락 축(미실증 부분).