# Step 1 보조 통계 (`scripts/step1_stats.py`)

`results/step1/<model>/<original|augmented>/eval/<eval_name>/predictions.csv`를 읽어 신뢰구간과 조건 간 차이를 `results/stats/`에 저장한다. 기존 코드와 결과는 수정하지 않는다. pandas, numpy만 사용한다.

## 실행

```bash
python scripts/step1_stats.py \
    --clean-input <test> --obfuscated-input <obfuscated_test> \
    [--kg-clean-input <kg_test> --kg-obfuscated-input <kg_obfuscated_test>]
```

| 인자 | 기본값 |
| --- | --- |
| `--results-root` | `results/step1` |
| `--models`, `--trainings` | `koelectra,mdeberta` / `original,augmented` |
| `--train-techniques` | `yamin_swap:0.7,symbol_insert:0.3` |
| `--n-boot`, `--seed` | `2000`, `20261002` |
| `--output-dir` | `results/stats` |
| `--skip-crosscheck` | 기존 `metrics.json`, analysis csv와의 대조 생략 |

## 입력 검증

아래 중 하나라도 맞지 않으면 목록을 출력하고 종료한다.

- `predictions.csv`와 평가 입력 파일의 `id`: 빠진 행, 중복 id, 여분 행
- 같은 `id`의 `label` (obfuscated는 `seed_id`, `technique`, `intensity`, `changed` 포함)
- obfuscated의 `seed_id`가 clean에 없음, 변형의 `label`이 원문과 다름
- 다시 계산한 confusion count가 `metrics.json`, `obfuscated_overall.csv`(`changed_only`)와 다름

## 지표

- 정의는 `evaluate.py`와 같다: label 1 = Attack, `zero_division=0`, 분모 0이면 0.0, FPR = FP/(FP+TN).
- clean은 전체 행, obfuscated는 `changed=true` 행. 원문 행은 `seed_id = id`.

## 신뢰구간 (95%)

| 대상 | 방법 |
| --- | --- |
| clean Recall, FPR | Wilson |
| obfuscated Recall, FPR | `seed_id` 클러스터 부트스트랩 |
| F1, Precision | `seed_id` 클러스터 부트스트랩 |
| Augmented − Original 차이 | 같은 `id` 결합, 같은 클러스터 재표본 |
| 원문 정답 공격 중 변형도 정답인 비율 | `seed_id` 클러스터 부트스트랩 |

- 부트스트랩: 반복 2000, 시드 고정, 백분위 구간. 같은 원문의 변형은 독립이 아니므로 `seed_id`를 재표본 단위로 한다.
- 성공이 0건 또는 전체: 구간 폭이 0이 되므로 `seed_id` 수를 n으로 한 Wilson을 쓴다. `ci_method = wilson_cluster_fallback`, 표에는 †.
- 재표본 결과가 모두 같음(F1, Precision, 차이): 구간을 만들지 않는다. `ci_method = not_computable`, 표에는 `산출 불가`.
- 해당 label 행이 없는 칸(분모 0)은 표에 `–`로 표시한다. csv의 값은 `evaluate.py`와 같이 0.0이다.
- 참고용 Wilson 구간은 `*_wilson_lo/hi` 열에 남긴다. 표에는 기본 구간만 표시한다.

## 분해

- 기법 그룹: 학습 기법·같은 강도 / 학습 기법·다른 강도 / 나머지 기법
- `source`별 Recall, FPR
- 정상 문장(label 0)의 난독화 FPR (전체, 기법별)

## 출력 (`results/stats/`)

`condition_metrics.csv`, `paired_original_vs_augmented.csv`, `paired_original_vs_variant.csv`, `technique_groups.csv`, `source_metrics.csv`, `benign_obfuscation_fpr.csv`, `step1_stats_tables.md`, `run_info.json`

구간 방식 열: `recall_ci_method`, `fpr_ci_method`, `f1_ci_method`, `precision_ci_method`. 차이, 유지율, 정상 오탐률은 `ci_method`. 값은 `wilson`, `cluster_bootstrap`, `wilson_cluster_fallback`, `not_computable`.

## 테스트

```bash
python tests/test_step1_stats.py    # 또는 pytest tests/
```

`evaluate.py`, `analyze_obfuscated_eval.py`로 만든 `predictions.csv`와 `metrics.json`, analysis csv의 점추정이 일치하는지, `changed` 정규화가 기존 함수와 같은지, 입력 불일치 시 종료하는지, 퇴화 사례(전부 맞힘, 오탐 0건, 차이 0)를 확인한다. `torch`, `transformers`가 없으면 `evaluate.py`를 쓰는 테스트는 건너뛴다.
