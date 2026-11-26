# BatchPace 실험 기록

[← 프로젝트 README](../README.md) · [초기 40 + 40 benchmark guide](../README_BENCHMARK_SUITE.md) · [로컬 추론 환경](../HANDOFF.md)

현재 저장소의 코드, 환경 manifest, seed별 metrics와 summary를 기준으로 정리한 실험 안내입니다. 초기 distribution-shift 실험 이후 stationary Beta, stationary LLM 세부 과목, stationary LLM 대분류 실험을 추가했습니다.

## 전체 실험 지도

| 실험군 | 환경 | Seed | 예산 | 정책 | 실행 기록 | 원시 결과 |
|---|---:|---:|---:|---:|---:|---|
| Synthetic shift | 40 | 10 | 8 | 7 | 22,400 | [synthetic_metrics.csv](../results/benchmark_suite/synthetic_metrics.csv) |
| LLM shift | 40 | 10 | 8 | 7 | 22,400 | [llm_metrics.csv](../results/benchmark_suite/llm_metrics.csv) |
| Stationary Beta | 40 | 10 | 8 | 6 | 19,200 | [metrics.csv](../results/stationary_beta_suite/metrics.csv) |
| Stationary LLM subjects | 40 | 10 | 8 | 7 | 22,400 | [metrics.csv](../results/stationary_llm_suite/metrics.csv) |
| Stationary LLM categories | 14 | 10 | 8 | 7 | 7,840 | [metrics.csv](../results/stationary_llm_broad_suite/metrics.csv) |
| **합계** | **174** | | | | **94,240** | 초기 Main Figure 1·2는 별도 |

각 실행은 24개 배치에 20개씩, 총 480개 질의 도착으로 구성됩니다. 공통 예산은 `0, 24, 48, 72, 96, 120, 144, 192`, calibration path 수는 200입니다. LLM 환경은 같은 1,500-query 캐시를 사용하므로 서로 독립된 174개 데이터셋으로 해석하지 않습니다.

## 1. Synthetic · distribution shift

**목적:** gain 분포와 배치 내 상관구조가 바뀔 때 전역 예산 배분 정책을 비교합니다.

`5 gain families × 4 shifts × 2 batch regimes = 40`개 환경입니다. 모든 환경에서 batch 12 이후 분포가 바뀝니다.

| Gain family | 변화 전 분포 |
|---|---|
| Sparse | 25% Beta(2, 5), 75%는 0 |
| Moderate | Beta(2.5, 4) |
| Bimodal | 35% Beta(6, 2), 65% Beta(1.5, 7) |
| Heavy-tail | Beta(0.75, 3.5) |
| Rare-jackpot | 8% Beta(12, 1.5), 92% Beta(0.5, 12) |

변화 종류는 `mean_up`, `mean_down`, `variance_up`, `tail_reversal`이며, 배치 구성은 `iid`와 `batch_bursty`입니다. 같은 family의 환경은 기본 분포를 공유하고 shift·regime으로 구분됩니다. 합성 realized gain은 `[0, 1]` 범위입니다.

[환경 manifest](../results/benchmark_suite/synthetic_environment_manifest.json) · [환경 구현](../src/benchmark_suite_environments.py) · [전체 80개 atlas](../output/pdf/batchpace_80_environment_atlas.pdf)

## 2. LLM-derived · distribution shift

**목적:** 실제 모델 correctness 차이에서 얻은 gain을 사용해 workload transition에 따른 라우팅 성능을 평가합니다.

MMLU-Pro calibration split의 과목별 평균 gain 차이로 10개 과목 쌍을 선택하고, 양방향 전환과 두 배치 구성을 적용합니다.

```text
10 category pairs × 2 directions × 2 regimes = 40 environments
batch 1–12: source category / batch 13–24: target category
```

IID는 해당 과목 pool에서 독립 추출합니다. Batch-bursty는 한 배치를 predicted-gain difficulty quartile 하나에서 추출해 구성합니다. 환경 선택에는 calibration gain을 사용하며 test gain을 사용하지 않습니다. Shift 위치와 전환 종류가 calibration protocol에 반영된 실험입니다.

LLM query의 gain은 다음과 같습니다.

```text
gain(q) = 1[strong correct] − 1[cheap correct]
       = +1  strong model만 정답
       =  0  두 모델의 correctness가 같음
       = −1  cheap model만 정답
```

[환경 manifest](../results/benchmark_suite/llm_environment_manifest.json) · [전체 summary](../results/benchmark_suite/all_summary.csv) · [선별 그림](../output/selected_llm_highlighted/selected_llm_2x4_ascending.pdf)

## 3. Synthetic · stationary Beta

**목적:** 시간에 따른 분포 변화와 batch-level latent 효과를 제거하고, 고정 분포의 모양에 따른 정책 차이를 확인합니다.

각 환경의 gain은 하나의 Beta(α, β)에서 IID로 생성합니다. Mixture, point mass, 중간 shift 없이 아래 조합을 전부 사용합니다.

```text
α ∈ {0.5, 1, 2, 4, 8}
β ∈ {0.5, 1, 2, 3, 5, 8, 12, 20}
5 × 8 = 40 environments
```

관측 점수는 gain에 평균 0인 Gaussian noise를 더한 뒤 `[0, 1]`로 clip합니다. Noise 표준편차는 `0.08 + 0.10 × (1 − |2 × gain − 1|)`입니다. 평균은 `α / (α + β)`이며, theoretical mean·variance가 환경 manifest와 결과에 기록됩니다.

BatchPace, Oracle BatchPace, Fixed Threshold, Equal Split, WISERouter-Offline, Robust Batch-Level의 **6개 정책**을 비교했습니다. Sequential Pacing은 이 실험군에서 실행하지 않았습니다.

[환경 manifest](../results/stationary_beta_suite/environment_manifest.json) · [summary](../results/stationary_beta_suite/summary.csv) · [전체 40개 atlas](../output/stationary_beta_40/stationary_beta_40_atlas.pdf) · [구현](../src/stationary_beta_environments.py)

## 4. LLM-derived · stationary subjects

**목적:** 세부 과목 하나를 유지한 채 실제 LLM gain에 대한 라우팅 정책을 비교합니다.

MMLU-Pro의 fine-grained `src` subject 중 split별 query pool 가용성을 기준으로 40개를 선택합니다. Test gain은 과목 선택에 사용하지 않습니다. 각 환경은 24개 배치 전체에서 동일한 subject를 사용하며, shift와 batch-bursty 구성이 없습니다.

각 batch는 split별 고정 subject pool에서 **복원추출**한 20개 질의로 구성됩니다. 따라서 480 arrivals는 서로 다른 480개 질문을 뜻하지 않습니다. 현재 40개 환경의 고유 test query pool 크기는 **3–43개**입니다. CI는 이 고정 pool을 다시 재생한 seed 간 변동을 나타내며 새로운 질문 모집단에 대한 불확실성을 모두 나타내지 않습니다.

원시 결과는 **7개 정책**을 보존하고 그림에는 BatchPace, Fixed Threshold, Equal Split, WISERouter-Offline, Robust Batch-Level의 5개를 표시합니다.

[실험별 README](../results/stationary_llm_suite/README.md) · [pool audit](../results/stationary_llm_suite/query_pool_audit.csv) · [환경 manifest](../results/stationary_llm_suite/environment_manifest.json) · [전체 40개 atlas](../output/stationary_llm_40/stationary_llm_40_atlas.pdf)

## 5. LLM-derived · stationary categories

**목적:** 세부 subject 대신 MMLU-Pro 대분류 단위로 고정 pool을 구성해 stationary 결과를 확인합니다.

대분류는 biology, business, chemistry, computer science, economics, engineering, health, history, law, math, other, philosophy, physics, psychology의 **14개**입니다. 각 환경은 같은 category에서 24개 배치 동안 IID 복원추출합니다.

고유 test query pool 크기는 **16–56개**입니다. 원시 결과에는 7개 정책이 있고, top-8 그림에는 stationary subjects와 같은 5개 정책을 표시합니다. 배치 12의 change point와 category transition은 없습니다.

[실험별 README](../results/stationary_llm_broad_suite/README.md) · [pool audit](../results/stationary_llm_broad_suite/query_pool_audit.csv) · [환경 manifest](../results/stationary_llm_broad_suite/environment_manifest.json) · [전체 summary](../results/stationary_llm_broad_suite/summary.csv)

## 초기 Main Figure 1·2

현재 5개 실험군과 설정이 다른 초기 통제·실제 LLM 실험도 저장되어 있습니다. 174개 환경과 94,240개 실행 수에는 포함하지 않았습니다.

| 실험 | 설계 | 산출물 |
|---|---|---|
| Main Figure 1 | episode당 1,000 queries, strong budget 50, 250 episodes; IID / bursty에서 predictor error와 batch size 변화 | [데이터 설명](../data/synthetic_main_figure_1/README.md) · [summary](../results/synthetic_main_figure_1/summary.json) · [PDF](../output/pdf/fig1_synthetic_controlled_study.pdf) |
| Main Figure 2 | 5개 train/calibration/test 분할, 각 20 order seeds, random / category-bursty; 예산 5%·10%·20% | [결과와 해석](../results/real_llm/README.md) · [10% 표](../results/real_llm/main_table_10pct.csv) · [PDF](../output/pdf/fig2_real_budget.pdf) |

Main Figure 2의 10% 예산에서 BatchPace의 cheap-only 대비 정확도 증가는 random **3.182 pp**, category-bursty **2.922 pp**입니다. Sequential은 각각 **3.162 pp**, **2.904 pp**로, 이 결과만으로 두 allocator 사이의 의미 있는 우위를 주장하기 어렵습니다. 이 초기 실험의 CI는 5개 split별 평균을 기준으로 계산했습니다.

## LLM 캐시와 프로토콜 구분

공통 모델은 cheap **Qwen3-0.6B-Q8_0**, strong **Qwen3-4B-Q4_K_M**이며, embedding은 **Qwen3-Embedding-0.6B-Q8_0, 1,024차원**입니다. MMLU-Pro 1,500개 질의를 사용하고 category-matched 5-shot prompt로 로컬 추론했습니다.

| 프로토콜 | 사용 실험 | 1,500개 질의 정확도 cheap / strong | 캐시 |
|---|---|---|---|
| 초기 deterministic constrained generation | Main Figure 2 | 22.93% / 40.67% | [기존 cache manifest](../data/real_llm/cache/manifest.json) |
| Constrained option-label scoring | LLM shift + stationary LLM 2개 실험군 | 14.07% / 36.07% | [cheap JSONL](../results/benchmark_suite/option_scores_qwen3_0.6b.jsonl) · [strong JSONL](../results/benchmark_suite/option_scores_qwen3_4b.jsonl) |

현재 option scoring은 grammar로 유효한 label을 제한하고 truncation sampler를 끈 상태에서 `log p(A | prompt), …, log p(J | prompt)`를 저장합니다. 최고 label likelihood를 답으로 사용하며, answer text를 파싱하지 않습니다. **Option 본문 전체 문자열의 conditional likelihood를 계산하는 실험은 아닙니다.**

현재 suite는 기존 Parquet에서 embedding을 읽고 correctness는 option-score JSONL에서 가져옵니다. 각 JSONL은 1,500개의 고유 `question_id`를 담습니다. Split seed와 원본 revision·hash는 [split manifest](../data/real_llm/splits/manifest.json)에 있습니다. 초기 generation과 현재 option-scoring 정확도는 서로 다른 프로토콜의 결과이므로 섞어 사용하지 않습니다.

## 정책 구현과 해석 범위

| 정책 | 구현 범위 |
|---|---|
| BatchPace | predicted gain과 position-aware batch Bellman table. 배치 내부 top-k와 미래 value의 합으로 k 결정 |
| Oracle BatchPace | 같은 online batch rule에 current true gain과 calibration true gain을 제공. 전체 test stream의 hindsight top-B와 구분 |
| Sequential Pacing | 같은 계열의 score·calibration에 query별 Bellman take/skip 적용. Stationary Beta에는 없음 |
| Fixed Threshold | calibration budget quantile을 threshold로 고정하고 global budget guard 적용 |
| Equal Split | 24개 배치에 균등 예산. 현재 예산은 모두 24의 배수 |
| WISERouter-Offline | offline adaptive LP를 비용 0/1의 두 action으로 축약. 16개 K-means context에서 train reward 추정. Online exploration 없음 |
| Robust Batch-Level | 두 모델·단위 incremental cost·capacity 비제약 설정. LLM은 100개 bootstrap refit의 10th-percentile lower bound; 합성은 Gaussian lower bound. 배치별 equal cap 적용 |

외부 baseline은 원 논문의 핵심 최적화를 반영한 구현이며, 저자 배포 코드나 전체 multi-model serving system의 재실행은 아닙니다. 현재 공통 정책 구현은 [benchmark_suite_policies.py](../src/benchmark_suite_policies.py)에 있습니다. 초기 Main Figure 2의 predictor·robust estimator 설정은 [해당 실험 README](../results/real_llm/README.md)에 별도로 기록했습니다.

## 결과 집계 방법

[프로젝트 README의 결과 표](../README.md#결과-요약)는 아래 원시 파일에서 실험군별로 `budget > 0`을 남긴 뒤 `method`별 `total_realized_gain`을 평균한 값입니다. 초기 benchmark는 `environment_type`으로 Synthetic과 LLM을 구분합니다.

| 실험군 | 원시 파일 | 환경·예산·정책별 평균 / CI |
|---|---|---|
| Synthetic + LLM shift | [all_metrics.parquet](../results/benchmark_suite/all_metrics.parquet) | [all_summary.csv](../results/benchmark_suite/all_summary.csv) |
| Stationary Beta | [metrics.parquet](../results/stationary_beta_suite/metrics.parquet) | [summary.csv](../results/stationary_beta_suite/summary.csv) |
| Stationary subjects | [metrics.parquet](../results/stationary_llm_suite/metrics.parquet) | [summary.csv](../results/stationary_llm_suite/summary.csv) |
| Stationary categories | [metrics.parquet](../results/stationary_llm_broad_suite/metrics.parquet) | [summary.csv](../results/stationary_llm_broad_suite/summary.csv) |

```python
import pandas as pd

metrics = pd.read_parquet("results/benchmark_suite/all_metrics.parquet")
aggregate = (
    metrics.loc[metrics["budget"].gt(0)]
    .groupby(["environment_type", "method"])["total_realized_gain"]
    .mean()
)
print(aggregate.round(3))
```

각 환경·예산의 `ci95`는 `1.96 × seed 간 표준편차 / √10`입니다. 이 평균의 단위는 **480 arrivals에서 얻은 total gain**입니다. 위 집계는 여러 예산을 평균한 진단 요약이며, 곡선 면적이나 통계적 유의성 검정 결과가 아닙니다. LLM에서는 strong이 틀리고 cheap이 맞으면 gain이 −1일 수 있습니다.

현재 5개 실험군의 원시 결과를 확인한 기준으로, 94,240개 기록의 `budget_used > budget` 위반과 `total_realized_gain` 누락은 각각 0건입니다.

## 그림 선택 방식

전체 환경 그림과 결과를 보존하면서 읽기 쉬운 선별 그림을 함께 제공합니다. 선별은 **저장된 평가 결과를 확인한 이후** 수행했으며, 별도 사전 고정 평가 집합으로 해석하지 않습니다.

| 그림 | 선택 방식 | 선택 기록 |
|---|---|---|
| Stationary Beta top-12 | 각 B > 0에서 BatchPace gain − 가장 강한 non-oracle baseline gain을 구한 후 예산 간 평균으로 순위 | [ranking](../output/stationary_beta_top12/selection_ranking.csv) |
| Stationary Beta positive-9 | 위 평균 margin이 양수인 9개. 모든 개별 예산에서 이긴다는 뜻은 아님 | [selection](../output/stationary_beta_positive9_zoomed/selection_positive9.csv) |
| Stationary subjects top-8 | 각 정책의 B > 0 평균을 먼저 구한 뒤, BatchPace 평균 − 가장 강한 표시 baseline의 평균으로 순위 | [ranking](../output/stationary_llm_top8/selection_ranking.csv) |
| Stationary categories top-8 | Stationary subjects와 같은 기준. 순위 내 음수 margin 환경도 포함 | [ranking](../output/stationary_llm_broad_top8/selection_ranking.csv) |
| LLM shift selected-8 | 초기 shift 환경 중 선별한 8개 사례 | [figure manifest](../output/selected_llm_highlighted/figure_manifest.csv) |
| Synthetic shift selected-12 | 초기 합성 환경 중 선별한 12개 사례 | [selection manifest](../output/selected_synthetic_highlighted/selection_manifest.json) |

Stationary LLM의 표시 baseline은 Fixed Threshold, Equal Split, WISERouter-Offline, Robust Batch-Level입니다. Top-8 순위에는 원시 결과에만 있는 Oracle·Sequential을 포함하지 않습니다. Beta positive-9는 [total gain](../output/stationary_beta_positive9_zoomed/total_gain_data.json)과 [동일 seed의 paired difference](../output/stationary_beta_positive9_zoomed/paired_difference_data.json)를 함께 제공합니다.

PNG는 빠른 확인에, SVG는 편집에, PDF는 벡터 기반 논문 삽입에 사용할 수 있습니다. [JSON deliverable](../deliverables/batchpace_graph_data_json_20260826/README_KO.md)은 초기 shift 실험과 그래프 데이터 패키지이며 이후 stationary 실험 전체를 포함하는 패키지는 아닙니다.

## 재현 명령

저장소 루트에서 의존성을 설치합니다. 저장된 캐시를 재사용하는 아래 정책 실험에는 GGUF와 llama.cpp가 필요하지 않습니다.

```bash
python -m pip install -r requirements-benchmark.txt
```

| 대상 | 실험 실행 | 전체 그림 또는 대분류 선별 그림 |
|---|---|---|
| Shift 40 + 40 | `python scripts/run_benchmark_suite.py --part all --seeds 10 --calibration-paths 200` | `python scripts/plot_benchmark_suite.py` |
| Stationary Beta 40 | `python scripts/run_stationary_beta_suite.py` | `python scripts/plot_stationary_beta_suite.py` |
| Stationary subjects 40 | `python scripts/run_stationary_llm_suite.py` | `python scripts/plot_stationary_llm_suite.py` |
| Stationary categories 14 | `python scripts/run_stationary_llm_broad_suite.py` | `python scripts/plot_top8_stationary_llm_broad.py` |

저장된 stationary 결과에서 선별 그림만 다시 만들려면 다음 명령을 사용합니다.

```bash
python scripts/plot_top12_stationary_beta.py
python scripts/plot_positive9_stationary_beta_zoomed.py
python scripts/plot_top8_stationary_llm.py
python scripts/plot_top8_stationary_llm_broad.py
```

원시 결과와 기본 경로의 그림은 재실행 시 갱신됩니다. 네 runner 모두 `--output`을 지원합니다. `plot_benchmark_suite.py`와 stationary Beta·subject 전체 plotter는 `--result-dir`과 `--output-dir`을 지원합니다. Top-8·top-12·positive-9 스크립트는 코드에 지정한 기본 경로를 읽고 씁니다.

전체 환경·정책 검증:

```bash
python -m unittest discover -s tests -v
```

처음부터 LLM option score를 생성하려면 [HANDOFF의 모델 배치·서버·scoring 명령](../HANDOFF.md#8-llm-option-score를-처음부터-재생성)을 참고하세요. GGUF 가중치는 `data/real_llm/models/`, llama.cpp Vulkan runtime은 `tools/llama.cpp-vulkan/`에 두며 둘 다 Git에서 제외되어 있습니다. 실제 scoring entry point는 [score_options_llamacpp.py](../scripts/score_options_llamacpp.py)입니다.

## 해석 시 함께 확인할 조건

- Shift 실험의 category transition과 change point는 calibration protocol에 반영됩니다. 예고 없는 change-point 탐지 실험과 구분됩니다.
- LLM batch-bursty는 자연 발생 serving log가 아니라 difficulty quartile로 구성한 workload입니다.
- Stationary LLM은 작은 고정 pool을 복원추출합니다. Pool audit와 replay CI의 범위를 함께 확인해야 합니다.
- Cheap는 Q8_0, strong은 Q4_K_M으로 quantization도 다릅니다.
- 합성 gain과 LLM correctness gain은 값의 의미와 범위가 달라 직접 같은 척도의 성능으로 비교하지 않습니다.
- Oracle BatchPace와 hindsight oracle의 정보 접근 범위가 다릅니다. 실제 배포 가능한 정책과 oracle을 분리해 읽어야 합니다.
- 여러 예산의 평균이나 선별 그림만으로 전체 우위·통계적 유의성을 판단하지 않습니다. 환경별 summary와 같은 seed의 차이를 함께 확인합니다.
