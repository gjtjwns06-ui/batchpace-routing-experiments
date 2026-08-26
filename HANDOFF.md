# BatchPace Routing Experiments - 핸드오프 문서

이 문서는 다른 연구자나 개발자가 현재 프로젝트를 바로 이어받을 수 있도록
실험 목적, 구현 상태, 재실행 순서, 산출물, 검증 결과와 주의점을 정리한다.

## 1. 현재 상태

- GitHub: <https://github.com/gjtjwns06-ui/batchpace-routing-experiments>
- 기본 브랜치: `main`
- 운영체제: Windows 11 / PowerShell
- 실험 장비: Intel Core Ultra 5 226V, 16 GB RAM, Intel integrated GPU
- 로컬 추론 런타임: llama.cpp Vulkan build
- 현재 상태: Synthetic 40개와 LLM-derived 40개 환경 실행 및 figure 생성 완료
- 최종 실험 행 수: 44,800
- 환경별 figure: PNG 80개 + SVG 80개
- 통합 산출물: 80-page PDF atlas

모델 가중치와 llama.cpp 실행 파일은 용량 때문에 Git에서 제외했다. 코드,
MMLU-Pro 데이터, query embedding cache, option-score cache, 정책별 결과와 figure는
저장소에 포함되어 있다.

## 2. 연구 질문

480개 query가 24개 batch로 도착하고 전체 strong-model 호출 예산 `B`가 주어졌을
때, 어떤 query를 strong model로 보내야 전체 realized gain이 최대가 되는지를
비교한다. 모든 환경은 batch 12 이후 workload distribution이 변한다.

공통 설정:

| 항목 | 값 |
|---|---:|
| Batches | 24 |
| Queries per batch | 20 |
| Total queries per environment | 480 |
| Shift point | batch 12 이후 |
| Budgets | 0, 24, 48, 72, 96, 120, 144, 192 |
| Test repetitions | 10 seeds |
| Calibration paths | 200 |
| Error band | normal-approximation 95% CI |

비교 정책:

1. BatchPace
2. Oracle BatchPace
3. Sequential Pacing
4. Fixed Threshold
5. Equal Split
6. WISERouter-Offline
7. Robust Batch-Level Query Routing

## 3. Synthetic 40개 환경

40개는 다음 Cartesian product다.

```text
5 gain families x 4 shifts x 2 batch regimes = 40 environments
```

### 3.1 기본 gain family

| Family | 구현 |
|---|---|
| Sparse | 25% `Beta(2,5)`, 75% 정확히 0 |
| Moderate | `Beta(2.5,4)` |
| Bimodal | 35% `Beta(6,2)`, 65% `Beta(1.5,7)` |
| Heavy-tail | `Beta(0.75,3.5)` |
| Rare-jackpot | 8% `Beta(12,1.5)`, 92% `Beta(0.5,12)` |

주의: 40개 환경마다 서로 다른 Beta 파라미터를 둔 것이 아니다. 같은 family의
8개 환경은 batch 12 이전의 기본 Beta 또는 mixture를 공유하고, shift 및 batch
regime으로 구분된다.

### 3.2 Batch 12 이후 shift

기본 gain을 `x`라고 할 때 결과는 항상 `[0,1]`로 clip한다.

| Shift | 변환 |
|---|---|
| Mean-up | `0.18 + 1.05*x` |
| Mean-down | `0.72*x - 0.03` |
| Variance-up | `0.5 + 1.55*(x - 0.5)` |
| Tail-reversal | 88%는 `0.45*x`, 12%는 `0.72 + 0.28*Uniform(0,1)` |

Batch regime은 query가 독립적으로 생성되는 `iid`와, batch 전체에 공통 latent
difficulty 효과를 추가하는 `batch_bursty` 두 가지다.

정확한 목록은
`results/benchmark_suite/synthetic_environment_manifest.json`에 있다.

## 4. LLM-derived 40개 환경

### 4.1 데이터와 모델

- Dataset: `TIGER-Lab/MMLU-Pro`
- 고정 split: train 500 / calibration 500 / test 500
- Split seed: `20260825`
- Cheap model: `Qwen3-0.6B-Q8_0`
- Strong model: `Qwen3-4B-Q4_K_M`
- Routing embedding: `Qwen3-Embedding-0.6B-Q8_0`, dimension 1024
- Prompting: category-matched 5-shot

데이터 및 모델 revision과 SHA-256은 다음 manifest에 기록되어 있다.

- `data/real_llm/splits/manifest.json`
- `data/real_llm/cache/manifest.json`

### 4.2 Option scoring과 realized gain

자유 생성 결과를 파싱하지 않는다. llama.cpp grammar로 유효한 option label만
허용하고 truncation sampler를 끈 뒤 다음 값을 직접 저장한다.

```text
log p(A | prompt), ..., log p(J | prompt)
```

가장 높은 option-label conditional log-likelihood의 label을 모델 답으로 사용한다.
따라서 현재 구현은 option **본문 전체 문자열**의 likelihood가 아니라 option
**label token**의 constrained likelihood다.

Query-level realized gain은 다음과 같다.

```text
gain(q) = 1[strong correct] - 1[cheap correct]
```

- `+1`: strong model만 정답
- `0`: 두 모델의 correctness가 같음
- `-1`: cheap model만 정답

최종 option-scoring cache 정확도:

| Model | Accuracy over 1,500 queries |
|---|---:|
| Qwen3-0.6B-Q8_0 | 14.07% |
| Qwen3-4B-Q4_K_M | 36.07% |

`data/real_llm/cache/manifest.json`의 22.93%/40.67% 값은 이전 Main Figure 2에서
사용한 constrained generation cache의 정확도다. 새 40-environment benchmark는
반드시 `results/benchmark_suite/option_scores_*.jsonl`의 option-scoring 결과를
사용한다. `run_benchmark_suite.py`는 legacy Parquet에서 embedding column만 읽는다.

### 4.3 40개 workload 구성

Calibration split에서 과목별 평균 realized gain을 구하고 차이가 큰 10개 과목
쌍을 선택한다. 각 쌍의 양방향을 사용해 20개 directed transition을 만들고,
각 transition에 `iid`와 `batch_bursty`를 적용한다.

```text
20 directed category transitions x 2 regimes = 40 environments
```

- Batches 1-12: source category
- Batches 13-24: target category
- IID: 해당 과목 pool에서 독립 표본 추출
- Batch-bursty: 한 batch를 predicted-gain difficulty quartile 하나에서 표본 추출

환경 선택에는 calibration label만 사용하며 test label은 사용하지 않는다. 정확한
목록은 `results/benchmark_suite/llm_environment_manifest.json`에 있다.

## 5. 정책 구현 요약

### BatchPace

Ridge gain predictor의 점수와 position-aware batch Bellman table을 사용한다.
현재 batch 내부에서는 top-k query를 고르고, 미래 value와 현재 cumulative score의
합이 최대가 되는 k를 선택한다.

### Oracle BatchPace

동일한 online batch Bellman rule을 사용하지만 현재 query score와 calibration
value를 true gain으로 대체한다. 전체 480개 query를 미리 보고 top-B를 고르는
hindsight oracle은 아니다.

### Sequential Pacing

한 번에 query 하나를 보며 position-aware Bellman take/skip 결정을 한다.

### Fixed Threshold

Calibration score의 budget quantile을 고정 threshold로 사용한다. Stream 앞부분에서
budget이 소진될 수 있으므로 hard global budget guard를 적용한다.

### Equal Split

전체 budget을 24개 batch에 균등 분할하고 각 batch에서 predicted score top-k를
선택한다. 현재 budgets는 모두 24의 배수라 batch cap은 정확히 정수다.

### WISERouter-Offline

원 논문의 offline adaptive LP를 두 action, 비용 0/1 설정으로 축약했다. Query
embedding을 16개 K-means context로 나누고 train split에서 context probability와
cheap/strong 평균 reward를 계산한다. Online exploration은 구현하지 않았다.

### Robust Batch-Level Query Routing

원 논문의 robust batch ILP를 두 모델, unit incremental cost, GPU capacity 없음
설정으로 축약했다. LLM 실험에서는 100개 bootstrap refit의 10th-percentile lower
bound를 사용하고 batch별 equal cap 아래에서 positive robust gain top-k를 고른다.
원 논문의 multi-model instance allocation과 GPU/concurrency constraint는 포함하지
않는다.

핵심 구현 파일:

- `src/benchmark_suite_environments.py`
- `src/benchmark_suite_policies.py`
- `scripts/run_benchmark_suite.py`

## 6. 저장소 구조

```text
data/
  real_llm/                 MMLU-Pro, splits, tracked embedding cache
  synthetic_main_figure_1/  이전 Main Figure 1 데이터
output/
  benchmark_suite/          환경별 PNG/SVG 80개
  figures/                  이전 Main Figure 1/2 PNG
  pdf/                      80-page atlas와 이전 PDF
results/
  benchmark_suite/          40+40 raw metrics, summaries, manifests, scores
  real_llm/                 이전 Main Figure 2 cache와 결과
  synthetic_main_figure_1/  이전 Main Figure 1 결과
scripts/                    데이터, inference, 실행, plotting entry points
src/                        정책 및 환경 구현
tests/                      budget/shape/environment-count tests
```

가장 중요한 산출물:

- `results/benchmark_suite/all_metrics.parquet`: 44,800 raw records
- `results/benchmark_suite/all_summary.csv`: environment-budget-policy별 mean/CI
- `output/benchmark_suite/figure_manifest.csv`: 80개 figure 경로
- `output/pdf/batchpace_80_environment_atlas.pdf`: 80-page 통합 PDF

## 7. 가장 빠른 재실행

Tracked option-score와 embedding cache를 그대로 사용하면 모델 inference 없이 정책
실험부터 재실행할 수 있다.

```powershell
python -m pip install -r requirements-benchmark.txt
python scripts/run_benchmark_suite.py --part all --seeds 10 --calibration-paths 200 --budgets 0 24 48 72 96 120 144 192
python scripts/plot_benchmark_suite.py
python -m unittest tests.test_benchmark_suite -v
```

이 명령은 결과와 figure를 동일한 경로에 덮어쓴다. 기존 결과를 보존해야 한다면
먼저 별도 branch를 만들거나 `--output`으로 다른 results 디렉터리를 지정한다.

## 8. LLM option score를 처음부터 재생성

### 8.1 제외된 파일 준비

다음 Hugging Face repository에서 GGUF를 받아 정확한 경로에 둔다.

| Repository | File | Local path |
|---|---|---|
| `Qwen/Qwen3-0.6B-GGUF` | `Qwen3-0.6B-Q8_0.gguf` | `data/real_llm/models/qwen3-0.6b/` |
| `Qwen/Qwen3-4B-GGUF` | `Qwen3-4B-Q4_K_M.gguf` | `data/real_llm/models/qwen3-4b/` |
| `Qwen/Qwen3-Embedding-0.6B-GGUF` | `Qwen3-Embedding-0.6B-Q8_0.gguf` | `data/real_llm/models/qwen3-embedding-0.6b/` |

llama.cpp Windows Vulkan build는 `tools/llama.cpp-vulkan/` 아래에 둔다. 모델과
runtime은 `.gitignore` 대상이다.

### 8.2 Cheap model scoring

터미널 1:

```powershell
.\tools\llama.cpp-vulkan\llama-server.exe `
  --model .\data\real_llm\models\qwen3-0.6b\Qwen3-0.6B-Q8_0.gguf `
  --host 127.0.0.1 --port 8088 --threads 8 --threads-batch 8 `
  --parallel 1 --ctx-size 8192 --batch-size 512 --jinja `
  --reasoning off --reasoning-format none --n-gpu-layers 99
```

터미널 2:

```powershell
python scripts/score_options_llamacpp.py `
  --input data/real_llm/splits/mmlu_pro_main_1500.parquet `
  --demonstrations data/real_llm/mmlu_pro/data/validation-00000-of-00001.parquet `
  --output results/benchmark_suite/option_scores_qwen3_0.6b.jsonl `
  --model-id Qwen3-0.6B-Q8_0 --shots 5
```

### 8.3 Strong model scoring

Cheap server를 종료하고 같은 명령의 model path를 4B 파일로 바꾼다. 서버를
8089에서 실행한다면 scoring 명령에 `--base-url http://127.0.0.1:8089`를 추가한다.

```powershell
python scripts/score_options_llamacpp.py `
  --input data/real_llm/splits/mmlu_pro_main_1500.parquet `
  --demonstrations data/real_llm/mmlu_pro/data/validation-00000-of-00001.parquet `
  --output results/benchmark_suite/option_scores_qwen3_4b.jsonl `
  --model-id Qwen3-4B-Q4_K_M --shots 5
```

Scoring script는 `question_id` 기준 restartable cache를 사용한다. 완료 후 각 JSONL은
정확히 1,500 unique question IDs여야 한다.

## 9. 검증된 결과

최종 검증:

- Synthetic: 22,400 records, 40 environments
- LLM: 22,400 records, 40 environments
- Combined: 44,800 records, 80 environments
- Seeds: 10
- Budgets: 8
- Methods: 7
- Budget violations: 0
- NaN: 0
- PDF pages: 80
- PNG/SVG: 각 80개
- Unit tests: 3 passed

모든 nonzero budget을 합친 참고용 평균 realized gain:

| Method | Synthetic | LLM-derived |
|---|---:|---:|
| Oracle BatchPace | 54.385 | 88.461 |
| BatchPace | 51.365 | 40.026 |
| Sequential Pacing | 51.299 | 40.425 |
| Fixed Threshold | 49.660 | 35.166 |
| Equal Split | 46.703 | 32.524 |
| WISERouter-Offline | 50.563 | 42.022 |
| Robust Batch-Level Query Routing | 46.666 | 32.272 |

이 평균은 환경과 여러 budget point를 함께 평균한 진단값이다. 논문 표에는
environment 또는 budget별 `all_summary.csv`를 사용해야 한다.

## 10. 중요한 해석상 주의점

1. LLM shift는 과목 변화로 gain distribution shift를 구현한 proxy다. 과목 자체가
   연구 대상은 아니다.
2. Shift 위치와 category transition은 calibration protocol에 반영된다. 예고 없는
   online change-point detection 실험이 아니다.
3. LLM `batch_bursty`는 자연 발생 로그가 아니라 predicted difficulty quartile로
   인위적으로 구성한 batch다.
4. Oracle BatchPace는 online batch oracle이며 global hindsight top-B가 아니다.
5. WISERouter와 Robust baseline은 원 논문의 핵심 optimization을 반영하지만 전체
   multi-model serving system을 그대로 복제하지는 않는다.
6. Quantization이 서로 다르다. Cheap는 Q8_0, strong은 Q4_K_M이다.
7. 95% CI는 10 seeds의 normal approximation이다. 출판 전 bootstrap CI 또는 더
   많은 seed를 추가하는 것이 바람직하다.
8. Public repository에는 GGUF와 llama.cpp binary가 없다. 원본 `논문실험.pdf`도
   workspace 밖 파일이므로 저장소에 포함되지 않았다.
9. `output/`과 `results/`는 재실행 시 같은 파일명으로 갱신된다.

## 11. 다음 작업 권장 순서

1. 논문에서 사용할 대표 환경과 aggregate table을 사전에 고정한다.
2. Full option-text conditional likelihood와 현재 label-token scoring을 비교한다.
3. 예상된 shift와 예고 없는 shift를 분리해 평가한다.
4. WISERouter/Robust의 원 논문 전체 설정을 별도 multi-model experiment로 재현한다.
5. seed 수와 confidence interval 방법을 확정한다.
6. 모델, 데이터셋, baseline paper의 license와 citation을 논문 appendix에 명시한다.
7. 최종 논문 수치 생성 시 새 Git tag와 immutable result manifest를 만든다.
