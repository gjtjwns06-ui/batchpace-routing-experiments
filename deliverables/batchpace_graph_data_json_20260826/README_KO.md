# BatchPace 그래프 데이터 전달본

이 압축파일은 BatchPace synthetic 및 LLM-derived 비교 실험에서 그래프를 생성할
때 사용한 데이터를 JSON 형식으로 정리한 전달본입니다.

## 핵심 구성

- 환경: Synthetic 40개 + LLM-derived 40개
- 환경당 batch: 24개
- Batch당 query: 20개
- Distribution shift: batch 12 이후
- Global budget: `0, 24, 48, 72, 96, 120, 144, 192`
- 비교 정책: 7개
- Random seed: 10개 (`0`~`9`)
- 원시 실험 결과: 44,800행
- 그래프 점: 4,480행

## 파일 설명

### `figure_points.json`

그래프를 그릴 때 직접 사용한 최종 집계 데이터입니다.

- `environment_type`: `synthetic` 또는 `llm`
- `environment_id`: 환경 식별자
- `budget`: x축 global budget
- `method`: 비교 정책
- `mean`: y축 total realized gain 평균
- `std`: 10개 seed 사이 표준편차
- `count`: 반복 횟수
- `ci95`: 그래프 음영으로 표시한 95% 신뢰구간 반폭

### `experiment_runs.json`

집계 전 seed별 원시 정책 실행 결과입니다.

- `seed`: 반복 실험 번호
- `budget_fraction`: 전체 480개 query 중 budget 비율
- `total_realized_gain`: 선택된 query에서 얻은 gain의 합
- `budget_used`: 실제 사용한 strong-model 호출 수
- `batch_allocations`: 24개 batch별 strong-model 할당 수

### `summarized_results.json`

환경·budget·정책별 평균, 표준편차 및 95% 신뢰구간을 저장합니다.
`figure_points.json`과 같은 실험 결과를 사용하며 열 이름이 더 설명적인 형태입니다.

### 환경 정의

- `synthetic_environments.json`: 40개 synthetic gain 환경
- `llm_environments.json`: 40개 LLM 과목 전환 환경

### LLM 입력 및 점수

- `llm_queries.json`: 고정된 MMLU-Pro 1,500개 query와 train/calibration/test split
- `option_scores_qwen3_0.6b.json`: cheap model option-scoring 결과
- `option_scores_qwen3_4b.json`: strong model option-scoring 결과
- `llm_split_manifest.json`: 데이터 split과 원본 파일의 provenance 및 hash

LLM query-level realized gain은 다음과 같이 계산했습니다.

```text
gain(q) = 1[strong model correct] - 1[cheap model correct]
```

- `+1`: strong model만 정답
- `0`: 두 모델의 correctness가 같음
- `-1`: cheap model만 정답

### 기타

- `package_metadata.json`: 전체 실험 규모, 정책, budget 및 파일 목록
- `SHA256SUMS.json`: 압축파일 내부 각 파일의 SHA-256 체크섬

## 그래프 재구성 방법

각 환경과 정책에 대해 `figure_points.json`의 `budget`을 x축, `mean`을 y축으로
연결합니다. 오차 영역은 `mean - ci95`에서 `mean + ci95`까지 표시합니다.

## 원본 저장소

<https://github.com/gjtjwns06-ui/batchpace-routing-experiments>

생성일: 2026-08-26
