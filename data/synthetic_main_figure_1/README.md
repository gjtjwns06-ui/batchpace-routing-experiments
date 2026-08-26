# Main Figure 1 synthetic data

This directory contains reproducible numerical query data for the BatchPace
synthetic controlled study. A synthetic query is not a text prompt. It is an
item with a latent `true_gain` and a predictor error draw `unit_noise`.

## Parameters

- Test queries per episode: `N = 1000`
- Test episodes per setting: `250`
- Total strong-model budget: `K = 50`
- Base batch size: `m = 20` (`T = 50` batches)
- High-opportunity probability: `0.20`
- Low-opportunity gain: `Beta(1, 12)`
- High-opportunity gain: `Beta(5, 2)`
- Prediction-error values: `0, 0.05, 0.10, 0.20, 0.30, 0.40`
- Batch-size values: `1, 5, 10, 20, 50, 100`
- Batch-size sweep error: `epsilon = 0.10`
- Independent calibration batches per setting: `S = 1000`

## Workloads

- `iid`: every query independently draws its high/low state.
- `bursty`: a high/low state is drawn once per batch, and every query in that
  batch draws from the corresponding gain distribution.

The marginal query-level distribution is therefore similar between workloads;
the within-batch clustering is different.

## Files

- `fig1a_test.npz`: fixed `m=20` test episodes for the prediction-error sweep.
- `fig1a_calibration.npz`: independent `m=20` calibration batches.
- `fig1b_test.npz`: test episodes for every batch size in the batch-size sweep.
- `fig1b_calibration.npz`: independent calibration batches for every batch size.
- `fig1a_preview.jsonl`: human-readable sample rows and all epsilon scores.
- `fig1b_preview.jsonl`: human-readable sample rows at `epsilon=0.10`.
- `manifest.json`: parameters, deterministic random-stream seeds, hashes, and
  validation statistics.

Test and calibration arrays are generated from distinct deterministic random
streams. Do not use test arrays to estimate thresholds or future-value tables.

## NPZ schema

Array keys follow this pattern:

```text
m{batch_size:03d}_{environment}_true_gain
m{batch_size:03d}_{environment}_high_state
m{batch_size:03d}_{environment}_unit_noise
```

Test arrays have shape `[episode, query_id]`. The corresponding batch is:

```python
batch_id = query_id // batch_size
query_in_batch = query_id % batch_size
```

Calibration arrays have shape `[calibration_batch_id, query_in_batch]`.

For any prediction-error level `epsilon`, derive the deployable predictor score
as follows:

```python
import numpy as np

data = np.load("fig1a_test.npz")
gain = data["m020_bursty_true_gain"]
noise = data["m020_bursty_unit_noise"]
epsilon = 0.10
score = np.clip(gain + epsilon * noise, 0.0, 1.0)
```

The same `true_gain` and `unit_noise` are reused across epsilon values. This is
intentional: it produces a paired prediction-error sweep in which only the
error magnitude changes.

## Regeneration

Run from the project root:

```powershell
python scripts/generate_synthetic_data.py
```

The root seed is fixed in the generator. Re-running it produces identical file
contents and hashes.
