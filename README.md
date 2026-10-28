# GridSentinel — Electricity Theft & Abnormal Consumption Detection

GridSentinel is a reproducible smart-meter risk-triage system. It generates or ingests hourly meter readings, injects realistic cyber-physical benchmark attacks when labels are unavailable, compares a calibrated Random Forest with a two-layer PyTorch LSTM on the same consumer-level holdout, and serves the resulting priority queue through a field-verification dashboard.

> **Evidence boundary:** the committed benchmark measures recovery of deterministic simulated attacks. It is not a claim of field accuracy or proof that a flagged consumer committed theft. Production use requires labeled inspection outcomes, regional/temporal validation, monitoring, and human verification.

## Verified benchmark

The default seeded run uses 320 consumers, 70 days of hourly readings, and 12.5% attacked consumers. Consumers—not rows—are split 60%/20%/20% into train, validation, and test cohorts, preventing a consumer's readings from leaking between cohorts.

| Model | Recall | Precision | F1 | AUC-PR | Top 10% capture | Top 20% capture |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 75.0% | 100.0% | 85.7% | 100.0% | 87.5% | 100.0% |
| LSTM | 75.0% | 100.0% | 85.7% | 100.0% | 87.5% | 100.0% |

The selected model passes both strict assertions:

- recall ≥ 75%: **75.0%**
- anomalies captured in the highest-risk 20% ≥ 60%: **100.0%**

Exact outputs are committed under [`artifacts/`](artifacts/), including metrics, PR/gains curve points, calibrated consumer rankings, and the top-priority load curves.

## Architecture

```text
CSV or seeded synthesis
        │
        ▼
hourly repair + attack injection
        │
        ├── engineered consumer features ── calibrated Random Forest
        │
        └── normalized 168-hour curves ──── 2-layer LSTM + Platt calibration
                                              │
                         shared consumer test split + benchmark assertions
                                              │
                         calibrated ranking + top-decile field dashboard
```

The attack simulator includes peak-hour sudden drops, low/flatline tampering, intermittent peak bypass, and peak/off-peak inversion. Benign missing readings are injected independently of class and short gaps are forward-filled/interpolated.

## Run locally

Python 3.11–3.13 is recommended.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python evaluate_models.py
python -m streamlit run app.py
```

`python evaluate_models.py` retrains both models and refreshes the committed benchmark artifacts. You only need it for the initial setup or when changing the data/model pipeline. For normal dashboard use, activate the venv and run only `python -m streamlit run app.py`.

Using `python -m` for installation and launch guarantees that both commands use the interpreter from the active `.venv`. Verify it before running the benchmark:

```powershell
python -c "import sys; print(sys.executable)"
python -m streamlit version
```

The executable should end in `CPI\.venv\Scripts\python.exe`.

Open `http://localhost:8501`. The dashboard provides:

- benchmark KPI cards and the top-10 priority queue;
- editable per-consumer field scheduling and 7-day sparklines;
- observed-versus-expected load curves with abnormal periods highlighted;
- anomaly-signal radar, PR curves, and cumulative gains curves;
- filterable CSV and PDF inspection manifests.

### Interface design

The UI follows a Vercel-inspired visual system: solid black surfaces, neutral gray borders, white typography, compact radii, and restrained monochrome charts. It deliberately avoids gradients, blended backgrounds, colorful decorative treatments, and emoji branding. Keep operational colors semantic only—for example, red is reserved for abnormal load periods.

### Streamlit Community Cloud

Deploy from [Streamlit Community Cloud](https://share.streamlit.io/) using these exact values:

- Repository: `Paramount50/GridSentinal`
- Branch: `main`
- Main file path: `app.py`

Sign in with GitHub and ensure Streamlit Community Cloud has access to the repository. Local Streamlit can prefill these values only when the app was launched from the Git checkout with the active branch pushed. The venv-verified runtime resolves this checkout as `Paramount50/GridSentinal` on `origin/main`.

If the local **Deploy** button reports that no remote repository is connected, use the Community Cloud link above and enter the three values manually. The local toolbar can fail during its GitHub handoff even when Git itself resolves the remote correctly. Confirm the repository state with:

```powershell
git remote -v
git status -sb
```

The remote should be `https://github.com/Paramount50/GridSentinal.git`, and `main` should not be ahead of `origin/main`.

## Use an external dataset

Pass a CSV containing at least `consumer_id`, `timestamp`, and `consumption_kwh`:

```powershell
python evaluate_models.py --data .\data\meter_readings.csv
```

When `label` is absent it defaults to benign, so supervised evaluation is not meaningful until confirmed labels are supplied. Optional columns are `expected_kwh`, `label`, `attack_type`, `primary_signal`, and `is_anomaly_period`.

## Reproduce and test

```powershell
python -m pytest -q
python evaluate_models.py --consumers 320 --days 70 --epochs 24
```

`evaluate_models.py` exits non-zero if the selected model misses either strict benchmark. Thresholds are chosen on validation data at the recall constraint; model selection uses validation AUC-PR; the test set remains untouched until final reporting.

## Repository map

- `data_loader.py` — ingestion, hourly repair, deterministic load synthesis, attack injection
- `feature_engineering.py` — statistical, temporal, behavioral, and anomaly features plus sequences
- `models.py` — calibrated Random Forest and weighted two-layer LSTM training
- `evaluate_models.py` — shared split, metrics, curves, assertions, calibrated rankings
- `app.py` — Streamlit field-verification and export dashboard
- `tests/` — deterministic data and evaluation helper tests
- `artifacts/` — benchmark evidence used by the dashboard

## Production hardening checklist

Before operational decisions, replace or augment injected labels with confirmed inspection outcomes; validate by geography, meter type, season, and tariff; encrypt identifiers; add role-based access and audit logs; monitor calibration and data drift; version thresholds; and route adverse actions through a documented human review policy.

