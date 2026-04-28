# Design Document — MLOps Churn Simulation

**Tanggal:** 2026-04-28
**Pemilik:** Gos Shafadonia (BNI ML Engineer)
**Status:** APPROVED (sections 1–7 by user, 2026-04-28)
**Companion docs:** [SoW + DoD](2026-04-28-mlops-churn-simulation-sow-dod.md)

---

## 0. Ringkasan Eksekutif

Simulasi MLOps end-to-end untuk Customer Churn Prediction (BNI banking) dengan 2 tujuan:
1. Pemahaman pribadi user tentang cara kerja MLOps secara hands-on
2. Demo visual untuk PM yang membutuhkan visibility ke ML operations

**Tool boundary (binding):**
- **MLflow** = surface OBSERVASI (experiment, registry, monitoring, audit)
- **Gradio** = surface AKSI (training trigger, live inference, A/B test)
- Tidak ada overlap — setiap capability hanya hidup di satu tool.

**Tech stack tervalidasi (Context7, 2026-04-28):**
Python 3.11 · uv (package mgr) · ruff (lint+format) · MLflow 3.1.4 · Gradio 6.0.1 · scikit-learn 1.7.1 · xgboost (latest) · scipy · pandas · numpy · matplotlib · pytest

**Prinsip yang dilock:**
- KISS, YAGNI, Single Source of Truth
- Use existing libraries (sklearn/scipy) — never reinvent
- Decisions binding upfront — no "ask later" hedging
- Explainable to non-technical PM in 1-2 plain sentences

---

## 1. Folder Structure & Module Boundaries

### Layout

```
test/
├── pyproject.toml                # uv-managed (deps + ruff config)
├── uv.lock                       # locked deps (committed)
├── README.md
├── .gitignore                    # mlruns/, mlflow.db, data/raw/, data/processed/, .venv/
├── run.sh                        # launcher: spawn MLflow + Gradio
│
├── src/mlops_churn/              # importable package — single source of truth
│   ├── __init__.py
│   ├── config.py                 # SSoT: paths, MLflow URI, aliases, HYPERPARAM_SPEC, threshold
│   ├── data.py                   # load + preprocess + split
│   ├── train.py                  # train_one(algo, params) → run_id
│   ├── registry.py               # alias mgmt (set/get/delete/list/history)
│   ├── serving.py                # predict() + predict_ab() + version-cache
│   └── monitoring.py             # log_batch_metrics, compute_drift (KS test)
│
├── scripts/                      # thin CLI wrappers
│   ├── seed_runs.py              # initial 9 baseline + register top 2
│   ├── promote.py                # alias mgmt (4 modes)
│   └── simulate_traffic.py       # generate traffic, log batch metrics
│
├── app/
│   └── gradio_app.py             # SATU Gradio app, 3 tabs
│
├── notebooks/01_eda.ipynb        # exploratory only
│
├── tests/                        # pytest, 5 file
│   ├── conftest.py
│   ├── test_data.py
│   ├── test_train.py
│   ├── test_registry.py
│   ├── test_serving.py
│   └── test_monitoring.py
│
├── data/                         # gitignored
│   ├── raw/Churn_Modelling.csv
│   └── processed/{train,val,test}.csv
│
├── mlruns/                       # MLflow artifacts (gitignored)
├── mlflow.db                     # MLflow SQLite (gitignored)
│
└── docs/superpowers/specs/       # design + plan + sow-dod
```

**Total: 6 src + 3 script + 5 test = 14 file kode.**

### Module Boundaries (kontrak public API)

| Module | Purpose | Public API |
|---|---|---|
| `config.py` | SSoT konstanta | `DATA_*`, `MLFLOW_*`, `EXPERIMENT_*`, `ALIAS_*`, `HYPERPARAM_SPEC`, `CUSTOMER_FEATURE_SCHEMA`, `AB_TEST_SAMPLES`, `DRIFT_SCORE_THRESHOLD` |
| `data.py` | Data lifecycle | `load_raw()`, `preprocess(df)`, `get_splits() → tuple` |
| `train.py` | Train + log | `train_one(algo: str, params: dict) → run_id: str` |
| `registry.py` | Alias mgmt | `register_run(run_id) → version`, `set_alias(alias, version)`, `get_version_by_alias(alias)`, `list_versions()`, `transition_history(version)` |
| `serving.py` | Inference | `predict(features, alias="production")`, `predict_ab(features)` |
| `monitoring.py` | Monitor + drift | `log_batch_metrics(run_id, step, batch)`, `compute_drift(ref, current, cols)` (uses `scipy.stats.ks_2samp`) |

### Prinsip Codebase
- Scripts thin wrappers; Gradio app glue; `src/` fat dengan logic
- `config.py` standalone (no internal imports)
- No global MLflow client singleton — tiap fungsi terima/buat sendiri (testable)
- Type hints + dataclass/TypedDict untuk kontrak modul

---

## 2. Data Flow

### Overview

```
Churn_Modelling.csv (raw)
  → data.preprocess + split
    → data/processed/{train,val,test}.csv
      → seed_runs.py (9 baseline) ──┐
      → Gradio Training Lab ────────┤
                                    ▼
                         MLflow experiment "churn-prediction"
                         (params, metrics, model artifact, tags)
                                    │
                         register + set_alias
                                    ▼
                         MLflow Model Registry
                         (churn-model @production, @staging, @archived)
                                    │
                                    ▼
              ┌──────────────┐  ┌────────────────────────┐
              │ Gradio       │  │ scripts/simulate_      │
              │ Inference +  │  │ traffic.py             │
              │ A/B (no log) │  │ (loop predict + log)   │
              └──────────────┘  └────────┬───────────────┘
                                         ▼
                              MLflow experiment
                              "production-monitoring"
                              (drift_score, latency, count, alert tag)
```

### Flow A — Initial Seed (sekali setelah clone)
```
manual: download Churn_Modelling.csv → data/raw/
$ uv run python -m scripts.seed_runs
  ├ data.preprocess + split → data/processed/
  ├ loop 9× (3 algo × 3 variant from HYPERPARAM_SPEC):
  │    train.train_one(algo, params)
  ├ pick top 2 by F1 → register
  └ set alias: production = top1, staging = top2
```

### Flow B — User-Driven Training (Gradio Training Lab)
```
User: pilih algo → adjust sliders → Train
↓
train.train_one(algo, params) → MLflow run (tag source=gradio-lab)
↓
Tampilkan metrics + run_id + link MLflow UI
```
**Tidak auto-register** — promosi eksplisit via MLflow UI atau `promote.py`.

### Flow C — Inference (Gradio Inference Tab)
```
User: isi form → Predict
↓
serving.predict(features, alias="production")
  ├ check version via MlflowClient (cheap call)
  ├ reload jika version berubah, else cached
  └ return {prob, label, latency_ms}
```
**Tidak log ke MLflow** (manual demo, bukan production volume).

### Flow D — A/B Test (Gradio A/B Tab)
```
User: pilih sample customer atau isi manual → Compare A/B
↓
serving.predict_ab(features) → {production: {...}, staging: {...}, agreement: bool}
↓
Tampilkan side-by-side
```
**Tidak log.**

### Flow E — Traffic Simulation (CLI)
```
$ uv run python -m scripts.simulate_traffic --mode {normal|drifted} --batches 10
↓
1 fresh MLflow run di experiment "production-monitoring"
↓
Loop batches (step = 0..N-1):
  generate batch → predict each → aggregate → log_metrics(step=batch_idx)
  if drift_score > threshold: set_tag("alert", "true")
```
**1 run per invocation** (bukan long-running).

### Flow F — Promotion
**Dua jalur, dua-duanya didukung:**
- **MLflow UI** — klik tab Models → version → Set Alias (untuk demo PM)
- **CLI** — `uv run python -m scripts.promote --version N --alias production`

Setelah promosi, panggilan `serving.predict()` berikutnya pakai version baru otomatis.

### Logging Map

| Source | Tujuan | Tipe |
|---|---|---|
| seed_runs.py | `churn-prediction`, 9 runs | params, metrics, model, tag `source=seed` |
| Gradio Training Lab | `churn-prediction`, 1 run | params, metrics, model, tag `source=gradio-lab` |
| Gradio Inference | (no logging) | — |
| Gradio A/B | (no logging) | — |
| simulate_traffic.py | `production-monitoring`, 1 run/invocation | metrics with step, tag `mode={normal,drifted}`, tag `alert` if needed |
| promote.py + UI | Registry transition history | timestamp + version + alias change (MLflow native) |

### Model Loading Strategy
Module-level cache di `serving.py` invalidated by version polling:
```python
_cache: dict | None = None

def predict(features, alias="production"):
    client = MlflowClient()
    current_version = client.get_model_version_by_alias("churn-model", alias).version
    if _cache is None or _cache["alias"] != alias or _cache["version"] != current_version:
        _cache = {
            "model": mlflow.pyfunc.load_model(f"models:/churn-model@{alias}"),
            "version": current_version,
            "alias": alias,
        }
    return _predict_with_latency(_cache["model"], features)
```

---

## 3. MLflow Logical Schema

### Experiments (hanya 2)
| Experiment | Isi |
|---|---|
| `churn-prediction` | Semua training run (seed + Gradio Lab) |
| `production-monitoring` | Setiap invocation `simulate_traffic.py` |

### Registered Model (hanya 1)
- Nama: `churn-model`
- Aliases yang dipakai: `production`, `staging`, `archived`

### Tag Conventions

**Training runs (`churn-prediction`):**
| Tag | Nilai |
|---|---|
| `source` | `seed` \| `gradio-lab` |
| `algo` | `logistic_regression` \| `random_forest` \| `xgboost` |

**Monitoring runs (`production-monitoring`):**
| Tag | Nilai |
|---|---|
| `mode` | `normal` \| `drifted` |
| `alert` | `true` (di-set jika drift > threshold) |
| `model_version` | versi model production saat run dijalankan |

### Run Naming
- Training: `{source}-{algo}-{compact-params}` (e.g., `gradio-lab-xgboost-lr0.05-md10-ne200`)
- Monitoring: `monitoring-{mode}-{timestamp}` (e.g., `monitoring-drifted-20260428-1505`)

### Metrics

**Training (semua run):**
- `accuracy`, `f1`, `roc_auc`, `precision`, `recall`

**Monitoring (per step):**
- `prediction_count`, `latency_p50_ms`, `latency_p95_ms`, `churn_rate`, `drift_score`

### Artifact per training run
- `model/` (MLflow model directory)
- `feature_schema.json`
- `confusion_matrix.png`

### Filesystem Layout (MLflow lokal)
- `test/mlflow.db` (SQLite metadata)
- `test/mlruns/` (artifacts per run)

---

## 4. Gradio App Spec

### Layout Global
- 1 Gradio app, 3 tabs: Training Lab / Inference / A/B Test
- Header tetap, footer link ke MLflow UI (`http://localhost:5000`)
- Single launch command via `run.sh`

### Tab 1: Training Lab
- Dropdown algoritma → toggle 3 column blocks (visible by selection)
- Sliders dynamic dari `HYPERPARAM_SPEC` di config.py (SSoT)
- Textbox tag opsional
- Button Train → call `train.train_one()` → tampilkan metrics + run_id + link MLflow
- Default: Random Forest

### Tab 2: Inference
- Status banner: "Model: churn-model v7 (production)" + Refresh button
- Form 2 kolom (10 fitur nasabah) — dari `CUSTOMER_FEATURE_SCHEMA` di config.py
- Predict button → `serving.predict()` → tampilkan label + probability progress bar + latency

### Tab 3: A/B Test
- Status banner: "Production v7 | Staging v9"
- Sample dropdown (3 preset di config.py `AB_TEST_SAMPLES`) atau isi manual
- Compare button → `serving.predict_ab()` → side-by-side display + agreement banner

### Edge Cases
- Alias tidak ada → friendly error + instruksi seed_runs.py
- MLflow server down → friendly error + cek run.sh
- Training error → tampilkan stacktrace dalam accordion

### Reusable Component
`render_customer_form()` dipakai di Tab 2 + Tab 3.

### Bootstrap
```python
def build_app() -> gr.Blocks:
    with gr.Blocks(title="BNI Churn MLOps Simulation") as demo:
        gr.Markdown("# 🏦 BNI Churn — MLOps Simulation")
        with gr.Tabs():
            with gr.Tab("🎛️ Training Lab"):  build_training_lab()
            with gr.Tab("🚀 Inference"):     build_inference()
            with gr.Tab("🔀 A/B Test"):       build_ab_test()
        gr.Markdown("📊 MLflow UI: [http://localhost:5000](http://localhost:5000)")
    return demo
```

---

## 5. Scripts Spec

### `run.sh` — Launcher
```bash
#!/usr/bin/env bash
set -e
uv run mlflow server --backend-store-uri sqlite:///mlflow.db \
                     --default-artifact-root ./mlartifacts \
                     --host 0.0.0.0 --port 5000 &
MLFLOW_PID=$!
sleep 2
uv run python app/gradio_app.py
trap "kill $MLFLOW_PID" EXIT
```

### `scripts/seed_runs.py`
```
uv run python -m scripts.seed_runs [--skip-preprocess] [--skip-register]
```
1. Validate `data/raw/Churn_Modelling.csv` ada
2. Preprocess → `data/processed/`
3. Loop 9 runs (3 algo × 3 variant from HYPERPARAM_SPEC)
4. Register top 2 by F1 → set aliases
5. Print summary

### `scripts/promote.py`
4 mode mutually exclusive:
- `--run-id <ID> --alias <NAME>` — register run + set alias
- `--version <N> --alias <NAME>` — move alias
- `--list` — print all versions + aliases
- `--remove --alias <NAME>` — delete alias

### `scripts/simulate_traffic.py`
```
uv run python -m scripts.simulate_traffic [--mode {normal,drifted}] [--batches N] [--batch-size N] [--alias ALIAS]
```
1 run per invocation di `production-monitoring`. Loop batches → predict → aggregate → log_metrics with step. Set tag alert jika drift > threshold.

### Script Dependency Order
1. `run.sh` (boot infra)
2. `seed_runs.py` (one-time, requires data/raw/)
3. `simulate_traffic.py` & Gradio inference (require alias `production` ada)
4. `promote.py` (kapan saja setelah ada training run)

---

## 6. Testing Strategy

### Filosofi
KISS — bukan 100% coverage. Setiap fungsi non-trivial punya test yang gagal saat logic-nya pecah.

### Stack
- pytest (existing)
- `sklearn.datasets.make_classification` untuk synthetic test data
- `tmp_path` fixture pytest + `file:` URI untuk isolated MLflow

### `conftest.py` Fixtures
- `tmp_mlflow_uri(tmp_path)` — isolated MLflow per test
- `synthetic_data` — make_classification(200 samples, 10 features, random_state=42)

### 13 Test Cases (5 file)

**`test_data.py` (3):**
- `test_load_raw_returns_dataframe`
- `test_preprocess_output_schema`
- `test_get_splits_ratios`

**`test_train.py` (2):**
- `test_train_one_returns_valid_run_id`
- `test_train_one_logs_5_metrics`

**`test_registry.py` (3):**
- `test_set_alias_assigns_correctly`
- `test_set_alias_atomic_move`
- `test_remove_alias`

**`test_serving.py` (2):**
- `test_predict_returns_dict_with_required_keys`
- `test_predict_ab_returns_both_versions`

**`test_monitoring.py` (3):**
- `test_drift_score_zero_for_identical`
- `test_drift_score_high_for_shifted`
- `test_log_batch_metrics_writes_5_metrics`

### Run
```bash
uv run pytest tests/ -v
```
Target durasi total: **< 30 detik**.

### Tidak Di-test
- Gradio UI behavior (manual visual)
- MLflow server startup (external tool)
- Library internals (sklearn/scipy/etc — sudah di-test maintainer)
- E2E demo flow (manual)
- Performance/load testing

---

## 7. Demo Flow & PM Story

### Pain Points → Solusi

| PM Pain | Komponen yang Menjawab |
|---|---|
| Tidak tahu model deploy | MLflow Registry alias `production` |
| Tidak tahu kapan update | Registry transition history (timestamp) |
| Tidak bisa lihat tren | Experiment Compare view |
| Tidak bisa jelaskan dampak | Run comparison metric diff |
| Tidak ada audit trail | Registry history + tags |
| Tidak bisa lihat masalah produksi | `production-monitoring` experiment |

### 7-Step Demo (≤15 menit)

**Persiapan:**
```bash
cd test/
uv sync
uv run python -m scripts.seed_runs
./run.sh &
```

**Step 1 (2m):** "Inilah masalahnya, inilah solusinya" — buka MLflow UI, tampilkan 2 experiments
**Step 2 (3m):** "Riwayat Training Lengkap" — Compare 3 runs side-by-side
**Step 3 (3m):** "Train Model Live" — Gradio Training Lab → adjust sliders → train → MLflow record
**Step 4 (2m):** "Promote Model" — Set alias staging via MLflow UI
**Step 5 (2m):** "Inference" — Gradio Inference tab, prediksi customer
**Step 6 (2m):** "A/B Test" — Compare production vs staging
**Step 7 (1m):** "Monitoring Produksi" — `simulate_traffic --mode drifted`, lihat drift_score chart + alert tag

### Closing Talking Points
- "MLflow = ingatan & governance. Gradio = kontrol & inference. Lokal dulu, scale ke server BNI kalau approve."
- Roadmap: minggu 1-2 adopsi tools, minggu 3+ auth + deploy.

### Mapping 20 Item In Scope ke Demo

| Item | Step | Bukti |
|---|---|---|
| 1-3 | Step 1, 2, 3 | seed_runs + 9 runs + Training Lab |
| 4 | All steps | MLflow UI :5000 |
| 5, 6, 15 | Step 4 | Models tab + transition history |
| 7 | Step 5 | Inference tab |
| 8-11, 14 | Step 7 | simulate_traffic + drift_score + alert |
| 12 | Step 3 | Training Lab as retraining trigger |
| 13 | Step 6 | A/B Test tab |
| 16-20 | Pre-demo | code, README, tests, design doc |

---

## Appendix A — Validation Trail (Context7, 2026-04-28)

| Library | Version | Validated |
|---|---|---|
| MLflow | v3.1.4 | ✅ Stages deprecated since 2.9.0 → use aliases |
| Gradio | v6.0.1 | ✅ `gr.Tabs`, `gr.Column(visible=...)`, `gr.update()` confirmed |
| scikit-learn | 1.7.1 | ✅ LogisticRegression, RandomForestClassifier confirmed |
| xgboost | latest | ✅ XGBClassifier (sklearn API) confirmed |
| scipy | latest | ✅ `scipy.stats.ks_2samp` confirmed (drift via KS test) |
| uv | latest | ✅ `uv init / add / run / sync` workflow confirmed |
| ruff | latest | ✅ linter + formatter unified |

## Appendix B — Decisions Locked Across Sections

1. Gradio = action only (3 tabs); MLflow = observation only (no overlap)
2. Use aliases (`production`/`staging`/`archived`), not deprecated stages
3. `simulate_traffic.py` = 1 run per invocation
4. Gradio Training Lab does NOT auto-register
5. Drift via `scipy.stats.ks_2samp` (KS test) — no custom PSI
6. Test data via `sklearn.datasets.make_classification` — no custom synthetic
7. uv as package manager; ruff as linter+formatter
8. `uv.lock` committed to git; `.venv/` gitignored
9. 6 src + 3 script + 5 test file = 14 file kode
10. 13 test cases, target < 30 detik runtime

## Appendix C — Out-of-Scope (Binding)

Items eksplisit OUT (lihat SoW Section 5):
Docker, Kubernetes, Cloud, CI/CD, Auth/RBAC, real-time streaming, Postgres/MySQL, feature store, Optuna automated tuning, custom MLflow plugins, SHAP/LIME explainability, FastAPI/Flask serving, multi-user collab, production-grade error handling.

Setiap item ini tidak diimplementasi titik. Kalau muncul kebutuhan saat implementasi, itu scope-change conversation eksplisit, bukan sneak-add.
