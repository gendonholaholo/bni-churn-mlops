# BNI Churn — MLOps Simulation

Hands-on MLOps simulation: customer churn classification using MLflow 3.x (tracking + registry + aliases) and Gradio 6.x (training trigger + inference + A/B test).

## Tech Stack
Python 3.11 · uv · ruff · MLflow 3.1.4 · Gradio 6.0.1 · scikit-learn 1.7.1 · xgboost · scipy

## Setup

1. Install uv: https://docs.astral.sh/uv/getting-started/installation/
2. Sync dependencies:
   ```bash
   uv sync
   ```
3. Download the Churn Modelling dataset from Kaggle:
   https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling
   Place `Churn_Modelling.csv` at `data/raw/Churn_Modelling.csv`.

## Quickstart Demo

```bash
# 1. Initial seed (preprocess + 9 baseline runs + register top 2)
uv run python -m scripts.seed_runs

# 2. Launch MLflow + Gradio
./run.sh

# Open in browser:
#  - Gradio app: http://localhost:7860
#  - MLflow UI: http://localhost:5000
```

## Demo Flow (≤ 15 min)
See [docs/superpowers/specs/2026-04-28-mlops-churn-simulation-design.md](docs/superpowers/specs/2026-04-28-mlops-churn-simulation-design.md) Section 7.

## Development

```bash
uv run pytest           # 14 tests, < 30s
uv run ruff check .     # lint
uv run ruff format .    # format
```
