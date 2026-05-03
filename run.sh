#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

# Tracking server is provided externally (Docker container on host port 5001 by
# default). Override with MLFLOW_TRACKING_URI when pointing elsewhere.
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://localhost:5001}"

echo "Using MLflow tracking server at $MLFLOW_TRACKING_URI"
if ! curl -sSf "$MLFLOW_TRACKING_URI/api/2.0/mlflow/experiments/search" \
     -X POST -H "Content-Type: application/json" -d '{"max_results":1}' \
     >/dev/null 2>&1; then
  echo "ERROR: MLflow server not reachable at $MLFLOW_TRACKING_URI"
  echo "Start it (e.g. docker compose up -d mlflow) or override MLFLOW_TRACKING_URI."
  exit 1
fi

echo "Booting Gradio app (port 7860)..."
echo "   - MLflow UI: $MLFLOW_TRACKING_URI"
echo "   - Gradio:    http://localhost:7860"
uv run python app/gradio_app.py
