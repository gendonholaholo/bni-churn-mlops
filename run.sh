#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo "🚀 Booting MLflow tracking server (port 5001)..."
# NOTE: macOS Control Center / AirPlay Receiver squats on port 5000 by default.
# We use 5001 to avoid the conflict — no need to disable AirPlay in System Settings.
uv run mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts \
  --host 0.0.0.0 --port 5001 &
MLFLOW_PID=$!

cleanup() {
  echo "🛑 Stopping MLflow (PID $MLFLOW_PID)..."
  kill $MLFLOW_PID 2>/dev/null || true
}
trap cleanup EXIT

sleep 3
echo "🎨 Booting Gradio app (port 7860)..."
echo "   - MLflow UI: http://localhost:5001"
echo "   - Gradio:    http://localhost:7860"
uv run python app/gradio_app.py
