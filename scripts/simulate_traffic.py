"""Generate inference batches + log monitoring metrics. 1 run per invocation."""

import argparse
import datetime as dt
import time

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd

from mlops_churn import config, data, monitoring, registry


def _shift_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply distribution shift to numeric features (drift simulation).

    NOTE: data/processed/ contains z-scored numerics, so values are in stdev units
    (~ -3 to +3 typical). Shift +1.5 stdev = noticeable drift (KS detects).
    """
    out = df.copy()
    for col in ("Age", "CreditScore", "Balance"):
        if col in out.columns:
            out[col] = out[col] + 1.5  # +1.5 sigma shift in scaled space
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["normal", "drifted"], default="normal")
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--alias", default=config.ALIAS_PRODUCTION)
    args = parser.parse_args()

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_MONITORING)

    train_df, _val, test_df = data.load_processed()
    target_col = config.TARGET
    feature_cols = [c for c in train_df.columns if c != target_col]
    numeric_cols = [c for c in config.NUMERIC_FEATURES if c in feature_cols]

    version = registry.get_version_by_alias(args.alias).version

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M")
    run_name = f"monitoring-{args.mode}-{timestamp}"

    print(f"🌊 Traffic sim: mode={args.mode}, batches={args.batches}, batch_size={args.batch_size}")
    print(f"   Run name: {run_name}")
    print(f"   Serving model: {config.REGISTERED_MODEL_NAME} v{version} (@{args.alias})")
    print()
    print("step  count  p50ms  p95ms  churn_rate  drift_score  alert")

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        mlflow.set_tag("mode", args.mode)
        mlflow.set_tag("model_version", str(version))
        mlflow.set_tag("alias_used", args.alias)

        rng = np.random.default_rng(42)
        for step in range(args.batches):
            sampled = test_df.sample(
                n=args.batch_size, replace=True, random_state=rng.integers(0, 10000)
            )
            X_batch = sampled.drop(columns=[target_col])
            if args.mode == "drifted":
                X_batch = _shift_features(X_batch)

            # Batch predict (much faster than per-row for 50+ samples)
            model_uri = f"models:/{config.REGISTERED_MODEL_NAME}@{args.alias}"

            # Load underlying model in native flavor for batch predict
            try:
                native_model = mlflow.sklearn.load_model(model_uri)
            except Exception:
                native_model = mlflow.xgboost.load_model(model_uri)

            t0 = time.perf_counter()
            labels_array = native_model.predict(X_batch)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            labels = [int(x) for x in labels_array]
            # Approximate per-prediction latency
            latencies = [elapsed_ms / len(labels)] * len(labels)

            drift_score = monitoring.compute_drift(
                train_df[numeric_cols], X_batch[numeric_cols], numeric_cols
            )
            monitoring.log_batch_metrics(
                run_id=run_id,
                step=step,
                prediction_count=len(labels),
                latency_ms_list=latencies,
                labels=labels,
                drift_score=drift_score,
            )

            alert = "🚨" if drift_score > config.DRIFT_SCORE_THRESHOLD else "  "
            p50 = float(np.percentile(latencies, 50))
            p95 = float(np.percentile(latencies, 95))
            churn = sum(labels) / len(labels)
            row_str = (
                f"{step + 1:<4}  {len(labels):<5}  {p50:<5.1f}  {p95:<5.1f}  "
                f"{churn:<10.3f}  {drift_score:<11.3f} {alert}"
            )
            print(row_str)

        print(f"\n✅ Run {run_id[:6]} logged.")
        print(f"🔗 {config.MLFLOW_UI_URL}/#/experiments/{run.info.experiment_id}/runs/{run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
