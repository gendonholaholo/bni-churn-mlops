"""Initial setup: preprocess + 9 baseline runs + register top 2 with aliases."""

import argparse

import mlflow

from mlops_churn import config, data, registry, train


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-register", action="store_true")
    args = parser.parse_args()

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

    # Preprocess
    if not args.skip_preprocess:
        print("[Preprocess] Loading", config.DATA_RAW_PATH)
        raw = data.load_raw()
        print(f"[Preprocess] {len(raw)} rows. Encoding + scaling...")
        processed = data.preprocess(raw)
        train_df, val_df, test_df = data.get_splits(processed)
        data.write_splits_to_disk(train_df, val_df, test_df)
        print(f"[Preprocess] Wrote splits to {config.DATA_PROCESSED_DIR}.")

    # Train 9 runs
    run_ids = []
    total = sum(len(v) for v in config.SEED_VARIANTS.values())
    i = 0
    for algo, variants in config.SEED_VARIANTS.items():
        for params in variants:
            i += 1
            print(f"[{i}/{total}] {algo}  {params}", end="  ")
            run_id = train.train_one(algo, params, source="seed")
            metrics = mlflow.get_run(run_id).data.metrics
            print(f"F1={metrics['f1']:.3f}  run={run_id[:6]}")
            run_ids.append((run_id, metrics["f1"], algo))

    # Register top 2 by F1
    if not args.skip_register:
        run_ids.sort(key=lambda x: x[1], reverse=True)
        top = run_ids[:2]
        v1 = registry.register_run(top[0][0])
        registry.set_alias(config.ALIAS_PRODUCTION, v1)
        v2 = registry.register_run(top[1][0])
        registry.set_alias(config.ALIAS_STAGING, v2)
        print("\n✅ Top 2 registered:")
        print(f"   v{v1} (run {top[0][0][:6]}, {top[0][2]}) → @production  F1={top[0][1]:.3f}")
        print(f"   v{v2} (run {top[1][0][:6]}, {top[1][2]}) → @staging     F1={top[1][1]:.3f}")

    print("\n🎯 Next: ./run.sh untuk buka Gradio + MLflow UI.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
