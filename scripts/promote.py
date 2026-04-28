"""Alias management CLI: register, move, list, remove."""

import argparse

import mlflow

from mlops_churn import config, registry


def cmd_list() -> int:
    versions = registry.list_versions()
    print(f"{config.REGISTERED_MODEL_NAME}:")
    if not versions:
        print("  (no versions registered)")
        return 0
    for mv in versions:
        aliases_str = ", ".join(mv.aliases) if mv.aliases else "(none)"
        print(f"  v{mv.version} (run {mv.run_id[:6]}) → aliases: {aliases_str}")
    return 0


def cmd_register_and_set(run_id: str, alias: str) -> int:
    print(f"Registering run {run_id} → {config.REGISTERED_MODEL_NAME}...")
    v = registry.register_run(run_id)
    print(f"✅ Registered as v{v}.")
    print(f"Setting alias '{alias}' → v{v}.")
    registry.set_alias(alias, v)
    print("✅ Done.")
    return 0


def cmd_move(version: str, alias: str) -> int:
    print(f"Setting alias '{alias}' → v{version}.")
    registry.set_alias(alias, version)
    print("✅ Done.")
    return 0


def cmd_remove(alias: str) -> int:
    print(f"Removing alias '{alias}'...")
    registry.remove_alias(alias)
    print("✅ Done.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Alias management for churn-model")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="List versions + aliases")
    group.add_argument("--run-id", help="Register run + set alias")
    group.add_argument("--version", help="Move existing alias to version")
    group.add_argument("--remove", action="store_true", help="Delete alias")
    parser.add_argument("--alias", help="Alias name (required for --run-id, --version, --remove)")
    args = parser.parse_args()

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

    if args.list:
        return cmd_list()
    if args.run_id:
        if not args.alias:
            parser.error("--run-id requires --alias")
        return cmd_register_and_set(args.run_id, args.alias)
    if args.version:
        if not args.alias:
            parser.error("--version requires --alias")
        return cmd_move(args.version, args.alias)
    if args.remove:
        if not args.alias:
            parser.error("--remove requires --alias")
        return cmd_remove(args.alias)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
