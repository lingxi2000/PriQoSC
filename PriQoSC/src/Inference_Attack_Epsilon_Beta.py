

from __future__ import annotations

import argparse
import copy
import os

import Inference_Attack_PoolSize_Representative as attack


DATASETS = ("QWS", "Normal")
EPSILON_VALUES = (1.0, 2.0, 3.0, 4.0, 5.0)
BETA_VALUES = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
DISTANCE_NAMES = ("hamming", "euclidean")
CLIP_BOUNDS = {"QWS": 0.4, "Normal": 0.2}
PROXY_POOL_SIZES = {"QWS": 400, "Normal": 100}
MECHANISM = "EM"
DEFAULT_RUNS = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one dataset's inference attack. Launch QWS and Normal in "
            "separate terminals for parallel execution."
        )
    )
    parser.add_argument(
        "dataset",
        choices=DATASETS,
        help="Dataset to run in this terminal.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Processes used to precompute public attack templates. Default: 4.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help="Independent pipeline runs per dataset. Default: 1.",
    )
    parser.add_argument(
        "--task-limit",
        type=int,
        default=None,
        help="Optional smoke-test task limit; omit for the complete experiment.",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Rebuild public attack-template caches before running.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Build or validate this dataset's attack-template cache only.",
    )
    return parser.parse_args()


def experiment_parameters() -> dict:
    """Create a private configuration copy without modifying dp_perturb.py."""
    parameters = copy.deepcopy(attack.dp_para)
    dataset_settings = parameters.setdefault("dataset_settings", {})
    for dataset, proxy_pool_size in PROXY_POOL_SIZES.items():
        current = dict(dataset_settings.get(dataset, {}))
        current["proxy_pool_size"] = int(proxy_pool_size)
        dataset_settings[dataset] = current
    return parameters


def run_dataset(
    dataset: str,
    args: argparse.Namespace,
) -> None:
    attack.run_poolsize_representative_attack(
        dataset=dataset,
        workers=args.workers,
        task_limit=args.task_limit,
        rebuild_cache=args.rebuild_cache,
        validate_only=args.validate_only,
        mechanisms_filter=(MECHANISM,),
        epsilons_filter=EPSILON_VALUES,
        clip_bounds_filter=(CLIP_BOUNDS[dataset],),
        delta_values_filter=BETA_VALUES,
        distances_filter=DISTANCE_NAMES,
        runs_override=args.runs,
        matched_control_enabled=True,
        compact_output=True,
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.workers <= 0:
        raise ValueError("--workers must be positive.")
    if args.runs <= 0:
        raise ValueError("--runs must be positive.")
    if args.task_limit is not None and args.task_limit <= 0:
        raise ValueError("--task-limit must be positive when provided.")


def main() -> None:
    args = parse_args()
    validate_args(args)

    original_parameters = attack.dp_para
    attack.dp_para = experiment_parameters()
    try:
        dataset = args.dataset
        print("\n" + "#" * 72)
        print(
            f"Dataset={dataset} | EM | clip={CLIP_BOUNDS[dataset]:g} | "
            f"proxy={PROXY_POOL_SIZES[dataset]}"
        )
        print("#" * 72)
        run_dataset(dataset, args)
        print(f"\n[{dataset}] inference experiment finished.")
    finally:
        attack.dp_para = original_parameters


if __name__ == "__main__":
    main()
