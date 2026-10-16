
from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

from tqdm import tqdm

import Inference_Attack_PoolSize_Representative as inference
from comparison_run_stats import (
    format_mean_std,
    load_completed_runs,
    save_completed_runs,
)
from config_loader import load_config
from differ_attack_core import (
    build_or_load_nested_attack_models,
    infer_constraint,
    load_training_constraint_bank,
    mean_metrics,
    shorten_task,
)
from dp_perturb import dataset_dp_settings, dp_para, validate_dp_para
from dp_qosc import (
    build_candidate_source,
    build_representative_pool,
    select_candidate,
    selection_probabilities,
)
from loadData_GA import load_sla_tasks


TASK_COUNT = 1000
PIPELINE_RUNS = 20
RUN_STATE_VERSION = 1
SEQUENCE_LENGTHS = {"QWS": 20, "Normal": 40}
HYPOTHESIS_COUNTS = {"QWS": (50,), "Normal": (50, 100)}
EPSILON_VALUES = (1.0, 2.0, 3.0, 4.0, 5.0)
BETA_VALUES = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
CLIP_BOUNDS = {"QWS": 0.4, "Normal": 0.2}
PROXY_POOL_SIZES = {"QWS": 400, "Normal": 100}
DISTANCE_NAME = "euclidean"
MECHANISM = "EM"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fixed-K Differ inference attack with Euclidean distance."
    )
    parser.add_argument("dataset", choices=tuple(SEQUENCE_LENGTHS))
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Retained for API compatibility; hypothesis construction is cheap.",
    )
    parser.add_argument(
        "--task-limit",
        type=int,
        default=None,
        help="Optional smoke-test task count; omit for all 1000 tasks.",
    )
    parser.add_argument(
        "--restart-runs",
        action="store_true",
        help="Discard completed run-level statistics and restart from run 1.",
    )
    return parser.parse_args()


def sample_with_uniform(probabilities, draw: float) -> int:
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += float(probability)
        if draw < cumulative:
            return index
    return len(probabilities) - 1


def result_path(dataset: str) -> Path:
    path = Path(__file__).resolve().parent / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{dataset}_compare_differ_results.txt"


def run_checkpoint_path(dataset: str, sequence_length: int, task_count: int) -> Path:
    directory = Path(__file__).resolve().parent / "results" / "run_checkpoints"
    return directory / (
        f"{dataset}_compare_differ_L{sequence_length}_N{task_count}_"
        f"R{PIPELINE_RUNS}_v{RUN_STATE_VERSION}.pkl"
    )


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def append_metric_table(lines, title, completed_runs, hypothesis_count):
    lines.extend(
        [
            "",
            f"[{title}]",
            "Beta\t"
            + "\t".join(f"PRA(eps={epsilon:g})" for epsilon in EPSILON_VALUES)
            + "\t"
            + "\t".join(
                f"NMAE(eps={epsilon:g})" for epsilon in EPSILON_VALUES
            ),
        ]
    )
    for beta in BETA_VALUES:
        lines.append(
            f"{beta:.2f}\t"
            + "\t".join(
                format_mean_std(
                    completed_runs,
                    lambda run, c=hypothesis_count, b=beta, e=epsilon: run[
                        "dp"
                    ][(c, b, e)]["attack_accuracy_percent"],
                    digits=4,
                )
                for epsilon in EPSILON_VALUES
            )
            + "\t"
            + "\t".join(
                format_mean_std(
                    completed_runs,
                    lambda run, c=hypothesis_count, b=beta, e=epsilon: run[
                        "dp"
                    ][(c, b, e)]["constraint_nmae"],
                    scale=1000.0,
                    digits=6,
                )
                for epsilon in EPSILON_VALUES
            )
        )


def render_results(
    dataset,
    sequence_length,
    proxy_pool_size,
    clip_bound,
    task_count,
    hypothesis_counts,
    completed_runs,
):
    lines = [
        "DIFFER INFERENCE ATTACK COMPARISON",
        f"Dataset: {dataset}",
        f"TasksPerRun: {task_count}",
        f"RunsCompleted: {len(completed_runs)}/{PIPELINE_RUNS}",
        "Statistic: run-level Mean±SampleSD",
        f"SequenceLength: {sequence_length}",
        f"ProxyPoolSize: {proxy_pool_size}",
        f"ClipBound: {clip_bound:g}",
        "Distance: euclidean",
        "NMAEScale: x10^-3",
        "",
        "[Non-DP]",
        "K\tPRA(%)\tNMAE(x10^-3)",
    ]
    for count in hypothesis_counts:
        lines.append(
            f"{count}\t"
            + format_mean_std(
                completed_runs,
                lambda run, c=count: run["non_dp"][c][
                    "attack_accuracy_percent"
                ],
                digits=4,
            )
            + "\t"
            + format_mean_std(
                completed_runs,
                lambda run, c=count: run["non_dp"][c]["constraint_nmae"],
                scale=1000.0,
                digits=6,
            )
        )
    lines.extend(["", "[Pool]", "Beta\tMeanRepresentativeCount"])
    for beta in BETA_VALUES:
        lines.append(
            f"{beta:.2f}\t"
            + format_mean_std(
                completed_runs,
                lambda run, b=beta: run["pool"][b],
                digits=4,
            )
        )
    for count in hypothesis_counts:
        append_metric_table(
            lines,
            f"Fixed K={count}, Euclidean",
            completed_runs,
            count,
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive.")
    if args.task_limit is not None and args.task_limit <= 0:
        raise ValueError("--task-limit must be positive.")

    dataset = args.dataset
    sequence_length = SEQUENCE_LENGTHS[dataset]
    hypothesis_counts = HYPOTHESIS_COUNTS[dataset]
    clip_bound = CLIP_BOUNDS[dataset]
    cfg = load_config(dataset=dataset)
    validate_dp_para(dp_para)

    all_tasks = load_sla_tasks(
        dataset,
        cfg.node_file,
        cfg.service_file,
        test_only=True,
        min_candidates=1,
    )
    if len(all_tasks) != TASK_COUNT:
        raise RuntimeError(
            f"Expected {TASK_COUNT} held-out tasks, found {len(all_tasks)}."
        )
    tasks = [shorten_task(task, sequence_length) for task in all_tasks]
    if args.task_limit is not None:
        tasks = tasks[: args.task_limit]

    repository = inference.load_public_service_repository(
        dataset, cfg.node_file, cfg.service_file
    )
    training_bank = load_training_constraint_bank(
        dataset, cfg.node_file, cfg.service_file, repository
    )
    models_by_k, _, _ = build_or_load_nested_attack_models(
        dataset=dataset,
        tasks=tasks,
        training_bank=training_bank,
        repository=repository,
        hypothesis_counts=hypothesis_counts,
        cfg=cfg,
        workers=args.workers,
    )

    optimal_count = int(dp_para["optimal_candidate_count"])
    proxy_pool_size = PROXY_POOL_SIZES[dataset]
    task_count = len(tasks)
    output = result_path(dataset)
    checkpoint = run_checkpoint_path(dataset, sequence_length, task_count)
    run_metadata = {
        "version": RUN_STATE_VERSION,
        "experiment": "compare-differ-euclidean",
        "dataset": dataset,
        "task_indices": [int(task.task_index) for task in tasks],
        "sequence_length": sequence_length,
        "hypothesis_counts": hypothesis_counts,
        "epsilon_values": EPSILON_VALUES,
        "beta_values": BETA_VALUES,
        "clip_bound": clip_bound,
        "proxy_pool_size": proxy_pool_size,
        "pipeline_runs": PIPELINE_RUNS,
        "ga_seed": int(cfg.ga_seed),
    }
    completed_runs = load_completed_runs(
        checkpoint,
        run_metadata,
        restart=args.restart_runs,
    )
    if len(completed_runs) > PIPELINE_RUNS:
        raise RuntimeError("Run checkpoint contains more runs than configured.")
    if completed_runs:
        atomic_write(
            output,
            render_results(
                dataset,
                sequence_length,
                proxy_pool_size,
                clip_bound,
                task_count,
                hypothesis_counts,
                completed_runs,
            ),
        )
        print(f"Resuming after {len(completed_runs)}/{PIPELINE_RUNS} completed runs.")
    else:
        atomic_write(
            output,
            "\n".join(
                [
                    "DIFFER INFERENCE ATTACK COMPARISON",
                    f"Dataset: {dataset}",
                    f"TasksPerRun: {task_count}",
                    f"RunsCompleted: 0/{PIPELINE_RUNS}",
                    "Status: run 1 is pending or in progress; no complete run is included.",
                    "",
                ]
            ),
        )

    for run_index in range(len(completed_runs) + 1, PIPELINE_RUNS + 1):
        run_started = time.time()
        non_dp_rows = {count: [] for count in hypothesis_counts}
        dp_rows = {
            (count, beta, epsilon): []
            for count in hypothesis_counts
            for beta in BETA_VALUES
            for epsilon in EPSILON_VALUES
        }
        representative_totals = {beta: 0 for beta in BETA_VALUES}

        for task in tqdm(
            tasks,
            desc=f"Compare Differ: {dataset} run {run_index}/{PIPELINE_RUNS}",
        ):
            task_index = int(task.task_index)
            base_model = models_by_k[hypothesis_counts[0]][task_index]
            victim_task = inference.make_hypothetical_task(
                base_task=task,
                hypothesis=base_model.true_hypothesis,
                repository=repository,
            )
            source_seed = cfg.ga_seed + run_index * 10_000_019 + task_index
            source = build_candidate_source(
                task=victim_task,
                cfg=cfg,
                seed=source_seed,
                optimal_count=optimal_count,
                proxy_pool_size=proxy_pool_size,
            )

            non_dp_candidate = select_candidate(
                inference.PoolView(
                    candidates=list(source.quality_candidates[:optimal_count])
                ),
                "Non_DP",
            )
            non_dp_observation = inference.selected_candidate_observation(
                non_dp_candidate, victim_task
            )
            for count in hypothesis_counts:
                non_dp_rows[count].append(
                    infer_constraint(
                        non_dp_observation,
                        models_by_k[count][task_index],
                        repository,
                        DISTANCE_NAME,
                    )
                )

            common_draw = random.Random(source_seed ^ 0xD1FF3E).random()
            for beta in BETA_VALUES:
                pool_parameters = dict(dp_para)
                pool_parameters["max_utility_degradation"] = beta
                pool_parameters["pool_construction_mechanism"] = MECHANISM
                pool_parameters["pool_construction_epsilon"] = min(EPSILON_VALUES)
                pool_parameters["pool_construction_clip_bound"] = clip_bound
                pool = build_representative_pool(
                    source=source,
                    optimal_count=optimal_count,
                    para=pool_parameters,
                )
                representative_totals[beta] += pool.representative_count
                for epsilon in EPSILON_VALUES:
                    probabilities = selection_probabilities(
                        candidates=pool.candidates,
                        mechanism=MECHANISM,
                        epsilon=epsilon,
                        clip_bound=clip_bound,
                    )
                    selected = pool.candidates[
                        sample_with_uniform(probabilities, common_draw)
                    ]
                    observation = inference.selected_candidate_observation(
                        selected, victim_task
                    )
                    for count in hypothesis_counts:
                        dp_rows[(count, beta, epsilon)].append(
                            infer_constraint(
                                observation,
                                models_by_k[count][task_index],
                                repository,
                                DISTANCE_NAME,
                            )
                        )

        completed_runs.append(
            {
                "run_index": run_index,
                "non_dp": {
                    count: mean_metrics(rows)
                    for count, rows in non_dp_rows.items()
                },
                "dp": {key: mean_metrics(rows) for key, rows in dp_rows.items()},
                "pool": {
                    beta: representative_totals[beta] / task_count
                    for beta in BETA_VALUES
                },
            }
        )
        save_completed_runs(checkpoint, run_metadata, completed_runs)
        atomic_write(
            output,
            render_results(
                dataset,
                sequence_length,
                proxy_pool_size,
                clip_bound,
                task_count,
                hypothesis_counts,
                completed_runs,
            ),
        )
        print(
            f"Run {run_index}/{PIPELINE_RUNS} completed and saved | "
            f"elapsed={time.time() - run_started:.2f}s | result={output}"
        )

    print(f"All {PIPELINE_RUNS} runs completed. Result file: {output}")


if __name__ == "__main__":
    main()
