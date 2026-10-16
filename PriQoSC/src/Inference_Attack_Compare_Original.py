
from __future__ import annotations

import argparse
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

import Inference_Attack_PoolSize_Representative as attack
from comparison_run_stats import (
    format_mean_std,
    load_completed_runs,
    save_completed_runs,
)
from config_loader import load_config
from differ_attack_core import shorten_task
from dp_perturb import dp_para, validate_dp_para
from dp_qosc import (
    build_candidate_source,
    build_representative_pool,
    selection_probabilities,
)
from loadData_GA import load_sla_tasks


TASK_COUNT = 1000
PIPELINE_RUNS = 20
RUN_STATE_VERSION = 1
SEQUENCE_LENGTHS = {"QWS": 20, "Normal": 40}
FIXED_HYPOTHESIS_COUNTS = {"QWS": (50,), "Normal": (50, 100)}
EPSILON_VALUES = (1.0, 2.0, 3.0, 4.0, 5.0)
BETA_VALUES = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
CLIP_BOUNDS = {"QWS": 0.4, "Normal": 0.2}
PROXY_POOL_SIZES = {"QWS": 400, "Normal": 100}
DISTANCE_NAME = "euclidean"
MECHANISM = "EM"
BASE_HYPOTHESIS_COUNT = 50
CACHE_VERSION = 1
CACHE_SAVE_EVERY_TASKS = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Algorithm-aware Euclidean inference attack with fixed-K and "
            "fixed/native sequence-length modes."
        )
    )
    parser.add_argument("dataset", choices=tuple(SEQUENCE_LENGTHS))
    parser.add_argument(
        "--length-mode",
        choices=("fixed", "native"),
        default="fixed",
        help=(
            "fixed uses QWS L=20 and Normal L=40; native preserves each "
            "task's original sequence length."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Processes used to build public service-composition templates.",
    )
    parser.add_argument(
        "--task-limit",
        type=int,
        default=None,
        help="Optional smoke-test task count; omit for all 1000 tasks.",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Discard this comparison script's independent template cache.",
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


def result_path(dataset: str, length_mode: str) -> Path:
    directory = Path(__file__).resolve().parent / "results"
    directory.mkdir(parents=True, exist_ok=True)
    if length_mode == "native":
        return directory / f"{dataset}_compare_original_native_length_results.txt"
    return directory / f"{dataset}_compare_original_results.txt"


def sequence_token(sequence_length) -> str:
    return "native" if sequence_length is None else str(int(sequence_length))


def sequence_label(sequence_length) -> str:
    return "native task lengths" if sequence_length is None else str(
        int(sequence_length)
    )


def run_checkpoint_path(dataset: str, sequence_length, task_count: int) -> Path:
    directory = Path(__file__).resolve().parent / "results" / "run_checkpoints"
    return directory / (
        f"{dataset}_compare_original_L{sequence_token(sequence_length)}_N{task_count}_"
        f"R{PIPELINE_RUNS}_v{RUN_STATE_VERSION}.pkl"
    )


def cache_path(dataset: str, sequence_length, task_count: int) -> Path:
    directory = Path(__file__).resolve().parent / "results" / "attack_cache"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / (
        f"{dataset}_compare_original_L{sequence_token(sequence_length)}_N{task_count}_"
        f"base{BASE_HYPOTHESIS_COUNT}_v{CACHE_VERSION}.pkl"
    )


def cache_metadata(dataset, cfg, tasks, sequence_length):
    metadata = attack.cache_metadata(
        dataset=dataset,
        cfg=cfg,
        tasks=tasks,
        base_count=BASE_HYPOTHESIS_COUNT,
    )
    if sequence_length is None:
        metadata.update(
            {
                "comparison_cache_version": CACHE_VERSION,
                "comparison_attack": "native-length-original-template-euclidean",
                "sequence_length": "native",
                "task_sequence_lengths": [
                    len(task.local_constraints) for task in tasks
                ],
            }
        )
    else:
        metadata.update(
            {
                "comparison_cache_version": CACHE_VERSION,
                "comparison_attack": "fixed-length-original-template-euclidean",
                "sequence_length": int(sequence_length),
            }
        )
    return metadata


def build_or_load_models(
    dataset,
    tasks,
    training_bank,
    repository,
    sequence_length,
    cfg,
    workers,
    rebuild_cache,
):
    path = cache_path(dataset, sequence_length, len(tasks))
    metadata = cache_metadata(dataset, cfg, tasks, sequence_length)
    if rebuild_cache:
        models = {}
    else:
        models, complete = attack.load_attack_cache(path, metadata)
        expected = {int(task.task_index) for task in tasks}
        if complete and set(models) == expected:
            return models, path, True

    expected = {int(task.task_index) for task in tasks}
    models = {
        int(task_index): model
        for task_index, model in models.items()
        if int(task_index) in expected
    }
    missing = [task for task in tasks if int(task.task_index) not in models]

    def save(complete):
        attack.save_attack_cache(path, metadata, models, complete)

    since_save = 0
    if workers <= 1:
        for task in tqdm(missing, desc="Build original attack templates"):
            models[int(task.task_index)] = attack.build_initial_attack_model(
                base_task=task,
                training_bank=training_bank,
                repository=repository,
                base_count=BASE_HYPOTHESIS_COUNT,
                cfg=cfg,
            )
            since_save += 1
            if since_save >= CACHE_SAVE_EVERY_TASKS:
                save(False)
                since_save = 0
    elif missing:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=attack.init_worker,
            initargs=(
                tasks,
                training_bank,
                repository,
                cfg,
                BASE_HYPOTHESIS_COUNT,
            ),
        ) as executor:
            futures = {
                executor.submit(attack.build_worker, int(task.task_index)): int(
                    task.task_index
                )
                for task in missing
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Build original attack templates",
            ):
                task_index = futures[future]
                models[task_index] = future.result()
                since_save += 1
                if since_save >= CACHE_SAVE_EVERY_TASKS:
                    save(False)
                    since_save = 0

    if set(models) != expected:
        raise RuntimeError("Original attack-template cache is incomplete.")
    save(True)
    return models, path, False


def add_metric(accumulators, key, metrics):
    accumulators.setdefault(
        key, attack.MetricAccumulator(attack.ATTACK_METRIC_KEYS)
    ).add(metrics)


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def append_fixed_table(lines, title, completed_runs, hypothesis_count):
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
                        "metrics"
                    ][("fixed", c, b, e)]["attack_accuracy_percent"],
                    digits=4,
                )
                for epsilon in EPSILON_VALUES
            )
            + "\t"
            + "\t".join(
                format_mean_std(
                    completed_runs,
                    lambda run, c=hypothesis_count, b=beta, e=epsilon: run[
                        "metrics"
                    ][("fixed", c, b, e)]["constraint_nmae"],
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
    fixed_counts,
    completed_runs,
):
    lines = [
        "ORIGINAL ALGORITHM-AWARE INFERENCE ATTACK COMPARISON",
        f"Dataset: {dataset}",
        f"TasksPerRun: {task_count}",
        f"RunsCompleted: {len(completed_runs)}/{PIPELINE_RUNS}",
        "Statistic: run-level Mean±SampleSD",
        f"SequenceLength: {sequence_label(sequence_length)}",
        f"ProxyPoolSize: {proxy_pool_size}",
        f"ClipBound: {clip_bound:g}",
        "Distance: euclidean",
        "NMAEScale: x10^-3",
        "",
        "[Non-DP - Strong-background inference]",
        "K\tPRA(%)\tNMAE(x10^-3)",
    ]
    for count in fixed_counts:
        lines.append(
            f"{count}\t"
            + format_mean_std(
                completed_runs,
                lambda run, c=count: run["metrics"][("conventional", c)][
                    "attack_accuracy_percent"
                ],
                digits=4,
            )
            + "\t"
            + format_mean_std(
                completed_runs,
                lambda run, c=count: run["metrics"][("conventional", c)][
                    "constraint_nmae"
                ],
                scale=1000.0,
                digits=6,
            )
        )
    for count in fixed_counts:
        append_fixed_table(
            lines,
            f"Fixed K={count}, Euclidean",
            completed_runs,
            count,
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    started_at = time.time()
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive.")
    if args.task_limit is not None and args.task_limit <= 0:
        raise ValueError("--task-limit must be positive.")

    dataset = args.dataset
    sequence_length = (
        SEQUENCE_LENGTHS[dataset] if args.length_mode == "fixed" else None
    )
    fixed_counts = FIXED_HYPOTHESIS_COUNTS[dataset]
    clip_bound = CLIP_BOUNDS[dataset]
    proxy_pool_size = PROXY_POOL_SIZES[dataset]
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
    tasks = (
        [shorten_task(task, sequence_length) for task in all_tasks]
        if sequence_length is not None
        else list(all_tasks)
    )
    if args.task_limit is not None:
        tasks = tasks[: args.task_limit]

    repository = attack.load_public_service_repository(
        dataset, cfg.node_file, cfg.service_file
    )
    training_bank = attack.load_training_requirement_bank(
        dataset, cfg.node_file, cfg.service_file, repository
    )
    models, model_cache, cache_hit = build_or_load_models(
        dataset=dataset,
        tasks=tasks,
        training_bank=training_bank,
        repository=repository,
        sequence_length=sequence_length,
        cfg=cfg,
        workers=args.workers,
        rebuild_cache=args.rebuild_cache,
    )
    print(
        f"Template cache: {model_cache} "
        f"({'HIT' if cache_hit else 'BUILT/UPDATED'})"
    )

    optimal_count = int(dp_para["optimal_candidate_count"])
    task_count = len(tasks)
    output = result_path(dataset, args.length_mode)
    checkpoint = run_checkpoint_path(dataset, sequence_length, task_count)
    run_metadata = {
        "version": RUN_STATE_VERSION,
        "experiment": "compare-original-euclidean",
        "dataset": dataset,
        "task_indices": [int(task.task_index) for task in tasks],
        "sequence_length": (
            int(sequence_length) if sequence_length is not None else "native"
        ),
        "fixed_counts": fixed_counts,
        "epsilon_values": EPSILON_VALUES,
        "beta_values": BETA_VALUES,
        "clip_bound": clip_bound,
        "proxy_pool_size": proxy_pool_size,
        "pipeline_runs": PIPELINE_RUNS,
        "ga_seed": int(cfg.ga_seed),
    }
    if sequence_length is None:
        run_metadata["task_sequence_lengths"] = [
            len(task.local_constraints) for task in tasks
        ]
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
                fixed_counts,
                completed_runs,
            ),
        )
        print(f"Resuming after {len(completed_runs)}/{PIPELINE_RUNS} completed runs.")
    else:
        atomic_write(
            output,
            "\n".join(
                [
                    "ORIGINAL ALGORITHM-AWARE INFERENCE ATTACK COMPARISON",
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
        accumulators = {}
        expanded_since_save = 0
        for task in tqdm(
            tasks,
            desc=(
                f"Compare original attack: {dataset} "
                f"run {run_index}/{PIPELINE_RUNS}"
            ),
        ):
            task_index = int(task.task_index)
            model = models[task_index]
            victim_task = attack.make_hypothetical_task(
                base_task=task,
                hypothesis=model.true_hypothesis,
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
            true_template = model.true_template()
            conventional_observation = attack.ReleaseObservation(
                service_ids=true_template.service_ids,
                qos_sequence=true_template.qos_sequence,
                q2_product=true_template.q2_product,
                q3_product=true_template.q3_product,
            )

            pools = {}
            maximum_required_k = max(fixed_counts)
            for beta in BETA_VALUES:
                pool_parameters = dict(dp_para)
                pool_parameters["max_utility_degradation"] = beta
                pool_parameters["pool_construction_mechanism"] = MECHANISM
                pool_parameters["pool_construction_epsilon"] = min(
                    EPSILON_VALUES
                )
                pool_parameters["pool_construction_clip_bound"] = clip_bound
                pool = build_representative_pool(
                    source=source,
                    optimal_count=optimal_count,
                    para=pool_parameters,
                )
                pools[beta] = pool

            added = attack.ensure_attack_template_count(
                attack_model=model,
                target_count=maximum_required_k,
                base_task=task,
                training_bank=training_bank,
                repository=repository,
                cfg=cfg,
            )
            if added:
                expanded_since_save += 1
                if expanded_since_save >= CACHE_SAVE_EVERY_TASKS:
                    attack.save_attack_cache(
                        model_cache,
                        cache_metadata(dataset, cfg, tasks, sequence_length),
                        models,
                        True,
                    )
                    expanded_since_save = 0

            for count in fixed_counts:
                add_metric(
                    accumulators,
                    ("conventional", count),
                    attack.attack_observation(
                        model,
                        conventional_observation,
                        count,
                        DISTANCE_NAME,
                        optimal_count,
                        0,
                    ),
                )

            common_draw = random.Random(source_seed ^ 0x0A11CE).random()
            for beta in BETA_VALUES:
                pool = pools[beta]
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
                    observation = attack.selected_candidate_observation(
                        selected, victim_task
                    )
                    for count in fixed_counts:
                        add_metric(
                            accumulators,
                            ("fixed", count, beta, epsilon),
                            attack.attack_observation(
                                model,
                                observation,
                                count,
                                DISTANCE_NAME,
                                len(pool.candidates),
                                pool.representative_count,
                            ),
                        )

        if expanded_since_save:
            attack.save_attack_cache(
                model_cache,
                cache_metadata(dataset, cfg, tasks, sequence_length),
                models,
                True,
            )

        completed_runs.append(
            {
                "run_index": run_index,
                "metrics": {
                    key: accumulator.mean()
                    for key, accumulator in accumulators.items()
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
                fixed_counts,
                completed_runs,
            ),
        )
        print(
            f"Run {run_index}/{PIPELINE_RUNS} completed and saved | "
            f"run_elapsed={time.time() - run_started:.2f}s | "
            f"total_elapsed={time.time() - started_at:.2f}s | result={output}"
        )

    print(f"All {PIPELINE_RUNS} runs completed. Result file: {output}")


if __name__ == "__main__":
    main()
