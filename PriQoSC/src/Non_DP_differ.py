

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import Inference_Attack_PoolSize_Representative as inference  # noqa: E402
from config_loader import load_config  # noqa: E402
from differ_attack_core import (  # noqa: E402
    build_or_load_nested_attack_models,
    infer_constraint,
    load_training_constraint_bank,
    mean_metrics,
    save_result_section,
    shorten_task,
)
from loadData_GA import load_sla_tasks  # noqa: E402
from sla_core import SLAOnDemandGA  # noqa: E402

TASK_COUNT = 1000
SEQUENCE_LENGTHS = {"QWS": 20, "Normal": 40}
HYPOTHESIS_COUNTS = {"QWS": (50,), "Normal": (50, 100)}
DISTANCE_NAMES = ("euclidean", "manhattan")


def parse_args() -> str:
    parser = argparse.ArgumentParser(
        description="Infer the synthetic constraint from a Non-DP release."
    )
    parser.add_argument("dataset", choices=tuple(SEQUENCE_LENGTHS))
    return parser.parse_args().dataset


def main() -> None:
    dataset = parse_args()
    sequence_length = SEQUENCE_LENGTHS[dataset]
    hypothesis_counts = HYPOTHESIS_COUNTS[dataset]
    cfg = load_config(dataset=dataset)
    all_tasks = load_sla_tasks(
        dataset=dataset,
        node_filename=cfg.node_file,
        service_filename=cfg.service_file,
        test_only=True,
        min_candidates=1,
    )
    if len(all_tasks) != TASK_COUNT:
        raise RuntimeError(
            f"Expected exactly {TASK_COUNT} test tasks, found {len(all_tasks)}."
        )
    tasks = [shorten_task(task, sequence_length) for task in all_tasks]

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
        workers=min(4, os.cpu_count() or 1),
    )

    attack_rows = {
        (hypothesis_count, name): []
        for hypothesis_count in hypothesis_counts
        for name in DISTANCE_NAMES
    }
    for task in tqdm(tasks, desc=f"Non-DP differ attack: {dataset}"):
        victim_model = models_by_k[hypothesis_counts[0]][int(task.task_index)]
        victim_task = inference.make_hypothetical_task(
            base_task=task,
            hypothesis=victim_model.true_hypothesis,
            repository=repository,
        )
        # The victim uses its composition algorithm to produce the release.
        # The attacker receives only the released service QoS values.
        victim_ga = SLAOnDemandGA(
            victim_task,
            population_size=cfg.ga_population,
            max_generations=cfg.ga_max_generations,
            stagnation_patience=cfg.ga_stagnation_patience,
            crossover_rate=cfg.ga_crossover_rate,
            mutation_probability=cfg.ga_mutation_probability,
            elite_count=cfg.ga_elite_count,
            seed=cfg.ga_seed + task.task_index,
        )
        released_sequence, released_evaluation = victim_ga.search()
        observation = inference.ReleaseObservation(
            service_ids=inference.service_identities(released_sequence),
            q2_product=float(released_evaluation.q2_product),
            q3_product=float(released_evaluation.q3_product),
        )
        for hypothesis_count in hypothesis_counts:
            model = models_by_k[hypothesis_count][int(task.task_index)]
            for distance_name in DISTANCE_NAMES:
                attack_rows[(hypothesis_count, distance_name)].append(
                    infer_constraint(observation, model, repository, distance_name)
                )
    attack_means = {
        key: mean_metrics(rows) for key, rows in attack_rows.items()
    }

    result_lines = [
        f"Tasks: {len(tasks)}",
        f"SequenceLength: {sequence_length}",
        "NMAEScale: x10^-3",
    ]
    for hypothesis_count in hypothesis_counts:
        result_lines.extend(
            [
                "",
                f"K={hypothesis_count}\tRandomGuessPRA(%)={100.0 / hypothesis_count:.4f}",
                "Distance\tPRA(%)\tNMAE(x10^-3)",
            ]
        )
        for name in DISTANCE_NAMES:
            values = attack_means[(hypothesis_count, name)]
            result_lines.append(
                f"{name}\t{values['attack_accuracy_percent']:.4f}\t"
                f"{1000.0 * values['constraint_nmae']:.6f}"
            )
    result_path = save_result_section(
        dataset=dataset,
        section_name="NON_DP",
        content="\n".join(result_lines),
    )

    print("\n" + "=" * 86)
    print("NON-DP CLOSED-SET USER-CONSTRAINT INFERENCE ATTACK")
    print("=" * 86)
    print(f"Dataset                 : {dataset}")
    print(f"Tasks                   : {len(tasks)}")
    print(f"Sequence length         : {sequence_length}")
    print(f"Constraint hypotheses K : {hypothesis_counts} (nested, shared truth)")
    print("Attacker knowledge       : K constraints + released per-service QoS only")
    print("Ground truth source      : random member of the training-only hypothesis set")
    print("Candidate construction   : same-category empirical constraints, locally feasible")
    print("Distance target          : released q2/q3 to hypothesized upper endpoints")
    print("Hypothesis storage       : in memory (no shared cache file)")
    print("-" * 86)
    print("K   | Attack distance   | PRA (%) | Constraint NMAE")
    for hypothesis_count in hypothesis_counts:
        for name in DISTANCE_NAMES:
            values = attack_means[(hypothesis_count, name)]
            print(
                f"{hypothesis_count:3d} | {name:17s} | "
                f"{values['attack_accuracy_percent']:7.4f} | "
                f"{values['constraint_nmae']:15.10f}"
            )
    print(f"Result file            : {result_path}")
    print("=" * 86)


if __name__ == "__main__":
    main()
