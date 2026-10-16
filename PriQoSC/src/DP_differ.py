"""Closed-set DP user-constraint inference attack.

Usage: python DP_differ.py QWS|Normal
"""

from __future__ import annotations

import argparse
import os
import random
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
from dp_perturb import dataset_dp_settings, dp_para, validate_dp_para  # noqa: E402
from dp_qosc import (  # noqa: E402
    build_candidate_source,
    build_representative_pool,
    selection_probabilities,
)
from loadData_GA import load_sla_tasks  # noqa: E402

TASK_COUNT = 1000
RUN_INDEX = 1
SEQUENCE_LENGTHS = {"QWS": 20, "Normal": 40}
HYPOTHESIS_COUNTS = {"QWS": (50,), "Normal": (50, 100)}
DISTANCE_NAMES = ("euclidean", "manhattan")
EPSILON_VALUES = (1.0, 2.0, 3.0, 4.0, 5.0)
UTILITY_DEGRADATION_BOUNDS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
DP_SETTINGS = {
    "QWS": {"mechanism": "EM", "clip_bound": 0.4, "proxy_pool_size": 1000},
    "Normal": {"mechanism": "EM", "clip_bound": 0.2},
}


def parse_args() -> str:
    parser = argparse.ArgumentParser(
        description="Infer the synthetic constraint from a DP release."
    )
    parser.add_argument("dataset", choices=tuple(SEQUENCE_LENGTHS))
    return parser.parse_args().dataset


def sample_with_uniform(probabilities, draw: float) -> int:
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += float(probability)
        if draw < cumulative:
            return index
    return len(probabilities) - 1


def main() -> None:
    dataset = parse_args()
    sequence_length = SEQUENCE_LENGTHS[dataset]
    hypothesis_counts = HYPOTHESIS_COUNTS[dataset]
    setting = DP_SETTINGS[dataset]
    clip_bound = float(setting["clip_bound"])
    mechanism = str(setting["mechanism"])
    cfg = load_config(dataset=dataset)
    validate_dp_para(dp_para)

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

    dataset_settings = dataset_dp_settings(dataset, dp_para)
    optimal_count = int(dp_para["optimal_candidate_count"])
    proxy_pool_size = int(
        setting.get("proxy_pool_size", dataset_settings["proxy_pool_size"])
    )
    attack_rows = {
        (hypothesis_count, beta, epsilon, distance): []
        for hypothesis_count in hypothesis_counts
        for beta in UTILITY_DEGRADATION_BOUNDS
        for epsilon in EPSILON_VALUES
        for distance in DISTANCE_NAMES
    }
    release_changes = {beta: 0 for beta in UTILITY_DEGRADATION_BOUNDS}
    pmf_total_variation = {beta: 0.0 for beta in UTILITY_DEGRADATION_BOUNDS}
    representative_count_total = {
        beta: 0 for beta in UTILITY_DEGRADATION_BOUNDS
    }

    for task in tqdm(tasks, desc=f"DP differ attack: {dataset}"):
        victim_model = models_by_k[hypothesis_counts[0]][int(task.task_index)]
        # The held-out task supplies only the public workflow structure and
        # global experimental context.  Its private local constraints are
        # replaced by the training-only synthetic ground-truth hypothesis.
        victim_task = inference.make_hypothetical_task(
            base_task=task,
            hypothesis=victim_model.true_hypothesis,
            repository=repository,
        )
        source_seed = cfg.ga_seed + RUN_INDEX * 10_000_019 + task.task_index
        source = build_candidate_source(
            task=victim_task,
            cfg=cfg,
            seed=source_seed,
            optimal_count=optimal_count,
            proxy_pool_size=proxy_pool_size,
        )
        common_draw = random.Random(source_seed ^ 0xD1FF3E).random()

        for beta in UTILITY_DEGRADATION_BOUNDS:
            pool_parameters = dict(dp_para)
            pool_parameters["max_utility_degradation"] = float(beta)
            pool_parameters["pool_construction_mechanism"] = mechanism
            pool_parameters["pool_construction_epsilon"] = min(EPSILON_VALUES)
            pool_parameters["pool_construction_clip_bound"] = clip_bound
            pool = build_representative_pool(
                source=source,
                optimal_count=optimal_count,
                para=pool_parameters,
            )
            representative_count_total[beta] += int(pool.representative_count)
            probabilities = {
                epsilon: selection_probabilities(
                    candidates=pool.candidates,
                    mechanism=mechanism,
                    epsilon=epsilon,
                    clip_bound=clip_bound,
                )
                for epsilon in EPSILON_VALUES
            }
            low = probabilities[min(EPSILON_VALUES)]
            high = probabilities[max(EPSILON_VALUES)]
            pmf_total_variation[beta] += 0.5 * sum(
                abs(float(left) - float(right)) for left, right in zip(low, high)
            )

            selected_indices = {}
            for epsilon in EPSILON_VALUES:
                selected_index = sample_with_uniform(probabilities[epsilon], common_draw)
                selected_indices[epsilon] = selected_index
                observation = inference.selected_candidate_observation(
                    pool.candidates[selected_index], victim_task
                )
                for hypothesis_count in hypothesis_counts:
                    model = models_by_k[hypothesis_count][int(task.task_index)]
                    for distance_name in DISTANCE_NAMES:
                        attack_rows[
                            (hypothesis_count, beta, epsilon, distance_name)
                        ].append(
                            infer_constraint(
                                observation, model, repository, distance_name
                            )
                        )
            if selected_indices[min(EPSILON_VALUES)] != selected_indices[max(EPSILON_VALUES)]:
                release_changes[beta] += 1

    result_lines = [
        f"Tasks: {len(tasks)}",
        f"SequenceLength: {sequence_length}",
        f"Mechanism: {mechanism}",
        f"ClipBound: {clip_bound:g}",
        f"ProxyPoolSize: {proxy_pool_size}",
        "NMAEScale: x10^-3",
        "",
        "[Pool diagnostics]",
        "Beta\tMeanRepresentativeCount\tReleaseChangedEps1To5\tMeanPMFTV",
    ]
    for beta in UTILITY_DEGRADATION_BOUNDS:
        result_lines.append(
            f"{beta:.2f}\t"
            f"{representative_count_total[beta] / len(tasks):.4f}\t"
            f"{release_changes[beta]}/{len(tasks)}\t"
            f"{pmf_total_variation[beta] / len(tasks):.10f}"
        )

    epsilon_headers = [f"PRA(eps={epsilon:g})" for epsilon in EPSILON_VALUES]
    nmae_headers = [f"NMAE(eps={epsilon:g})" for epsilon in EPSILON_VALUES]
    for hypothesis_count in hypothesis_counts:
        for distance_name in DISTANCE_NAMES:
            result_lines.extend(
                [
                    "",
                    (
                        f"[K={hypothesis_count}, Distance={distance_name}, "
                        f"RandomGuessPRA={100.0 / hypothesis_count:.4f}%]"
                    ),
                    "Beta\t" + "\t".join(epsilon_headers + nmae_headers),
                ]
            )
            for beta in UTILITY_DEGRADATION_BOUNDS:
                pra_values = []
                nmae_values = []
                for epsilon in EPSILON_VALUES:
                    attack = mean_metrics(
                        attack_rows[
                            (hypothesis_count, beta, epsilon, distance_name)
                        ]
                    )
                    pra_values.append(
                        f"{attack['attack_accuracy_percent']:.4f}"
                    )
                    nmae_values.append(
                        f"{1000.0 * attack['constraint_nmae']:.6f}"
                    )
                result_lines.append(
                    f"{beta:.2f}\t" + "\t".join(pra_values + nmae_values)
                )
    result_path = save_result_section(
        dataset=dataset,
        section_name="DP",
        content="\n".join(result_lines),
    )

    print("\n" + "=" * 112)
    print("DP CLOSED-SET USER-CONSTRAINT INFERENCE ATTACK")
    print("=" * 112)
    print(f"Dataset                 : {dataset}")
    print(f"Tasks                   : {len(tasks)}")
    print(f"Sequence length         : {sequence_length}")
    print(f"Constraint hypotheses K : {hypothesis_counts} (nested, shared truth)")
    print("Attacker knowledge       : K constraints + released per-service QoS only")
    print("Ground truth source      : random member of the training-only hypothesis set")
    print("Candidate construction   : same-category empirical constraints, locally feasible")
    print("Distance target          : released q2/q3 to hypothesized upper endpoints")
    print(f"Release mechanism       : {mechanism}")
    print(f"Epsilon values          : {EPSILON_VALUES}")
    print(f"Clip bound              : {clip_bound}")
    print(f"Proxy pool size         : {proxy_pool_size}")
    print(f"Beta values             : {UTILITY_DEGRADATION_BOUNDS}")
    print("Hypothesis storage      : in memory (no shared cache file)")

    for beta in UTILITY_DEGRADATION_BOUNDS:
        print("-" * 112)
        print(
            f"Beta={beta:.2f} | release changed eps 1->5: "
            f"{release_changes[beta]}/{len(tasks)} | mean PMF TV: "
            f"{pmf_total_variation[beta] / len(tasks):.10f}"
        )
        print(
            "Epsilon | Attack distance | PRA (%) | Constraint NMAE"
        )
        for hypothesis_count in hypothesis_counts:
            for epsilon in EPSILON_VALUES:
                for distance_name in DISTANCE_NAMES:
                    attack = mean_metrics(
                        attack_rows[
                            (hypothesis_count, beta, epsilon, distance_name)
                        ]
                    )
                    print(
                        f"K={hypothesis_count:3d} | {epsilon:7.1f} | "
                        f"{distance_name:15s} | "
                        f"{attack['attack_accuracy_percent']:7.4f} | "
                        f"{attack['constraint_nmae']:15.10f}"
                    )
    print(f"Result file: {result_path}")
    print("=" * 112)


if __name__ == "__main__":
    main()
