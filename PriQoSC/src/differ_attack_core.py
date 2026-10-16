"""Low-background closed-set constraint attack for *_differ.py.

The K hypotheses are built exclusively from the first 75% training split.
They are constructed symmetrically; only after construction is one hypothesis
chosen as the synthetic victim constraint.  The held-out task contributes its
public workflow categories, but its private local QoS intervals are never read
by the attacker model.  Online inference uses only released per-service QoS
values and does not run or know the service-composition algorithm.
"""

from __future__ import annotations

import math
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

from tqdm import tqdm

import Inference_Attack_PoolSize_Representative as inference
from loadData_GA import SLATask, Service


DISTANCE_NAMES = ("euclidean", "manhattan", "mae")
DIST_TOL = 1e-12
RESULT_SECTION_ORDER = ("NON_DP", "DP")


@contextmanager
def _exclusive_result_lock(path: Path):
    """Hold a small cross-process lock while updating one dataset result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)

        if os.name == "nt":
            import msvcrt

            while True:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.05)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def _result_markers(section_name: str) -> Tuple[str, str]:
    section_name = str(section_name).upper()
    if section_name not in RESULT_SECTION_ORDER:
        raise ValueError(f"Unsupported differ result section: {section_name}")
    return (
        f"===== BEGIN {section_name} =====",
        f"===== END {section_name} =====",
    )


def _extract_result_section(text: str, section_name: str) -> Optional[str]:
    begin, end = _result_markers(section_name)
    start = text.find(begin)
    if start < 0:
        return None
    finish = text.find(end, start + len(begin))
    if finish < 0:
        return None
    return text[start + len(begin):finish].strip()


def save_result_section(dataset: str, section_name: str, content: str) -> Path:
    """Atomically replace one section and preserve NON_DP-before-DP ordering."""
    dataset = str(dataset)
    section_name = str(section_name).upper()
    _result_markers(section_name)

    result_dir = Path(__file__).resolve().parent / "results"
    result_path = result_dir / f"{dataset}_differ_results.txt"
    lock_path = result_dir / f".{dataset}_differ_results.lock"

    with _exclusive_result_lock(lock_path):
        previous = (
            result_path.read_text(encoding="utf-8")
            if result_path.is_file()
            else ""
        )
        sections = {
            name: _extract_result_section(previous, name)
            for name in RESULT_SECTION_ORDER
        }
        sections[section_name] = str(content).strip()

        lines = [
            "DIFFER INFERENCE ATTACK RESULTS",
            f"Dataset: {dataset}",
            "",
        ]
        for name in RESULT_SECTION_ORDER:
            section = sections[name]
            if section is None:
                continue
            begin, end = _result_markers(name)
            lines.extend([begin, section, end, ""])

        temporary = result_path.with_name(
            f".{result_path.name}.{os.getpid()}.tmp"
        )
        temporary.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        os.replace(temporary, result_path)

    return result_path


def load_training_constraint_bank(
    dataset: str,
    node_filename: str,
    service_filename: str,
    repository: Dict[int, Tuple[Service, ...]],
) -> inference.TrainingRequirementBank:
    """Load category-matched, locally feasible constraints from training only."""
    return inference.load_training_requirement_bank(
        dataset=dataset,
        node_filename=node_filename,
        service_filename=service_filename,
        repository=repository,
    )


def shorten_task(task: SLATask, sequence_length: int) -> SLATask:
    if len(task.services) < sequence_length:
        raise ValueError(
            f"Task {task.task_index} has length {len(task.services)}, "
            f"shorter than requested length {sequence_length}."
        )
    return SLATask(
        task_index=task.task_index,
        services=task.services[:sequence_length],
        global_q2=task.global_q2,
        global_q3=task.global_q3,
        local_constraints=task.local_constraints[:sequence_length],
    )


def build_closed_set_model(
    base_task: SLATask,
    training_bank: inference.TrainingRequirementBank,
    repository: Dict[int, Tuple[Service, ...]],
    hypothesis_count: int,
    cfg,
) -> inference.ExpandableAttackModel:
    """Build K symmetric empirical hypotheses, then choose a synthetic truth."""
    hypothesis_count = int(hypothesis_count)
    if hypothesis_count < 2:
        raise ValueError("hypothesis_count must be at least 2.")

    # For each public workflow position, choose a training-only anchor from the
    # same service category and order the distinct empirical constraints around
    # it.  Rank r across all positions forms hypothesis H_r.  Consequently all
    # K profiles follow exactly the same construction rule and every local
    # interval has at least one public service in its own category.
    ordered_by_position = inference.position_neighbor_lists(
        base_task=base_task,
        training_bank=training_bank,
        cfg=cfg,
    )
    available_ranks = min(len(values) for values in ordered_by_position)
    if available_ranks < hypothesis_count:
        raise RuntimeError(
            f"Task {base_task.task_index}: only {available_ranks} category-matched "
            f"training ranks are available for K={hypothesis_count}."
        )

    hypotheses = [
        inference.profile_from_neighbor_rank(ordered_by_position, rank)
        for rank in range(hypothesis_count)
    ]
    if len({inference.hypothesis_key(value) for value in hypotheses}) != hypothesis_count:
        raise RuntimeError(
            f"Task {base_task.task_index}: training hypotheses are not distinct."
        )
    for hypothesis in hypotheses:
        for requirement in hypothesis.positions:
            if not inference.filter_public_services(repository, requirement):
                raise RuntimeError(
                    f"Task {base_task.task_index}: a category-matched training "
                    "constraint unexpectedly has no feasible public service."
                )

    # Do not read base_task.local_constraints here.  The synthetic truth is
    # designated only after all hypotheses have been built symmetrically.
    rng = random.Random(
        int(cfg.ga_seed) + int(base_task.task_index) * 1_000_003 + 0xD1FFE2
    )
    truth = hypotheses[rng.randrange(hypothesis_count)]

    templates = [
        inference.AttackTemplate(
            hypothesis=hypothesis,
            service_ids=(),
            qos_sequence=(),
            q2_product=0.0,
            q3_product=0.0,
        )
        for hypothesis in hypotheses
    ]

    # Conceal the index of the real constraint in the closed set.
    rng.shuffle(templates)
    return inference.ExpandableAttackModel(
        task_index=int(base_task.task_index),
        true_hypothesis=truth,
        templates=templates,
        next_profile_rank=0,
    )


def build_or_load_attack_models(
    dataset: str,
    tasks: Sequence[SLATask],
    training_bank: inference.TrainingRequirementBank,
    repository: Dict[int, Tuple[Service, ...]],
    hypothesis_count: int,
    cfg,
    workers: int = 4,
):
    """Construct the inexpensive constraint hypotheses in memory.

    Earlier versions cached GA-generated service templates.  The current
    low-background attack has no such templates, and serializing one million
    small interval objects creates a large, contention-prone Windows cache for
    work that is cheaper to rebuild.  Therefore no shared file is read or
    written here.
    """
    tasks = list(tasks)
    if not tasks:
        raise ValueError("At least one task is required.")
    models = {
        int(task.task_index): build_closed_set_model(
            task, training_bank, repository, hypothesis_count, cfg
        )
        for task in tqdm(tasks, desc="Build in-memory constraint hypotheses")
    }
    return models, None, False


def build_or_load_nested_attack_models(
    dataset: str,
    tasks: Sequence[SLATask],
    training_bank: inference.TrainingRequirementBank,
    repository: Dict[int, Tuple[Service, ...]],
    hypothesis_counts: Sequence[int],
    cfg,
    workers: int = 4,
):
    """Build controlled nested K sets with one shared synthetic truth per task.

    The smallest-K templates are a subset of every larger-K set.  The shared
    truth is selected from the smallest set, so changing K adds decoys without
    changing the victim constraint or the released composition.
    """
    del dataset, workers  # Kept in the API for symmetry with the original builder.
    counts = tuple(sorted({int(value) for value in hypothesis_counts}))
    if not counts or counts[0] < 2:
        raise ValueError("hypothesis_counts must contain values >= 2.")

    tasks = list(tasks)
    if not tasks:
        raise ValueError("At least one task is required.")

    models_by_k = {count: {} for count in counts}
    maximum_count = counts[-1]
    minimum_count = counts[0]

    for task in tqdm(tasks, desc="Build nested in-memory constraint hypotheses"):
        # Preserve the original K=50 experiment exactly: its rank-0..49
        # hypotheses, truth-selection RNG, and template shuffle all come from
        # the existing builder.  Larger K values only append decoys.
        baseline = build_closed_set_model(
            task,
            training_bank,
            repository,
            minimum_count,
            cfg,
        )
        templates = list(baseline.templates)
        truth = baseline.true_hypothesis

        if maximum_count > minimum_count:
            ordered_by_position = inference.position_neighbor_lists(
                base_task=task,
                training_bank=training_bank,
                cfg=cfg,
            )
            available_ranks = min(len(values) for values in ordered_by_position)
            if available_ranks < maximum_count:
                raise RuntimeError(
                    f"Task {task.task_index}: only {available_ranks} category-matched "
                    f"training ranks are available for K={maximum_count}."
                )
            seen = {
                inference.hypothesis_key(template.hypothesis)
                for template in templates
            }
            for rank in range(minimum_count, maximum_count):
                hypothesis = inference.profile_from_neighbor_rank(
                    ordered_by_position, rank
                )
                key = inference.hypothesis_key(hypothesis)
                if key in seen:
                    raise RuntimeError(
                        f"Task {task.task_index}: duplicate hypothesis at rank {rank}."
                    )
                seen.add(key)
                templates.append(
                    inference.AttackTemplate(
                        hypothesis=hypothesis,
                        service_ids=(),
                        qos_sequence=(),
                        q2_product=0.0,
                        q3_product=0.0,
                    )
                )

        for count in counts:
            selected_templates = list(templates[:count])
            if not any(
                inference.same_hypothesis(template.hypothesis, truth)
                for template in selected_templates
            ):
                raise RuntimeError(
                    f"Task {task.task_index}: shared truth is absent from K={count}."
                )
            models_by_k[count][int(task.task_index)] = (
                inference.ExpandableAttackModel(
                    task_index=int(task.task_index),
                    true_hypothesis=truth,
                    templates=selected_templates,
                    next_profile_rank=0,
                )
            )

    return models_by_k, None, False


def service_from_identity(
    identity: inference.ServiceIdentity,
    repository: Dict[int, Tuple[Service, ...]],
) -> Service:
    category_id, concrete_index = map(int, identity)
    service = repository[category_id][concrete_index]
    if (int(service[4]), int(service[5])) != (category_id, concrete_index):
        raise RuntimeError("Public service identity lookup is inconsistent.")
    return service


def release_constraint_distance(
    observation: inference.ReleaseObservation,
    hypothesis: inference.RequirementHypothesis,
    repository: Dict[int, Tuple[Service, ...]],
    distance_name: str,
) -> float:
    """Compare released q2/q3 values directly with a constraint hypothesis.

    All distances compare released reliability/availability values with the
    corresponding upper constraint endpoints.  Manhattan is the raw L1 norm,
    MAE is its dimension-normalized form, and Euclidean is the L2 norm.
    """
    if len(observation.service_ids) != len(hypothesis.positions):
        raise ValueError("Release and constraint hypothesis lengths differ.")
    if not observation.service_ids:
        raise ValueError("Release sequence cannot be empty.")

    absolute = 0.0
    squared = 0.0
    for identity, requirement in zip(
        observation.service_ids, hypothesis.positions
    ):
        service = service_from_identity(identity, repository)
        if int(service[4]) != int(requirement.category_id):
            raise RuntimeError("Release and hypothesis workflows do not match.")
        q2 = float(service[2])
        q3 = float(service[3])
        q2_error = q2 - float(requirement.q2_upper)
        q3_error = q3 - float(requirement.q3_upper)
        absolute += abs(q2_error) + abs(q3_error)
        squared += q2_error * q2_error + q3_error * q3_error
    if distance_name == "euclidean":
        return math.sqrt(squared)
    if distance_name == "manhattan":
        return absolute
    if distance_name == "mae":
        return absolute / (2.0 * len(observation.service_ids))
    raise ValueError(f"Unsupported differ attack distance: {distance_name}")


def infer_constraint(
    observation: inference.ReleaseObservation,
    model: inference.ExpandableAttackModel,
    repository: Dict[int, Tuple[Service, ...]],
    distance_name: str,
):
    templates = inference.public_templates(model, len(model.templates))
    distances = tuple(
        release_constraint_distance(
            observation, template.hypothesis, repository, distance_name
        )
        for template in templates
    )
    best = min(distances)
    winners = tuple(
        index for index, distance in enumerate(distances)
        if abs(distance - best) <= DIST_TOL
    )
    result = inference.InferenceResult(winner_indices=winners, distances=distances)
    return inference.evaluate_inference(
        attack_model=model,
        templates=templates,
        inference=result,
        release_pool_size=1,
        representative_count=0,
    )


def mean_metrics(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    if not rows:
        raise ValueError("Cannot average empty results.")
    return {
        key: sum(float(row[key]) for row in rows) / len(rows)
        for key in rows[0]
    }
