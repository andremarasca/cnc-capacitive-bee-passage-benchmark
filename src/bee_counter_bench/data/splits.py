"""Deterministic session-grouped experimental split protocols."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from bee_counter_bench.config import load_yaml
from bee_counter_bench.data.sequence_io import (
    file_sha256,
    load_preprocessed_sequence,
)


@dataclass(frozen=True)
class SessionUnit:
    session_id: str
    collection_id: str
    gcode_id: str
    motion_family: str
    frames: int
    events: int
    low_speed_events: int
    medium_speed_events: int
    high_speed_events: int


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=True, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def _speed_bin(speed_mm_min: float) -> str:
    if speed_mm_min < 1500.0:
        return "low"
    if speed_mm_min < 2500.0:
        return "medium"
    return "high"


def load_session_units(dataset_dir: str | Path) -> tuple[SessionUnit, ...]:
    """Load immutable split units and their pre-training metadata."""
    root = Path(dataset_dir).resolve()
    manifest = _read_json(root / "manifest.json")
    units: list[SessionUnit] = []
    for record in manifest["sequences"]:
        sequence = load_preprocessed_sequence(root / record["output"])
        speeds = Counter(
            _speed_bin(event.mean_speed_mm_min) for event in sequence.events
        )
        units.append(
            SessionUnit(
                session_id=str(record["session_id"]),
                collection_id=str(record["collection_id"]),
                gcode_id=str(record["gcode_id"]),
                motion_family=str(record["motion_family"]),
                frames=len(sequence.signal),
                events=len(sequence.events),
                low_speed_events=int(speeds["low"]),
                medium_speed_events=int(speeds["medium"]),
                high_speed_events=int(speeds["high"]),
            )
        )
    return tuple(sorted(units, key=lambda unit: unit.session_id))


def _feature_schema(
    units: Sequence[SessionUnit],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    collections = tuple(sorted({unit.collection_id for unit in units}))
    families = tuple(sorted({unit.motion_family for unit in units}))
    return collections, families


def _feature_vector(
    unit: SessionUnit,
    collections: Sequence[str],
    families: Sequence[str],
    weights: dict[str, float],
) -> NDArray[np.float64]:
    values = [
        unit.events * weights["events"],
        unit.frames * weights["frames"],
    ]
    values.extend(
        unit.events * weights["collections"] if unit.collection_id == value else 0.0
        for value in collections
    )
    values.extend(
        unit.events * weights["motion_families"] if unit.motion_family == value else 0.0
        for value in families
    )
    values.extend(
        value * weights["speed_bins"]
        for value in (
            unit.low_speed_events,
            unit.medium_speed_events,
            unit.high_speed_events,
        )
    )
    return np.asarray(values, dtype=np.float64)


def _vectors(
    units: Sequence[SessionUnit],
    weights: dict[str, float],
) -> dict[str, NDArray[np.float64]]:
    collections, families = _feature_schema(units)
    return {
        unit.session_id: _feature_vector(unit, collections, families, weights)
        for unit in units
    }


def _balance_score(
    groups: Sequence[Sequence[SessionUnit]],
    vectors: dict[str, NDArray[np.float64]],
) -> float:
    all_units = [unit for group in groups for unit in group]
    if not all_units:
        return 0.0
    total = sum(
        (vectors[unit.session_id] for unit in all_units),
        start=np.zeros_like(next(iter(vectors.values()))),
    )
    target = total / len(groups)
    scale = np.maximum(np.abs(target), 1.0)
    deviations = []
    for group in groups:
        value = sum(
            (vectors[unit.session_id] for unit in group),
            start=np.zeros_like(target),
        )
        deviations.append(np.square((value - target) / scale))
    return float(np.mean(np.stack(deviations)))


def _main_partitions(
    units: Sequence[SessionUnit],
    capacities: Sequence[int],
    vectors: dict[str, NDArray[np.float64]],
) -> tuple[tuple[SessionUnit, ...], ...]:
    if sum(capacities) != len(units):
        raise ValueError("Partition capacities must equal the session count.")
    groups: list[list[SessionUnit]] = [[] for _ in capacities]
    for unit in sorted(units, key=lambda value: (-value.events, value.session_id)):
        candidates = []
        for index, capacity in enumerate(capacities):
            if len(groups[index]) >= capacity:
                continue
            proposal = [list(group) for group in groups]
            proposal[index].append(unit)
            candidates.append((_balance_score(proposal, vectors), index))
        groups[min(candidates)[1]].append(unit)

    while True:
        current = _balance_score(groups, vectors)
        best: tuple[float, int, int, int, int] | None = None
        for left_group in range(len(groups)):
            for right_group in range(left_group + 1, len(groups)):
                for left_index in range(len(groups[left_group])):
                    for right_index in range(len(groups[right_group])):
                        proposal = [list(group) for group in groups]
                        (
                            proposal[left_group][left_index],
                            proposal[right_group][right_index],
                        ) = (
                            proposal[right_group][right_index],
                            proposal[left_group][left_index],
                        )
                        candidate = (
                            _balance_score(proposal, vectors),
                            left_group,
                            right_group,
                            left_index,
                            right_index,
                        )
                        if candidate[0] < current - 1e-12 and (
                            best is None or candidate < best
                        ):
                            best = candidate
        if best is None:
            break
        _, left_group, right_group, left_index, right_index = best
        groups[left_group][left_index], groups[right_group][right_index] = (
            groups[right_group][right_index],
            groups[left_group][left_index],
        )
    return tuple(
        tuple(sorted(group, key=lambda unit: unit.session_id)) for group in groups
    )


def _selection_score(
    selected: Sequence[SessionUnit],
    available: Sequence[SessionUnit],
    vectors: dict[str, NDArray[np.float64]],
) -> float:
    target = sum(
        (vectors[unit.session_id] for unit in available),
        start=np.zeros_like(next(iter(vectors.values()))),
    ) * (len(selected) / len(available))
    value = sum(
        (vectors[unit.session_id] for unit in selected),
        start=np.zeros_like(target),
    )
    return float(np.mean(np.square((value - target) / np.maximum(np.abs(target), 1.0))))


def _select_validation(
    available: Sequence[SessionUnit],
    count: int,
    vectors: dict[str, NDArray[np.float64]],
) -> tuple[SessionUnit, ...]:
    if not 0 < count < len(available):
        raise ValueError("Validation must leave at least one training session.")
    selected: list[SessionUnit] = []
    remaining = list(sorted(available, key=lambda unit: unit.session_id))
    while len(selected) < count:
        chosen = min(
            remaining,
            key=lambda unit: (
                _selection_score(selected + [unit], available, vectors),
                unit.session_id,
            ),
        )
        selected.append(chosen)
        remaining.remove(chosen)

    while True:
        current = _selection_score(selected, available, vectors)
        best: tuple[float, str, str, SessionUnit, SessionUnit] | None = None
        for old in selected:
            for new in remaining:
                proposal = [new if unit == old else unit for unit in selected]
                candidate = (
                    _selection_score(proposal, available, vectors),
                    old.session_id,
                    new.session_id,
                    old,
                    new,
                )
                if candidate[0] < current - 1e-12 and (
                    best is None or candidate[:3] < best[:3]
                ):
                    best = candidate
        if best is None:
            break
        _, _, _, old, new = best
        selected.remove(old)
        remaining.remove(new)
        selected.append(new)
        remaining.append(old)
    return tuple(sorted(selected, key=lambda unit: unit.session_id))


def _pairings(items: tuple[str, ...]) -> Iterable[tuple[tuple[str, str], ...]]:
    if not items:
        yield ()
        return
    first = items[0]
    for index in range(1, len(items)):
        pair = tuple(sorted((first, items[index])))
        remainder = items[1:index] + items[index + 1 :]
        for suffix in _pairings(remainder):
            yield tuple(sorted((pair,) + suffix))


def _gcode_groups(
    units: Sequence[SessionUnit],
    vectors: dict[str, NDArray[np.float64]],
) -> tuple[tuple[str, str], ...]:
    gcode_ids = tuple(sorted({unit.gcode_id for unit in units}))
    if len(gcode_ids) != 10:
        raise ValueError(f"Expected 10 exact G-code IDs, found {len(gcode_ids)}.")
    by_gcode: dict[str, list[SessionUnit]] = defaultdict(list)
    for unit in units:
        by_gcode[unit.gcode_id].append(unit)
    scored = []
    for pairing in _pairings(gcode_ids):
        groups = [by_gcode[left] + by_gcode[right] for left, right in pairing]
        scored.append((_balance_score(groups, vectors), pairing))
    return min(scored, key=lambda item: (item[0], item[1]))[1]


def _ids(units: Iterable[SessionUnit]) -> list[str]:
    return sorted(unit.session_id for unit in units)


def _fold(
    fold_id: str,
    train: Sequence[SessionUnit],
    validation: Sequence[SessionUnit],
    test: Sequence[SessionUnit],
    excluded: Sequence[SessionUnit] = (),
) -> dict[str, Any]:
    payload = {
        "fold_id": fold_id,
        "train_session_ids": _ids(train),
        "validation_session_ids": _ids(validation),
        "test_session_ids": _ids(test),
        "excluded_session_ids": _ids(excluded),
    }
    sets = [
        set(payload[key])
        for key in (
            "train_session_ids",
            "validation_session_ids",
            "test_session_ids",
            "excluded_session_ids",
        )
    ]
    if any(
        left & right for index, left in enumerate(sets) for right in sets[index + 1 :]
    ):
        raise ValueError(f"Session leakage in {fold_id}.")
    return payload


def _main_folds(
    units: Sequence[SessionUnit],
    experiment: dict[str, Any],
    vectors: dict[str, NDArray[np.float64]],
) -> tuple[list[dict[str, Any]], list[list[str]]]:
    settings = experiment["splits"]["main_session_grouped"]
    partitions = _main_partitions(
        units,
        tuple(int(value) for value in settings["partition_session_counts"]),
        vectors,
    )
    folds = []
    for index, test in enumerate(partitions):
        validation_index = (index + 1) % len(partitions)
        validation = partitions[validation_index]
        train = tuple(
            unit
            for group_index, group in enumerate(partitions)
            if group_index not in {index, validation_index}
            for unit in group
        )
        folds.append(_fold(f"fold_{index + 1}", train, validation, test))
    return folds, [_ids(partition) for partition in partitions]


def _collection_folds(
    units: Sequence[SessionUnit],
    vectors: dict[str, NDArray[np.float64]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    collections = tuple(sorted({unit.collection_id for unit in units}))
    leave_one_out = []
    pairwise = []
    for test_collection in collections:
        test = tuple(unit for unit in units if unit.collection_id == test_collection)
        development = tuple(
            unit for unit in units if unit.collection_id != test_collection
        )
        validation = _select_validation(
            development, math.ceil(0.20 * len(development)), vectors
        )
        validation_ids = set(_ids(validation))
        train = tuple(
            unit for unit in development if unit.session_id not in validation_ids
        )
        leave_one_out.append(_fold(f"test_{test_collection}", train, validation, test))

    for train_collection in collections:
        for test_collection in collections:
            if train_collection == test_collection:
                continue
            source = tuple(
                unit for unit in units if unit.collection_id == train_collection
            )
            test = tuple(
                unit for unit in units if unit.collection_id == test_collection
            )
            excluded = tuple(
                unit
                for unit in units
                if unit.collection_id not in {train_collection, test_collection}
            )
            validation = _select_validation(
                source, math.ceil(0.20 * len(source)), vectors
            )
            validation_ids = set(_ids(validation))
            train = tuple(
                unit for unit in source if unit.session_id not in validation_ids
            )
            pairwise.append(
                _fold(
                    f"{train_collection}_to_{test_collection}",
                    train,
                    validation,
                    test,
                    excluded,
                )
            )
    return leave_one_out, pairwise


def _gcode_folds(
    units: Sequence[SessionUnit],
    vectors: dict[str, NDArray[np.float64]],
) -> tuple[list[dict[str, Any]], tuple[tuple[str, str], ...]]:
    groups = _gcode_groups(units, vectors)
    folds = []
    for index, test_group in enumerate(groups):
        validation_group = groups[(index + 1) % len(groups)]
        test_ids = set(test_group)
        validation_ids = set(validation_group)
        test = tuple(unit for unit in units if unit.gcode_id in test_ids)
        validation = tuple(unit for unit in units if unit.gcode_id in validation_ids)
        train = tuple(
            unit for unit in units if unit.gcode_id not in test_ids | validation_ids
        )
        payload = _fold(f"gcode_fold_{index + 1}", train, validation, test)
        payload.update(
            {
                "train_gcode_ids": sorted({unit.gcode_id for unit in train}),
                "validation_gcode_ids": sorted(validation_ids),
                "test_gcode_ids": sorted(test_ids),
            }
        )
        folds.append(payload)
    return folds, groups


def build_split_manifests(
    *,
    project_root: str | Path,
    dataset_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write all frozen protocols from the same session units."""
    root = Path(project_root).resolve()
    dataset_root = Path(dataset_dir).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to reuse non-empty output: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    experiment = load_yaml(root / "configs" / "experiment.yaml")
    weights = {
        key: float(value)
        for key, value in experiment["splits"]["balance_components"].items()
    }
    units = load_session_units(dataset_root)
    vectors = _vectors(units, weights)
    main, partitions = _main_folds(units, experiment, vectors)
    leave_one_out, pairwise = _collection_folds(units, vectors)
    gcode, gcode_groups = _gcode_folds(units, vectors)
    common = {
        "schema_version": 1,
        "dataset_manifest_sha256": file_sha256(dataset_root / "manifest.json"),
        "session_count": len(units),
        "session_units": [asdict(unit) for unit in units],
    }
    payloads = {
        "main_session_grouped": {
            **common,
            "protocol": "main_session_grouped_5fold",
            "partitions": partitions,
            "folds": main,
        },
        "leave_one_collection_out": {
            **common,
            "protocol": "leave_one_collection_out",
            "folds": leave_one_out,
        },
        "pairwise_collection_transfer": {
            **common,
            "protocol": "pairwise_collection_transfer",
            "folds": pairwise,
        },
        "gcode_blocked": {
            **common,
            "protocol": "exact_gcode_blocked_5fold",
            "gcode_groups": [list(group) for group in gcode_groups],
            "folds": gcode,
        },
    }
    paths = {}
    for name, payload in payloads.items():
        path = destination / f"{name}.json"
        _write_json(path, payload)
        paths[name] = path
    return paths
