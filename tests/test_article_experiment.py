"""CPU-only tests for the frozen RF165 article campaign."""

from pathlib import Path
from typing import Any

import pytest

from bee_counter_bench.evaluation.context_safety import (
    build_context_safety_audit,
)
from bee_counter_bench.provenance import find_project_root
from bee_counter_bench.training.article_experiment import (
    ArticleExperimentTask,
    build_article_tasks,
    load_article_experiment_plan,
)
from bee_counter_bench.training import article_experiment


def _plan():
    root = find_project_root(Path(__file__))
    return load_article_experiment_plan(root / "configs" / "article_rf165.yaml")


def test_article_plan_expands_to_exact_frozen_matrix() -> None:
    plan = _plan()
    tasks = build_article_tasks(plan)

    assert len(tasks) == 48
    assert plan.maximum_parallel_gpu_tasks == 4
    assert len({task.task_id for task in tasks}) == 48
    assert sum(task.seed == 42 for task in tasks) == 16
    assert sum(task.protocol == "main_session_grouped" for task in tasks) == 30
    assert sum(task.protocol == "leave_one_collection_out" for task in tasks) == 18
    assert sum(
        task.normalization_variant == "training_population" for task in tasks
    ) == 24
    assert sum(
        task.normalization_variant == "causal_adaptive_tau60" for task in tasks
    ) == 24


def test_context_gate_excludes_labeled_neighboring_events() -> None:
    report = build_context_safety_audit(_plan())
    isolation = report["event_isolation"]

    assert report["receptive_field"]["total_frames"] == 165
    assert report["receptive_field"]["past_frames"] == 82
    assert report["receptive_field"]["future_frames"] == 82
    assert isolation["minimum_out_gap_frames"] == 85
    assert isolation["minimum_margin_frames"] == 3
    assert isolation["event_frame_neighbor_violations"] == 0
    assert isolation["strict_neural_event_isolation"] is True
    assert len(report["normalization_memory"]) == 2
    assert "does not claim complete statistical independence" in report[
        "claim_boundary"
    ]


def test_article_runner_routes_both_normalization_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()
    calls: list[dict[str, Any]] = []

    monkeypatch.setattr(
        article_experiment,
        "ensure_campaign_manifest",
        lambda selected_plan: {},
    )
    monkeypatch.setattr(
        article_experiment,
        "run_training_task",
        lambda **kwargs: calls.append(kwargs) or {"event_metrics": {}},
    )
    base = {
        "protocol": "main_session_grouped",
        "fold_id": "fold_1",
        "seed": 42,
    }

    for variant in ("training_population", "causal_adaptive_tau60"):
        article_experiment.run_article_task(
            plan=plan,
            task=ArticleExperimentTask(
                normalization_variant=variant,
                **base,
            ),
            device="cpu",
            resume=True,
            progress=False,
        )

    assert calls[0]["normalization_config_path"] is None
    assert calls[1]["normalization_config_path"] == (
        plan.project_root / "configs" / "normalization_causal_tau60.yaml"
    )
    assert all(call["model_path"] == plan.model_path for call in calls)
    assert all(call["core_frames"] == 900 for call in calls)
