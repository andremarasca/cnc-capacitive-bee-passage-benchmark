"""CPU-only tests for architecture, halos, masks and loss behavior."""

from pathlib import Path

import torch

from bee_counter_bench.config import load_config
from bee_counter_bench.data.schemas import Direction, PassageEvent
from bee_counter_bench.models.receptive_field import mstcnpp_receptive_field
from bee_counter_bench.provenance import find_project_root, source_code_sha256
from bee_counter_bench.training.dataset import safe_core_ranges
from bee_counter_bench.training.losses import MSTCNPPMultiStageLoss
from bee_counter_bench.training.trainer import build_model


def test_model_matches_frozen_size_and_receptive_field() -> None:
    config = load_config()
    model = build_model(config)

    assert sum(parameter.numel() for parameter in model.parameters()) == 83_625
    assert mstcnpp_receptive_field(6, 4, 2, 30.0).total_frames == 285

    inputs = torch.zeros((2, 1, 285), dtype=torch.float32)
    mask = torch.ones((2, 285), dtype=torch.bool)
    outputs = model(inputs, mask)
    assert outputs.shape == (3, 2, 3, 285)


def test_rf165_model_matches_context_safe_contract() -> None:
    root = find_project_root(Path(__file__))
    config = load_config(model_path=root / "configs" / "model_rf165.yaml")
    model = build_model(config)
    receptive_field = mstcnpp_receptive_field(5, 4, 2, 30.0)

    assert sum(parameter.numel() for parameter in model.parameters()) == 75_337
    assert receptive_field.total_frames == 165
    assert receptive_field.past_frames == 82
    assert receptive_field.future_frames == 82

    inputs = torch.zeros((1, 1, 165), dtype=torch.float32)
    mask = torch.ones((1, 165), dtype=torch.bool)
    outputs = model(inputs, mask)
    assert outputs.shape == (3, 1, 3, 165)


def test_rf165_center_ignores_inputs_outside_nominal_field() -> None:
    root = find_project_root(Path(__file__))
    config = load_config(model_path=root / "configs" / "model_rf165.yaml")
    model = build_model(config).eval()
    torch.manual_seed(90210)
    inputs = torch.randn((1, 1, 205), dtype=torch.float32)
    changed = inputs.clone()
    changed[..., :20] += 100.0
    changed[..., 185:] -= 100.0
    mask = torch.ones((1, 205), dtype=torch.bool)

    with torch.no_grad():
        reference = model(inputs, mask)[..., 102]
        comparison = model(changed, mask)[..., 102]

    torch.testing.assert_close(reference, comparison, rtol=0.0, atol=0.0)


def test_loss_excludes_halo_frames() -> None:
    logits = torch.zeros((3, 1, 3, 8), dtype=torch.float32, requires_grad=True)
    targets = torch.tensor([[2, 2, 1, 1, 1, 1, 2, 2]], dtype=torch.long)
    core_mask = torch.tensor(
        [[False, False, True, True, True, True, False, False]], dtype=torch.bool
    )
    criterion = MSTCNPPMultiStageLoss()

    reference = criterion(logits, targets, core_mask).total
    changed = targets.clone()
    changed[:, :2] = 0
    changed[:, -2:] = 0
    comparison = criterion(logits, changed, core_mask).total

    assert torch.equal(reference, comparison)


def test_safe_cores_shift_boundaries_without_exceeding_limit() -> None:
    event = PassageEvent(895, 930, Direction.RIGHT, 1000.0, 900.0, 1100.0)

    ranges = safe_core_ranges(1800, (event,), 900)

    assert len(ranges) == 3
    assert all(end - start <= 900 for start, end in ranges)
    assert all(end - start >= 500 for start, end in ranges)
    assert not any(
        event.start_idx < boundary <= event.end_idx for _, boundary in ranges[:-1]
    )


def test_source_code_provenance_is_stable_from_nested_path() -> None:
    root = find_project_root(Path(__file__))

    direct_hash = source_code_sha256(root)
    nested_hash = source_code_sha256(root / "outputs" / "processed" / "dataset")

    assert direct_hash == nested_hash
    assert len(direct_hash) == 64
    assert all(character in "0123456789abcdef" for character in direct_hash)
