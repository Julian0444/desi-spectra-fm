from pathlib import Path

import numpy as np
import torch

from desi_fm.model import DESIFoundationModel, DESIFoundationModelConfig
from desi_fm.train import (
    build_z_bin_weights,
    compute_z_histogram,
    is_better_checkpoint,
    load_compatible_checkpoint,
)


def _tiny_config(n_z_bins: int) -> DESIFoundationModelConfig:
    return DESIFoundationModelConfig(
        n_pixels=64,
        n_tokens=8,
        d_model=32,
        n_layers=1,
        n_heads=4,
        dropout=0.0,
        n_z_bins=n_z_bins,
        z_max=6.0,
    )


def test_compute_z_histogram_counts_every_valid_label_once():
    counts, edges = compute_z_histogram(
        np.asarray([0.0, 0.1, 0.5, 1.0, 3.0, np.nan, -1.0]),
        n_bins=10,
        z_max=6.0,
    )
    assert counts.shape == (10,)
    assert edges.shape == (11,)
    assert int(counts.sum()) == 5


def test_sqrt_inverse_weights_ignore_empty_bins():
    counts = torch.tensor([100.0, 25.0, 0.0])
    weights = build_z_bin_weights(
        counts,
        mode="sqrt_inverse",
        min_weight=0.5,
        max_weight=3.0,
    )
    assert weights[2].item() == 0.0
    assert 0.5 <= weights[0].item() < weights[1].item() <= 3.0


def test_warm_start_loads_encoder_but_skips_incompatible_head(tmp_path: Path):
    source = DESIFoundationModel(_tiny_config(n_z_bins=0))
    with torch.no_grad():
        first_encoder_param = next(source.encoder.parameters())
        first_encoder_param.fill_(0.123)
    checkpoint = tmp_path / "v1.pt"
    torch.save(
        {"model": source.state_dict(), "config": source.config.to_dict()},
        checkpoint,
    )

    target = DESIFoundationModel(_tiny_config(n_z_bins=16))
    report = load_compatible_checkpoint(target, checkpoint, torch.device("cpu"))

    assert torch.allclose(
        next(target.encoder.parameters()),
        torch.full_like(next(target.encoder.parameters()), 0.123),
    )
    assert "redshift_head.4.weight" in report["skipped"]
    assert target.redshift_head[-1].out_features == 16


def test_best_checkpoint_requires_nonempty_map_improvement():
    assert not is_better_checkpoint(
        {"examples": 0, "eta15_map": 0.0},
        best_score=0.5,
    )
    assert is_better_checkpoint(
        {"examples": 2000, "eta15_map": 0.25},
        best_score=0.5,
    )
    assert not is_better_checkpoint(
        {"examples": 2000, "eta15_map": 0.60},
        best_score=0.5,
    )
