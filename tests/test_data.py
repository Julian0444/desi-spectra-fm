import numpy as np
from datasets import IterableDataset

from desi_fm.data import HFDESISpectra, SpectrumPreprocessConfig


def _example(z: float) -> dict:
    wavelength = np.linspace(3600.0, 9800.0, 16, dtype=np.float32)
    return {
        "Z": z,
        "spectrum": {
            "flux": np.ones(16, dtype=np.float32),
            "ivar": np.ones(16, dtype=np.float32),
            "lambda": wavelength,
            "mask": np.zeros(16, dtype=bool),
        },
    }


def _attach_stream(monkeypatch, dataset, rows):
    def load_base_stream():
        return IterableDataset.from_generator(lambda: iter(rows))

    monkeypatch.setattr(dataset, "_load_base_stream", load_base_stream)


def test_take_then_shuffle_keeps_train_and_heldout_disjoint(monkeypatch):
    rows = [
        _example(float("nan")),
        _example(-1.0),
        *[_example(float(z)) for z in range(6)],
    ]
    train = HFDESISpectra(
        max_examples=4,
        skip_examples=0,
        shuffle_buffer=4,
        seed=17,
        preprocess=SpectrumPreprocessConfig(n_pixels=16),
    )
    heldout = HFDESISpectra(
        max_examples=2,
        skip_examples=4,
        shuffle_buffer=0,
        preprocess=SpectrumPreprocessConfig(n_pixels=16),
    )
    _attach_stream(monkeypatch, train, rows)
    _attach_stream(monkeypatch, heldout, rows)

    train_z = [float(row["z"]) for row in train]
    heldout_z = [float(row["z"]) for row in heldout]

    assert set(train_z) == {0.0, 1.0, 2.0, 3.0}
    assert heldout_z == [4.0, 5.0]
    assert set(train_z).isdisjoint(heldout_z)


def test_training_membership_is_fixed_but_order_changes_by_epoch(monkeypatch):
    rows = [_example(float(z)) for z in range(10)]
    dataset = HFDESISpectra(
        max_examples=10,
        skip_examples=0,
        shuffle_buffer=10,
        seed=42,
        preprocess=SpectrumPreprocessConfig(n_pixels=16),
    )
    _attach_stream(monkeypatch, dataset, rows)

    dataset.set_epoch(0)
    epoch_0 = [float(row["z"]) for row in dataset]
    dataset.set_epoch(1)
    epoch_1 = [float(row["z"]) for row in dataset]
    dataset.set_epoch(0)
    epoch_0_repeat = [float(row["z"]) for row in dataset]

    assert set(epoch_0) == set(range(10))
    assert set(epoch_1) == set(range(10))
    assert epoch_0 != epoch_1
    assert epoch_0_repeat == epoch_0
