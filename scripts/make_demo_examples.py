"""Generate examples/*.npz for the HF Spaces demo.

Default mode exports **real DESI spectra from the canonical held-out split**
(the same `filter(valid z) -> skip(80000) -> take(...)` stream used by
`desi_fm.evaluate`, so example i here is row i+1 of
`runs/desi_80k_classhead_v21/predictions.csv`). Four spectra are selected to
be honest and diverse: three across the redshift range that v2.1 predicts
well, plus one catastrophic outlier (low `z_confidence`) so the demo shows a
real failure mode. Raw `flux/ivar/wavelength/mask` are saved BEFORE any
preprocessing, with the DESI pipeline `z_true`.

    python3 scripts/make_demo_examples.py [output_dir]

Streaming through the 80k-example skip takes 15 min - 2.5 h depending on the
day. For a quick offline smoke test, `--synthetic` writes toy spectra instead
(known z, but far out-of-distribution for the model — not used by the demo).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# (name, held-out stream index) — index i corresponds to row i+1 of
# runs/desi_80k_classhead_v21/predictions.csv. Selected on demo-protocol
# predictions (mask_ratio=0, deterministic): z_true 0.20 / 0.83 predicted well,
# z_true 2.87 -> 2.44 (above v1's old z≈2 ceiling, delta shown honestly), and
# index 40 (z_true 1.57 -> z_pred_map 0.96, z_confidence 0.18) as the honest
# catastrophic outlier in the still-weak z in [1.5, 2.5) band.
SELECTED = [
    ("heldout_z020", 12),
    ("heldout_z083", 27),
    ("heldout_z287", 10),
    ("heldout_lowconf_z157", 40),
]

REST = {"OII": 3727.0, "CaK": 3934.0, "CaH": 3969.0, "Hb": 4861.0,
        "OIII": 5007.0, "Ha": 6563.0}


def synth(z: float, seed: int, n: int = 6000) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    w = np.linspace(3600.0, 9800.0, n).astype(np.float32)
    f = 0.5 + 0.15 * np.sin(w / 700.0) + rng.normal(0, 0.05, n)
    for lam in REST.values():
        obs = lam * (1 + z)
        if w[0] <= obs <= w[-1]:
            f += rng.uniform(0.4, 1.2) * np.exp(-0.5 * ((w - obs) / 5.0) ** 2)
    return w, f.astype(np.float32)


def write_synthetic(out: Path) -> None:
    for name, z, seed in [("galaxy_z010", 0.10, 1), ("galaxy_z042", 0.42, 2),
                          ("emission_z080", 0.80, 3), ("noisy_z025", 0.25, 4)]:
        w, f = synth(z, seed)
        np.savez(out / f"{name}.npz", flux=f, wavelength=w, z_true=np.float32(z))
        print(f"wrote {out / name}.npz (synthetic, z_true={z:.2f})")


def write_heldout(out: Path) -> None:
    from datasets import load_dataset

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from desi_fm.data import extract_mmu_desi_example, has_valid_redshift

    wanted = {idx: name for name, idx in SELECTED}
    last = max(wanted)
    stream = load_dataset(
        "MultimodalUniverse/desi", data_dir="edr_sv3", split="train",
        streaming=True,
    )
    stream = stream.filter(has_valid_redshift).skip(80000).take(last + 1)
    for i, example in enumerate(stream):
        if i not in wanted:
            continue
        flux, ivar, wavelength, mask, redshift = extract_mmu_desi_example(example)
        path = out / f"{wanted[i]}.npz"
        np.savez(
            path,
            flux=flux.astype(np.float32),
            ivar=ivar.astype(np.float32),
            wavelength=wavelength.astype(np.float32),
            mask=mask.astype(bool),
            z_true=np.float32(redshift),
            object_id=str(example.get("object_id", "")),
        )
        print(f"wrote {path} (held-out index {i}, z_true={redshift:.4f})")


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--synthetic"]
    out = Path(args[0]) if args else Path("examples")
    out.mkdir(exist_ok=True)
    if "--synthetic" in sys.argv[1:]:
        write_synthetic(out)
    else:
        write_heldout(out)


if __name__ == "__main__":
    main()
