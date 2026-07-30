from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset

from desi_fm.data import extract_redshift, has_valid_redshift
from desi_fm.train import compute_z_histogram


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="MultimodalUniverse/desi")
    parser.add_argument("--data-dir", default="edr_sv3")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-examples", type=int, default=80000)
    parser.add_argument("--n-z-bins", type=int, default=100)
    parser.add_argument("--z-max", type=float, default=6.0)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stream = load_dataset(
        args.dataset,
        data_dir=args.data_dir,
        split=args.split,
        streaming=True,
    )
    train_window = stream.filter(has_valid_redshift).take(args.max_examples)
    redshifts = [extract_redshift(example) for example in train_window]

    counts, edges = compute_z_histogram(
        np.asarray(redshifts),
        n_bins=args.n_z_bins,
        z_max=args.z_max,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        counts=counts,
        edges=edges,
        n_examples=np.int64(len(redshifts)),
        n_z_bins=np.int64(args.n_z_bins),
        z_max=np.float64(args.z_max),
    )
    print(json.dumps({
        "output": str(output),
        "n_examples": len(redshifts),
        "n_bins": args.n_z_bins,
        "nonempty_bins": int((counts > 0).sum()),
        "counted": int(counts.sum()),
    }, indent=2))


if __name__ == "__main__":
    main()
