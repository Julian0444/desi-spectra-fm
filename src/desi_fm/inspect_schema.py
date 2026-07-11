from __future__ import annotations

import argparse
import json
import os
import sys

from .data import extract_mmu_desi_example, summarize_mmu_schema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="MultimodalUniverse/desi")
    parser.add_argument("--data-dir", default="edr_sv3")
    parser.add_argument("--split", default="train")
    return parser.parse_args()


def main() -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Install the data dependencies first: pip install -r requirements.txt"
        ) from exc

    args = parse_args()
    kwargs = {"split": args.split, "streaming": True}
    if args.data_dir:
        kwargs["data_dir"] = args.data_dir
    stream = load_dataset(args.dataset, **kwargs)
    example = next(iter(stream))

    summary = summarize_mmu_schema(example)
    flux, ivar, wavelength, mask, redshift = extract_mmu_desi_example(example)
    summary.update(
        {
            "flux_shape": list(flux.shape),
            "ivar_shape": list(ivar.shape),
            "wavelength_shape": list(wavelength.shape),
            "mask_shape": list(mask.shape),
            "redshift": redshift,
            "wavelength_min": float(wavelength.min()),
            "wavelength_max": float(wavelength.max()),
        }
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
