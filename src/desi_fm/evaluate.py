from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import (
    HFDESISpectra,
    SpectrumPreprocessConfig,
    SyntheticSpectra,
    collate_spectra,
)
from .model import DESIFoundationModel, DESIFoundationModelConfig
from .train import pick_device


def write_metrics_json(metrics: dict[str, float], output: Path) -> None:
    payload = json.dumps(metrics, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(payload)
    temporary.replace(output)


def load_model(checkpoint_path: Path, device: torch.device) -> DESIFoundationModel:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = DESIFoundationModelConfig(**checkpoint["config"])
    model = DESIFoundationModel(config).to(device)
    state_dict = checkpoint["model"]
    if "pos_embed" in state_dict and "learned_pos_embed" not in state_dict:
        state_dict["learned_pos_embed"] = state_dict.pop("pos_embed")
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def build_eval_loader(args: argparse.Namespace, config: DESIFoundationModelConfig) -> DataLoader:
    preprocess = SpectrumPreprocessConfig(
        n_pixels=config.n_pixels,
        lambda_min=config.lambda_min,
        lambda_max=config.lambda_max,
        wavelength_grid=config.wavelength_grid,
    )
    if args.synthetic:
        dataset = SyntheticSpectra(
            num_examples=args.max_examples,
            seed=args.seed,
            preprocess=preprocess,
        )
    else:
        dataset = HFDESISpectra(
            dataset_name=args.dataset,
            split=args.split,
            data_dir=args.data_dir,
            max_examples=args.max_examples,
            skip_examples=args.skip_examples,
            shuffle_buffer=0,
            seed=args.seed,
            preprocess=preprocess,
        )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_spectra,
    )


@torch.no_grad()
def evaluate_and_write_outputs(
    model: DESIFoundationModel,
    loader: DataLoader,
    device: torch.device,
    predictions_csv: Path | None = None,
    reconstructions_npz: Path | None = None,
    num_reconstructions: int = 8,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_z = 0.0
    total_abs_z = 0.0
    total_abs_z_norm = 0.0
    total_recon_sse = 0.0
    total_recon_pixels = 0.0
    total_abs_z_map = 0.0
    total_abs_z_norm_map = 0.0
    total_conf = 0.0
    n_out15 = 0
    n_out15_map = 0
    has_map = False
    batches = 0
    examples = 0
    flux_rows = []
    valid_rows = []
    recon_rows = []
    mask_rows = []
    z_true_rows = []
    z_pred_rows = []

    writer = None
    csv_file = None
    if predictions_csv is not None:
        predictions_csv.parent.mkdir(parents=True, exist_ok=True)
        csv_file = predictions_csv.open("w", newline="")
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["z_true", "z_pred", "abs_dz", "abs_dz_norm", "z_pred_map", "z_confidence"],
            restval="",
        )
        writer.writeheader()

    for batch in loader:
        flux = batch["flux"].to(device)
        valid = batch["valid"].to(device)
        z = batch["z"].to(device)
        out = model(flux, valid, z=z)
        bsz = flux.shape[0]
        total_loss += float(out["loss"].detach().cpu())
        total_recon += float(out["reconstruction_loss"].detach().cpu())
        total_z += float(out["redshift_loss"].detach().cpu())
        total_recon_sse += float(out["reconstruction_sse"].detach().cpu())
        total_recon_pixels += float(out["reconstruction_valid_pixels"].detach().cpu())
        dz = (out["z_pred"] - z).abs()
        n_out15 += int(((dz / (1.0 + z)) > 0.15).sum().cpu())
        if "z_pred_map" in out:
            has_map = True
            dz_map = (out["z_pred_map"] - z).abs()
            total_abs_z_map += float(dz_map.sum().cpu())
            total_abs_z_norm_map += float((dz_map / (1.0 + z)).sum().cpu())
            n_out15_map += int(((dz_map / (1.0 + z)) > 0.15).sum().cpu())
            total_conf += float(out["z_confidence"].sum().cpu())
        total_abs_z += float(dz.sum().detach().cpu())
        total_abs_z_norm += float((dz / (1.0 + z)).sum().detach().cpu())
        batches += 1
        examples += bsz

        z_cpu = z.cpu().tolist()
        z_pred_cpu = out["z_pred"].cpu().tolist()
        if writer is not None:
            z_map_cpu = out["z_pred_map"].cpu().tolist() if "z_pred_map" in out else [None] * bsz
            conf_cpu = out["z_confidence"].cpu().tolist() if "z_confidence" in out else [None] * bsz
            for z_true_v, z_pred_v, z_map_v, conf_v in zip(z_cpu, z_pred_cpu, z_map_cpu, conf_cpu):
                row = {
                    "z_true": z_true_v,
                    "z_pred": z_pred_v,
                    "abs_dz": abs(z_pred_v - z_true_v),
                    "abs_dz_norm": abs(z_pred_v - z_true_v) / (1.0 + z_true_v),
                }
                if z_map_v is not None:
                    row["z_pred_map"] = z_map_v
                    row["z_confidence"] = conf_v
                writer.writerow(row)

        if reconstructions_npz is not None and len(flux_rows) < num_reconstructions:
            recon = model.unpatchify(out["patch_pred"])
            patch_mask = out["spectrum_mask"].unsqueeze(-1).expand(
                -1, -1, model.config.patch_size
            )
            pixel_mask = model.unpatchify(patch_mask.float()) > 0.5
            remaining = num_reconstructions - len(flux_rows)
            for i in range(min(flux.shape[0], remaining)):
                flux_rows.append(flux[i].cpu().numpy())
                valid_rows.append(valid[i].cpu().numpy())
                recon_rows.append(recon[i].cpu().numpy())
                mask_rows.append(pixel_mask[i].cpu().numpy())
                z_true_rows.append(float(z[i].cpu()))
                z_pred_rows.append(float(out["z_pred"][i].cpu()))

    if csv_file is not None:
        csv_file.close()
    if reconstructions_npz is not None and flux_rows:
        reconstructions_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            reconstructions_npz,
            flux=np.stack(flux_rows),
            valid=np.stack(valid_rows),
            reconstruction=np.stack(recon_rows),
            masked_pixels=np.stack(mask_rows),
            z_true=np.asarray(z_true_rows, dtype=np.float32),
            z_pred=np.asarray(z_pred_rows, dtype=np.float32),
        )

    denom = max(batches, 1)
    ex_denom = max(examples, 1)
    return {
        "loss": total_loss / denom,
        "reconstruction_loss": total_recon / denom,
        "reconstruction_rmse_masked": float(
            np.sqrt(total_recon_sse / max(total_recon_pixels, 1.0))
        ),
        "masked_valid_pixels": total_recon_pixels,
        "redshift_loss": total_z / denom,
        "redshift_mae": total_abs_z / ex_denom,
        "redshift_mae_norm": total_abs_z_norm / ex_denom,
        "eta15": n_out15 / ex_denom,
        **({
            "redshift_mae_map": total_abs_z_map / ex_denom,
            "redshift_mae_norm_map": total_abs_z_norm_map / ex_denom,
            "eta15_map": n_out15_map / ex_denom,
            "mean_z_confidence": total_conf / ex_denom,
        } if has_map else {}),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="MultimodalUniverse/desi")
    parser.add_argument("--data-dir", default="edr_sv3")
    parser.add_argument("--split", default="train")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=512)
    parser.add_argument("--skip-examples", type=int, default=0,
                        help="skip the first N stream examples (use the training count "
                             "to evaluate on truly held-out spectra)")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--lambda-min", type=float, default=3600.0)
    parser.add_argument("--lambda-max", type=float, default=9800.0)
    parser.add_argument("--predictions-csv", default="")
    parser.add_argument("--reconstructions-npz", default="")
    parser.add_argument("--num-reconstructions", type=int, default=8)
    parser.add_argument(
        "--metrics-json",
        default="",
        help="write the final metrics as standalone JSON; logs remain independent",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    model = load_model(Path(args.checkpoint), device)
    config = model.config
    loader = build_eval_loader(args, config)
    metrics = evaluate_and_write_outputs(
        model,
        loader,
        device,
        Path(args.predictions_csv) if args.predictions_csv else None,
        Path(args.reconstructions_npz) if args.reconstructions_npz else None,
        args.num_reconstructions,
    )
    if args.metrics_json:
        write_metrics_json(metrics, Path(args.metrics_json))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
