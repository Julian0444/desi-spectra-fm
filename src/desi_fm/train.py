from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import (
    HFDESISpectra,
    SpectrumPreprocessConfig,
    SyntheticSpectra,
    collate_spectra,
)
from .model import DESIFoundationModel, DESIFoundationModelConfig


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_loader(args: argparse.Namespace, *, train: bool) -> DataLoader:
    preprocess = SpectrumPreprocessConfig(
        n_pixels=args.n_pixels,
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
        wavelength_grid=args.wavelength_grid,
    )
    if args.synthetic:
        dataset = SyntheticSpectra(
            num_examples=args.max_train_examples if train else args.val_examples,
            seed=args.seed + (0 if train else 10_000),
            preprocess=preprocess,
        )
    else:
        dataset = HFDESISpectra(
            dataset_name=args.dataset,
            split=args.split,
            data_dir=args.data_dir,
            max_examples=args.max_train_examples if train else args.val_examples,
            skip_examples=0 if train else args.max_train_examples,
            shuffle_buffer=args.shuffle_buffer if train else 0,
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
def evaluate(model: DESIFoundationModel, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_z = 0.0
    total_abs_z = 0.0
    total_abs_z_norm = 0.0
    total_recon_sse = 0.0
    total_recon_pixels = 0.0
    batches = 0
    examples = 0
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
        total_abs_z += float(dz.sum().detach().cpu())
        total_abs_z_norm += float((dz / (1.0 + z)).sum().detach().cpu())
        batches += 1
        examples += bsz
    denom = max(batches, 1)
    ex_denom = max(examples, 1)
    return {
        "loss": total_loss / denom,
        "reconstruction_loss": total_recon / denom,
        "reconstruction_rmse_masked": math.sqrt(
            total_recon_sse / max(total_recon_pixels, 1.0)
        ),
        "masked_valid_pixels": total_recon_pixels,
        "redshift_loss": total_z / denom,
        "redshift_mae": total_abs_z / ex_denom,
        "redshift_mae_norm": total_abs_z_norm / ex_denom,
    }


def estimate_total_steps(args: argparse.Namespace) -> int:
    if args.max_steps > 0:
        return args.max_steps
    steps_per_epoch = max(math.ceil(args.max_train_examples / args.batch_size), 1)
    return max(steps_per_epoch * args.epochs, 1)


def build_scheduler(
    optimizer: torch.optim.Optimizer, args: argparse.Namespace
) -> torch.optim.lr_scheduler.LambdaLR | None:
    if args.lr_schedule == "constant":
        return None

    total_steps = estimate_total_steps(args)
    warmup_steps = args.warmup_steps
    if warmup_steps <= 0:
        warmup_steps = int(total_steps * args.warmup_ratio)
    warmup_steps = min(max(warmup_steps, 1), total_steps)

    def lr_lambda(step: int) -> float:
        current = step + 1
        if current <= warmup_steps:
            return current / warmup_steps
        progress = (current - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    path: Path,
    model: DESIFoundationModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None,
    config: DESIFoundationModelConfig,
    step: int,
    epoch: int,
    save_optimizer: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": model.state_dict(),
        "config": config.to_dict(),
        "step": step,
        "epoch": epoch,
    }
    if save_optimizer:
        checkpoint["optimizer"] = optimizer.state_dict()
        checkpoint["scheduler"] = scheduler.state_dict() if scheduler is not None else None
    torch.save(checkpoint, path)


def append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(payload) + "\n")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_z_histogram(
    redshifts: np.ndarray,
    *,
    n_bins: int,
    z_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(redshifts, dtype=np.float64)
    valid = np.isfinite(values) & (values >= 0.0)
    encoded = np.log1p(np.minimum(values[valid], z_max))
    edges = np.linspace(0.0, math.log1p(z_max), n_bins + 1)
    counts, _ = np.histogram(encoded, bins=edges)
    return counts.astype(np.int64), edges.astype(np.float64)


def build_z_bin_weights(
    counts: torch.Tensor,
    *,
    mode: str,
    min_weight: float,
    max_weight: float,
) -> torch.Tensor:
    counts = counts.float()
    if counts.ndim != 1:
        raise ValueError("counts must be one-dimensional")
    if mode == "none":
        return torch.ones_like(counts)
    if mode != "sqrt_inverse":
        raise ValueError(f"unknown z weighting mode: {mode}")
    observed = counts > 0
    if not bool(observed.any()):
        raise ValueError("histogram has no observed bins")
    mean_count = counts[observed].mean()
    weights = torch.zeros_like(counts)
    weights[observed] = torch.sqrt(mean_count / counts[observed]).clamp(
        min=min_weight,
        max=max_weight,
    )
    return weights


def load_compatible_checkpoint(
    model: DESIFoundationModel,
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    source = dict(checkpoint["model"])
    if "pos_embed" in source and "learned_pos_embed" not in source:
        source["learned_pos_embed"] = source.pop("pos_embed")
    target = model.state_dict()
    compatible = {
        key: value
        for key, value in source.items()
        if key in target and target[key].shape == value.shape
    }
    skipped = sorted(key for key in source if key not in compatible)
    model.load_state_dict(compatible, strict=False)
    return {"loaded": len(compatible), "skipped": skipped}


def is_better_checkpoint(
    metrics: dict[str, float],
    *,
    best_score: float,
) -> bool:
    return (
        metrics.get("examples", 0.0) > 0
        and "eta15_map" in metrics
        and math.isfinite(metrics["eta15_map"])
        and metrics["eta15_map"] < best_score
    )


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"

    config = DESIFoundationModelConfig(
        n_pixels=args.n_pixels,
        n_tokens=args.n_tokens,
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
        wavelength_grid=args.wavelength_grid,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        mask_ratio=args.mask_ratio,
        redshift_loss_weight=args.redshift_loss_weight,
        reconstruction_loss_weight=args.reconstruction_loss_weight,
        use_wavelength_embedding=not args.no_wavelength_embedding,
        use_learned_position=not args.no_learned_position,
        n_z_bins=args.n_z_bins,
        z_max=args.z_max,
    )
    (out_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2))

    model = DESIFoundationModel(config).to(device)
    if args.z_rebalance and args.n_z_bins > 0:
        import csv as _csv
        zs = [float(r["z_true"]) for r in _csv.DictReader(open("runs/desi_50k_big/predictions.csv"))]
        enc = torch.log1p(torch.tensor(zs).clamp(min=0.0))
        hist = torch.histc(enc, bins=args.n_z_bins, min=0.0, max=math.log1p(args.z_max)) + 1.0
        w = (hist.sum() / args.n_z_bins) / hist
        model.z_bin_weights.copy_(w.clamp(0.3, 10.0).to(model.z_bin_weights.device))
    print(f"trainable_parameters={count_parameters(model):,}")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = build_scheduler(optimizer, args)

    train_loader = build_loader(args, train=True)
    val_loader = build_loader(args, train=False) if args.val_examples > 0 else None

    step = 0
    start = time.time()
    last_epoch = 0
    try:
        for epoch in range(args.epochs):
            last_epoch = epoch
            model.train()
            if hasattr(train_loader.dataset, "set_epoch"):
                train_loader.dataset.set_epoch(epoch)
            print(f"dataset_epoch={epoch} shuffle_seed={args.seed + epoch}")
            progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}")
            for batch in progress:
                flux = batch["flux"].to(device)
                valid = batch["valid"].to(device)
                z = batch["z"].to(device)

                out = model(flux, valid, z=z)
                loss = out["loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                step += 1
                lr = optimizer.param_groups[0]["lr"]
                train_metrics = {
                    "phase": "train",
                    "epoch": epoch,
                    "step": step,
                    "lr": lr,
                    "loss": float(loss.detach().cpu()),
                    "reconstruction_loss": float(
                        out["reconstruction_loss"].detach().cpu()
                    ),
                    "reconstruction_rmse": float(
                        out["reconstruction_rmse"].detach().cpu()
                    ),
                    "redshift_loss": float(out["redshift_loss"].detach().cpu()),
                }
                progress.set_postfix(
                    loss=f"{train_metrics['loss']:.4f}",
                    recon=f"{train_metrics['reconstruction_loss']:.4f}",
                    z=f"{train_metrics['redshift_loss']:.4f}",
                    lr=f"{lr:.2e}",
                )
                if step == 1 or step % args.log_every_steps == 0:
                    append_jsonl(metrics_path, train_metrics)
                if args.save_every_steps > 0 and step % args.save_every_steps == 0:
                    save_checkpoint(
                        out_dir / f"checkpoint_step_{step}.pt",
                        model,
                        optimizer,
                        scheduler,
                        config,
                        step,
                        epoch,
                        save_optimizer=args.save_optimizer,
                    )
                if args.max_steps and step >= args.max_steps:
                    break

            if val_loader is not None:
                metrics = evaluate(model, val_loader, device)
                print("validation", json.dumps(metrics, indent=2))
                append_jsonl(
                    metrics_path,
                    {"phase": "validation", "epoch": epoch, "step": step, **metrics},
                )

            save_checkpoint(
                out_dir / "checkpoint_last.pt",
                model,
                optimizer,
                scheduler,
                config,
                step,
                epoch,
                save_optimizer=args.save_optimizer,
            )
            if args.max_steps and step >= args.max_steps:
                break
    except KeyboardInterrupt:
        if step > 0:
            interrupted_path = out_dir / "checkpoint_interrupted.pt"
            save_checkpoint(
                interrupted_path,
                model,
                optimizer,
                scheduler,
                config,
                step,
                last_epoch,
                save_optimizer=args.save_optimizer,
            )
            append_jsonl(
                metrics_path,
                {"phase": "interrupted", "epoch": last_epoch, "step": step},
            )
            print(f"\nInterrupted. Saved {interrupted_path} at step={step}.")
        raise

    elapsed = time.time() - start
    print(f"done step={step} elapsed_sec={elapsed:.1f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="MultimodalUniverse/desi")
    parser.add_argument("--data-dir", default="edr_sv3")
    parser.add_argument("--split", default="train")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--output-dir", default="runs/desi_fm")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--shuffle-buffer", type=int, default=2048)

    parser.add_argument("--n-pixels", type=int, default=7081)
    parser.add_argument("--lambda-min", type=float, default=3600.0)
    parser.add_argument("--lambda-max", type=float, default=9800.0)
    parser.add_argument("--wavelength-grid", choices=["log", "linear"], default="log")
    parser.add_argument("--n-tokens", type=int, default=273)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=6)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--no-wavelength-embedding", action="store_true")
    parser.add_argument("--no-learned-position", action="store_true")
    parser.add_argument("--n-z-bins", type=int, default=0)
    parser.add_argument("--z-max", type=float, default=6.0)
    parser.add_argument("--z-rebalance", action="store_true",
                        help="pesos inverso-frecuencia por bin, estimados de runs/desi_50k_big/predictions.csv")

    parser.add_argument("--mask-ratio", type=float, default=0.35)
    parser.add_argument("--redshift-loss-weight", type=float, default=2.0)
    parser.add_argument("--reconstruction-loss-weight", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-train-examples", type=int, default=1000)
    parser.add_argument("--val-examples", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-schedule", choices=["cosine", "constant"], default="cosine")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every-steps", type=int, default=25)
    parser.add_argument("--save-every-steps", type=int, default=0)
    parser.add_argument("--save-optimizer", action="store_true")
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
