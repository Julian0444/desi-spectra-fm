from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import IterableDataset


@dataclass(frozen=True)
class SpectrumPreprocessConfig:
    n_pixels: int = 7081
    lambda_min: float = 3600.0
    lambda_max: float = 9800.0
    wavelength_grid: str = "log"
    clip_value: float = 8.0
    eps: float = 1e-6

    @property
    def target_wavelength(self) -> np.ndarray:
        if self.wavelength_grid == "linear":
            return np.linspace(
                self.lambda_min, self.lambda_max, self.n_pixels, dtype=np.float32
            )
        if self.wavelength_grid == "log":
            return np.geomspace(
                self.lambda_min, self.lambda_max, self.n_pixels
            ).astype(np.float32)
        raise ValueError(
            f"Unknown wavelength_grid={self.wavelength_grid!r}; use 'log' or 'linear'."
        )


def _as_float_array(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _as_bool_array(x: Any, length: int) -> np.ndarray:
    if x is None:
        return np.zeros(length, dtype=bool)
    return np.asarray(x, dtype=bool)


def extract_mmu_desi_example(
    example: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Extract the DESI columns used by MultimodalUniverse/desi.

    Expected schema:
      spectrum.flux, spectrum.ivar, spectrum.lambda, spectrum.mask, and Z/redshift.

    The function is intentionally tolerant because Hugging Face streaming can return
    nested values as Python lists, NumPy arrays, or Arrow-backed objects.
    """
    if "spectrum" not in example:
        raise KeyError("Expected a 'spectrum' field in the DESI example.")

    spectrum = example["spectrum"]
    flux = _as_float_array(spectrum["flux"])
    ivar = _as_float_array(spectrum.get("ivar", np.ones_like(flux)))
    wavelength_key = "lambda" if "lambda" in spectrum else "wavelength"
    wavelength = _as_float_array(spectrum[wavelength_key])
    mask = _as_bool_array(spectrum.get("mask"), len(flux))

    redshift_key = next((key for key in ("Z", "redshift", "z") if key in example), None)
    if redshift_key is None:
        raise KeyError("Expected a redshift field named 'Z', 'redshift', or 'z'.")
    redshift = float(example[redshift_key])
    return flux, ivar, wavelength, mask, redshift


def summarize_mmu_schema(example: dict[str, Any]) -> dict[str, Any]:
    spectrum = example.get("spectrum", {})
    redshift_key = next((key for key in ("Z", "redshift", "z") if key in example), None)
    return {
        "top_level_keys": sorted(example.keys()),
        "spectrum_keys": sorted(spectrum.keys()) if isinstance(spectrum, dict) else [],
        "redshift_key": redshift_key,
        "object_id": example.get("object_id"),
    }


def preprocess_spectrum(
    flux: np.ndarray,
    ivar: np.ndarray,
    wavelength: np.ndarray,
    mask: np.ndarray,
    config: SpectrumPreprocessConfig,
) -> dict[str, np.ndarray | float]:
    """Interpolate and robustly normalize one observed spectrum.

    AION projects spectra onto a latent wavelength grid before tokenization. This
    implementation uses the same idea in simpler form: every input spectrum is
    interpolated onto one fixed wavelength grid so the transformer sees aligned
    patches. Bad pixels are excluded from the robust normalization and from the
    reconstruction loss.
    """
    flux = np.nan_to_num(flux.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    ivar = np.nan_to_num(ivar.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    wavelength = wavelength.astype(np.float32)
    mask = mask.astype(bool)

    order = np.argsort(wavelength)
    wavelength = wavelength[order]
    flux = flux[order]
    ivar = ivar[order]
    mask = mask[order]

    good = (~mask) & np.isfinite(flux) & np.isfinite(ivar) & (ivar > 0)
    target = config.target_wavelength

    if good.sum() < 8:
        flux_i = np.zeros(config.n_pixels, dtype=np.float32)
        valid_i = np.zeros(config.n_pixels, dtype=np.float32)
    else:
        valid = good.astype(np.float32)
        flux_i = np.interp(target, wavelength, np.where(good, flux, 0.0)).astype(
            np.float32
        )
        valid_i = np.interp(target, wavelength, valid, left=0.0, right=0.0).astype(
            np.float32
        )
        valid_i = (valid_i > 0.5).astype(np.float32)

    valid_values = flux_i[valid_i > 0.5]
    if valid_values.size < 8:
        center = 0.0
        scale = 1.0
    else:
        center = float(np.median(valid_values))
        q25, q75 = np.percentile(valid_values, [25.0, 75.0])
        scale = float((q75 - q25) / 1.349)
        if not math.isfinite(scale) or scale < config.eps:
            scale = float(np.std(valid_values))
        scale = max(scale, config.eps)

    norm_flux = np.arcsinh((flux_i - center) / scale).astype(np.float32)
    norm_flux = np.clip(norm_flux, -config.clip_value, config.clip_value)
    norm_flux = norm_flux * valid_i

    return {
        "flux": norm_flux.astype(np.float32),
        "valid": valid_i.astype(np.float32),
        "center": np.float32(center),
        "scale": np.float32(scale),
    }


class HFDESISpectra(IterableDataset):
    """Streaming PyTorch dataset for MultimodalUniverse/desi."""

    def __init__(
        self,
        dataset_name: str = "MultimodalUniverse/desi",
        split: str = "train",
        data_dir: str | None = "edr_sv3",
        max_examples: int | None = None,
        skip_examples: int = 0,
        shuffle_buffer: int = 0,
        seed: int = 42,
        preprocess: SpectrumPreprocessConfig | None = None,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.split = split
        self.data_dir = data_dir
        self.max_examples = max_examples
        self.skip_examples = skip_examples
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.preprocess = preprocess or SpectrumPreprocessConfig()

    def _load_stream(self) -> Iterable[dict[str, Any]]:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "Install the data dependencies first: pip install -r requirements.txt"
            ) from exc

        kwargs: dict[str, Any] = {"split": self.split, "streaming": True}
        if self.data_dir:
            kwargs["data_dir"] = self.data_dir

        stream = load_dataset(self.dataset_name, **kwargs)
        if self.shuffle_buffer > 0:
            stream = stream.shuffle(buffer_size=self.shuffle_buffer, seed=self.seed)
        return stream

    def __iter__(self):
        stream = self._load_stream()
        emitted = 0
        for idx, example in enumerate(stream):
            if idx < self.skip_examples:
                continue
            flux, ivar, wavelength, mask, redshift = extract_mmu_desi_example(example)
            processed = preprocess_spectrum(flux, ivar, wavelength, mask, self.preprocess)
            if not math.isfinite(redshift) or redshift < 0:
                continue
            yield {
                "flux": processed["flux"],
                "valid": processed["valid"],
                "z": np.float32(redshift),
            }
            emitted += 1
            if self.max_examples is not None and emitted >= self.max_examples:
                break


class SyntheticSpectra(IterableDataset):
    """Small synthetic spectra for smoke tests and pipeline debugging.

    This is not scientifically meaningful. It creates emission/absorption lines
    shifted by z so the model can overfit a tiny run and prove the code path works.
    """

    rest_lines = np.array([3727.0, 3934.0, 3969.0, 4861.0, 5007.0, 6563.0])

    def __init__(
        self,
        num_examples: int = 1024,
        seed: int = 13,
        preprocess: SpectrumPreprocessConfig | None = None,
        z_range: tuple[float, float] = (0.0, 0.8),
    ):
        self.num_examples = num_examples
        self.seed = seed
        self.preprocess = preprocess or SpectrumPreprocessConfig()
        self.z_range = z_range

    def __iter__(self):
        rng = np.random.default_rng(self.seed + random.randint(0, 10_000))
        wave = self.preprocess.target_wavelength
        for _ in range(self.num_examples):
            z = float(rng.uniform(*self.z_range))
            flux = rng.normal(0.0, 0.05, size=wave.shape).astype(np.float32)
            continuum = 0.5 + 0.15 * np.sin(wave / 700.0)
            flux += continuum.astype(np.float32)
            for line in self.rest_lines:
                obs = line * (1.0 + z)
                if wave[0] <= obs <= wave[-1]:
                    amp = float(rng.uniform(-0.6, 1.2))
                    width = float(rng.uniform(2.0, 7.0))
                    flux += amp * np.exp(-0.5 * ((wave - obs) / width) ** 2)
            ivar = np.ones_like(flux, dtype=np.float32) * 100.0
            mask = np.zeros_like(flux, dtype=bool)
            processed = preprocess_spectrum(flux, ivar, wave, mask, self.preprocess)
            yield {
                "flux": processed["flux"],
                "valid": processed["valid"],
                "z": np.float32(z),
            }


def collate_spectra(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    flux = torch.tensor(np.stack([b["flux"] for b in batch]), dtype=torch.float32)
    valid = torch.tensor(np.stack([b["valid"] for b in batch]), dtype=torch.float32)
    z = torch.tensor([b["z"] for b in batch], dtype=torch.float32)
    return {"flux": flux, "valid": valid, "z": z}
