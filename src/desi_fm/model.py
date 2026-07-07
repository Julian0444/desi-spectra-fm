from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class DESIFoundationModelConfig:
    n_pixels: int = 7081
    n_tokens: int = 273
    lambda_min: float = 3600.0
    lambda_max: float = 9800.0
    wavelength_grid: str = "log"
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    mask_ratio: float = 0.35
    redshift_loss_weight: float = 2.0
    reconstruction_loss_weight: float = 1.0
    use_wavelength_embedding: bool = True
    use_learned_position: bool = True
    n_z_bins: int = 0        # 0 = cabeza escalar (v1); >0 = clasificación sobre bins de log(1+z)
    z_max: float = 6.0       # techo del rango de bins
    z_label_smoothing: float = 0.05

    @property
    def patch_size(self) -> int:
        return (self.n_pixels + self.n_tokens - 1) // self.n_tokens

    @property
    def padded_pixels(self) -> int:
        return self.patch_size * self.n_tokens

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_wavelength_sinusoidal_embedding(
    config: DESIFoundationModelConfig,
) -> torch.Tensor:
    """Build a fixed token embedding from physical log-wavelength coordinates."""
    log_min = torch.log(torch.tensor(float(config.lambda_min)))
    log_max = torch.log(torch.tensor(float(config.lambda_max)))
    pixel_pos = torch.linspace(0.0, 1.0, config.padded_pixels)

    if config.wavelength_grid == "log":
        log_lambda = log_min + pixel_pos * (log_max - log_min)
    elif config.wavelength_grid == "linear":
        wavelength = config.lambda_min + pixel_pos * (config.lambda_max - config.lambda_min)
        log_lambda = torch.log(wavelength)
    else:
        raise ValueError(
            f"Unknown wavelength_grid={config.wavelength_grid!r}; use 'log' or 'linear'."
        )

    patch_log_lambda = log_lambda.reshape(config.n_tokens, config.patch_size).mean(dim=1)
    coord = (patch_log_lambda - log_min) / (log_max - log_min).clamp_min(1e-6)

    half_dim = max(config.d_model // 2, 1)
    freq = torch.exp(
        -torch.log(torch.tensor(10_000.0))
        * torch.arange(half_dim, dtype=torch.float32)
        / max(half_dim - 1, 1)
    )
    angles = 2.0 * torch.pi * coord[:, None] * freq[None, :]
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if embedding.shape[-1] < config.d_model:
        embedding = F.pad(embedding, (0, config.d_model - embedding.shape[-1]))
    return embedding[:, : config.d_model].unsqueeze(0)


class SpectrumPatchTokenizer(nn.Module):
    """Turn one spectrum into a sequence of continuous spectral tokens."""

    def __init__(self, config: DESIFoundationModelConfig):
        super().__init__()
        self.config = config
        self.patch_dim = config.patch_size * 2
        self.proj = nn.Sequential(
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, config.d_model),
        )

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.config.padded_pixels - x.shape[-1]
        if pad > 0:
            x = F.pad(x, (0, pad))
        return x[..., : self.config.padded_pixels]

    def patchify_pair(
        self, flux: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flux = self._pad(flux)
        valid = self._pad(valid)
        bsz = flux.shape[0]
        p = self.config.patch_size
        n = self.config.n_tokens
        flux_patches = flux.reshape(bsz, n, p)
        valid_patches = valid.reshape(bsz, n, p)
        token_input = torch.cat([flux_patches, valid_patches], dim=-1)
        return token_input, flux_patches, valid_patches

    def forward(
        self, flux: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        token_input, flux_patches, valid_patches = self.patchify_pair(flux, valid)
        return self.proj(token_input), flux_patches, valid_patches


class DESIFoundationModel(nn.Module):
    """Masked spectral-token model with joint redshift learning.

    The redshift token is always masked. The model therefore never receives the
    answer as input; the z loss shapes the transformer representation directly.
    """

    def __init__(self, config: DESIFoundationModelConfig):
        super().__init__()
        self.config = config
        self.tokenizer = SpectrumPatchTokenizer(config)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.z_mask_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.learned_pos_embed = (
            nn.Parameter(torch.zeros(1, config.n_tokens + 1, config.d_model))
            if config.use_learned_position
            else None
        )
        if config.use_wavelength_embedding:
            wavelength_embed = build_wavelength_sinusoidal_embedding(config)
            z_wavelength_embed = torch.zeros(1, 1, config.d_model)
            wavelength_embed = torch.cat([wavelength_embed, z_wavelength_embed], dim=1)
        else:
            wavelength_embed = torch.zeros(1, config.n_tokens + 1, config.d_model)
        self.register_buffer(
            "wavelength_pos_embed", wavelength_embed, persistent=False
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=int(config.d_model * config.mlp_ratio),
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        self.norm = nn.LayerNorm(config.d_model)

        self.reconstruction_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.patch_size),
        )
        head_out_dim = config.n_z_bins if config.n_z_bins > 0 else 1
        self.redshift_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, head_out_dim),
        )
        if config.n_z_bins > 0:
            edges = torch.linspace(0.0, math.log1p(config.z_max), config.n_z_bins + 1)
            self.register_buffer("z_bin_edges", edges, persistent=False)
            self.register_buffer(
                "z_bin_centers", 0.5 * (edges[:-1] + edges[1:]), persistent=False
            )
            # pesos de rebalanceo (los setea train.py; ones = sin rebalanceo)
            self.register_buffer(
                "z_bin_weights", torch.ones(config.n_z_bins), persistent=False
            )
        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.z_mask_token, std=0.02)
        if self.learned_pos_embed is not None:
            nn.init.trunc_normal_(self.learned_pos_embed, std=0.02)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def random_spectrum_mask(
        self, batch_size: int, device: torch.device, mask_ratio: float | None = None
    ) -> torch.Tensor:
        ratio = self.config.mask_ratio if mask_ratio is None else mask_ratio
        return torch.rand(batch_size, self.config.n_tokens, device=device) < ratio

    @staticmethod
    def encode_redshift(z: torch.Tensor) -> torch.Tensor:
        return torch.log1p(torch.clamp(z, min=0.0))

    @staticmethod
    def decode_redshift(encoded_z: torch.Tensor) -> torch.Tensor:
        return torch.expm1(encoded_z).clamp(min=0.0)

    def forward(
        self,
        flux: torch.Tensor,
        valid: torch.Tensor,
        z: torch.Tensor | None = None,
        spectrum_mask: torch.Tensor | None = None,
        mask_ratio: float | None = None,
    ) -> dict[str, torch.Tensor]:
        bsz = flux.shape[0]
        device = flux.device
        tokens, target_patches, valid_patches = self.tokenizer(flux, valid)

        if spectrum_mask is None:
            spectrum_mask = self.random_spectrum_mask(bsz, device, mask_ratio)
        spectrum_mask = spectrum_mask.bool()

        masked_tokens = torch.where(
            spectrum_mask.unsqueeze(-1),
            self.mask_token.expand(bsz, self.config.n_tokens, -1),
            tokens,
        )
        z_token = self.z_mask_token.expand(bsz, 1, -1)
        sequence = torch.cat([masked_tokens, z_token], dim=1)
        sequence = sequence + self.wavelength_pos_embed.to(dtype=sequence.dtype)
        if self.learned_pos_embed is not None:
            sequence = sequence + self.learned_pos_embed

        hidden = self.norm(self.encoder(sequence))
        spectrum_hidden = hidden[:, : self.config.n_tokens]
        z_hidden = hidden[:, self.config.n_tokens]

        patch_pred = self.reconstruction_head(spectrum_hidden)
        head_out = self.redshift_head(z_hidden)
        z_logits = None
        if self.config.n_z_bins > 0:
            z_logits = head_out                                   # (B, n_bins)
            p = z_logits.softmax(-1)
            z_encoded_pred = (p * self.z_bin_centers).sum(-1)     # esperanza en log(1+z)
            z_pred = self.decode_redshift(z_encoded_pred)
            z_pred_map = self.decode_redshift(self.z_bin_centers[z_logits.argmax(-1)])
            entropy = -(p.clamp_min(1e-9).log() * p).sum(-1) / math.log(self.config.n_z_bins)
        else:
            z_encoded_pred = head_out.squeeze(-1)
            z_pred = self.decode_redshift(z_encoded_pred)

        out: dict[str, torch.Tensor] = {
            "patch_pred": patch_pred,
            "z_pred": z_pred,
            "spectrum_mask": spectrum_mask,
        }
        if self.config.n_z_bins > 0:
            out["z_logits"] = z_logits
            out["z_pred_map"] = z_pred_map                        # argmax: el anti-outliers
            out["z_confidence"] = 1.0 - entropy                   # 1 = posterior concentrada

        valid_masked = valid_patches * spectrum_mask.unsqueeze(-1).float()
        valid_count = valid_masked.sum()
        recon_sse = (((patch_pred - target_patches) ** 2) * valid_masked).sum()
        recon_loss = recon_sse / valid_count.clamp_min(1.0)
        out["reconstruction_loss"] = recon_loss
        out["reconstruction_sse"] = recon_sse
        out["reconstruction_valid_pixels"] = valid_count
        out["reconstruction_rmse"] = torch.sqrt(recon_loss.clamp_min(0.0))

        if z is not None:
            if self.config.n_z_bins > 0:
                target_bin = torch.bucketize(
                    self.encode_redshift(z), self.z_bin_edges[1:-1]
                )
                z_loss = F.cross_entropy(
                    z_logits,
                    target_bin,
                    weight=self.z_bin_weights,
                    label_smoothing=self.config.z_label_smoothing,
                )
            else:
                z_loss = F.smooth_l1_loss(z_encoded_pred, self.encode_redshift(z))
            total = (
                self.config.reconstruction_loss_weight * recon_loss
                + self.config.redshift_loss_weight * z_loss
            )
            out["redshift_loss"] = z_loss
            out["loss"] = total
        else:
            out["loss"] = recon_loss

        return out

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        pixels = patches.reshape(patches.shape[0], self.config.padded_pixels)
        return pixels[:, : self.config.n_pixels]

    @torch.no_grad()
    def reconstruct_full_spectrum(
        self, flux: torch.Tensor, valid: torch.Tensor, mask_ratio: float = 1.0
    ) -> torch.Tensor:
        mask = self.random_spectrum_mask(flux.shape[0], flux.device, mask_ratio)
        out = self.forward(flux, valid, spectrum_mask=mask)
        return self.unpatchify(out["patch_pred"])
