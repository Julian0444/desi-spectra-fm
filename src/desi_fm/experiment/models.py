"""Fixed experiment models and train-only class weights (no historical weights)."""
from __future__ import annotations

from dataclasses import asdict
import math
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..model import DESIFoundationModel, DESIFoundationModelConfig
from .artifacts import digest


def default_config():
    model = DESIFoundationModelConfig(d_model=512, n_layers=8, n_heads=8,
        n_z_bins=100, z_max=6., mask_ratio=.5, redshift_loss_weight=1.,
        reconstruction_loss_weight=1., z_label_smoothing=0., normalize_redshift_ce=True)
    return {'schema': 'batch4-training-v1', 'model': asdict(model),
            'micro_batch': 8, 'accumulation': 4, 'updates': 30000,
            'learning_rate': .0003, 'weight_decay': .01, 'betas': [.9, .999],
            'adam_epsilon': 1e-8, 'clip_grad_norm': 1., 'warmup_updates': 200,
            'minimum_lr_fraction': .1, 'validation_every': 500, 'checkpoint_every': 500,
            'cpu_threads': 4, 'precision': 'float32', 'num_workers': 0,
            'class_weights': 'inverse_sqrt_max_count_1_mean_one',
            'normalization': 'visible_only_median_iqr_std_eps_identity_if_lt8',
            'seeds': [17, 42, 101]}


def validate_config(config):
    if set(config) != set(default_config()) or config['schema'] != 'batch4-training-v1':
        raise ValueError('Incompatible experiment configuration schema')
    for key in ('micro_batch', 'accumulation', 'updates', 'validation_every', 'checkpoint_every', 'cpu_threads'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'Invalid {key}')
    if config['updates'] > 30000 or config['precision'] != 'float32' or config['num_workers'] != 0 or config['seeds'] != [17,42,101]:
        raise ValueError('Outside frozen experiment scope')
    m = DESIFoundationModelConfig(**config['model'])
    if m.n_z_bins != 100 or m.mask_ratio != .5 or m.redshift_loss_weight != 1 or m.reconstruction_loss_weight != 1 or m.z_label_smoothing != 0 or not m.normalize_redshift_ce:
        raise ValueError('A/B loss, masking or bins changed')
    if m.z_max != 6 or config['class_weights'] != default_config()['class_weights'] or config['normalization'] != default_config()['normalization']:
        raise ValueError('Unsupported weighting/normalization/range')
    if not 0 <= config['warmup_updates'] < config['updates'] or not 0 < config['minimum_lr_fraction'] <= 1:
        raise ValueError('Invalid scheduler horizon')
    return m


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def target_bins(z, edges):
    if not torch.isfinite(z).all() or (z < 0).any():
        raise ValueError('Ineligible training label')
    return torch.bucketize(torch.log1p(z), edges[1:-1])


def class_weights(train, config):
    if train.role != 'train':
        raise PermissionError('Class weights require train only')
    edges = torch.linspace(0., math.log1p(config['model']['z_max']), 101)
    labels = torch.tensor(train.z, dtype=torch.float32)
    counts = torch.bincount(target_bins(labels, edges), minlength=100)
    weights = counts.clamp_min(1).float().rsqrt()
    weights /= weights.mean()
    manifest = {'train_identity': train.identity, 'train_n': len(train),
                'counts': counts.tolist(), 'weights': weights.tolist(),
                'out_of_range': int((labels > config['model']['z_max']).sum()),
                'rule': config['class_weights'], 'bin_edges': edges.tolist()}
    manifest['sha256'] = digest(manifest)
    return weights, manifest


class FixedAdaptivePool1d(nn.Module):
    """Exactly the adaptive-average windows, expressed as a fixed linear map.

    MPS does not implement non-divisible adaptive pooling. This performs the
    same floor/ceil windows on-device and supports autograd without CPU copies.
    """
    def __init__(self, input_size, output_size=64):
        super().__init__()
        matrix = torch.zeros(input_size,output_size)
        for i in range(output_size):
            first = i*input_size//output_size
            last = ((i+1)*input_size+output_size-1)//output_size
            matrix[first:last,i] = 1./(last-first)
        self.register_buffer('matrix',matrix,persistent=False)

    def forward(self, x):
        return x @ self.matrix.to(dtype=x.dtype)


class SpectralCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        layers, previous = [], 3
        for channels in (32,64,128):
            layers.extend([nn.Conv1d(previous, channels, kernel_size=9, stride=4, padding=4), nn.GELU()])
            previous = channels
        self.features = nn.Sequential(*layers)
        positions = config.n_pixels
        for _ in range(3):
            positions = (positions+3)//4
        self.pool = FixedAdaptivePool1d(positions,64)
        self.head = nn.Linear(128*64, 100)
        # The base grid is logarithmic: this is the affine-normalized log lambda.
        self.register_buffer('log_lambda', torch.linspace(0., 1., config.n_pixels), persistent=False)
        edges = torch.linspace(0., math.log1p(config.z_max), 101)
        self.register_buffer('z_bin_edges', edges, persistent=False)
        self.register_buffer('z_bin_centers', (edges[:-1]+edges[1:])/2, persistent=False)
        self.register_buffer('z_bin_weights', torch.ones(100), persistent=False)
        if sum(p.numel() for p in self.parameters()) >= 1_000_000:
            raise ValueError('CNN exceeds fixed parameter budget')

    def forward(self, flux, valid, **unused):
        inputs = torch.stack([flux, valid, self.log_lambda.expand_as(flux)], dim=1)
        logits = self.head(self.pool(self.features(inputs)).flatten(1))
        p = logits.softmax(-1)
        return {'z_logits': logits,
                'z_pred_map': torch.expm1(self.z_bin_centers[logits.argmax(-1)]),
                'z_pred': torch.expm1((p*self.z_bin_centers).sum(-1)),
                'z_confidence': 1 + (p*p.clamp_min(1e-9).log()).sum(-1)/math.log(100)}


def make_model(config, variant, seed, weights):
    model_config = validate_config(config)
    if variant not in ('A','B','CNN') or seed not in config['seeds']:
        raise ValueError('Unknown variant/seed')
    seed_everything(seed)
    if variant == 'CNN':
        model = SpectralCNN(model_config)
    else:
        model_config.reconstruction_loss_weight = 1. if variant == 'A' else 0.
        model = DESIFoundationModel(model_config)
    model.z_bin_weights.copy_(weights)
    return model


def classification_sum(logits, z, model):
    bins = target_bins(z, model.z_bin_edges)
    return F.cross_entropy(logits, bins, weight=model.z_bin_weights, reduction='sum') / math.log(100)


def lr_multiplier(update, config):
    # LambdaLR starts at update=0; update n is the LR for the next optimizer step.
    warmup, total = config['warmup_updates'], config['updates']
    if warmup and update < warmup:
        return (update + 1) / warmup
    fraction = min(1., max(0., (update-warmup)/max(1,total-warmup)))
    minimum = config['minimum_lr_fraction']
    return minimum + (1-minimum)*.5*(1+math.cos(math.pi*fraction))
