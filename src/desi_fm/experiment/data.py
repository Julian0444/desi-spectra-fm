"""Fast local role-scoped arrays; test labels are never loaded by development."""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np

from ..data import SpectrumPreprocessConfig, normalize_grid
from .artifacts import read_json, verify_record, digest

GRID_CONTRACT = "batch4-grid-v2:unnormalized-spectra-input-v3"
TEST_PHRASE = "OPEN_FROZEN_BATCH4_TEST_ONCE"


class ExperimentData:
    def __init__(self, root, role, *, test_authorization=None):
        if role not in ('train', 'validation', 'test'):
            raise ValueError("Unknown split")
        if role == 'test' and test_authorization != TEST_PHRASE:
            raise PermissionError("Test is closed; use the frozen final evaluator")
        self.root, self.role = Path(root), role
        self.manifest = read_json(self.root / 'manifest.json')
        if self.manifest.get('status') != 'complete' or self.manifest.get('contract') != GRID_CONTRACT:
            raise ValueError("Incomplete or incompatible derivative")
        entry = self.manifest['splits'][role]
        for name, record in entry['files'].items():
            if Path(name).name != name:
                raise ValueError("Invalid derivative path")
            verify_record(self.root / role / name, record)
        self.ids = read_json(self.root / role / 'ids.json')
        self.flux = np.load(self.root / role / 'flux.npy', mmap_mode='r', allow_pickle=False)
        self.valid = np.load(self.root / role / 'valid.npy', mmap_mode='r', allow_pickle=False)
        with np.load(self.root / role / 'metadata.npz', allow_pickle=False) as data:
            self.z = data['z'].copy()
            self.zwarn = data['zwarn'].copy()
            self.groups = data['groups'].copy()
        self.config = SpectrumPreprocessConfig(**self.manifest['preprocess'])
        shape = (len(self.ids), self.config.n_pixels)
        if self.flux.shape != shape or self.valid.shape != shape or len(set(self.ids)) != len(self.ids) or len(self.z) != len(self.ids):
            raise ValueError("Invalid derivative dimensions/IDs")
        self.identity = digest({"contract": GRID_CONTRACT, "preprocess": self.manifest['preprocess'], "split": role, "files": entry['files']})

    def __len__(self):
        return len(self.ids)

    def batch(self, indices, *, seed=0, draw_positions=None, n_tokens=273, masked=False):
        indices = np.asarray(indices, dtype=np.int64)
        if masked and (self.role != 'train' or draw_positions is None):
            raise ValueError("Artificial masks are training-only and require draw positions")
        raw = np.array(self.flux[indices], dtype=np.float32)
        valid = np.array(self.valid[indices], dtype=np.float32)
        masks = np.zeros((len(indices), n_tokens), dtype=bool)
        if masked:
            masks = sample_masks([self.ids[i] for i in indices], draw_positions, seed, n_tokens)
        patch_size = (self.config.n_pixels + n_tokens - 1) // n_tokens
        hidden = np.repeat(masks, patch_size, axis=1)[:, :self.config.n_pixels]
        normalized = [normalize_grid(f, v, self.config, m, insufficient='identity') for f, v, m in zip(raw, valid, hidden)]
        return {"ids": [self.ids[i] for i in indices], "flux": np.stack([r['flux'] for r in normalized]),
                "valid": valid, "z": self.z[indices].astype(np.float32), "mask": masks,
                "fallbacks": int(sum(r['normalization_fallback'] for r in normalized)),
                "centers": np.array([r['center'] for r in normalized]), "scales": np.array([r['scale'] for r in normalized]),
                "hidden_valid": int((hidden * valid).sum())}


def sample_masks(ids, draw_positions, seed, n_tokens):
    if len(ids) != len(draw_positions):
        raise ValueError("Draw/ID count mismatch")
    result = []
    for oid, position in zip(ids, draw_positions):
        key = hashlib.sha256(f'batch4-mask-v1:{seed}:{int(position)}:{oid}'.encode()).digest()
        rng = np.random.Generator(np.random.PCG64(int.from_bytes(key[:16], 'little')))
        result.append(rng.random(n_tokens) < .5)
    return np.asarray(result, dtype=bool)


class PairedSampler:
    """Infinite shuffled epochs; RNG-independent draw order includes epoch tails."""
    def __init__(self, size, seed, position=0):
        if size <= 0 or position < 0:
            raise ValueError("Invalid sampler")
        self.size, self.seed, self.position = size, seed, position
        self._epoch, self._order = None, None

    def take(self, count):
        positions = np.arange(self.position, self.position + count, dtype=np.int64)
        indices = []
        for position in positions:
            epoch, offset = divmod(int(position), self.size)
            if epoch != self._epoch:
                self._order = np.random.Generator(np.random.PCG64(np.random.SeedSequence([self.seed, epoch]))).permutation(self.size)
                self._epoch = epoch
            indices.append(self._order[offset])
        self.position += count
        return np.array(indices, dtype=np.int64), positions
