"""Paired training, bounded profiling and durable resumable state."""
from __future__ import annotations

import contextlib
import hashlib
import os
from pathlib import Path
import random
import resource
import signal
import time
import uuid
import fcntl

import numpy as np
import torch

from .artifacts import atomic_write, digest, environment, file_record, read_json, source_identity, verify_record, write_json
from .data import ExperimentData, PairedSampler
from .metrics import save_predictions, selection_key
from .models import class_weights, classification_sum, lr_multiplier, make_model, target_bins, validate_config


def synchronize(device):
    if str(device).startswith('mps'):
        torch.mps.synchronize()
    elif str(device).startswith('cuda'):
        torch.cuda.synchronize()


def memory(device):
    import sys
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {'peak_host_rss_bytes': int(rss * (1 if sys.platform == 'darwin' else 1024))}
    if str(device) == 'mps':
        result.update(mps_current_allocated_bytes=torch.mps.current_allocated_memory(),
                      mps_driver_allocated_bytes=torch.mps.driver_allocated_memory())
    return result


def capture_rng(device):
    state = np.random.get_state()
    result = {'python': random.getstate(), 'numpy': [state[0], state[1].tolist(), state[2], state[3], state[4]],
              'torch_cpu': torch.get_rng_state()}
    if str(device) == 'mps':
        result['torch_mps'] = torch.mps.get_rng_state()
    return result


def restore_rng(state, device):
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state['torch_cpu'])
    if str(device) == 'mps':
        torch.mps.set_rng_state(state['torch_mps'])


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k,v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


@contextlib.contextmanager
def run_lock(directory):
    with (Path(directory)/'.writer.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Another writer owns this run') from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def save_checkpoint(directory, state, *, best=False):
    """Publish a new immutable payload before its atomic pointer; retain fallback."""
    directory = Path(directory)
    old = read_json(directory/'checkpoints.json') if (directory/'checkpoints.json').exists() else {}
    name = f"state-{state['step']:06d}-{uuid.uuid4().hex}.pt"
    path = directory/name
    atomic_write(path, lambda stream: torch.save(cpu_tree(state), stream))
    entry = {'path': name, 'step': state['step'], **file_record(path)}
    previous = old.get('last')
    if previous:
        try:
            verify_record(directory/previous['path'],{k:previous[k] for k in ('bytes','sha256')})
        except (ValueError,OSError):
            previous = old.get('previous')
    pointer = {'last': entry, 'previous': previous, 'best': entry if best else old.get('best')}
    write_json(directory/'checkpoints.json', pointer)
    keep = {v['path'] for v in pointer.values() if v}
    # Only our previous, now-unreferenced generations are removed. Orphan files
    # left by a crash are preserved for diagnosis.
    for record in old.values():
        if record and record['path'] not in keep:
            (directory/record['path']).unlink(missing_ok=True)
    return entry


def load_checkpoint(directory, expected_identity, *, which='last'):
    directory = Path(directory)
    pointers = read_json(directory/'checkpoints.json')
    candidates = [which, 'previous'] if which == 'last' else [which]
    errors = []
    for key in candidates:
        record = pointers.get(key)
        if not record:
            continue
        try:
            verify_record(directory/record['path'], {k: record[k] for k in ('bytes','sha256')})
            state = torch.load(directory/record['path'], map_location='cpu', weights_only=True)
        except (ValueError, OSError, EOFError, RuntimeError) as exc:
            errors.append(str(exc))
            continue
        if state['identity'] != expected_identity:
            raise ValueError('Incompatible checkpoint identity (config/code/data/seed/device/mode)')
        return state, key
    raise ValueError('No intact checkpoint generation: ' + '; '.join(errors))


@torch.no_grad()
def predict(model, data, indices, batch_size, device):
    """Full spectrum only, including for train-only evaluation profiling."""
    model.eval()
    outputs = {'prediction': [], 'secondary': [], 'confidence': []}
    for start in range(0,len(indices),batch_size):
        batch = data.batch(indices[start:start+batch_size], n_tokens=model.config.n_tokens, masked=False)
        out = model(torch.from_numpy(batch['flux']).to(device), torch.from_numpy(batch['valid']).to(device),
                    spectrum_mask=torch.from_numpy(batch['mask']).to(device), mask_ratio=0.)
        for field, key in [('prediction','z_pred_map'),('secondary','z_pred'),('confidence','z_confidence')]:
            outputs[field].extend(out[key].cpu().numpy().astype(float).tolist())
    return outputs


class Trainer:
    def __init__(self, data, config, variant, seed, directory, device='cpu', *, profile=False, resume=False):
        validate_config(config)
        if data.role != 'train':
            raise PermissionError('Training/profiling requires train only')
        if str(device) not in ('cpu','mps'):
            raise ValueError('This local protocol supports CPU or MPS')
        self.data, self.config, self.variant, self.seed = data, config, variant, seed
        self.directory, self.device, self.profile = Path(directory), torch.device(device), profile
        for key in ('n_pixels', 'lambda_min', 'lambda_max', 'wavelength_grid'):
            if getattr(data.config,key) != config['model'][key]:
                raise ValueError('Model grid differs from verified data')
        self.step, self.best, self.fallbacks, self.trace = 0, None, 0, 'initial'
        self.timings = []
        torch.set_num_threads(config['cpu_threads'])
        torch.use_deterministic_algorithms(str(device) == 'cpu')
        weights, self.weight_manifest = class_weights(data, config)
        self.identity = {'schema': 'batch4-run-v1', 'mode': 'profile' if profile else 'training',
                         'config': config, 'variant': variant, 'seed': seed, 'device': str(device),
                         'train_identity': data.identity,
                         'validation_identity': None if profile else digest({'contract':data.manifest['contract'], 'preprocess':data.manifest['preprocess'], 'split':'validation', 'files':data.manifest['splits']['validation']['files']}),
                         'class_weights_sha256': self.weight_manifest['sha256'],
                         'source_sha256': source_identity()['sha256'], 'environment': environment()}
        self.model = make_model(config,variant,seed,weights).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config['learning_rate'],
            betas=tuple(config['betas']), eps=config['adam_epsilon'], weight_decay=config['weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: lr_multiplier(step,config))
        self.sampler = PairedSampler(len(data), seed)
        self.recovery_seconds, self.recovered_from = 0., None
        self._resume_pointer = None
        if resume:
            start = time.perf_counter()
            manifest = read_json(self.directory/'run.json')
            if manifest['identity'] != self.identity:
                raise ValueError('Incompatible run identity')
            with run_lock(self.directory):
                state, self.recovered_from = load_checkpoint(self.directory,self.identity)
                self._resume_pointer = read_json(self.directory/'checkpoints.json')
            self.model.load_state_dict(state['model'], strict=True)
            saved_weights = state['z_bin_weights']
            if not torch.equal(saved_weights, weights):
                raise ValueError('Train-only class weights were not preserved')
            self.model.z_bin_weights.copy_(saved_weights.to(self.device))
            self.optimizer.load_state_dict(state['optimizer'])
            self.scheduler.load_state_dict(state['scheduler'])
            self.step, self.best = state['step'], state['best']
            self.sampler = PairedSampler(len(data), seed, state['sampler_position'])
            expected_position = self.step * config['micro_batch'] * config['accumulation']
            if self.sampler.position != expected_position:
                raise ValueError('Sampler position does not match completed updates')
            self.fallbacks, self.trace, self.timings = state['fallbacks'], state['trace'], state['timings']
            restore_rng(state['rng'], self.device)
            synchronize(self.device)
            self.recovery_seconds = time.perf_counter()-start
        else:
            self.directory.mkdir(parents=True, exist_ok=False)
            write_json(self.directory/'run.json', {'status': 'incomplete', 'identity': self.identity,
                'source': source_identity(), 'class_weights': self.weight_manifest,
                'parameters': sum(p.numel() for p in self.model.parameters()),
                'reproducibility': 'CPU exact resume tested; MPS numeric limitations recorded in acceptance'})

        self._rng = capture_rng(self.device)

    def state(self):
        return {'identity': self.identity, 'model': self.model.state_dict(),
                'z_bin_weights': self.model.z_bin_weights,  # historical buffer stays nonpersistent
                'optimizer': self.optimizer.state_dict(), 'scheduler': self.scheduler.state_dict(),
                'step': self.step, 'sampler_position': self.sampler.position,
                'rng': capture_rng(self.device), 'best': self.best,
                'fallbacks': self.fallbacks, 'trace': self.trace, 'timings': self.timings}

    def update(self):
        config = self.config
        self.model.train()
        synchronize(self.device)
        start = time.perf_counter()
        indices, positions = self.sampler.take(config['micro_batch']*config['accumulation'])
        batch = self.data.batch(indices, seed=self.seed, draw_positions=positions,
                                n_tokens=self.model.config.n_tokens, masked=self.variant != 'CNN')
        read_seconds = time.perf_counter()-start
        self.trace = digest({'previous': self.trace, 'ids': batch['ids'],
                             'positions': positions.tolist(), 'masks': hashlib.sha256(batch['mask'].tobytes()).hexdigest()})
        self.fallbacks += batch['fallbacks']
        z = torch.from_numpy(batch['z']).to(self.device)
        denominator = self.model.z_bin_weights[target_bins(z,self.model.z_bin_edges)].sum()
        recon_denominator = max(1,batch['hidden_valid'])
        self.optimizer.zero_grad(set_to_none=True)
        seconds_forward, seconds_backward, loss_z, loss_rec = 0.,0.,0.,0.
        for first in range(0,len(indices),config['micro_batch']):
            last = first+config['micro_batch']
            synchronize(self.device); start = time.perf_counter()
            out = self.model(torch.from_numpy(batch['flux'][first:last]).to(self.device),
                torch.from_numpy(batch['valid'][first:last]).to(self.device),
                spectrum_mask=torch.from_numpy(batch['mask'][first:last]).to(self.device))
            lz = classification_sum(out['z_logits'],z[first:last],self.model)/denominator
            lr = out.get('reconstruction_sse', torch.zeros((),device=self.device))/recon_denominator
            # Multiply by 0 for B, retaining the identical graph/auxiliary head.
            loss = lz + (1. if self.variant == 'A' else 0.)*lr
            synchronize(self.device); seconds_forward += time.perf_counter()-start
            start = time.perf_counter(); loss.backward(); synchronize(self.device)
            seconds_backward += time.perf_counter()-start
            loss_z += float(lz.detach()); loss_rec += float(lr.detach())
        start = time.perf_counter()
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),config['clip_grad_norm'],error_if_nonfinite=True)
        self.optimizer.step(); self.scheduler.step(); synchronize(self.device)
        update_seconds = time.perf_counter()-start
        self.step += 1
        record = {'step': self.step, 'read_preprocess_seconds': read_seconds,
                  'forward_seconds': seconds_forward, 'backward_seconds': seconds_backward,
                  'optimizer_seconds': update_seconds, 'loss_z': loss_z, 'loss_recon': loss_rec,
                  'gradient_norm': float(norm), 'normalization_fallbacks': batch['fallbacks'],
                  **memory(self.device)}
        self.timings.append(record)
        return record

    def checkpoint(self, best=False):
        synchronize(self.device); start = time.perf_counter()
        entry = save_checkpoint(self.directory,self.state(),best=best)
        return {**entry, 'write_seconds': time.perf_counter()-start}

    def validate(self, validation):
        if self.profile or validation.role != 'validation':
            raise PermissionError('Profiles cannot load validation; checkpoint selection is validation-only')
        synchronize(self.device); start = time.perf_counter()
        outputs = predict(self.model,validation,np.arange(len(validation)),self.config['micro_batch'],self.device)
        path = self.directory/f'validation-{self.step:06d}.npz'
        summary = save_predictions(path,validation,np.arange(len(validation)),outputs,
            {'seed':self.seed,'variant':self.variant,'step':self.step,'run_identity':digest(self.identity)})
        key = selection_key(summary,self.step)
        improved = self.best is None or key < tuple(self.best['key'])
        if improved:
            self.best = {'key': list(key), 'predictions': path.name, 'data_identity': validation.identity}
        return improved, time.perf_counter()-start

    def run(self, *, validation=None, until=None, install_signals=True):
        target = self.config['updates'] if until is None else int(until)
        if target < self.step or target > self.config['updates'] or (self.profile and target > 200):
            raise ValueError('Invalid run budget; profiles are limited to 200 updates')
        if not self.profile and (validation is None or validation.role != 'validation'):
            raise ValueError('Full training requires validation')
        if self.profile and validation is not None:
            raise PermissionError('Profiles are train-only')
        if not self.profile and validation.identity != self.identity['validation_identity']:
            raise ValueError('Validation identity differs from the frozen run')
        requested = [False]
        def stop(signum, frame):
            requested[0] = True
        old_handlers = {}
        if install_signals:
            for sig in (signal.SIGINT, signal.SIGTERM):
                old_handlers[sig] = signal.signal(sig,stop)
        checkpoints, validations = [], []
        try:
            with run_lock(self.directory):
                current = read_json(self.directory/'checkpoints.json') if (self.directory/'checkpoints.json').exists() else None
                if current != self._resume_pointer:
                    raise ValueError('Checkpoint changed after initialization; reload before resuming')
                restore_rng(self._rng,self.device)
                if current is None:
                    checkpoints.append(self.checkpoint())
                while self.step < target and not requested[0]:
                    record = self.update()
                    print({'variant':self.variant,'seed':self.seed,'step':self.step,
                           'seconds':sum(record[k] for k in ('read_preprocess_seconds','forward_seconds','backward_seconds','optimizer_seconds'))},flush=True)
                    best = False
                    if not self.profile and (self.step % self.config['validation_every']==0 or self.step==self.config['updates']):
                        best, seconds = self.validate(validation)
                        validations.append({'step':self.step,'seconds':seconds})
                    if best or self.step % self.config['checkpoint_every']==0:
                        checkpoints.append(self.checkpoint(best=best))
                checkpoints.append(self.checkpoint())
                manifest = read_json(self.directory/'run.json')
                manifest.update(status='complete' if not requested[0] and self.step==target and (self.profile or target==self.config['updates']) else 'interrupted',
                                step=self.step, best=self.best, sampler_position=self.sampler.position,
                                fallbacks=self.fallbacks, trace=self.trace, timings=self.timings,
                                checkpoint_events=manifest.get('checkpoint_events',[])+checkpoints,
                                validation_events=manifest.get('validation_events',[])+validations,
                                recovery_seconds=self.recovery_seconds,recovered_from=self.recovered_from,**memory(self.device))
                write_json(self.directory/'run.json',manifest)
                self._resume_pointer = read_json(self.directory/'checkpoints.json')
                self._rng = capture_rng(self.device)
                return manifest
        finally:
            for sig,handler in old_handlers.items():
                signal.signal(sig,handler)
