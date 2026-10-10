"""Metrics from persisted predictions; pairing preserves objects across seeds."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from ..historical import metrics
from .artifacts import atomic_write, read_json, write_json, file_record


def selection_key(summary, step):
    return (summary['primary']['eta15'], summary['primary']['mae_norm'], int(step))


def save_predictions(path, data, indices, outputs, identity):
    indices = np.asarray(indices, dtype=np.int64)
    ids = np.asarray([data.ids[i] for i in indices])
    if len(set(ids)) != len(ids) or not len(ids):
        raise ValueError('Empty or duplicate prediction IDs')
    arrays = {'ids': ids, 'groups': data.groups[indices], 'z_true': data.z[indices], 'zwarn': data.zwarn[indices],
              'coverage': np.asarray(data.valid[indices].mean(axis=1), dtype=np.float64),
              'prediction': np.asarray(outputs['prediction'], dtype=np.float64),
              'secondary': np.asarray(outputs.get('secondary', outputs['prediction']), dtype=np.float64),
              'confidence': np.asarray(outputs.get('confidence', np.full(len(ids), np.nan)), dtype=np.float64)}
    if arrays['prediction'].shape != arrays['z_true'].shape or not np.isfinite(arrays['prediction']).all():
        raise ValueError('Missing/nonfinite predictions; cannot silently drop objects')
    path = Path(path)
    atomic_write(path, lambda stream: np.savez(stream, **arrays))
    write_json(path.with_suffix('.json'), {'role': data.role, 'data_identity': data.identity,
               'requested': len(indices), 'evaluated': len(ids), 'rejected': 0,
               'artificial_mask': False, 'identity': identity, 'file': file_record(path)})
    return summarize_predictions(path)


def load_predictions(path):
    from .artifacts import verify_record
    manifest = read_json(Path(path).with_suffix('.json'))
    verify_record(path, manifest['file'])
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    n = len(arrays['ids'])
    if not n or len(set(arrays['ids'])) != n or manifest['requested'] != n or manifest['evaluated'] != n or manifest['rejected'] != 0:
        raise ValueError('Invalid prediction accounting')
    if any(len(v) != n for v in arrays.values()) or not np.isfinite(arrays['prediction']).all():
        raise ValueError('Invalid prediction arrays')
    return arrays, manifest


def summarize_predictions(path):
    a, manifest = load_predictions(path)
    z, prediction = a['z_true'], a['prediction']
    summary = {'accounting': {k: manifest[k] for k in ('requested','evaluated','rejected')},
               'primary': metrics(z, prediction), 'secondary': metrics(z, a['secondary']), 'strata': {}}
    axes = {'z': [(f'[{lo},{hi})', (z >= lo) & (z < hi)) for lo,hi in zip([0,.5,1,1.5,2,2.5],[.5,1,1.5,2,2.5,np.inf])],
            'quality': [('original_warning', a['zwarn']==0), ('no_original_warning', a['zwarn']==1)],
            'coverage': [(f'[{lo},{hi})', (a['coverage'] >= lo) & (a['coverage'] < hi)) for lo,hi in zip([0,.5,.9,.99],[.5,.9,.99,1.000001])]}
    for axis, entries in axes.items():
        summary['strata'][axis] = {key: metrics(z[keep], prediction[keep]) for key,keep in entries}
    if np.isfinite(a['confidence']).all():
        order = np.lexsort((a['ids'], -a['confidence']))
        errors = (np.abs(prediction-z)/(1+z) > .15)[order]
        summary['risk_coverage'] = {'retained': list(range(1,len(z)+1)),
                                    'eta15': (np.cumsum(errors)/np.arange(1,len(z)+1)).tolist(),
                                    'coverage': (np.arange(1,len(z)+1)/len(z)).tolist(),
                                    'minimum_score': a['confidence'][order].tolist()}
    else:
        summary['risk_coverage'] = None  # ridge has no entropy/concentration score
    write_json(Path(path).with_suffix('.summary.json'), summary)
    return summary


def paired_analysis(paths_a, paths_b, *, seeds=(17,42,101), replicates=5000, bootstrap_seed=20261009):
    if len(paths_a) != 3 or len(paths_b) != 3 or tuple(seeds) != (17,42,101):
        raise ValueError('The comparison requires all three paired seeds')
    pairs, reference = [], None
    for seed, pa, pb in zip(seeds,paths_a,paths_b):
        a, am = load_predictions(pa); b, bm = load_predictions(pb)
        for arr in (a,b):
            if reference is None:
                reference = arr
            for key in ('ids','groups','z_true','zwarn','coverage'):
                if not np.array_equal(arr[key], reference[key]):
                    raise ValueError('Unpaired IDs, groups or labels across models/seeds')
        if am['data_identity'] != bm['data_identity'] or am['identity']['seed'] != seed or bm['identity']['seed'] != seed or am['identity'].get('variant') != 'A' or bm['identity'].get('variant') != 'B':
            raise ValueError('Unpaired provenance')
        z = a['z_true']
        ra, rb = (a['prediction']-z)/(1+z), (b['prediction']-z)/(1+z)
        pairs.append((ra,rb))
    ra, rb = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    ea, eb = np.abs(ra) > .15, np.abs(rb) > .15
    per_seed = (ea.mean(axis=1)-eb.mean(axis=1))
    delta_by_object = (ea.astype(float)-eb).mean(axis=0)
    unique, inverse = np.unique(reference['groups'], return_inverse=True)
    sums = np.bincount(inverse, weights=delta_by_object); counts = np.bincount(inverse)
    rng = np.random.Generator(np.random.PCG64(bootstrap_seed))
    draws = np.empty(replicates)
    for i in range(replicates):
        chosen = rng.integers(0, len(unique), size=len(unique))
        draws[i] = sums[chosen].sum()/counts[chosen].sum()
    interval = np.quantile(draws,[.025,.975]).tolist()
    mae_delta = float(np.abs(ra).mean()-np.abs(rb).mean())
    strata = {}
    masks = [('warning',reference['zwarn']==0), ('no_warning',reference['zwarn']==1)]
    masks += [(f'z[{lo},{hi})', (reference['z_true']>=lo)&(reference['z_true']<hi)) for lo,hi in zip([0,.5,1,1.5,2,2.5],[.5,1,1.5,2,2.5,np.inf])]
    for key, mask in masks:
        stratum_interval = None
        if mask.any():
            gs, inv = np.unique(reference['groups'][mask],return_inverse=True)
            ns = np.bincount(inv); ds = np.bincount(inv,weights=delta_by_object[mask])
            sampled = np.empty(replicates)
            for j in range(replicates):
                selected = rng.integers(0,len(gs),size=len(gs))
                sampled[j] = ds[selected].sum()/ns[selected].sum()
            stratum_interval = np.quantile(sampled,[.025,.975]).tolist()
        strata[key] = {'n': int(mask.sum()), 'paired_mean_interval95':stratum_interval,
                       'delta_eta_per_seed': (ea[:,mask].mean(axis=1)-eb[:,mask].mean(axis=1)).tolist() if mask.any() else [None]*3}
    return {'seeds': list(seeds), 'n_objects': len(reference['ids']), 'n_observed_groups': len(unique),
            'delta_eta_per_seed': per_seed.tolist(), 'mean_delta_eta': float(per_seed.mean()),
            'paired_95_interval': interval, 'mean_delta_mae_norm': mae_delta,
            'criterion_met': bool((per_seed<0).all() and interval[1]<0 and mae_delta<=0),
            'strata': strata, 'bootstrap_replicates': replicates, 'bootstrap_seed': bootstrap_seed,
            'interval_scope': 'group/object sampling conditional on these three training seeds; physical independence of singleton IDs is unverified',
            'pairing': 'one sampled group multiplicity shared by A/B and every seed; seeds are not extra objects'}
