"""Validation-only freezing and explicitly gated, once-per-experiment final test."""
from __future__ import annotations

from pathlib import Path
import math
import time
import numpy as np
import torch

from .artifacts import atomic_write, digest, environment, file_record, read_json, source_identity, verify_record, write_json
from .data import ExperimentData, TEST_PHRASE
from .metrics import load_predictions, paired_analysis, save_predictions, selection_key
from .models import make_model
from .ridge import feature_matrix, ridge_predict
from .training import load_checkpoint, predict, run_lock


def common_identity(identity):
    """The execution conditions are part of the experiment, including on resume."""
    keys = ('config', 'train_identity', 'validation_identity',
            'class_weights_sha256', 'source_sha256', 'device', 'environment')
    if any(key not in identity for key in keys) or identity['device'] not in ('cpu', 'mps'):
        raise ValueError('Missing or unsupported experiment device/environment identity')
    return {key: identity[key] for key in keys}


def check_ridge_environment(ridge, neural_environment):
    # Ridge uses NumPy on the host CPU even when the neural candidates use MPS.
    # It has no neural device field. The recorded Python/platform/dependencies
    # must still match: identical version strings are our conservative contract.
    if ridge['identity'].get('environment') != neural_environment:
        raise ValueError('Ridge environment differs from the neural experiment')


def eta_intervals(arrays, *, replicates=2000, seed=20261009):
    """Group bootstrap of eta with explicit denominators, including warning strata."""
    outlier = np.abs((arrays['prediction']-arrays['z_true'])/(1+arrays['z_true'])) > .15
    result = {}
    for label,mask in [('all',np.ones(len(outlier),dtype=bool)),('original_warning',arrays['zwarn']==0),('no_original_warning',arrays['zwarn']==1)]:
        n = int(mask.sum())
        if not n:
            result[label] = {'n':0,'outliers':0,'eta15':None,'interval95':None}
            continue
        unique, inverse = np.unique(arrays['groups'][mask],return_inverse=True)
        counts = np.bincount(inverse); sums = np.bincount(inverse,weights=outlier[mask])
        rng = np.random.default_rng(seed)
        samples = np.empty(replicates)
        for i in range(replicates):
            groups = rng.integers(0,len(unique),size=len(unique))
            samples[i] = sums[groups].sum()/counts[groups].sum()
        result[label] = {'n':n,'outliers':int(outlier[mask].sum()),'eta15':float(outlier[mask].mean()),
                         'interval95':np.quantile(samples,[.025,.975]).tolist(),'observed_groups':len(unique)}
    return {'metrics':result,'replicates':replicates,'seed':seed,
            'scope':'group/object sampling at fixed checkpoint; physical independence not established'}


def freeze(runs_root, ridge_directory, data_root, output):
    """Only validation files are read here; no test labels or spectra are opened."""
    output = Path(output)
    if output.exists():
        raise ValueError('Use a new frozen manifest destination')
    data_manifest = read_json(Path(data_root)/'manifest.json')
    entries, common, validation_identity = [], None, None
    paired_paths = {'A':[],'B':[]}
    traces = {}
    for variant in ('A','B','CNN'):
        for seed in (17,42,101):
            directory = Path(runs_root)/f'{variant}-{seed}'
            report = read_json(directory/'run.json'); identity = report['identity']
            if report['status']!='complete' or identity['mode']!='training' or identity['variant']!=variant or identity['seed']!=seed or report['step']!=identity['config']['updates']:
                raise ValueError('Missing complete planned run; profiles are not final candidates')
            shared = common_identity(identity)
            if common is not None and common != shared:
                raise ValueError('Runs do not share the same recipe/data/code/device/environment')
            common = shared
            if common['source_sha256'] != source_identity()['sha256']:
                raise ValueError('Experiment source changed since training')
            traces[(variant,seed)] = report['trace']
            state,_ = load_checkpoint(directory,identity,which='best')
            if not torch.equal(state['z_bin_weights'],torch.tensor(report['class_weights']['weights'])):
                raise ValueError('Checkpoint did not preserve train-only class weights')
            best = state['best']
            path = directory/best['predictions']
            arrays, provenance = load_predictions(path)
            if provenance['role']!='validation' or provenance['artificial_mask'] or provenance['identity']!={'seed':seed,'variant':variant,'step':state['step'],'run_identity':digest(identity)}:
                raise ValueError('Checkpoint lacks matching full validation predictions')
            if validation_identity is not None and validation_identity != provenance['data_identity']:
                raise ValueError('Validation membership changed between candidates')
            validation_identity = provenance['data_identity']
            # Confirm selection over every planned common evaluation step from disk.
            keys = []
            total = identity['config']['updates']; cadence = identity['config']['validation_every']
            for step in sorted(set(range(cadence,total+1,cadence))|{total}):
                candidate, prov = load_predictions(directory/f'validation-{step:06d}.npz')
                if prov['data_identity']!=validation_identity or prov['identity']!={'seed':seed,'variant':variant,'step':step,'run_identity':digest(identity)}:
                    raise ValueError('Missing/mismatched validation selection record')
                from ..historical import metrics
                score = metrics(candidate['z_true'],candidate['prediction'])
                keys.append((score['eta15'],score['mae_norm'],step))
            if tuple(best['key']) != min(keys):
                raise ValueError('Selected checkpoint violates the frozen tie-break')
            record = read_json(directory/'checkpoints.json')['best']
            entries.append({'variant':variant,'seed':seed,'run_identity':identity,
                            'checkpoint':str((directory/record['path']).resolve()),'file':{k:record[k] for k in ('bytes','sha256')},
                            'step':state['step'],'validation_key':best['key'],
                            'validation_predictions':str(path.resolve()),'validation_file':file_record(path)})
            if variant in paired_paths:
                paired_paths[variant].append(path)
    for seed in (17,42,101):
        if traces[('A',seed)] != traces[('B',seed)]:
            raise ValueError('A/B did not consume identical IDs and masks')
    ridge_directory = Path(ridge_directory)
    ridge = read_json(ridge_directory/'ridge.json')
    if common['environment'] != environment():
        raise ValueError('Current environment differs from the neural experiment')
    check_ridge_environment(ridge, common['environment'])
    if ridge['status']!='complete' or ridge['fits']!=3 or ridge['identity']['train']!=common['train_identity'] or ridge['identity']['validation']!=validation_identity:
        raise ValueError('Ridge has not completed its train/validation-only three-fit plan')
    if ridge['identity']['source']['sha256'] != common['source_sha256']:
        raise ValueError('Ridge and neural code identities differ')
    ridge_best = ridge['best']
    ridge_path = ridge_directory/ridge_best['path']; verify_record(ridge_path,ridge_best['file'])
    sensitivity = []
    n_test = data_manifest['splits']['test']['n']
    for seed,pa,pb in zip((17,42,101),paired_paths['A'],paired_paths['B']):
        a,_ = load_predictions(pa); b,_ = load_predictions(pb)
        if not np.array_equal(a['ids'],b['ids']):
            raise ValueError('Unpaired validation IDs')
        ea = abs((a['prediction']-a['z_true'])/(1+a['z_true'])) > .15
        eb = abs((b['prediction']-b['z_true'])/(1+b['z_true'])) > .15
        p = float((ea!=eb).mean())
        sensitivity.append({'seed':seed,'validation_discordance':p,
            'assumed_test_n':n_test,'approx_null_95_half_width':1.96*math.sqrt(p/n_test),
            'approx_80pct_power_detectable_delta':(1.96+.841621)*math.sqrt(p/n_test)})
    frozen = {'schema':'batch4-final-v1','data_root':str(Path(data_root).resolve()),'data_manifest':file_record(Path(data_root)/'manifest.json'),
              'source':source_identity(),'neural':entries,
              'ridge':{'path':str(ridge_path.resolve()),'file':file_record(ridge_path),'alpha':ridge_best['alpha'],
                       'ridge_manifest':str((ridge_directory/'ridge.json').resolve()),'manifest_file':file_record(ridge_directory/'ridge.json')},
              'sensitivity':{'estimates':sensitivity,'assumptions':'Validation discordance transports to test; independent observed units; normal approximation. Selected validation is optimistic. Unknown physical dependencies can widen intervals; this is not a power guarantee.'},
              'criterion':'all three eta_A-eta_B<0, paired mean interval95 upper<0, mean MAE_norm_A<=B',
              'paired_bootstrap':{'replicates':5000,'seed':20261009,'unit':'observed group, jointly over models and seeds'},
              'scope':'test remains closed until separate explicit evaluation command'}
    write_json(output,frozen)
    return frozen


def evaluate_final(frozen_path, output, authorization, *, resume=False, device='mps'):
    if authorization != TEST_PHRASE:
        raise PermissionError('Explicit final-test opening phrase required')
    frozen = read_json(frozen_path)
    if frozen.get('schema')!='batch4-final-v1' or frozen['source']['sha256']!=source_identity()['sha256']:
        raise ValueError('Incompatible frozen protocol/code')
    expected = {(v,s) for v in ('A','B','CNN') for s in (17,42,101)}
    if {(r['variant'],r['seed']) for r in frozen['neural']} != expected or len(frozen['neural'])!=9:
        raise ValueError('Final evaluation requires all nine neural candidates')
    root, output = Path(frozen['data_root']), Path(output).resolve()
    verify_record(root/'manifest.json',frozen['data_manifest'])
    common = None
    current_environment = environment()
    # Complete this preflight for ALL candidates, including already-persisted
    # predictions on resume, before registering, creating output or opening test.
    for row in frozen['neural']:
        run_identity = row['run_identity']
        shared = common_identity(run_identity)
        if common is not None and shared != common:
            raise ValueError('Frozen candidates differ in recipe/data/code/device/environment')
        common = shared
        if (run_identity['mode'] != 'training' or run_identity['variant'] != row['variant']
                or run_identity['seed'] != row['seed']
                or shared['source_sha256'] != frozen['source']['sha256']):
            raise ValueError('Incompatible frozen candidate identity')
        if shared['device'] != device:
            raise ValueError('Final inference device differs from frozen training/validation device')
        if shared['environment'] != current_environment:
            raise ValueError('Current environment differs from frozen training/validation environment')
        verify_record(row['checkpoint'],row['file'])
        verify_record(row['validation_predictions'],row['validation_file'])
        state = torch.load(row['checkpoint'], map_location='cpu', weights_only=True)
        if state['identity'] != run_identity:
            raise ValueError('Frozen checkpoint identity mismatch')
        del state
    verify_record(frozen['ridge']['path'],frozen['ridge']['file'])
    verify_record(frozen['ridge']['ridge_manifest'],frozen['ridge']['manifest_file'])
    ridge = read_json(frozen['ridge']['ridge_manifest'])
    check_ridge_environment(ridge, current_environment)
    if (ridge['identity']['source']['sha256'] != common['source_sha256']
            or ridge['identity']['train'] != common['train_identity']
            or ridge['identity']['validation'] != common['validation_identity']):
        raise ValueError('Ridge and frozen neural experiment identities differ')
    if device == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('Frozen MPS device is unavailable')
    identity = {'freeze':file_record(frozen_path),'output':str(output),'device':device}
    # One durable opening registration shared by every destination for this data.
    with run_lock(root):
        registration = root/'test-opening.json'
        if registration.exists():
            if not resume or read_json(registration)!=identity:
                raise PermissionError('Test already opened; only resume the same frozen evaluation')
        else:
            if resume or output.exists():
                raise ValueError('New final evaluation needs a new destination')
            output.mkdir(parents=True)
            write_json(registration,identity)
    with run_lock(output):
        data = ExperimentData(root,'test',test_authorization=authorization)
        paths = {'A':[],'B':[]}; summaries = {}
        write_json(output/'final.json',{'status':'incomplete','identity':identity})
        for row in frozen['neural']:
            key = f"{row['variant']}-{row['seed']}"; path = output/(key+'.npz')
            if not (path.exists() and path.with_suffix('.json').exists()):
                state = torch.load(row['checkpoint'],map_location='cpu',weights_only=True)
                if state['identity']!=row['run_identity']:
                    raise ValueError('Frozen checkpoint identity mismatch')
                if state['identity']['device'] != device:
                    raise ValueError('Final inference device differs from frozen training/validation device')
                torch.set_num_threads(state['identity']['config']['cpu_threads'])
                torch.use_deterministic_algorithms(device=='cpu')
                model = make_model(state['identity']['config'],row['variant'],row['seed'],state['z_bin_weights'])
                model.load_state_dict(state['model'],strict=True); model.to(device)
                start = time.perf_counter()
                outputs = predict(model,data,np.arange(len(data)),state['identity']['config']['micro_batch'],device)
                save_predictions(path,data,np.arange(len(data)),outputs,
                    {'seed':row['seed'],'variant':row['variant'],'step':row['step'],'frozen':digest(frozen)})
                write_json(output/(key+'.cost.json'),{'seconds':time.perf_counter()-start})
                del model
            arrays, provenance = load_predictions(path)
            if provenance['data_identity']!=data.identity or provenance['identity'].get('frozen')!=digest(frozen):
                raise ValueError('Incompatible persisted final predictions')
            summaries[key] = eta_intervals(arrays)
            if row['variant'] in paths:
                paths[row['variant']].append(path)
        path = output/'ridge.npz'
        if not (path.exists() and path.with_suffix('.json').exists()):
            with np.load(frozen['ridge']['path'],allow_pickle=False) as saved:
                model = {k:saved[k] for k in saved.files}
            outputs = ridge_predict(model,feature_matrix(data))
            save_predictions(path,data,np.arange(len(data)),{'prediction':outputs},{'variant':'ridge','frozen':digest(frozen)})
        arrays,provenance = load_predictions(path)
        if provenance['identity'].get('frozen')!=digest(frozen) or provenance['data_identity']!=data.identity:
            raise ValueError('Incompatible ridge final predictions')
        summaries['ridge'] = eta_intervals(arrays)
        result = {'status':'complete','identity':identity,'intervals':summaries,
                  'paired':paired_analysis(paths['A'],paths['B']),
                  'files':{p.name:file_record(p) for p in sorted(output.glob('*.npz'))}}
        write_json(output/'final.json',result)
        return result
