"""Read-only CNN-17 engineering check in the exact historical checkout.

No test, training, new model selection, v3 adaptation, FAISS or serving changes.
Run this harness in a fresh process; its historical imports are verified first.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HISTORICAL_HEAD = '4387dd36c6ad640f00c6d9fc3bf6448ef3644d50'
CHECKPOINT = {'bytes':11874874,'sha256':'335b396c3e2ee737a5e62d51851eedb596b42fb815f9a3f4861e42a2cdc9d6f8'}


def record(path):
    p=Path(path);h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return {'bytes':p.stat().st_size,'sha256':h.hexdigest()}


def deny_test_open(data_root):
    """Process-wide open audit: even a direct NumPy read of test is forbidden."""
    test=os.path.realpath(str(Path(data_root)/'test'))
    def hook(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            target=os.path.realpath(os.fsdecode(args[0]))
            if target==test or target.startswith(test+os.sep):
                raise PermissionError('5B-2 test access is forbidden')
    sys.addaudithook(hook)


def require_historical_source(root, expected):
    root=Path(root).resolve()
    head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    if head!=HISTORICAL_HEAD:
        raise ValueError('Exact historical HEAD required; no bypass')
    if len(expected)!=25 or any(record(root/p)['sha256']!=h for p,h in expected.items()):
        raise ValueError('Historical 25-source identity mismatch')
    return head


def validate_batch(batch, n_pixels):
    import numpy as np
    f,v=batch['flux'],batch['valid']
    if f.ndim!=2 or f.shape!=v.shape or f.shape[1]!=n_pixels or not np.isfinite(f).all() or not np.isfinite(v).all():
        raise ValueError('Invalid normalized input shape or values')
    if not np.isin(v,[0,1]).all() or (v.sum(1)<8).any():
        raise ValueError('Invalid model-grid support')


class DevelopmentScope:
    def __init__(self, data, allowed):
        if data.role not in ('train','validation'):
            raise PermissionError('Only development train/validation roles allowed')
        self.data=data
        mapping={oid:i for i,oid in enumerate(data.ids)}
        if not allowed or len(allowed)!=len(set(allowed)) or any(x not in mapping for x in allowed):
            raise ValueError('Development membership mismatch')
        self.allowed={mapping[x] for x in allowed}
        self.indices={x:mapping[x] for x in allowed}
    def batch(self, indices, **kwargs):
        if any(int(i) not in self.allowed for i in indices) or kwargs.get('masked',False):
            raise PermissionError('Only fixed development IDs without artificial masks')
        batch=self.data.batch(indices,**kwargs)
        validate_batch(batch,self.data.config.n_pixels)
        return batch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('historical-root','frozen','development','input-dir','backup-status','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    harness_root=Path(__file__).resolve().parents[1]
    for filename in ('scripts/check_batch5b_cnn.py','scripts/batch5b_development.py'):
        committed=subprocess.check_output(['git','-C',str(harness_root),'show','HEAD:'+filename])
        if (harness_root/filename).read_bytes()!=committed:
            raise ValueError('Commit the exact harness before operational measurement')
    from batch5b_development import read_development, render_recipe
    cases=read_development(args.development)
    frozen=json.loads(args.frozen.read_text())
    require_historical_source(args.historical_root,frozen['source']['files'])
    if any(k=='desi_fm' or k.startswith('desi_fm.') for k in sys.modules):
        raise ValueError('Historical imports require a fresh process')
    sys.path.insert(0,str(args.historical_root/'src'))
    import numpy as np
    import torch
    from desi_fm.experiment import artifacts, models, training
    from desi_fm.experiment.data import ExperimentData, GRID_CONTRACT
    from desi_fm.data import preprocess_spectrum
    from desi_fm.inputs import SpectrumInputError
    cnn=next(x for x in frozen['neural'] if x['variant']=='CNN' and x['seed']==17)
    identity=cnn['run_identity']
    if cnn['step']!=6000 or cnn['file']!=CHECKPOINT or identity['source_sha256']!=frozen['source']['sha256']:
        raise ValueError('Frozen CNN-17 identity mismatch')
    if artifacts.source_identity()['files']!=frozen['source']['files'] or artifacts.environment()!=identity['environment']:
        raise ValueError('Historical code/environment mismatch; stop without relaxing')
    if identity['device']!='mps' or not torch.backends.mps.is_available():
        raise ValueError('Historical MPS device unavailable')
    imported={n:m.__file__ for n,m in sys.modules.items() if n=='desi_fm' or n.startswith('desi_fm.')}
    if any(not Path(path).resolve().is_relative_to(args.historical_root.resolve()) for path in imported.values()):
        raise ValueError('Mixed historical imports')
    backup=json.loads(args.backup_status.read_text())
    if not backup.get('status'):
        raise ValueError('Backup status must be recorded before checkpoint access')
    for immutable in (args.frozen.parent,Path(frozen['data_root']),args.historical_root):
        if args.output.resolve().is_relative_to(immutable.resolve()):
            raise ValueError('Output must be outside historical and data trees')
    args.output.mkdir(mode=0o700,parents=True,exist_ok=False)
    artifacts.write_json(args.output/'backup-status-before-checkpoint.json',backup)
    data_root=Path(frozen['data_root']);deny_test_open(data_root)
    artifacts.verify_record(data_root/'manifest.json',frozen['data_manifest'])
    artifacts.verify_record(cnn['checkpoint'],CHECKPOINT)
    artifacts.verify_record(cnn['validation_predictions'],cnn['validation_file'])
    torch.set_num_threads(identity['config']['cpu_threads'])
    start=time.perf_counter()
    state=torch.load(cnn['checkpoint'],map_location='cpu',weights_only=True)
    if state['identity']!=identity or state['step']!=6000:
        raise ValueError('Checkpoint payload identity mismatch')
    model=models.make_model(identity['config'],'CNN',17,state['z_bin_weights'])
    model.load_state_dict(state['model'],strict=True);model.eval();model.to('mps');torch.mps.synchronize()
    load_seconds=time.perf_counter()-start
    scopes={}
    for role in ('train','validation'):
        data=ExperimentData(data_root,role)
        if data.identity!=identity[role+'_identity'] or data.manifest['contract']!=GRID_CONTRACT:
            raise ValueError('Derivative identity mismatch')
        allowed=[c['object_id'] for c in cases if c['kind']=='real' and c['cnn_exposure']==role]
        if len(allowed)!=24:raise ValueError('Frozen 24-per-role membership required')
        scopes[role]=DevelopmentScope(data,allowed)
    # Only the 24 development rows are selected; no labels are inspected/scored.
    with np.load(cnn['validation_predictions'],allow_pickle=False) as saved:
        mapping={str(x):i for i,x in enumerate(saved['ids'])}
        parity={c['object_id']:{k:float(saved[k][mapping[c['object_id']]]) for k in ('prediction','secondary','confidence')}
                for c in cases if c['kind']=='real' and c['cnn_exposure']=='validation'}
    validation_meta=artifacts.read_json(Path(cnn['validation_predictions']).with_suffix('.json'))
    if (validation_meta['role']!='validation' or validation_meta['artificial_mask'] or
            validation_meta['data_identity']!=identity['validation_identity'] or
            validation_meta['identity']!={'seed':17,'variant':'CNN','step':6000,'run_identity':artifacts.digest(identity)}):
        raise ValueError('Persisted validation provenance mismatch')
    rows=[{'case_id':c['case_id'],'kind':c['kind'],'role':c['cnn_exposure'],
           'family':c.get('family','real'),'status':'pending','repetitions':[]} for c in cases]
    artifacts.write_json(args.output/'rows.json',rows)
    for pos,case in enumerate(cases):
        row=rows[pos]
        try:
            if case['kind']=='real':
                scope=scopes[case['cnn_exposure']];idx=scope.indices[case['object_id']]
                b=scope.batch([idx],n_tokens=model.config.n_tokens,masked=False)
                with np.load(args.input_dir/f'input_{pos:04d}.npz',allow_pickle=False) as raw:
                    processed=preprocess_spectrum(raw['flux'],raw['ivar'],raw['wavelength'],raw['mask'],scope.data.config)
                if not np.array_equal(processed['flux'],b['flux'][0]) or not np.array_equal(processed['valid'],b['valid'][0]):
                    raise ValueError('Raw historical preprocessing differs from frozen derivative')
                coverage=processed['coverage'];row['preprocessing_exact']=True
                def predict_once():
                    return {k:v[0] for k,v in training.predict(model,scope,np.array([idx]),1,'mps').items()}
            else:
                arrays,_=render_recipe(case)
                processed=preprocess_spectrum(arrays['flux'],arrays['ivar'],arrays['wavelength'],arrays['mask'],scopes['train'].data.config)
                if case['family'] in ('reversed_grid','permuted_grid'):
                    order=np.argsort(arrays['wavelength'])
                    a={k:v[order] for k,v in arrays.items()}
                    ordered=preprocess_spectrum(a['flux'],a['ivar'],a['wavelength'],a['mask'],scopes['train'].data.config)
                    row['ordering_equivalent']=all(np.array_equal(processed[k],ordered[k]) for k in ('flux','valid'))
                b={'flux':processed['flux'][None,:],'valid':processed['valid'][None,:]}
                validate_batch(b,model.config.n_pixels);coverage=processed['coverage']
                def predict_once():
                    with torch.no_grad():
                        out=model(torch.from_numpy(b['flux']).to('mps'),torch.from_numpy(b['valid']).to('mps'))
                    return {field:float(out[key].cpu()[0]) for field,key in [('prediction','z_pred_map'),('secondary','z_pred'),('confidence','z_confidence')]}
            row['coverage']=coverage
            for repetition in range(3):
                torch.mps.synchronize();start=time.perf_counter();out=predict_once();torch.mps.synchronize()
                latency=time.perf_counter()-start
                if not all(np.isfinite(v) for v in out.values()) or not 0<=out['prediction']<=6 or not 0<=out['confidence']<=1:
                    raise ValueError('Invalid CNN output')
                row['repetitions'].append(dict(out,seconds=latency,phase='first_pass' if repetition==0 else 'warm'))
            first=row['repetitions'][0]
            row['repeatable']=all(all(np.isclose(x[k],first[k],atol=1e-6,rtol=1e-6) for k in out) for x in row['repetitions'][1:])
            if case['kind']=='real' and case['cnn_exposure']=='validation':
                row['validation_parity']=all(np.isclose(first[k],v,atol=1e-6,rtol=1e-6) for k,v in parity[case['object_id']].items())
                row['parity_max_abs']=max(abs(first[k]-v) for k,v in parity[case['object_id']].items())
            row['status']='processed'
        except SpectrumInputError:
            row['status']='invalid_input'
        except Exception as exc:
            row['status']='technical_error';row['error_type']=type(exc).__name__
        artifacts.write_json(args.output/'rows.json',rows)
    real=[r for r in rows if r['kind']=='real'];synth=[r for r in rows if r['kind']!='real']
    def latency(group,phase):
        values=[x['seconds'] for r in group for x in r['repetitions'] if x['phase']==phase]
        return dict(n=len(values),p50=float(np.quantile(values,.5)),p95=float(np.quantile(values,.95)),maximum=max(values)) if values else {'n':0}
    root=Path(__file__).resolve().parents[1]
    harness_files={str(p.relative_to(root)):record(p) for p in [Path(__file__).resolve(),Path(__file__).with_name('batch5b_development.py')]}
    report={'scope':'Operational development only; validation selected CNN-17. No accuracy comparison, test, deployment, FAISS or reconstruction.',
        'historical_head':HISTORICAL_HEAD,'historical_source':artifacts.source_identity(),'imports':imported,
        'harness_head':subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
        'harness_files':harness_files,'environment':artifacts.environment(),'device':'mps','cpu_threads':torch.get_num_threads(),
        'checkpoint':CHECKPOINT,'strict_load':True,'contract':GRID_CONTRACT,'normalization':identity['config']['normalization'],
        'development':record(args.development),'preprocessing':scopes['train'].data.manifest['preprocess'],
        'backup_status':backup['status'],'model_parameters':sum(p.numel() for p in model.parameters()),
        'checkpoint_and_model_load_seconds':load_seconds,'first_request_seconds':rows[0]['repetitions'][0]['seconds'] if rows[0]['repetitions'] else None,
        'cold_load_plus_first_request_seconds':load_seconds+rows[0]['repetitions'][0]['seconds'] if rows[0]['repetitions'] else None,
        'timing_conditions':'Synchronized MPS, single case requests, historical CPU normalization; per-request includes normalization for real cases. Derivative verification and setup reported outside request latency. First-pass cases separated, only first overall request is truly cold.',
        'latency':{name:{phase:latency(group,phase) for phase in ('first_pass','warm')} for name,group in [('real',real),('synthetic',synth)]},
        'peak_host_rss_bytes':training.memory('mps')['peak_host_rss_bytes'],
        'mps_current_allocated_bytes':torch.mps.current_allocated_memory(),'mps_driver_allocated_bytes':torch.mps.driver_allocated_memory(),
        'status_counts':{status:sum(r['status']==status for r in rows) for status in ('processed','invalid_input','technical_error','pending')},
        'real_requested':len(real),'real_processed':sum(r['status']=='processed' for r in real),
        'synthetic_requested':len(synth),'synthetic_processed':sum(r['status']=='processed' for r in synth),
        'synthetic_invalid_input':sum(r['status']=='invalid_input' for r in synth),
        'repeatable':sum(r.get('repeatable',False) for r in rows),'parity_applicable':len(parity),
        'parity_passed':sum(r.get('validation_parity',False) for r in rows),
        'parity_max_abs':max(r.get('parity_max_abs',0) for r in rows),
        'ordering_equivalent':sum(r.get('ordering_equivalent',False) for r in rows),
        'test_access':'process-wide open audit forbids data_root/test; no test inference',
        'capabilities':['classification MAP redshift','secondary posterior moment','uncalibrated posterior concentration'],
        'limitations':['No reconstruction head','No embedding compatibility with transformer FAISS','No superiority/generalization evidence','v2.1 remains default']}
    report['gate_passes']=report['real_processed']==48 and report['synthetic_processed']==42 and report['synthetic_invalid_input']==6 and report['repeatable']==90 and report['parity_passed']==24 and report['ordering_equivalent']==12
    artifacts.write_json(args.output/'summary.json',report)
    print(json.dumps({k:report[k] for k in ('real_processed','synthetic_processed','synthetic_invalid_input','parity_passed','repeatable','gate_passes')}))
    if not report['gate_passes']:raise SystemExit(1)


if __name__=='__main__':main()
