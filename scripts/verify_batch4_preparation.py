"""Read-only integrity/accounting acceptance; metadata only for the sealed test."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import numpy as np
import torch

from desi_fm.assets import sha256
from desi_fm.local_data import LocalDESIStore
from desi_fm.experiment.artifacts import file_record, read_json, source_identity, verify_record, write_json
from desi_fm.experiment.splits import make_splits


def verify(args):
    root,index,data=Path(args.raw),Path(args.index),Path(args.data)
    output=Path(args.output)
    if output.exists():raise ValueError('Use a new acceptance report')
    store=LocalDESIStore(root,index)
    manifest=read_json(data/'manifest.json')
    verify_record(root/'snapshot.json',manifest['snapshot']);verify_record(index,manifest['index'])
    for name,record in manifest['files'].items():verify_record(data/name,record)
    decision=read_json(data/'decision-before-split.json')
    with sqlite3.connect(f'file:{index}?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        rows=[dict(r) for r in db.execute('SELECT * FROM objects WHERE label_eligible AND input_eligible ORDER BY ordinal')]
    regenerated,report=make_splits(rows,decision['protected_ids'])
    if report!=read_json(data/'splits.json'):raise AssertionError('Nonreproducible split manifest')
    memberships={}
    for role,entry in manifest['splits'].items():
        for name,record in entry['files'].items():verify_record(data/role/name,record)
        ids=read_json(data/role/'ids.json')
        if ids!=[r['object_id'] for r in regenerated[role]]:raise AssertionError('Split IDs changed')
        memberships[role]=set(ids)
        # Metadata inspection for partition integrity is permitted. No test
        # arrays are preprocessed, predictions made or learned quantities fitted.
        with np.load(data/role/'metadata.npz',allow_pickle=False) as saved:
            for key,source in [('z','z'),('zwarn','zwarn'),('groups','group_id')]:
                np.testing.assert_array_equal(saved[key],np.array([r[source] for r in regenerated[role]]))
    if set(decision['protected_ids'])&memberships['test']:raise AssertionError('Historical leakage into test')
    exclusions=read_json(data/'exclusions.json')
    if len(exclusions)+sum(len(v) for v in memberships.values())!=100000:raise AssertionError('Unaccounted snapshot rows')
    if (data/'test-opening.json').exists():raise AssertionError('Real test was opened')
    assets={}
    for name,repo in [('fm',Path(__file__).resolve().parents[1]),('copilot',Path(args.copilot))]:
        original=read_json(repo/'docs/evidence/historical-assets.json')
        for row in original['files']:
            verify_record(repo/row['path'],{k:row[k] for k in ('bytes','sha256')})
        assets[name]={'verified':len(original['files']),'manifest':file_record(repo/'docs/evidence/historical-assets.json'),
                      'branch':subprocess.check_output(['git','branch','--show-current'],cwd=repo,text=True).strip(),
                      'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()}
    state=Path(args.profiles)
    a=read_json(state/'profile-A/profile.json');b=read_json(state/'profile-B/profile.json');cnn=read_json(state/'profile-CNN-fixed/profile.json')
    assert a['trace']==b['trace'] and a['class_weights']==b['class_weights']
    initial=[]
    for role in ('A','B'):
        directory=state/('profile-'+role);entry=read_json(directory/'checkpoints.json')['previous']
        saved=torch.load(directory/entry['path'],map_location='cpu',weights_only=True)
        assert saved['step']==0
        initial.append(saved)
    assert all(torch.equal(v,initial[1]['model'][k]) for k,v in initial[0]['model'].items())
    assert torch.equal(initial[0]['z_bin_weights'],initial[1]['z_bin_weights'])
    from desi_fm.experiment.models import make_model
    initialization = {}
    for seed in (17,42,101):
        fingerprints=[]
        for variant in ('A','B'):
            model=make_model(a['identity']['config'],variant,seed,torch.tensor(a['class_weights']['weights']))
            h=hashlib.sha256()
            for key,value in sorted(model.state_dict().items()):
                h.update(key.encode());h.update(value.detach().numpy().tobytes())
            fingerprints.append(h.hexdigest())
            del model
        assert fingerprints[0]==fingerprints[1]
        initialization[str(seed)]=fingerprints[0]
    profile_counts={}
    for label,folder in [('A','profile-A'),('B','profile-B'),('CNN','profile-CNN-fixed')]:
        r=read_json(state/folder/'profile.json')
        with np.load(state/folder/'profile-train.npz',allow_pickle=False) as predictions:
            assert set(predictions['ids'])<=memberships['train']
        assert r['identity']['mode']=='profile' and r['step']<=200 and not r['validation_events']
        profile_counts[label]={'updates':r['step'],'evaluation_role':'train','evaluation_n':r['evaluation_profile']['objects'],
                               'parameters':r['parameters'],'manifest':file_record(state/folder/'profile.json')}
    result={'status':'passed','source':source_identity(),'raw_bytes_verified':sum(r['bytes'] for r in store.manifest['files']),
            'raw_files_without_write_bits':all(not (root/f['path']).stat().st_mode&0o222 for f in store.manifest['files']),
            'index':file_record(index),'derivative_manifest':file_record(data/'manifest.json'),
            'all_derivative_file_hashes_verified':True,'regenerated_splits_identical':True,
            'counts':{k:len(v) for k,v in memberships.items()},'exclusions':len(exclusions),
            'protected_n':len(decision['protected_ids']),'protected_development':{k:len(v&set(decision['protected_ids'])) for k,v in memberships.items()},
            'metadata_only_test_integrity_check':True,'real_test_opening_absent':True,
            'profiles':profile_counts,'real_ab_initial_tensors_exact':True,'ab_trace_and_class_weights_exact':True, 'full_architecture_paired_initialization_sha256':initialization,
            'historical_assets':assets,'hardware':subprocess.check_output(['sysctl','-n','hw.model','hw.memsize','hw.physicalcpu','hw.logicalcpu','machdep.cpu.brand_string'],text=True).splitlines(),
            'untracked_historical_csv':file_record(Path(__file__).resolve().parents[1]/'runs/desi_50k_big/predictions_heldout.csv')}
    write_json(output,result);print({k:v for k,v in result.items() if k not in ('source','profiles','historical_assets')})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('raw','index','data','copilot','profiles','output'):p.add_argument('--'+key,required=True)
    verify(p.parse_args())
