"""Verify actual random access against the unchanged batch-3 commit implementation."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import types
import numpy as np

from desi_fm.assets import sha256
from desi_fm.data import extract_mmu_desi_example
from desi_fm.local_data import LocalDESIStore
from desi_fm.experiment.artifacts import file_record, write_json
from desi_fm.experiment.data import ExperimentData, sample_masks
from desi_fm.experiment.ridge import feature_matrix, fit_three


def check(args):
    output=Path(args.output)
    if output.exists():raise ValueError('Use a new report destination')
    store=LocalDESIStore(args.root,args.index)
    data=ExperimentData(args.data,'train')
    baseline='008ce03da7d8c3d74d522b03eedbf3d61b7e861a'
    source=subprocess.check_output(['git','show',baseline+':src/desi_fm/data.py'],text=True)
    module=types.ModuleType('desi_fm._batch3_reference_data')
    sys.modules[module.__name__]=module
    exec(compile(source,baseline+'/data.py','exec'),module.__dict__)
    config=module.SpectrumPreprocessConfig(**asdict(data.config))
    lookup={oid:i for i,oid in enumerate(data.ids)}
    with sqlite3.connect(f'file:{args.index}?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        rows=[dict(r) for r in db.execute('SELECT * FROM objects WHERE label_eligible AND input_eligible ORDER BY object_id') if r['object_id'] in lookup]
    by_file={}
    for row in rows:by_file.setdefault(row['file'],[]).append(row)
    selected=set()
    for group in by_file.values():
        for quality in (0,1):
            sub=[r for r in group if r['zwarn']==quality]
            if sub:
                selected.update([min(sub,key=lambda r:r['valid_fraction'])['object_id'],max(sub,key=lambda r:r['z'])['object_id']])
    rng=np.random.default_rng(20261009)
    selected.update(rng.choice(data.ids,32,replace=False).tolist())
    ids=sorted(selected);rng.shuffle(ids)  # deliberately changes row group/shard on most reads
    indices=np.array([lookup[oid] for oid in ids]);positions=np.arange(len(ids))
    start=time.perf_counter();actual=data.batch(indices,seed=17,draw_positions=positions,masked=True)
    derived_masked_seconds=time.perf_counter()-start
    start=time.perf_counter();full=data.batch(indices,masked=False);derived_full_seconds=time.perf_counter()-start
    read_seconds,pre_seconds=0.,0.
    examples=[]
    for i,oid in enumerate(ids):
        start=time.perf_counter();raw=extract_mmu_desi_example(store.raw(oid));read_seconds+=time.perf_counter()-start
        start=time.perf_counter()
        reference=module.preprocess_spectrum(*raw[:4],config)
        hidden=np.repeat(actual['mask'][i],26)[:7081]
        masked=module.preprocess_spectrum(*raw[:4],config,normalization_mask=hidden)
        pre_seconds+=time.perf_counter()-start
        np.testing.assert_array_equal(full['flux'][i],reference['flux'])
        np.testing.assert_array_equal(actual['flux'][i],masked['flux'])
        np.testing.assert_array_equal(full['valid'][i],reference['valid'])
        assert actual['centers'][i]==masked['center'] and actual['scales'][i]==masked['scale']
        examples.append({'object_id':oid,'file':store.rows_for_ids([oid])[0]['file'],
                         'coverage':reference['coverage']['model_valid_fraction'],'zwarn':int(data.zwarn[indices[i]])})
    random_indices=rng.choice(len(data),2048,replace=False)
    start=time.perf_counter();raw_flux=np.array(data.flux[random_indices]);raw_valid=np.array(data.valid[random_indices]);random_read=time.perf_counter()-start
    start=time.perf_counter()
    for first in range(0,len(random_indices),32):data.batch(random_indices[first:first+32],masked=False)
    random_pre=time.perf_counter()-start
    # Three small train-only ridge fits measure cost, never select by validation.
    ridge_indices=random_indices[:1024]
    start=time.perf_counter();features=feature_matrix(data,ridge_indices);feature_seconds=time.perf_counter()-start
    start=time.perf_counter();models=fit_three(features,data.z[ridge_indices]);fit_seconds=time.perf_counter()-start
    report={'status':'passed','baseline_commit':baseline,'reference_source_sha256':__import__('hashlib').sha256(source.encode()).hexdigest(),
            'derivative_manifest':file_record(Path(args.data)/'manifest.json'),'train_identity':data.identity,
            'representative_n':len(ids),'sample':examples,'exact_full_and_masked_normalization_validity':True,
            'sample_order':'seeded random, across every shard, both ZWARN values, worst coverage and high z plus random IDs; train only',
            'timings_seconds':{'batch3_random_read':read_seconds,'batch3_full_and_masked_preprocess':pre_seconds,
                               'derived_representative_full':derived_full_seconds,'derived_representative_masked':derived_masked_seconds,
                               'random_2048_memmap_only':random_read,'random_2048_read_and_normalize':random_pre},
            'random_note':'Real nonconsecutive gathers in a warm OS cache; not a cold-disk or concurrent-load guarantee',
            'ridge_profile':{'role':'train','n':len(ridge_indices),'features_seconds':feature_seconds,'three_fits_seconds':fit_seconds,
                             'fit_count':len(models),'scope':'bounded cost only; no validation selection, models discarded'},
            'test_predictions_or_metrics':False}
    write_json(output,report)
    print({k:v for k,v in report.items() if k!='sample'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','index','data','output'):p.add_argument('--'+key,required=True)
    check(p.parse_args())
