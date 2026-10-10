"""Compare a complete offline repetition with its complete historical run."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from desi_fm.assets import sha256


def compare(first,second,output):
    first,second,output=map(Path,(first,second,output))
    if output.exists():raise ValueError('Use a new comparison output')
    a=json.loads((first/'manifest.json').read_text());b=json.loads((second/'manifest.json').read_text())
    for manifest,directory in ((a,first),(b,second)):
        assert manifest['status']=='complete'
        assert manifest['scope']=='historical diagnostic, not an independent test'
        for name,row in manifest['artifacts'].items():
            path=directory/name
            assert path.stat().st_size==row['bytes'] and sha256(path)==row['sha256']
    for key in ('requested_ids','selection_sha256','code_sha256','dataset','checkpoint_lock_sha256','preprocess','baseline_parameters','environment','device','batch_size','tolerances','accounting'):
        assert a[key]==b[key],f'Protocol mismatch: {key}'
    assert b['offline_socket_block'] and b['blocked_network_attempts']==0
    deltas={}
    for name in ('inputs.npz','predictions.npz','reconstruction.npz'):
        with np.load(first/name,allow_pickle=False) as left,np.load(second/name,allow_pickle=False) as right:
            assert set(left.files)==set(right.files)
            for key in left.files:
                x,y=left[key],right[key]
                if name=='inputs.npz' or key not in ('v1','v21','v1_scalar','v21_map','v21_secondary','v21_confidence'):
                    np.testing.assert_array_equal(x,y)
                else:
                    np.testing.assert_allclose(x,y,rtol=1e-6,atol=1e-6)
                    deltas[f'{name}:{key}']=float(np.max(abs(x-y))) if x.size else 0.
    assert json.loads((first/'exclusions.json').read_text())==json.loads((second/'exclusions.json').read_text())
    sa=json.loads((first/'summary.json').read_text());sb=json.loads((second/'summary.json').read_text())
    def recursive(x,y,path='summary'):
        if isinstance(x,dict):
            assert x.keys()==y.keys()
            for k in x:recursive(x[k],y[k],path+':'+k)
        elif isinstance(x,float):assert abs(x-y)<=1e-8,f'Metric tolerance exceeded: {path}'
        else:assert x==y,f'Mismatch: {path}'
    recursive(sa,sb)
    for name in ('predictions.csv','reconstruction.csv','risk-coverage.csv','paired-transitions.csv'):
        if not (first/name).exists():continue
        with (first/name).open() as f:rows1=list(csv.DictReader(f))
        with (second/name).open() as f:rows2=list(csv.DictReader(f))
        assert len(rows1)==len(rows2)
        for r1,r2 in zip(rows1,rows2):
            assert r1.keys()==r2.keys()
            for key in r1:
                if key in ('object_id','transition'):assert r1[key]==r2[key]
                else:
                    try:np.testing.assert_allclose(float(r1[key]),float(r2[key]),rtol=1e-6,atol=1e-6)
                    except ValueError:assert r1[key]==r2[key]
    result={'status':'passed','first':str(first),'offline_repeat':str(second),'n':len(a['requested_ids']),
            'manifest_sha256':[sha256(first/'manifest.json'),sha256(second/'manifest.json')],
            'exact_inputs_IDs_masks_order_and_exclusions':True,'prediction_reconstruction_max_abs_difference':deltas,
            'metric_atol':1e-8,'network_connections_blocked':True,'attempted_network_connections':b['blocked_network_attempts']}
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('first','second','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();compare(a.first,a.second,a.output)
