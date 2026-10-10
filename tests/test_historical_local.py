from test_local_data import local_snapshot

import numpy as np
import pytest
from desi_fm.data import preprocess_spectrum,SpectrumPreprocessConfig
from desi_fm.historical import metrics,fixed_masks,visible_baselines,masked_rmse


def test_metric_boundary_and_denominator():
    value=metrics(np.array([0.,0.]),np.array([.15,.16]))
    assert value['eta15']==.5 and value['strict_accuracy']==0 and value['n']==2
    assert metrics([],[])['eta15'] is None
    assert masked_rmse([1.,2.],[2.,4.],[False,False])=={'masked_valid_pixels':0,'sse':0.,'rmse':None}
    assert masked_rmse([1.,2.],[2.,4.],[False,True])['rmse']==2


def test_masks_follow_id_not_batch_order():
    ids=['9007199254740993','39627776344457978']
    original=fixed_masks(ids,273)
    np.testing.assert_array_equal(original,fixed_masks(ids,273))
    np.testing.assert_array_equal(original[::-1],fixed_masks(ids[::-1],273))
    assert not np.array_equal(original,fixed_masks(ids,273,seed=1))


def test_hidden_values_never_enter_baselines_or_normalization():
    wave=np.geomspace(3600,9800,64).astype(np.float32);flux=np.sin(wave/500).astype(np.float32)
    valid=np.ones(64);hidden=np.zeros(64,dtype=bool);hidden[16:40]=True
    changed=flux.copy();changed[hidden]=100000
    a=visible_baselines(flux,valid,hidden);b=visible_baselines(changed,valid,hidden)
    for name in a:np.testing.assert_array_equal(a[name],b[name])
    config=SpectrumPreprocessConfig(n_pixels=64)
    p=preprocess_spectrum(flux,valid,wave,np.zeros(64,bool),config,normalization_mask=hidden)
    q=preprocess_spectrum(changed,valid,wave,np.zeros(64,bool),config,normalization_mask=hidden)
    assert p['center']==q['center'] and p['scale']==q['scale']
    np.testing.assert_array_equal(p['flux'][~hidden],q['flux'][~hidden])
    with pytest.raises(ValueError):preprocess_spectrum(flux,valid,wave,np.zeros(64,bool),config,normalization_mask=np.ones(64,bool))


def test_small_end_to_end_persisted_metrics_exclusions_and_no_overwrite(local_snapshot,tmp_path,monkeypatch):
    import importlib.util,json,socket
    from argparse import Namespace
    from pathlib import Path
    import torch
    from desi_fm.assets import sha256
    from desi_fm.model import DESIFoundationModel,DESIFoundationModelConfig
    root,index,config,rows,_=local_snapshot
    lock={}
    for name,bins in [('v1',0),('v21',8)]:
        cfg=DESIFoundationModelConfig(n_pixels=64,n_tokens=8,d_model=16,n_layers=1,n_heads=2,n_z_bins=bins)
        model=DESIFoundationModel(cfg);path=tmp_path/(name+'.pt')
        torch.save({'model':model.state_dict(),'config':cfg.to_dict(),'step':1},path)
        lock[name]={'path':str(path),'bytes':path.stat().st_size,'sha256':sha256(path),'selection':'fixture'}
    checkpoints=tmp_path/'checkpoints.json';checkpoints.write_text(json.dumps(lock))
    selection=tmp_path/'ids.json';selection.write_text(json.dumps({'ids':[x['object_id'] for x in rows], 'requested':5,
                      'snapshot_sha256':sha256(root/'snapshot.json'),'index_sha256':sha256(index)}))
    script=Path(__file__).resolve().parents[1]/'scripts/evaluate_historical_local.py'
    spec=importlib.util.spec_from_file_location('local_evaluation',script);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for attr in ('connect','connect_ex'):monkeypatch.setattr(socket.socket,attr,getattr(socket.socket,attr))
    monkeypatch.setattr(socket,'create_connection',socket.create_connection)
    args=Namespace(root=str(root),index=str(index),ids=str(selection),output=str(tmp_path/'run'),checkpoints=str(checkpoints),device='cpu',batch_size=2,limit=None,offline=True)
    module.run(args)
    output=Path(args.output);manifest=json.loads((output/'manifest.json').read_text());summary=json.loads((output/'summary.json').read_text())
    assert manifest['status']=='complete' and manifest['blocked_network_attempts']==0
    assert summary['accounting']=={'requested':5,'redshift_evaluated':3,'redshift_rejected':2,'reconstruction_evaluated':3,'reconstruction_rejected':2}
    before=(output/'predictions.csv').read_bytes()
    with pytest.raises(FileExistsError):module.run(args)
    assert (output/'predictions.csv').read_bytes()==before
