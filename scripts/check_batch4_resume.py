"""Small synthetic CPU/MPS interruption experiment; never opens real test data."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch

from desi_fm.data import SpectrumPreprocessConfig
from desi_fm.experiment.artifacts import file_record, write_json
from desi_fm.experiment.data import ExperimentData, GRID_CONTRACT
from desi_fm.experiment.models import default_config
from desi_fm.experiment.training import Trainer


def check(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    root=output/'synthetic';(root/'train').mkdir(parents=True)
    cfg=SpectrumPreprocessConfig(n_pixels=256)
    rng=np.random.default_rng(991)
    d=root/'train'
    write_json(d/'ids.json',[str(i) for i in range(25)])
    np.save(d/'flux.npy',rng.normal(size=(25,256)).astype('float32'))
    np.save(d/'valid.npy',np.ones((25,256),dtype='uint8'))
    np.savez(d/'metadata.npz',z=(np.arange(25)//3)/20,zwarn=np.arange(25)%2,groups=np.array([str(i) for i in range(25)]))
    write_json(root/'manifest.json',{'status':'complete','contract':GRID_CONTRACT,'preprocess':asdict(cfg),
                                   'splits':{'train':{'n':25,'files':{p.name:file_record(p) for p in d.iterdir()}}}})
    config=default_config();config['model'].update(n_pixels=256,n_tokens=8,d_model=16,n_layers=1,n_heads=2,mlp_ratio=2.)
    config.update(micro_batch=2,accumulation=2,updates=4,warmup_updates=1,checkpoint_every=2,validation_every=2,cpu_threads=1)
    data=ExperimentData(root,'train');results={}
    for device in ['cpu']+(['mps'] if torch.backends.mps.is_available() else []):
        whole=Trainer(data,config,'A',17,output/(device+'-whole'),device,profile=True)
        whole.run(until=4,install_signals=False)
        saved={k:v.detach().cpu().clone() for k,v in whole.model.state_dict().items()}
        trace=whole.trace;del whole
        first=Trainer(data,config,'A',17,output/(device+'-resumed'),device,profile=True)
        first.run(until=2,install_signals=False);del first
        resumed=Trainer(data,config,'A',17,output/(device+'-resumed'),device,profile=True,resume=True)
        resumed.run(until=4,install_signals=False)
        deltas=[float((saved[k]-v.detach().cpu()).abs().max()) for k,v in resumed.model.state_dict().items()]
        exact=all(delta==0 for delta in deltas)
        close=all(torch.allclose(saved[k],v.detach().cpu(),atol=1e-6,rtol=1e-5) for k,v in resumed.model.state_dict().items())
        if device=='cpu' and not exact:raise AssertionError('CPU resume must be bit-exact')
        if not close:raise AssertionError('Resume exceeds declared atol 1e-6 rtol 1e-5')
        if trace!=resumed.trace:raise AssertionError('Sampler/mask mismatch')
        results[device]={'updates_per_trajectory':4,'executed_updates':8,'exact_tensors':exact,'max_abs_tensor_difference':max(deltas),
                         'allclose_atol1e-6_rtol1e-5':close,'trace_identical':True,'class_weights_nonunit':not torch.equal(resumed.model.z_bin_weights.cpu(),torch.ones(100)),
                         'recovery_seconds':resumed.recovery_seconds}
        del resumed
    report={'status':'passed','scope':'small synthetic interruption; no real test or validation inputs','config':config,'devices':results,
            'limit':'MPS agreement only for this small run on this host. No bitwise guarantee for full-size runs, other devices or torch versions.'}
    write_json(output/'resume-check.json',report);print(report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    check(p.parse_args().output)
