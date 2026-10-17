"""5A behavioral intervention tests; adapted from inspected Claude probes.

All data/checkpoints are tiny synthetic fixtures. No historical run is opened.
"""
from copy import deepcopy
import math
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from desi_fm.experiment.data import ExperimentData
from desi_fm.experiment.training import Trainer
from desi_fm.experiment.metrics import save_predictions, load_predictions, paired_analysis
from desi_fm.experiment.final import eta_intervals
from test_experiment_training import experiment_root, small_config


@pytest.mark.parametrize('seed',[17,42,101])
def test_ab_losses_gradients_and_consumed_inputs(experiment_root,small_config,tmp_path,monkeypatch,seed):
    config=deepcopy(small_config);config['model']['dropout']=0.
    data=ExperimentData(experiment_root,'train')
    a=Trainer(data,config,'A',seed,tmp_path/'A',profile=True)
    b=Trainer(data,config,'B',seed,tmp_path/'B',profile=True)
    assert all(torch.equal(v,b.model.state_dict()[k]) for k,v in a.model.state_dict().items())
    reference=deepcopy(a.model)
    batches=[]; real_batch=data.batch
    def capture_batch(*args,**kwargs):
        batch=real_batch(*args,**kwargs)
        batches.append((np.array(args[0]),np.array(kwargs['draw_positions']),deepcopy(batch)))
        return batch
    monkeypatch.setattr(data,'batch',capture_batch)
    gradients=[]; losses=[]; real_backward=torch.Tensor.backward
    def backward(loss,*args,**kwargs):
        losses.append(float(loss.detach()));return real_backward(loss,*args,**kwargs)
    monkeypatch.setattr(torch.Tensor,'backward',backward)
    for trainer in (a,b):
        def capture(parameters,*args,**kwargs):
            gradients.append({k:torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for k,p in trainer.model.named_parameters()})
            return torch.tensor(0.)  # inspect the un-clipped gradient, not optimizer behavior
        monkeypatch.setattr(torch.nn.utils,'clip_grad_norm_',capture)
        trainer.update()
    for x,y in zip(batches[0][:2],batches[1][:2]):np.testing.assert_array_equal(x,y)
    ba,bb=batches[0][2],batches[1][2]
    assert ba['ids']==bb['ids']
    for k in ['mask','flux','valid','z','centers','scales']:np.testing.assert_array_equal(ba[k],bb[k])
    # Independent full-batch reference: score reconstructed pixels with MSE,
    # not the trainer's reconstruction_sse/denominator expression.
    flux=torch.from_numpy(ba['flux']);valid=torch.from_numpy(ba['valid'])
    masks=torch.from_numpy(ba['mask']);out=reference(flux,valid,spectrum_mask=masks)
    hidden=masks.repeat_interleave(32,dim=1)&valid.bool()
    recon=F.mse_loss(out['patch_pred'].flatten(1)[hidden],flux[hidden])
    labels=torch.bucketize(torch.log1p(torch.from_numpy(ba['z'])),reference.z_bin_edges[1:-1])
    ce=F.cross_entropy(out['z_logits'],labels,weight=reference.z_bin_weights)/math.log(100)
    params=dict(reference.named_parameters())
    recon_grads=torch.autograd.grad(recon,tuple(params.values()),allow_unused=True,retain_graph=True)
    expected={k:torch.zeros_like(params[k]) if g is None else g for k,g in zip(params,recon_grads)}
    n=config['accumulation']
    assert sum(losses[:n])==pytest.approx(float((ce+recon).detach()),abs=1e-6)
    assert sum(losses[n:])==pytest.approx(float(ce.detach()),abs=1e-6)
    ga,gb=gradients
    assert any(g.abs().sum()>0 for k,g in expected.items() if k.startswith('encoder.'))
    for k,g in expected.items():
        torch.testing.assert_close(ga[k]-gb[k],g,atol=2e-6,rtol=2e-4)
    heads=[k for k in ga if k.startswith('reconstruction_head.')]
    assert heads and sum(ga[k].abs().sum() for k in heads)>0
    assert all(torch.count_nonzero(gb[k])==0 for k in heads)
    assert any(gb[k].abs().sum()>0 for k in gb if k.startswith('redshift_head.'))


def test_hidden_pixels_cannot_choose_batch_statistics(experiment_root):
    data=ExperimentData(experiment_root,'train');indices=np.arange(8);positions=np.arange(8)
    base=data.batch(indices,seed=17,draw_positions=positions,n_tokens=8,masked=True)
    hidden=np.repeat(base['mask'],32,axis=1)
    changed=np.array(data.flux);changed[indices]=np.where(hidden,1e6,changed[indices]);data.flux=changed
    after=data.batch(indices,seed=17,draw_positions=positions,n_tokens=8,masked=True)
    assert hidden.any() and (~hidden).sum(axis=1).min()>=8
    for key in ['mask','centers','scales']:np.testing.assert_array_equal(base[key],after[key])
    np.testing.assert_array_equal(base['flux'][~hidden],after['flux'][~hidden])
    assert not np.array_equal(base['flux'][hidden],after['flux'][hidden])


@pytest.mark.parametrize('error', [.125,.15,.16])
def test_exact_eta_boundary_in_paired_and_interval(experiment_root,tmp_path,error):
    data=ExperimentData(experiment_root,'train');data.z=np.zeros(len(data))
    paths={'A':[],'B':[]}
    # Constant outcomes make the expected interval exact, independent of RNG.
    for seed in (17,42,101):
        for v in paths:
            path=tmp_path/f'{v}-{seed}.npz'
            save_predictions(path,data,np.arange(len(data)),{'prediction':np.full(len(data),error if v=='A' else 0.)},{'variant':v,'seed':seed})
            paths[v].append(path)
    expected=1. if error>.15 else 0.
    paired=paired_analysis(paths['A'],paths['B'],replicates=40)
    assert paired['delta_eta_per_seed']==[expected]*3
    assert paired['paired_95_interval']==[expected]*2
    arrays,_=load_predictions(paths['A'][0]);ci=eta_intervals(arrays,replicates=40)
    assert ci['metrics']['all']['interval95']==[expected]*2
