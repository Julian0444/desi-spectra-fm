from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from desi_fm.data import SpectrumPreprocessConfig
from desi_fm.experiment.artifacts import file_record, write_json
from desi_fm.experiment.data import ExperimentData, GRID_CONTRACT
from desi_fm.experiment.models import default_config, class_weights, make_model
from desi_fm.experiment.training import Trainer, load_checkpoint, predict


@pytest.fixture
def experiment_root(tmp_path):
    root = tmp_path/'data'; root.mkdir()
    rng = np.random.default_rng(91)
    cfg = SpectrumPreprocessConfig(n_pixels=256)
    manifest = {'status':'complete','contract':GRID_CONTRACT,'preprocess':asdict(cfg),'splits':{}}
    for role, n in [('train',25),('validation',9),('test',7)]:
        d = root/role; d.mkdir()
        write_json(d/'ids.json',[role+str(i) for i in range(n)])
        np.save(d/'flux.npy',rng.normal(size=(n,256)).astype('float32'))
        np.save(d/'valid.npy',np.ones((n,256),dtype='uint8'))
        np.savez(d/'metadata.npz',z=(np.arange(n)//3)/20,zwarn=np.arange(n)%2,groups=np.array([role+str(i) for i in range(n)]))
        manifest['splits'][role] = {'n':n,'files':{p.name:file_record(p) for p in d.iterdir()}}
    write_json(root/'manifest.json',manifest)
    return root


@pytest.fixture
def small_config():
    config = default_config()
    config['model'].update(n_pixels=256,n_tokens=8,d_model=16,n_layers=1,n_heads=2,mlp_ratio=2.)
    config.update(micro_batch=2,accumulation=2,updates=4,warmup_updates=1,checkpoint_every=2,validation_every=2,cpu_threads=1)
    return config


def test_no_test_access_weights_train_only_and_full_inference(experiment_root,small_config,monkeypatch):
    with pytest.raises(PermissionError,match='closed'):
        ExperimentData(experiment_root,'test')
    original_load = np.load
    def guarded(path,*args,**kwargs):
        assert '/test/' not in str(path)
        return original_load(path,*args,**kwargs)
    monkeypatch.setattr(np,'load',guarded)
    train = ExperimentData(experiment_root,'train')
    val = ExperimentData(experiment_root,'validation')
    weights,_ = class_weights(train,small_config)
    with pytest.raises(PermissionError):
        class_weights(val,small_config)
    model = make_model(small_config,'A',17,weights)
    original_forward = model.forward
    def checked(flux,valid,**kwargs):
        assert not kwargs['spectrum_mask'].any() and kwargs['mask_ratio']==0
        return original_forward(flux,valid,**kwargs)
    monkeypatch.setattr(model,'forward',checked)
    outputs = predict(model,val,np.arange(len(val)),2,'cpu')
    assert len(outputs['prediction']) == len(val)
    with pytest.raises(ValueError,match='training-only'):
        val.batch([0],masked=True,draw_positions=[0])


@pytest.mark.parametrize('seed',[17,42,101])
def test_ab_initialization_draws_masks_and_weighting_identical(experiment_root,small_config,tmp_path,seed):
    train = ExperimentData(experiment_root,'train')
    a = Trainer(train,small_config,'A',seed,tmp_path/f'A{seed}',profile=True)
    b = Trainer(train,small_config,'B',seed,tmp_path/f'B{seed}',profile=True)
    assert a.model.state_dict().keys() == b.model.state_dict().keys()
    assert all(torch.equal(v,b.model.state_dict()[k]) for k,v in a.model.state_dict().items())
    a.run(until=2,install_signals=False)
    b.run(until=2,install_signals=False)
    assert a.trace == b.trace and a.sampler.position==b.sampler.position
    assert torch.equal(a.model.z_bin_weights,b.model.z_bin_weights)
    assert a.weight_manifest == b.weight_manifest


def test_cpu_exact_resume_optimizer_scheduler_rng_sampler_and_weights(experiment_root,small_config,tmp_path):
    train = ExperimentData(experiment_root,'train')
    whole = Trainer(train,small_config,'A',17,tmp_path/'whole',profile=True)
    whole.run(until=4,install_signals=False)
    interrupted = Trainer(train,small_config,'A',17,tmp_path/'resumed',profile=True)
    interrupted.run(until=2,install_signals=False)
    resumed = Trainer(train,small_config,'A',17,tmp_path/'resumed',profile=True,resume=True)
    resumed.run(until=4,install_signals=False)
    assert whole.trace == resumed.trace
    for key,value in whole.model.state_dict().items():
        assert torch.equal(value,resumed.model.state_dict()[key]),key
    assert torch.equal(whole.model.z_bin_weights,resumed.model.z_bin_weights)
    assert whole.scheduler.state_dict() == resumed.scheduler.state_dict()
    for key, state in whole.optimizer.state_dict()['state'].items():
        for k,v in state.items():
            assert torch.equal(v,resumed.optimizer.state_dict()['state'][key][k])
    assert whole.sampler.position == resumed.sampler.position == 16
    assert [x['loss_z'] for x in whole.timings] == [x['loss_z'] for x in resumed.timings]
    assert not torch.equal(resumed.model.z_bin_weights,torch.ones(100))
    with pytest.raises(ValueError,match='Incompatible'):
        Trainer(train,small_config,'B',17,tmp_path/'resumed',profile=True,resume=True)
    with pytest.raises(FileExistsError):
        Trainer(train,small_config,'A',17,tmp_path/'resumed',profile=True)


def test_atomic_checkpoint_failure_retains_previous_and_corruption_fallback(experiment_root,small_config,tmp_path,monkeypatch):
    from desi_fm.experiment import training
    train = ExperimentData(experiment_root,'train')
    t = Trainer(train,small_config,'B',42,tmp_path/'atomic',profile=True)
    t.run(until=2,install_signals=False)
    before = (t.directory/'checkpoints.json').read_bytes()
    original_save = torch.save
    def crash(state,stream):
        stream.write(b'partial'); raise OSError('simulated interruption')
    monkeypatch.setattr(torch,'save',crash)
    with pytest.raises(OSError):
        t.checkpoint()
    assert (t.directory/'checkpoints.json').read_bytes() == before
    monkeypatch.setattr(torch,'save',original_save)
    pointers = json.loads(before)
    (t.directory/pointers['last']['path']).write_bytes(b'corrupt')
    state, generation = load_checkpoint(t.directory,t.identity)
    assert generation=='previous' and state['step']==2
    assert torch.equal(state['z_bin_weights'],t.model.z_bin_weights)


def test_profile_guards_and_cnn_size(experiment_root,small_config,tmp_path):
    train = ExperimentData(experiment_root,'train')
    val = ExperimentData(experiment_root,'validation')
    t = Trainer(train,small_config,'CNN',101,tmp_path/'cnn',profile=True)
    assert sum(p.numel() for p in t.model.parameters()) == 912548
    with pytest.raises(PermissionError):
        t.run(validation=val,until=1,install_signals=False)
    with pytest.raises(ValueError):
        t.run(until=201,install_signals=False)
    t.run(until=1,install_signals=False)
    assert t.fallbacks==0


def test_validation_selection_and_resume_from_selected_state(experiment_root,small_config,tmp_path):
    train = ExperimentData(experiment_root,'train'); val = ExperimentData(experiment_root,'validation')
    t = Trainer(train,small_config,'B',42,tmp_path/'select',profile=False)
    report = t.run(validation=val,install_signals=False)
    assert report['status']=='complete' and report['best']
    state,_ = load_checkpoint(t.directory,t.identity,which='best')
    assert state['best']==report['best']
    assert (t.directory/state['best']['predictions']).exists()


def test_resume_rejects_changed_horizon_and_class_weight_tampering(experiment_root,small_config,tmp_path):
    from desi_fm.experiment.training import save_checkpoint
    train=ExperimentData(experiment_root,'train')
    trainer=Trainer(train,small_config,'A',17,tmp_path/'incompatible',profile=True)
    trainer.run(until=1,install_signals=False)
    with pytest.raises(ValueError,match='Incompatible'):
        Trainer(train,{**small_config,'updates':5},'A',17,trainer.directory,profile=True,resume=True)
    state=trainer.state(); state['z_bin_weights']=torch.ones(100)
    save_checkpoint(trainer.directory,state)
    with pytest.raises(ValueError,match='class weights'):
        Trainer(train,small_config,'A',17,trainer.directory,profile=True,resume=True)


def test_class_bin_boundary_out_of_range_and_weighted_ce_normalization():
    import math
    from desi_fm.experiment.models import target_bins, classification_sum
    cfg=default_config(); weights=torch.arange(1.,101.)
    model=make_model(cfg,'CNN',17,weights)
    labels=torch.tensor([0.,6.,8.])
    bins=target_bins(labels,model.z_bin_edges)
    assert bins.tolist()==[0,99,99]
    logits=torch.zeros(3,100)
    assert torch.allclose(classification_sum(logits,labels,model)/weights[bins].sum(),torch.tensor(1.))
    with pytest.raises(ValueError):target_bins(torch.tensor([-1.]),model.z_bin_edges)


@pytest.mark.parametrize('length',[4,111,128])
def test_fixed_pool_matches_adaptive_windows_and_gradients(length):
    from desi_fm.experiment.models import FixedAdaptivePool1d
    torch.manual_seed(19)
    x=torch.randn(2,3,length,requires_grad=True)
    actual=FixedAdaptivePool1d(length)(x)
    reference=torch.nn.functional.adaptive_avg_pool1d(x,64)
    torch.testing.assert_close(actual,reference,atol=1e-6,rtol=1e-6)
    gradient=torch.randn_like(actual)
    left=torch.autograd.grad(actual,x,gradient,retain_graph=True)[0]
    right=torch.autograd.grad(reference,x,gradient)[0]
    torch.testing.assert_close(left,right,atol=1e-6,rtol=1e-6)
