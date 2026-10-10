from pathlib import Path
import numpy as np
import pytest

from desi_fm.experiment.artifacts import read_json, write_json, file_record
from desi_fm.experiment.data import ExperimentData
from desi_fm.experiment.metrics import save_predictions, load_predictions, paired_analysis, selection_key
from desi_fm.experiment.ridge import features, fit_three, ridge_predict, run_ridge
from desi_fm.experiment.final import evaluate_final, freeze, eta_intervals
from desi_fm.experiment.training import Trainer
from test_experiment_training import experiment_root, small_config


def test_metric_boundary_accounting_selection_and_risk(experiment_root,tmp_path):
    data=ExperimentData(experiment_root,'train')
    indices=np.arange(3)
    prediction=data.z[indices]+np.array([.15,.16,0])*(1+data.z[indices])
    summary=save_predictions(tmp_path/'pred.npz',data,indices,{'prediction':prediction,'confidence':[.1,.3,.9]},{'seed':17})
    assert summary['primary']['n']==3 and summary['primary']['outliers']==1
    assert summary['primary']['eta15']==1/3
    assert summary['accounting']=={'requested':3,'evaluated':3,'rejected':0}
    assert summary['risk_coverage']['eta15']==[0.,.5,1/3]
    assert selection_key(summary,2)<selection_key(summary,3)
    better={'primary':{'eta15':1/3,'mae_norm':0}}
    assert selection_key(better,100)<selection_key(summary,2)
    with pytest.raises(ValueError,match='duplicate'):
        save_predictions(tmp_path/'bad.npz',data,[0,0],{'prediction':[0,0]}, {})
    with pytest.raises(ValueError,match='nonfinite'):
        save_predictions(tmp_path/'bad.npz',data,[0],{'prediction':[float('nan')]}, {})


def test_paired_bootstrap_keeps_three_seeds_one_object_unit_and_rejects_misalignment(experiment_root,tmp_path):
    data=ExperimentData(experiment_root,'train'); indices=np.arange(len(data))
    paths={'A':[],'B':[]}
    for seed in (17,42,101):
        for variant in ('A','B'):
            p=tmp_path/f'{variant}-{seed}.npz'
            prediction=data.z.copy()
            if variant=='B':prediction[::2]+=(1+prediction[::2])*.3
            save_predictions(p,data,indices,{'prediction':prediction},{'seed':seed,'variant':variant})
            paths[variant].append(p)
    result=paired_analysis(paths['A'],paths['B'],replicates=1000)
    assert result['criterion_met'] and result['n_objects']==25 and result['n_observed_groups']==25
    assert result['delta_eta_per_seed']==[-13/25]*3
    assert result['paired_95_interval'][0]<-.6 and result['paired_95_interval'][1]>-.4
    assert result['paired_95_interval'][1]<0
    arrays,_=load_predictions(paths['A'][0])
    intervals=eta_intervals(arrays,replicates=100)
    assert intervals['metrics']['all']['n']==25
    assert intervals['metrics']['all']['interval95']==[0.,0.]
    save_predictions(paths['B'][2],data,indices[::-1],{'prediction':data.z[::-1]},{'seed':101,'variant':'B'})
    with pytest.raises(ValueError,match='Unpaired'):
        paired_analysis(paths['A'],paths['B'],replicates=10)


def test_ridge_fixed_bands_train_scaling_three_fits_and_intercept():
    flux=np.tile(np.arange(256,dtype=float),(12,1));valid=np.ones_like(flux)
    valid[:,:2]=0
    x=features(flux,valid)
    assert x.shape==(12,256) and x[0,0]==0 and x[0,128]==0 and x[0,1]==2.5
    rng=np.random.default_rng(8);x=rng.normal(size=(200,256)); y=np.expm1(.4+.03*x[:,0])
    models=fit_three(x,y)
    assert [float(m['alpha']) for m in models]==[.1,1.,10.]
    for model in models:
        np.testing.assert_array_equal(model['mean'],x.mean(0))
        assert ridge_predict(model,x).shape==(200,)
    assert np.mean(abs(ridge_predict(models[0],x)-y))<.002


def test_full_synthetic_pipeline_freezes_all_candidates_and_gates_once(experiment_root,small_config,tmp_path):
    # Synthetic arrays only. This never opens the real batch-4 test directory.
    train=ExperimentData(experiment_root,'train'); validation=ExperimentData(experiment_root,'validation')
    runs=tmp_path/'runs'
    cfg={**small_config,'updates':2,'warmup_updates':0,'validation_every':1,'checkpoint_every':1}
    for variant in ('A','B','CNN'):
        for seed in (17,42,101):
            Trainer(train,cfg,variant,seed,runs/f'{variant}-{seed}').run(validation=validation,install_signals=False)
    ridge=tmp_path/'ridge'; report=run_ridge(train,validation,ridge)
    assert report['fits']==3
    with pytest.raises(FileExistsError):run_ridge(train,validation,ridge)
    with pytest.raises(PermissionError):run_ridge(validation,train,tmp_path/'bad-ridge')
    frozen=tmp_path/'frozen.json'
    spec=freeze(runs,ridge,experiment_root,frozen)
    assert len(spec['neural'])==9 and len(spec['sensitivity']['estimates'])==3
    assert not (experiment_root/'test-opening.json').exists()
    from desi_fm.experiment.data import TEST_PHRASE
    with pytest.raises(PermissionError):evaluate_final(frozen,tmp_path/'test-output','yes',device='cpu')
    assert not (experiment_root/'test-opening.json').exists()
    result=evaluate_final(frozen,tmp_path/'test-output',TEST_PHRASE,device='cpu')
    assert result['status']=='complete' and len(result['files'])==10 and result['paired']['n_objects']==7
    with pytest.raises(PermissionError):evaluate_final(frozen,tmp_path/'second-output',TEST_PHRASE,device='cpu')
    again=evaluate_final(frozen,tmp_path/'test-output',TEST_PHRASE,resume=True,device='cpu')
    assert result==again
