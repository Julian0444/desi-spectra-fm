"""Missing-coordinate support must not interpolate across two neighboring holes."""
import numpy as np
import pytest
from test_experiment_training import experiment_root
from desi_fm.data import resample_spectrum, SpectrumPreprocessConfig
from desi_fm.inputs import SpectrumInputError, read_npz

@pytest.mark.parametrize('encoding',['absent','nan','inf','mask','ivar'])
@pytest.mark.parametrize('order',['forward','reverse','shuffle'])
def test_two_holes_separated_by_one_sample(encoding,order):
    cfg=SpectrumPreprocessConfig(n_pixels=64);w=cfg.target_wavelength
    f=np.sin(w/500);v=np.ones(64);m=np.zeros(64,bool)
    holes=np.zeros(64,bool);holes[20:28]=True;holes[29:37]=True
    if encoding=='absent': f,w,v,m=(x[~holes] for x in (f,w,v,m))
    elif encoding in ('nan','inf'):w=w.copy();w[holes]=float(encoding)
    elif encoding=='mask':m[holes]=True
    else:v[holes]=0
    ix=np.arange(len(w))
    if order=='reverse':ix=ix[::-1]
    if order=='shuffle':ix=np.random.default_rng(5).permutation(ix)
    result=resample_spectrum(f[ix],v[ix],w[ix],m[ix],cfg)
    np.testing.assert_array_equal(result['valid'].astype(bool),~holes)
    assert result['valid'][28]==1 and (result['flux'][holes]==0).all()


def test_missing_wavelength_field_rejected(tmp_path):
    path=tmp_path/'missing.npz';np.savez(path,flux=np.ones(64))
    with pytest.raises(SpectrumInputError,match='flux and wavelength'):read_npz(path)


@pytest.mark.parametrize('fault',['all_nonfinite_wave','seven_usable','no_grid_support'])
def test_no_supported_prediction(fault):
    cfg=SpectrumPreprocessConfig(n_pixels=64);w=cfg.target_wavelength.copy()
    f=np.ones(64);v=np.ones(64);m=np.zeros(64,bool)
    if fault=='all_nonfinite_wave':w[:]=np.nan
    elif fault=='seven_usable':v[7:]=0
    else:w=np.linspace(3600,3601,64)
    with pytest.raises(SpectrumInputError,match='usable'):resample_spectrum(f,v,w,m,cfg)


def test_historical_derivative_is_not_silently_reinterpreted(experiment_root,monkeypatch):
    import json
    from desi_fm.experiment.data import ExperimentData
    path=experiment_root/'manifest.json'
    manifest=json.loads(path.read_text());manifest['contract']='batch4-grid-v1:unnormalized-spectra-input-v2'
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('Opened incompatible arrays'))
    with pytest.raises(ValueError,match='incompatible derivative'):
        ExperimentData(experiment_root,'train')


def test_prepare_rejects_old_eligibility_before_access_or_output(tmp_path,monkeypatch):
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace
    spec=importlib.util.spec_from_file_location('prepare_batch4',Path(__file__).resolve().parents[1]/'scripts/prepare_batch4.py')
    prepare=importlib.util.module_from_spec(spec);spec.loader.exec_module(prepare)
    monkeypatch.setattr(prepare,'LocalDESIStore',lambda *a:SimpleNamespace(index_contract='spectra-input-v2'))
    monkeypatch.setattr(prepare,'read_json',lambda *a:pytest.fail('Read protected IDs before contract check'))
    out=tmp_path/'new'
    with pytest.raises(ValueError,match='different input contract'):
        prepare.prepare(SimpleNamespace(root=tmp_path,index=tmp_path/'index.sqlite',output=out))
    assert not out.exists()
