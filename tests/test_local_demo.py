"""Local wheel demo must use the official prediction and reject bad inputs."""
import numpy as np
import pytest


def test_demo_uses_official_map_and_input_contract(tmp_path,monkeypatch):
    from desi_fm import demo
    from desi_fm.inputs import SpectrumInputError
    path=tmp_path/'spectrum.npz';np.savez(path,flux=np.ones(32),wavelength=np.linspace(3600,9800,32))
    monkeypatch.setattr(demo,'get_model',lambda:None)
    monkeypatch.setattr(demo,'predict_file_single',lambda *a,**k:{'z_pred_map':.42,'z_pred':.50,
        'z_confidence':.6,'coverage':{'model_valid_fraction':1.},'reconstruction_input_grid':np.ones(32)})
    text,figure=demo.analyze(str(path),0.)
    assert 'model_z = 0.4200' in text and 'secondary' in text and figure is not None
    np.savez(path,flux=np.ones(32),wavelength=np.linspace(3600,9800,32),mask=np.ones(32,dtype=bool))
    with pytest.raises(SpectrumInputError):demo.analyze(str(path),0.)
