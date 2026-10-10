import io
import numpy as np
import pytest
import torch
from contract_fixtures import spectra
from desi_fm.data import preprocess_spectrum, SpectrumPreprocessConfig
from desi_fm.inputs import parse_arrays, read_npz, SpectrumInputError
from desi_fm.model import DESIFoundationModel, DESIFoundationModelConfig
from desi_fm.predict import predict_spectrum, predict_spectra_batch, predict_file_single

@pytest.fixture
def model():
    torch.manual_seed(17)
    return DESIFoundationModel(DESIFoundationModelConfig(n_pixels=64,n_tokens=8,d_model=16,n_layers=1,n_heads=2,dropout=.4,n_z_bins=16))

def test_nonfinite_equals_masked_and_preserves_gap(model):
    fixture=spectra()
    a=predict_spectrum(**fixture['nonfinite'],model=model)
    b=predict_spectrum(**fixture['masked'],model=model)
    np.testing.assert_array_equal(a['normalized_input'],b['normalized_input'])
    assert not a['valid_model_grid'][20:28].any()
    assert a['z_pred']==b['z_pred']
    assert a['coverage']['n_valid_model_pixels']==56

@pytest.mark.parametrize('name',['no_coverage','bad_mask','bad_ivar','empty','strings','container'])
def test_invalid_is_domain_error(name,model):
    with pytest.raises(SpectrumInputError):
        predict_spectrum(**spectra()[name],model=model)

@pytest.mark.parametrize('value',[True,'1',1.5,-.5,float('nan'),float('inf')])
def test_invalid_mask_ratio(value,model):
    with pytest.raises(SpectrumInputError):
        predict_spectrum(**spectra()['valid'],model=model,mask_ratio=value)

def test_reorder_units_partial_and_shared_broadcast(model):
    f=spectra();a=predict_spectrum(**f['valid'],model=model)
    b=predict_spectrum(**f['reordered'],model=model)
    assert a['z_pred']==b['z_pred']
    c=predict_spectrum(f['valid']['flux'],f['valid']['wavelength']/10,wavelength_unit='nm',model=model)
    np.testing.assert_allclose(a['normalized_input'],c['normalized_input'],atol=1e-5)
    assert 0<predict_spectrum(**f['partial'],model=model)['coverage']['model_valid_fraction']<1
    batch=predict_spectra_batch(np.stack([f['valid']['flux']]*2),f['valid']['wavelength'],masks=np.zeros(64,bool),ivars=np.ones(64),model=model)
    np.testing.assert_array_equal(batch['z_pred'],[a['z_pred']]*2)
    with pytest.raises(SpectrumInputError): parse_arrays(np.ones((2,64)),np.ones((1,64)))

def test_file_and_batch_contract(model,tmp_path):
    p=tmp_path/'input.npz';np.savez(p,**spectra()['masked'],z_true=99)
    a=predict_file_single(p,model=model);b=predict_spectrum(**spectra()['masked'],model=model)
    assert a['z_pred']==b['z_pred'];assert a['coverage']==b['coverage']
    np.savez(p,flux=np.ones((2,64)),wavelength=spectra()['valid']['wavelength'])
    with pytest.raises(SpectrumInputError): predict_file_single(p,model=model)
