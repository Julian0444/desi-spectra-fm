import numpy as np
import pytest
import torch
from contract_fixtures import spectra
from desi_fm.model import DESIFoundationModel, DESIFoundationModelConfig
from desi_fm.predict import predict_spectrum, embed_spectrum, load_model_from_checkpoint
from desi_fm.evaluate import load_model, evaluate_and_write_outputs
from desi_fm.train import evaluate

@pytest.fixture
def model():
    torch.manual_seed(17)
    return DESIFoundationModel(DESIFoundationModelConfig(n_pixels=64,n_tokens=8,d_model=16,n_layers=1,n_heads=2,dropout=.6,n_z_bins=16,mask_ratio=0))

@pytest.mark.parametrize('mutation',['missing','unexpected','shape'])
def test_strict_checkpoint(model,tmp_path,mutation):
    state=dict(model.state_dict());key=next(iter(state))
    if mutation=='missing': del state[key]
    if mutation=='unexpected': state['invented']=torch.ones(1)
    if mutation=='shape': state[key]=torch.ones(1)
    p=tmp_path/'weights.pt';torch.save(dict(config=model.config.to_dict(),model=state),p)
    for loader in (load_model,load_model_from_checkpoint):
        with pytest.raises(RuntimeError): loader(p,torch.device('cpu'))

def test_explicit_migration_preserves_every_tensor(model,tmp_path):
    state=dict(model.state_dict());state['pos_embed']=state.pop('learned_pos_embed')
    p=tmp_path/'weights.pt';torch.save(dict(config=model.config.to_dict(),model=state),p)
    before=p.read_bytes();loaded=load_model_from_checkpoint(p,torch.device('cpu'))
    for k,v in model.state_dict().items(): torch.testing.assert_close(loaded.state_dict()[k],v,rtol=0,atol=0)
    assert p.read_bytes()==before;assert loaded.checkpoint_identity['migration']=='pos_embed rename'

def test_dropout_inference_restores_mixed_module_modes(model):
    model.train();model.redshift_head.eval()
    modes=[m.training for m in model.modules()]
    kw=dict(**spectra()['valid'],model=model)
    a=predict_spectrum(**kw);b=predict_spectrum(**kw)
    assert a['z_pred']==b['z_pred'];np.testing.assert_array_equal(a['normalized_reconstruction'],b['normalized_reconstruction'])
    np.testing.assert_array_equal(embed_spectrum(**kw),embed_spectrum(**kw))
    assert [m.training for m in model.modules()]==modes

@pytest.mark.parametrize('fn',[evaluate,evaluate_and_write_outputs])
def test_empty_evaluation_rejected(fn,model,tmp_path):
    p=tmp_path/'results.csv';p.write_text('existing evidence')
    kwargs={'predictions_csv':p} if fn is evaluate_and_write_outputs else {}
    with pytest.raises(ValueError,match='no examples'): fn(model,[],torch.device('cpu'),**kwargs)
    assert p.read_text()=='existing evidence'

@pytest.mark.parametrize('fn',[evaluate,evaluate_and_write_outputs])
def test_metrics_denominators_and_no_mask(fn,model):
    f=torch.zeros(2,64);v=torch.ones_like(f);z=torch.tensor([0.,1.])
    expected=model.eval()(f,v,z=z)
    r=fn(model,[dict(flux=f,valid=v,z=z)],torch.device('cpu'))
    assert r['n_examples']==2 and r['n_batches']==1 and r['redshift_denominator']==2
    assert r['reconstruction_denominator']==0 and r['reconstruction_rmse_masked'] is None
    assert r['redshift_mae']==pytest.approx(float((expected['z_pred']-z).abs().mean().detach()))
    assert r['eta15']==float(((expected['z_pred']-z).abs()/(1+z)>.15).float().mean())
    import json
    assert json.loads(json.dumps(r,allow_nan=False))['reconstruction_rmse_masked'] is None


def test_classifier_formulas_and_configured_range(model):
    import math
    p=torch.arange(1,17,dtype=torch.float32);p/=p.sum()
    with torch.no_grad():
        model.redshift_head[-1].weight.zero_();model.redshift_head[-1].bias.copy_(p.log())
    model.eval()
    out=model(torch.zeros(1,64),torch.ones(1,64),spectrum_mask=torch.zeros(1,8,dtype=torch.bool))
    expected=torch.expm1((p*model.z_bin_centers).sum())
    arithmetic=(p*torch.expm1(model.z_bin_centers)).sum()
    assert out['z_pred'].item()==pytest.approx(expected.item(),abs=1e-6)
    assert abs(expected-arithmetic)>.1
    assert out['z_confidence'].item()==pytest.approx(1-float(-(p*p.log()).sum())/math.log(16),abs=1e-6)
    assert 3.5<out['z_pred_map'].item()<model.config.z_max
    assert model.redshift_head[-1].out_features==16

@pytest.mark.parametrize('fn',[evaluate,evaluate_and_write_outputs])
def test_masked_rmse_has_actual_pixel_denominator(fn,model):
    model.config.mask_ratio=1
    f=torch.zeros(2,64);v=torch.ones_like(f);z=torch.tensor([0.,1.])
    out=model.eval()(f,v,z=z)
    r=fn(model,[dict(flux=f,valid=v,z=z)],torch.device('cpu'))
    assert r['n_examples']==2 and r['reconstruction_denominator']==128
    expected=float(torch.sqrt(out['reconstruction_sse']/out['reconstruction_valid_pixels']).detach())
    assert r['reconstruction_rmse_masked']==pytest.approx(expected)
    assert r['loss']==pytest.approx(float(out['loss'].detach()))
