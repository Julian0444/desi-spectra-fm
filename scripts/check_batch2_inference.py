"""Bounded contract smoke with preserved local v1/v2.1 checkpoints. No accuracy eval."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from desi_fm.predict import load_model_from_checkpoint, predict_spectrum


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--copilot-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    reference=json.loads((root/'docs/evidence/batch2-real-reference.json').read_text())
    source=args.copilot_root/'eval/cases/case_000.npz'
    assert sha(source)==reference['fixture_sha256']
    assets={e['path']:e['sha256'] for e in json.loads((root/'docs/evidence/historical-assets.json').read_text())['files']}
    with np.load(source,allow_pickle=False) as d:
        raw={k:d[k] for k in ('flux','wavelength','ivar','mask')}
    report={'protocol':'cpu/eval/mask_ratio=0; bounded contract smoke, no accuracy estimate',
            'fixture_sha256':sha(source),'loaded':{},'inference':{}}
    for name in reference['fully_usable_derived_expected']:
        for kind in ('best','last'):
            path=root/'runs'/name/f'checkpoint_{kind}.pt'
            if not path.exists():continue
            before=sha(path);assert before==assets[str(path.relative_to(root))]
            model=load_model_from_checkpoint(path,torch.device('cpu'))
            report['loaded'][str(path.relative_to(root))]=model.checkpoint_identity
            assert sha(path)==before
        path=root/'runs'/name/'checkpoint_last.pt'
        model=load_model_from_checkpoint(path,torch.device('cpu'))
        clean=dict(raw,ivar=np.ones_like(raw['flux']),mask=np.zeros_like(raw['mask'],bool))
        actual=predict_spectrum(**clean,model=model)
        expected=reference['fully_usable_derived_expected'][name]
        np.testing.assert_allclose([actual[k] for k in expected],list(expected.values()),rtol=0,atol=1e-6)
        served=predict_spectrum(**raw,model=model)
        if 'z_pred_map' in served:
            assert served['z_pred_map']==reference['historical_raw_expected'][name]['z_pred_map']
        # The raw historical sample has 19 invalid pixels. Contract v2 changes
        # their support; record the difference, never label it an accuracy gain.
        report['inference'][name]={'fully_usable_derived':{k:actual[k] for k in expected},
            'historical_raw':{k:served[k] for k in expected},
            'raw_delta_from_contract_v1':{k:served[k]-v for k,v in reference['historical_raw_expected'][name].items()},
            'coverage':served['coverage']}
        masked=dict(raw,flux=raw['flux'].copy(),mask=raw['mask'].copy())
        nonfinite=dict(raw,flux=raw['flux'].copy())
        indices=np.flatnonzero((raw['ivar']>0)&~raw['mask'])[:8]
        masked['mask'][indices]=True;nonfinite['flux'][indices]=np.nan
        a=predict_spectrum(**masked,model=model);b=predict_spectrum(**nonfinite,model=model)
        np.testing.assert_array_equal(a['normalized_input'],b['normalized_input'])
        assert a['z_pred']==b['z_pred']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2,allow_nan=False))

if __name__=='__main__':main()
