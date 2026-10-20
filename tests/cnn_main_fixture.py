"""Run a temporary copy of CNN main with invented identities and inert models.

No real checkpoint, data tree, test split, model inference or network is used.
This launcher exists only for main-path guard and gate regression tests.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch


def run(source, root, scenario):
    root=Path(root);harness=root/'harness';scripts=harness/'scripts';scripts.mkdir(parents=True)
    shutil.copyfile(source,scripts/'check_batch5b_cnn.py')
    historical=root/'historical';package=historical/'src/desi_fm';(package/'experiment').mkdir(parents=True)
    data_root=root/'invented-data';data_root.mkdir();inputs=root/'inputs';inputs.mkdir()
    fixture=root/'fixture.json'
    common="import json\nfrom pathlib import Path\nfrom types import SimpleNamespace\nimport numpy as np\nFIXTURE=Path("+repr(str(fixture))+")\ndef fixture():return json.loads(FIXTURE.read_text())\n"
    files={
        '__init__.py':'', 'experiment/__init__.py':'',
        'inputs.py':'class SpectrumInputError(ValueError):pass\n',
        'data.py':common+'''from .inputs import SpectrumInputError
def preprocess_spectrum(flux,ivar,wave,mask,config):
    if flux[0]<0:raise SpectrumInputError('Invented invalid input')
    return {'flux':np.ones(8),'valid':np.ones(8),'coverage':{'fixture':True}}
''',
        'experiment/data.py':common+'''GRID_CONTRACT='invented-grid'
class ExperimentData:
    def __init__(self,root,role):
        assert role in ('train','validation')
        self.role=role;self.ids=[role+str(i) for i in range(24)]
        self.identity={'role':role};self.config=SimpleNamespace(n_pixels=8,n_tokens=4)
        self.manifest={'contract':GRID_CONTRACT,'preprocess':'invented-noop'}
    def batch(self,indices,**kwargs):
        return {'flux':np.ones((len(indices),8)),'valid':np.ones((len(indices),8))}
''',
        'experiment/artifacts.py':common+'''import hashlib
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
def source_identity():
    value=fixture()['source']
    if fixture()['scenario']=='source':value['files']['__init__.py']='altered'
    return value
def environment():return {'version':'wrong' if fixture()['scenario']=='environment' else 'invented'}
def verify_record(path,expected):
    data=Path(path).read_bytes()
    if {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}!=expected:
        raise ValueError('Invented file identity mismatch')
def read_json(path):return json.loads(Path(path).read_text())
def write_json(path,value):Path(path).write_text(json.dumps(value))
''',
        'experiment/models.py':common+'''import torch
class InertModel:
    config=SimpleNamespace(n_tokens=4,n_pixels=8)
    def load_state_dict(self,state,strict):assert strict is True
    def eval(self):return self
    def to(self,device):return self
    def parameters(self):return []
    def __call__(self,flux,valid):
        return {'z_pred_map':torch.tensor([.3]),'z_pred':torch.tensor([.4]),'z_confidence':torch.tensor([.5])}
def make_model(*args):return InertModel()
''',
        'experiment/training.py':common+'''def predict(*args):return {'prediction':np.array([.3]),'secondary':np.array([.4]),'confidence':np.array([.5])}
def memory(*args):return {'peak_host_rss_bytes':0}
'''}
    for name,text in files.items():(package/name).write_text(text)
    for i in range(25-len(files)):(package/f'identity_only_{i}.py').write_text('# invented identity fixture\n')
    sources={str(f.relative_to(historical)):hashlib.sha256(f.read_bytes()).hexdigest() for f in package.rglob('*.py')}
    assert len(sources)==25
    cases=[{'case_id':'invented-'+role+str(i),'kind':'real','cnn_exposure':role,'object_id':role+str(i)} for role in ('train','validation') for i in range(24)]
    cases += [{'case_id':'invented-synthetic-'+str(i),'kind':'synthetic','cnn_exposure':'none_synthetic',
               'family':'invalid_support' if i<6 else 'reversed_grid' if i<12 else 'permuted_grid' if i<18 else 'emission'} for i in range(48)]
    development=root/'development.json';development.write_text(json.dumps(cases))
    (scripts/'batch5b_development.py').write_text('''import json
from pathlib import Path
import numpy as np
def read_development(path):return json.loads(Path(path).read_text())
def render_recipe(case):
    return {'flux':np.full(8,-1 if case['family']=='invalid_support' else 1),
            'wavelength':np.arange(8.),'ivar':np.ones(8),'mask':np.zeros(8,dtype=bool)},[]
''')
    for i in range(48):np.savez(inputs/f'input_{i:04d}.npz',flux=np.ones(8),ivar=np.ones(8),wavelength=np.arange(8.),mask=np.zeros(8,dtype=bool))
    checkpoint=root/'invented-checkpoint.bin';checkpoint.write_bytes(b'invented checkpoint identity only')
    manifest=data_root/'manifest.json';manifest.write_text('{}')
    validation=root/'invented-validation.npz'
    predictions=np.full(24,.3)
    if scenario=='parity':predictions[0]=.8
    np.savez(validation,ids=np.array(['validation'+str(i) for i in range(24)]),prediction=predictions,secondary=np.full(24,.4),confidence=np.full(24,.5))
    def record(path):
        value=Path(path).read_bytes();return {'bytes':len(value),'sha256':hashlib.sha256(value).hexdigest()}
    source_record={'files':sources,'sha256':'invented-source-digest'}
    identity={'source_sha256':source_record['sha256'],'environment':{'version':'invented'},'device':'mps',
              'config':{'cpu_threads':1,'normalization':'invented-noop'},'train_identity':{'role':'train'},'validation_identity':{'role':'validation'}}
    payload={'identity':identity,'step':6000,'z_bin_weights':None,'model':{}}
    frozen={'source':source_record,'data_root':str(data_root),'data_manifest':record(manifest),'neural':[{
        'variant':'CNN','seed':17,'step':6000,'file':record(checkpoint),'checkpoint':str(checkpoint),'run_identity':identity,
        'validation_predictions':str(validation),'validation_file':record(validation)}]}
    expected_checkpoint=record(checkpoint)
    if scenario=='frozen':frozen['neural'][0]['file']['sha256']='altered-frozen-record'
    if scenario=='checkpoint':checkpoint.write_bytes(b'changed after identity was recorded')
    frozen_dir=root/'frozen';frozen_dir.mkdir();frozen_path=frozen_dir/'frozen.json';frozen_path.write_text(json.dumps(frozen))
    fixture.write_text(json.dumps({'source':source_record,'scenario':scenario}))
    digest=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    validation.with_suffix('.json').write_text(json.dumps({'role':'validation','artificial_mask':False,
        'data_identity':identity['validation_identity'],'identity':{'seed':17,'variant':'CNN','step':6000,'run_identity':digest}}))
    backup=root/'backup.json';backup.write_text('{"status":"invented fixture; no real checkpoint"}')
    module_path=scripts/'check_batch5b_cnn.py'
    spec=importlib.util.spec_from_file_location('cnn_under_test',module_path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.CHECKPOINT=expected_checkpoint
    def git(command,**kwargs):
        if 'show' in command:return (harness/command[-1].split(':',1)[1]).read_bytes()
        return module.HISTORICAL_HEAD if str(historical) in command else 'invented-harness-head'
    module.subprocess.check_output=git
    sys.path.insert(0,str(scripts))
    torch.backends.mps.is_available=lambda:True
    torch.mps.synchronize=lambda:None
    torch.mps.current_allocated_memory=lambda:0
    torch.mps.driver_allocated_memory=lambda:0
    # Keep all computations on CPU and return fixed fixture values, no model inference.
    original_to=torch.Tensor.to
    torch.Tensor.to=lambda self,*args,**kwargs:self if args and args[0]=='mps' else original_to(self,*args,**kwargs)
    loads=[]
    def load(path,**kwargs):loads.append(str(path));return payload
    torch.load=load
    output=root/'output'
    sys.argv=['cnn','--historical-root',str(historical),'--frozen',str(frozen_path),'--development',str(development),
              '--input-dir',str(inputs),'--backup-status',str(backup),'--output',str(output)]
    result={'exception':None,'checkpoint_loads':0}
    try:module.main()
    except (ValueError,SystemExit) as exc:result.update(exception=type(exc).__name__,message=str(exc))
    result['checkpoint_loads']=len(loads)
    if (output/'summary.json').exists():result['summary']=json.loads((output/'summary.json').read_text())
    print(json.dumps(result))


if __name__=='__main__':run(*sys.argv[1:])
