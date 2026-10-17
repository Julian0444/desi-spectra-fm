"""Bounded 5A regression probes in new disposable source copies; no real data.

Usage: python scripts/check_batch5a_mutations.py --copilot /path/to/copilot
    --output /new/private/directory
Run with the test environment's interpreter. Baselines must pass; exit 1 means
an effective mutant survived or a probe had an infrastructure/collection error.
Only the temporary copies are edited. The original repositories are read-only.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def run(fm, copilot, output):
    output.mkdir(parents=True,exist_ok=False)
    copies={}
    for name,root,dirs in [('fm',fm,('src','scripts','tests')),('cp',copilot,('copilot','tests','references'))]:
        dest=output/name
        subprocess.run(['git','clone','--shared','--no-checkout',str(root),str(dest)],check=True,capture_output=True)
        for directory in dirs:
            if (root/directory).exists():
                shutil.copytree(root/directory,dest/directory,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
        copies[name]=dest
    env=os.environ.copy()
    env.update(PYTHONPATH=str(copies['fm']/'src')+os.pathsep+str(copies['cp']),
               SPECTRA_FM_SOURCE_ROOT=str(copies['fm']),PYTHONDONTWRITEBYTECODE='1',
               HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1')
    env.pop('ANTHROPIC_API_KEY',None)
    behavioral='tests/test_experiment_regressions.py'
    guard='tests/test_experiment_guard_regressions.py'
    gaps='tests/test_gap_regressions.py'
    interfaces='tests/test_interface_contract.py::test_quality_reaches_model_through_real_handlers'
    train='src/desi_fm/experiment/training.py';data='src/desi_fm/experiment/data.py'
    final='src/desi_fm/experiment/final.py';metrics='src/desi_fm/experiment/metrics.py'
    coefficient="1. if self.variant == 'A' else 0."
    rows=[
        ('B_uses_reconstruction','fm',train,coefficient,'1.',behavioral,True),
        ('A_ignores_reconstruction','fm',train,coefficient,'0.',behavioral,True),
        ('coefficients_swapped','fm',train,coefficient,"1. if self.variant == 'B' else 0.",behavioral,True),
        ('hidden_normalization','fm',data,"normalize_grid(f, v, self.config, m, insufficient='identity')","normalize_grid(f, v, self.config, None, insufficient='identity')",behavioral,True),
        ('paired_threshold_010','fm',metrics,'ea, eb = np.abs(ra) > .15, np.abs(rb) > .15','ea, eb = np.abs(ra) > .10, np.abs(rb) > .10',behavioral,True),
        ('paired_inclusive_boundary','fm',metrics,'ea, eb = np.abs(ra) > .15, np.abs(rb) > .15','ea, eb = np.abs(ra) >= .15, np.abs(rb) >= .15',behavioral,True),
        ('interval_threshold_010','fm',final,"outlier = np.abs((arrays['prediction']-arrays['z_true'])/(1+arrays['z_true'])) > .15","outlier = np.abs((arrays['prediction']-arrays['z_true'])/(1+arrays['z_true'])) > .10",behavioral,True),
        ('interval_inclusive_boundary','fm',final,"outlier = np.abs((arrays['prediction']-arrays['z_true'])/(1+arrays['z_true'])) > .15","outlier = np.abs((arrays['prediction']-arrays['z_true'])/(1+arrays['z_true'])) >= .15",behavioral,True),
        ('remove_final_current_environment','fm',final,"        if shared['environment'] != current_environment:\n            raise ValueError('Current environment differs from frozen training/validation environment')\n",'',guard+'::test_probe_b2_final_rejects_neural_environment_even_if_ridge_matches',True),
        ('remove_freeze_current_environment','fm',final,"    if common['environment'] != environment():\n        raise ValueError('Current environment differs from the neural experiment')\n",'',guard+'::test_probe_b8_freeze_rejects_homogeneous_foreign_environment',True),
        ('unpaired_initialization','fm',train,'make_model(config,variant,seed,weights)','make_model(config,variant,seed+int(variant=="B"),weights)',behavioral,True),
        ('unpaired_samples','fm',train,'self.sampler = PairedSampler(len(data), seed)','self.sampler = PairedSampler(len(data), seed+int(variant=="B"))',behavioral,True),
        ('unpaired_masks','fm',train,'batch = self.data.batch(indices, seed=self.seed,','batch = self.data.batch(indices, seed=self.seed+int(self.variant=="B"),',behavioral,True),
        ('old_gap_rule','fm','src/desi_fm/data.py','local = np.minimum(','local = np.maximum(',gaps,True),
        ('demo_loses_quality','fm','demo/app.py','result = predict_file_single(path, model=MODEL, mask_ratio=mask_ratio)','result = predict_spectrum(flux=flux, wavelength=wavelength, model=MODEL, mask_ratio=mask_ratio)',interfaces,True),
        ('api_loses_mask','fm','src/desi_fm/api.py','mask=s.mask,','mask=None,',interfaces,True),
        ('api_loses_ivar','fm','src/desi_fm/api.py','ivar=s.ivar,','ivar=None,',interfaces,True),
        ('tools_lose_quality','cp','copilot/tools.py','r = predict_spectrum(flux=flux, wavelength=wave, ivar=ivar, mask=mask, model=_model())','r = predict_spectrum(flux=flux, wavelength=wave, model=_model())',interfaces,True),
        ('equivalent_coefficient_spelling','fm',train,coefficient,"1.0 if self.variant == 'A' else 0.0",behavioral,False),
    ]
    # Demo source is needed for real handler imports, but never launched.
    for directory in ('demo','api'):
        (copies['fm']/directory).mkdir()
        shutil.copy2(fm/directory/'app.py',copies['fm']/directory/'app.py')
    results=[]
    def pytest(name,repo,tests):
        command=[sys.executable,'-m','pytest',*tests.split(),'-q','-p','no:cacheprovider']
        log=output/(name+'.log')
        with log.open('w') as stream:
            try:
                code=subprocess.run(command,cwd=copies[repo],env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=90).returncode
            except subprocess.TimeoutExpired:
                code=124;stream.write('\nTIMEOUT: failed probe\n')
        text=log.read_text()
        return {'returncode':code,'log_sha256':hashlib.sha256(log.read_bytes()).hexdigest(),
                'summary':text.splitlines()[-1] if text.splitlines() else '',
                'failed_tests':[line for line in text.splitlines() if line.startswith('FAILED ')]}
    for name,repo,tests in [('baseline_fm','fm',behavioral+' '+guard+' '+gaps),('baseline_interfaces','cp',interfaces)]:
        result={'name':name,**pytest(name,repo,tests)};results.append(result)
        if result['returncode']!=0:raise RuntimeError(f'Baseline failed: {result}')
    for name,repo,path,before,after,test,effective in rows:
        target=copies[repo]/path;original=target.read_text()
        if original.count(before)!=1:raise ValueError(f'Ambiguous mutation {name}: {original.count(before)}')
        target.write_text(original.replace(before,after))
        try:
            result={'name':name,'file':path,'before':before,'after':after,'effective':effective,
                    **pytest(name,'cp' if test==interfaces else 'fm',test)}
        finally:target.write_text(original)
        result['expected_outcome']=(result['returncode']==1 and bool(result['failed_tests'])) if effective else result['returncode']==0
        results.append(result)
        (output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
        print(name,result['summary'],result['expected_outcome'],flush=True)
    return all(row.get('expected_outcome',True) for row in results)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--copilot',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    sys.exit(0 if run(Path(__file__).resolve().parents[1],args.copilot.resolve(),args.output.resolve()) else 1)
