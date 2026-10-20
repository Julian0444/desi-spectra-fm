"""Bounded scope and source-identity regressions; no historical inference here."""
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('batch5b_cnn',ROOT/'scripts/check_batch5b_cnn.py')
cnn=importlib.util.module_from_spec(spec);spec.loader.exec_module(cnn)


def test_historical_head_and_all_25_sources_are_required(tmp_path,monkeypatch):
    expected={}
    for i in range(25):
        p=tmp_path/f'source_{i}.py';p.write_text(f'value={i}')
        expected[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
    monkeypatch.setattr(cnn.subprocess,'check_output',lambda *a,**k:'wrong-head')
    with pytest.raises(ValueError,match='HEAD'):cnn.require_historical_source(tmp_path,expected)
    monkeypatch.setattr(cnn.subprocess,'check_output',lambda *a,**k:cnn.HISTORICAL_HEAD)
    assert cnn.require_historical_source(tmp_path,expected)==cnn.HISTORICAL_HEAD
    (tmp_path/'source_7.py').write_text('changed')
    with pytest.raises(ValueError,match='25-source'):cnn.require_historical_source(tmp_path,expected)


def test_no_test_or_non_development_ids_even_through_direct_batch_calls():
    data=SimpleNamespace(role='test',ids=['invented-a'],config=SimpleNamespace(n_pixels=8))
    with pytest.raises(PermissionError):cnn.DevelopmentScope(data,['invented-a'])
    data.role='train';data.ids=['invented-a','invented-b']
    calls=[]
    def batch(indices,**kwargs):
        calls.append(indices);return {'flux':np.ones((1,8)),'valid':np.ones((1,8))}
    data.batch=batch
    scope=cnn.DevelopmentScope(data,['invented-a'])
    with pytest.raises(PermissionError):scope.batch([1],masked=False)
    with pytest.raises(PermissionError):scope.batch([0],masked=True)
    assert calls==[]
    scope.batch([0],masked=False);assert len(calls)==1
    with pytest.raises(ValueError):cnn.DevelopmentScope(data,['invented-c'])


@pytest.mark.parametrize('kind',['shape','nan','no_support','nonbinary'])
def test_invalid_batch_rejected_before_inference(kind):
    b={'flux':np.ones((1,8)),'valid':np.ones((1,8))}
    if kind=='shape':b['flux']=np.ones((8,))
    if kind=='nan':b['flux'][0,0]=np.nan
    if kind=='no_support':b['valid'][:]=0
    if kind=='nonbinary':b['valid'][0,0]=.5
    with pytest.raises(ValueError):cnn.validate_batch(b,8)


def test_audit_blocks_direct_and_symlink_test_reads(tmp_path):
    root=tmp_path/'data';(root/'test').mkdir(parents=True);(root/'train').mkdir()
    (root/'test/secret.npy').write_bytes(b'sealed')
    (root/'train/alias.npy').symlink_to(root/'test/secret.npy')
    program='''import importlib.util, pathlib
spec=importlib.util.spec_from_file_location('guard',PATH)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.deny_test_open(ROOT)
for path in [pathlib.Path(ROOT)/'test/secret.npy', pathlib.Path(ROOT)/'train/alias.npy']:
    try:path.read_bytes()
    except PermissionError:continue
    raise AssertionError('Read of forbidden test succeeded')
print('blocked both')
'''.replace('PATH',repr(str(ROOT/'scripts/check_batch5b_cnn.py'))).replace('ROOT',repr(str(root)))
    r=subprocess.run([sys.executable,'-c',program],capture_output=True,text=True)
    assert r.returncode==0,r.stderr
    assert r.stdout.strip()=='blocked both'


def test_audit_blocks_hardlink_and_open_descriptor_aliases(tmp_path):
    import os
    root=tmp_path/'data';(root/'test').mkdir(parents=True);(root/'train').mkdir()
    source=root/'test/invented.npy';source.write_bytes(b'invented forbidden bytes')
    alias=root/'train/hardlink.npy';os.link(source,alias)
    program='''import importlib.util, os, pathlib
spec=importlib.util.spec_from_file_location('guard',PATH)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
fd=os.open(ALIAS,os.O_RDONLY)
m.deny_test_open(ROOT)
for target in (ALIAS,fd):
    try:
        with open(target,'rb',closefd=False) if isinstance(target,int) else open(target,'rb') as stream:
            stream.read()
    except PermissionError:continue
    raise AssertionError('Forbidden inode alias was read')
os.close(fd)
'''.replace('PATH',repr(str(ROOT/'scripts/check_batch5b_cnn.py'))).replace('ROOT',repr(str(root))).replace('ALIAS',repr(str(alias)))
    result=subprocess.run([sys.executable,'-c',program],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
