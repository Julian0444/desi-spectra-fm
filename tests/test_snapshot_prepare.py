import importlib.util
import io
import json
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from desi_fm.assets import sha256


def test_download_verified_reuse_and_corrupt_rejection(tmp_path,monkeypatch):
    script=Path(__file__).resolve().parents[1]/'scripts/prepare_local_snapshot.py'
    spec=importlib.util.spec_from_file_location('snapshot_prepare',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    source=tmp_path/'source.parquet';pq.write_table(pa.table({'object_id':['1'],'Z':[.1]}),source)
    lock=tmp_path/'lock.json'
    lock.write_text(json.dumps({'repository':'fixture','revision':'immutable','data_dir':'edr_sv3',
         'files':[{'path':'edr_sv3/data.parquet','bytes':source.stat().st_size,'sha256':sha256(source)}]}))
    calls=[]
    def download(url,timeout):
        calls.append(url)
        assert '/resolve/immutable/edr_sv3/data.parquet' in url
        return io.BytesIO(source.read_bytes())
    monkeypatch.setattr(module.urllib.request,'urlopen',download)
    root=tmp_path/'snapshot'
    manifest=module.prepare(root,lock)
    assert manifest['status']=='complete' and manifest['files'][0]['rows']==1
    before=(root/'snapshot.json').read_bytes()
    module.prepare(root,lock)
    assert len(calls)==1 and (root/'snapshot.json').read_bytes()==before
    target=root/'edr_sv3/data.parquet';target.chmod(0o644);target.write_bytes(b'bad')
    with pytest.raises(ValueError,match='identity'):module.prepare(root,lock)
    assert len(calls)==1
