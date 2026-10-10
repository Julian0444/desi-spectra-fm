import json
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from torch.utils.data import DataLoader
from desi_fm.assets import sha256
from desi_fm.data import SpectrumPreprocessConfig, extract_mmu_desi_example, preprocess_spectrum
from desi_fm.local_data import LocalDESIStore, LocalDESISpectra, build_index, object_id


@pytest.fixture
def local_snapshot(tmp_path):
    root = tmp_path/'raw';root.mkdir()
    config = SpectrumPreprocessConfig(n_pixels=64)
    w = np.geomspace(3600,9800,64).astype('float32')
    rows=[]
    for i in range(5):
        rows.append({'object_id':str(2**60+i),'Z':float(i)/10 if i!=4 else -1., 'ZWARN':i%2==0,
                     'spectrum':{'flux':np.sin(w/500).tolist(),'ivar':np.ones(64).tolist(),'lambda':w.tolist(),'mask':[i==3]*64}})
    path=root/'data.parquet';pq.write_table(pa.Table.from_pylist(rows),path,row_group_size=2)
    (root/'snapshot.json').write_text(json.dumps({'status':'complete','revision':'fixture','files':[{'path':path.name,'bytes':path.stat().st_size,'sha256':sha256(path)}]}))
    index=tmp_path/'index.sqlite';report=build_index(root,index,config)
    return root,index,config,rows,report


def test_order_raw_identity_contract_and_eligibility(local_snapshot):
    root,index,config,rows,report=local_snapshot
    assert report['counts']=={'rows':5,'label_eligible':4,'input_eligible':4,'label_and_input_eligible':3}
    store=LocalDESIStore(root,index)
    ids=[rows[i]['object_id'] for i in [2,0,1]]
    assert [x['object_id'] for x in store.rows_for_ids(ids)]==ids
    for i in [2,0,1]:
        raw=store.raw(rows[i]['object_id'])
        a=extract_mmu_desi_example(raw);b=extract_mmu_desi_example(rows[i])
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
        x=preprocess_spectrum(*a[:4],config);y=preprocess_spectrum(*b[:4],config)
        np.testing.assert_array_equal(x['flux'],y['flux'])
        np.testing.assert_array_equal(x['valid'],y['valid'])
    with pytest.raises(KeyError):store.rows_for_ids(['missing'])
    with pytest.raises(ValueError,match='Duplicate'):store.rows_for_ids([ids[0],ids[0]])
    with pytest.raises(ValueError):object_id(float(2**60))


@pytest.mark.parametrize('workers',[0,2])
def test_map_dataset_workers_preserve_membership_and_order(local_snapshot,workers):
    root,index,config,rows,_=local_snapshot
    ids=[rows[i]['object_id'] for i in [2,0,1]]
    dataset=LocalDESISpectra(root,index,ids,config)
    loader=DataLoader(dataset,batch_size=2,num_workers=workers)
    got=[oid for batch in loader for oid in batch['object_id']]
    assert got==ids and len(got)==len(set(got))


def test_reject_tampering_and_overwrite(local_snapshot):
    root,index,config,_,_=local_snapshot
    with pytest.raises(ValueError,match='new index'):build_index(root,index,config)
    with (root/'data.parquet').open('ab') as f:f.write(b'x')
    with pytest.raises(ValueError,match='identity'):LocalDESIStore(root,index)


def test_duplicate_raw_ids_leave_index_incomplete(local_snapshot,tmp_path):
    import sqlite3
    root,index,config,rows,_=local_snapshot
    rows[1]['object_id']=rows[0]['object_id']
    path=root/'data.parquet';pq.write_table(pa.Table.from_pylist(rows),path,row_group_size=2)
    manifest=json.loads((root/'snapshot.json').read_text())
    manifest['files'][0].update(bytes=path.stat().st_size,sha256=sha256(path))
    (root/'snapshot.json').write_text(json.dumps(manifest))
    bad=tmp_path/'duplicate.sqlite'
    with pytest.raises(sqlite3.IntegrityError):build_index(root,bad,config)
    assert not bad.with_suffix('.json').exists()
