"""Check local ID identity, HF parquet decoding parity and complete worker membership."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from desi_fm.assets import sha256
from desi_fm.data import extract_mmu_desi_example, preprocess_spectrum, SpectrumPreprocessConfig
from desi_fm.local_data import LocalDESIStore, LocalDESISpectra, object_id
from desi_fm.inputs import CONTRACT_VERSION


def check(root,index,selection,output):
    from datasets import Dataset
    root,index,selection,output=map(Path,(root,index,selection,output))
    if output.exists():raise ValueError('Use a new parity evidence destination')
    manifest=json.loads(selection.read_text());ids=manifest['ids']
    store=LocalDESIStore(root,index)
    selected=[ids[i] for i in (0,1,10,12,27,40,500,1000,1999)]
    locations=store.rows_for_ids(selected)
    assert len({x['file'] for x in locations})==1
    reference=Dataset.from_parquet(str(root/locations[0]['file']))
    reference_ids=[]
    for location in locations:
        oid=location['object_id'];raw=store.raw(oid);ref=reference[location['row_index']]
        assert object_id(ref['object_id'])==oid
        a=extract_mmu_desi_example(raw);b=extract_mmu_desi_example(ref)
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
        p=preprocess_spectrum(*a[:4],SpectrumPreprocessConfig());q=preprocess_spectrum(*b[:4],SpectrumPreprocessConfig())
        for key in ('flux','valid','center','scale'):np.testing.assert_array_equal(p[key],q[key])
        reference_ids.append(oid)
    del reference
    signatures=[]
    for workers in (0,2):
        dataset=LocalDESISpectra(root,index,ids)
        loader=DataLoader(dataset,batch_size=32,num_workers=workers)
        got=[];digest=hashlib.sha256()
        for batch in loader:
            got.extend(batch['object_id'])
            digest.update(batch['flux'].numpy().tobytes());digest.update(batch['valid'].numpy().tobytes());digest.update(batch['z'].numpy().tobytes())
        assert got==ids and len(got)==len(set(got))==2000
        signatures.append({'workers':workers,'n':len(got),'arrays_sha256':digest.hexdigest()})
    assert signatures[0]['arrays_sha256']==signatures[1]['arrays_sha256']
    report={'selection_sha256':sha256(selection),'reference':f'HF Dataset.from_parquet, verified fixed-shard bytes; shared MMU extraction and {CONTRACT_VERSION}',
            'raw_and_preprocess_exact_ID_matches':reference_ids,'workers':signatures,'membership_order_arrays_exact':True,'remote_streaming_required_for_evaluation':False}
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','index','selection','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();check(a.root,a.index,a.selection,a.output)
