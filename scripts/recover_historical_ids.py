"""Reconstruct the historical diagnostic IDs; require audit and export identity evidence."""
import argparse
import csv
import json
from pathlib import Path
import sqlite3
import numpy as np
from desi_fm.assets import sha256
from desi_fm.data import extract_mmu_desi_example, preprocess_spectrum, SpectrumPreprocessConfig
from desi_fm.local_data import LocalDESIStore, object_id


def recover(root, index, audit, copilot, output):
    root,index,audit,copilot,output=map(Path,(root,index,audit,copilot,output))
    if output.exists() or output.with_suffix('.evidence.json').exists():raise ValueError('Use a new selection destination')
    store=LocalDESIStore(root,index)
    with sqlite3.connect(index) as db:
        db.row_factory=sqlite3.Row
        allrows=[dict(x) for x in db.execute('SELECT * FROM objects ORDER BY ordinal')]
        rows=[dict(x) for x in db.execute('SELECT * FROM objects WHERE label_eligible=1 ORDER BY ordinal LIMIT 2000 OFFSET 80000')]
    previous=json.loads(audit.read_text())
    assert len(allrows)==len(previous)==100000
    assert [x['object_id'] for x in allrows]==[object_id(x['object_id']) for x in previous], 'Audit IDs differ from raw snapshot'
    assert [x['z'] for x in allrows]==[x['Z'] for x in previous], 'Audit labels differ'
    assert [bool(x['zwarn']) for x in allrows]==[x['ZWARN'] for x in previous], 'Audit warning flags differ'
    assert len(rows)==2000
    ids=[x['object_id'] for x in rows]
    assert len(set(ids))==2000
    fm=Path(__file__).resolve().parents[1]
    csv_evidence=[]
    for path in [fm/'runs/calibration/predictions_v1_heldout_canonical.csv',fm/'runs/desi_80k_classhead_v21/predictions_last.csv',copilot/'eval/heldout_predictions_v21.csv']:
        with path.open() as f:z=np.array([float(x['z_true']) for x in csv.DictReader(f)])
        np.testing.assert_array_equal(z,np.array([x['z'] for x in rows]))
        csv_evidence.append({'path':str(path),'sha256':sha256(path),'label_order_exact':True})
    exports=[]
    with (copilot/'eval/cases/labels.csv').open() as f:labels=list(csv.DictReader(f))
    for label in labels:
        pos=int(label['heldout_index']);path=copilot/'eval/cases'/f"{label['case']}.npz"
        with np.load(path,allow_pickle=False) as source:
            oid=object_id(source['object_id'].item())
            assert oid==ids[pos], f'Export ID mismatch at {pos}'
            raw=store.raw(oid)
            a=extract_mmu_desi_example(raw)
            b=(source['flux'],source['ivar'],source['wavelength'],source['mask'],float(source['z_true']))
            for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
            p=preprocess_spectrum(*a[:4],SpectrumPreprocessConfig());q=preprocess_spectrum(*b[:4],SpectrumPreprocessConfig())
            for key in ('flux','valid','center','scale'):np.testing.assert_array_equal(p[key],q[key])
        exports.append({'case':label['case'],'heldout_index':pos,'object_id':oid,'sha256':sha256(path)})
    selection={'schema':1,'scope':'historical diagnostic, not an independent test','dataset_revision':store.manifest['revision'],
               'snapshot_sha256':sha256(root/'snapshot.json'),'index_sha256':sha256(index),
               'method':'lexical shard order -> finite nonnegative label filter -> eligible positions [80000,82000)',
               'requested':len(ids),'ids':ids,'objects':rows}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(selection,indent=2)+'\n')
    report={'audit_metadata_sha256':sha256(audit),'audit_ID_label_warning_matches':len(allrows),'csv_checks':csv_evidence,
            'export_ID_raw_preprocessing_matches':len(exports),'exports':exports,'selection_sha256':sha256(output),
            'limitation':'IDs reconstructed from fixed snapshot/order/filter and recovered audit; original full-2000 CSV has no IDs or contemporaneous per-row training log. Direct historical raw/ID anchors cover 150 exported objects; remaining 1850 inherit recipe-level correspondence, not individual historical pixel evidence.'}
    output.with_suffix('.evidence.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('audit_ID_label_warning_matches','export_ID_raw_preprocessing_matches','selection_sha256','limitation')},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','index','audit','copilot','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();recover(a.root,a.index,a.audit,a.copilot,a.output)
