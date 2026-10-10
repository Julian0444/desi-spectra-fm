"""Download only the seven locked EDR parquet files; verify before reuse."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request
from desi_fm.assets import verify_file, sha256


def prepare(root, lock_path):
    root, lock_path = Path(root), Path(lock_path)
    lock = json.loads(lock_path.read_text())
    total = sum(x['bytes'] for x in lock['files'])
    root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(root).free
    missing = sum(x['bytes'] for x in lock['files'] if not (root/x['path']).exists())
    if free < missing + 4_000_000_000:
        raise ValueError('Insufficient space: raw snapshot plus 4 GB derived/results reserve required')
    records = []
    start = time.perf_counter()
    import pyarrow.parquet as pq
    for row in lock['files']:
        target = root/row['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            partial = target.with_suffix('.download')
            if partial.exists():
                raise ValueError(f'Partial download requires explicit recovery: {partial}')
            url = f"https://huggingface.co/datasets/{lock['repository']}/resolve/{lock['revision']}/{row['path']}"
            with urllib.request.urlopen(url, timeout=60) as response, partial.open('xb') as output:
                shutil.copyfileobj(response, output, length=4*1024*1024)
            verify_file(partial, row['bytes'], row['sha256'])
            os.rename(partial, target)
        verify_file(target, row['bytes'], row['sha256'])
        target.chmod(0o444)
        parquet = pq.ParquetFile(target)
        records.append({**row, 'rows': parquet.metadata.num_rows, 'row_groups': parquet.metadata.num_row_groups,
                        'schema': str(parquet.schema_arrow)})
        print(json.dumps({'verified':row['path'],'rows':parquet.metadata.num_rows,'elapsed_s':time.perf_counter()-start}), flush=True)
    manifest = {**lock, 'files': records, 'lock_sha256':sha256(lock_path), 'raw_bytes':total,
                'free_before_bytes':free, 'derived_results_reserve_bytes':4_000_000_000,
                'elapsed_s':time.perf_counter()-start, 'status':'complete'}
    out = root/'snapshot.json'
    if out.exists():
        old = json.loads(out.read_text())
        if old['files'] != records or old['revision'] != lock['revision']:
            raise ValueError('Existing immutable manifest differs')
    else:
        out.write_text(json.dumps(manifest,indent=2)+'\n')
        out.chmod(0o444)
    return manifest

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--lock',type=Path,default=Path(__file__).resolve().parents[1]/'docs/evidence/desi-snapshot-lock.json')
    args=parser.parse_args();prepare(args.root,args.lock)
