"""Materialize metadata splits and unnormalized grid matrices, entirely offline."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import shutil
import sqlite3
import time

import numpy as np
import pyarrow.parquet as pq

from desi_fm.assets import sha256
from desi_fm.data import SpectrumPreprocessConfig, extract_mmu_desi_example, resample_spectrum
from desi_fm.local_data import LocalDESIStore, spectrum_rows
from desi_fm.experiment.artifacts import write_json, read_json, file_record, source_identity, environment
from desi_fm.experiment.data import GRID_CONTRACT
from desi_fm.experiment.splits import make_splits
from desi_fm.inputs import CONTRACT_VERSION


def prepare(args):
    started = time.perf_counter()
    root, index, output = Path(args.root), Path(args.index), Path(args.output)
    if output.exists():
        raise ValueError('Use a new derivative destination')
    store = LocalDESIStore(root, index)  # full raw + index verification before use
    if store.index_contract != CONTRACT_VERSION:
        raise ValueError('Index eligibility uses a different input contract; use registered 4B code for historical derivatives')
    historical = read_json(args.historical)
    if historical['index_sha256'] != sha256(index) or historical['snapshot_sha256'] != sha256(root/'snapshot.json'):
        raise ValueError('Historical IDs belong to different data')
    demos, unidentified = [], []
    for directory in args.examples:
        for path in sorted(Path(directory).glob('*.npz')):
            with np.load(path, allow_pickle=False) as data:
                if 'object_id' in data:
                    demos.append({'path': str(path), 'object_id': str(data['object_id'].item()), **file_record(path)})
                else:
                    unidentified.append({'path': str(path), 'keys': data.files, 'flux_shape': list(data['flux'].shape), **file_record(path)})
    protected = sorted(set(historical['ids']) | {d['object_id'] for d in demos})
    with sqlite3.connect(f'file:{index}?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        all_rows = [dict(r) for r in db.execute('SELECT * FROM objects ORDER BY ordinal')]
    eligible = [r for r in all_rows if r['label_eligible'] and r['input_eligible']]
    excluded = [{'object_id': r['object_id'], 'label_eligible': bool(r['label_eligible']),
                 'input_eligible': bool(r['input_eligible']), 'input_cause': r['cause'],
                 'label_cause': None if r['label_eligible'] else ('nonfinite' if r['z'] is None else 'negative')}
                for r in all_rows if not (r['label_eligible'] and r['input_eligible'])]
    # Read metadata only. Equality of photometry is inspected, not promoted to a
    # repeated-source identifier. TARGETID encodes spatial bricks; that locality
    # is not a repeated observation and does not establish physical independence.
    schemas, seen, duplicate_vectors, degenerate = [], {}, [], 0
    for file in store.manifest['files']:
        pf = pq.ParquetFile(root/file['path'])
        columns = pf.schema_arrow.names
        schemas.append({'file': file['path'], 'columns': columns, 'schema': str(pf.schema_arrow)})
        explicit = set(columns) & {'TARGETID', 'targetid', 'source_id', 'parent_object_id', 'RA', 'DEC', 'ra', 'dec'}
        if explicit:
            raise ValueError(f'New relationship metadata requires review before splitting: {explicit}')
        photo = [k for k in columns if 'FLUX' in k or k == 'EBV']
        for row in pf.read(columns=['object_id', *photo]).to_pylist():
            key = tuple(row[k] for k in photo)
            if not np.isfinite(key).all() or not any(key):
                degenerate += 1
                continue
            if key in seen:
                duplicate_vectors.append({'ids': [seen[key], row['object_id']], 'values': dict(zip(photo, key))})
            else:
                seen[key] = row['object_id']
    config = SpectrumPreprocessConfig()
    expected_bytes = len(eligible) * config.n_pixels * 5
    free = shutil.disk_usage(output.parent if output.parent.exists() else root.parent).free
    if free < expected_bytes + 8_000_000_000:
        raise ValueError('Insufficient derivative/checkpoint reserve')
    output.mkdir(parents=True)
    # This file is durable before the split generator is called.
    write_json(output/'decision-before-split.json', {
        'protected_policy': 'all historical and identifiable demo IDs in development; propagate to related groups',
        'historical_manifest': file_record(args.historical), 'protected_ids': protected,
        'demos': demos, 'unidentified_examples': unidentified,
        'grouping': 'No explicit relationship or coordinate fields; object identity only; physical independence unverified',
        'schemas': schemas, 'photometry_duplicate_vectors': duplicate_vectors, 'degenerate_photometry_rows': degenerate,
        'seed': 20261009, 'expected_payload_bytes': expected_bytes, 'free_before_bytes': free,
        'memory_estimate_bytes': 1_000_000_000, 'time_allowance_estimate_seconds': 600,
        'source': source_identity(), 'environment': environment()})
    splits, split_report = make_splits(eligible, protected)
    write_json(output/'splits.json', split_report)
    write_json(output/'exclusions.json', excluded)
    manifest = {'status': 'incomplete', 'contract': GRID_CONTRACT, 'preprocess': asdict(config),
                'snapshot': file_record(root/'snapshot.json'), 'index': file_record(index),
                'split_report': split_report, 'splits': {}, 'source': source_identity(), 'environment': environment()}
    write_json(output/'manifest.json', manifest)
    lookup, arrays = {}, {}
    for role, rows in splits.items():
        directory = output/role
        directory.mkdir()
        ids = [r['object_id'] for r in rows]
        write_json(directory/'ids.json', ids)
        np.savez(directory/'metadata.npz', z=np.array([r['z'] for r in rows], dtype=np.float64),
                 zwarn=np.array([r['zwarn'] for r in rows], dtype=np.uint8), groups=np.array([r['group_id'] for r in rows]))
        arrays[role] = (np.lib.format.open_memmap(directory/'flux.npy', mode='w+', dtype='float32', shape=(len(rows), config.n_pixels)),
                        np.lib.format.open_memmap(directory/'valid.npy', mode='w+', dtype='uint8', shape=(len(rows), config.n_pixels)))
        lookup.update({oid: (role, i) for i, oid in enumerate(ids)})
    seen_ids = set()
    stage = time.perf_counter()
    for file in store.manifest['files']:
        parquet = pq.ParquetFile(root/file['path'])
        for group in range(parquet.num_row_groups):
            for example in spectrum_rows(parquet.read_row_group(group)):
                oid = str(example['object_id'])
                if oid not in lookup:
                    continue
                if oid in seen_ids:
                    raise ValueError('Duplicate raw identity')
                seen_ids.add(oid)
                flux, ivar, wave, mask, _ = extract_mmu_desi_example(example)
                grid = resample_spectrum(flux, ivar, wave, mask, config)
                role, position = lookup[oid]
                arrays[role][0][position] = grid['flux']
                arrays[role][1][position] = grid['valid']
        print(f"materialized {file['path']}: {len(seen_ids)} eligible", flush=True)
    if seen_ids != set(lookup):
        raise ValueError('Missing raw objects')
    for role in arrays:
        for array in arrays[role]:
            array.flush()
        manifest['splits'][role] = {'n': len(splits[role]), 'files': {p.name: file_record(p) for p in sorted((output/role).iterdir())}}
    manifest.update(status='complete', materialization_seconds=time.perf_counter()-stage,
                    total_seconds=time.perf_counter()-started, files={p.name: file_record(p) for p in (output/'splits.json', output/'exclusions.json', output/'decision-before-split.json')})
    write_json(output/'manifest.json', manifest)
    print({'output': str(output), 'counts': split_report['actual'], 'total_seconds': manifest['total_seconds']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--index', required=True)
    parser.add_argument('--historical', required=True)
    parser.add_argument('--examples', action='append', default=[])
    parser.add_argument('--output', required=True)
    prepare(parser.parse_args())
