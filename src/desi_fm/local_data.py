"""Verified immutable parquet snapshot and ordered, lossless object-ID access."""
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3

import numpy as np
import pyarrow.parquet as pq
from torch.utils.data import Dataset

from .assets import sha256, verify_file
from .data import SpectrumPreprocessConfig, extract_mmu_desi_example, preprocess_spectrum
from .inputs import SpectrumInputError


def object_id(value):
    # Never pass identifiers through floating point, including numpy floats.
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return str(value)
    if isinstance(value, str) and value:
        return value
    raise ValueError("object_id must be a nonempty string or integer")


def verify_snapshot(root):
    root = Path(root)
    manifest = json.loads((root / "snapshot.json").read_text())
    if manifest.get("status") != "complete":
        raise ValueError("Incomplete snapshot")
    for row in manifest["files"]:
        path = (root / row["path"]).resolve()
        if root.resolve() not in path.parents:
            raise ValueError("Snapshot path escapes root")
        verify_file(path, row["bytes"], row["sha256"])
    return manifest


def spectrum_rows(table):
    """Arrow list slices avoid expanding every spectral pixel to Python objects."""
    spectra = table["spectrum"].combine_chunks()
    fields = {}
    for name in ("flux", "ivar", "lambda", "wavelength", "mask"):
        if name not in spectra.type.names:
            continue
        array = spectra.field(name)
        if hasattr(array, "offsets"):
            offsets = array.offsets.to_numpy()
        else:
            offsets = np.arange(len(array)+1) * array.type.list_size
        fields[name] = (array.values.to_numpy(zero_copy_only=False), offsets)
    metadata = table.drop(["spectrum"]).to_pylist()
    for i, row in enumerate(metadata):
        row["spectrum"] = {name: values[offsets[i]:offsets[i+1]] for name, (values, offsets) in fields.items()}
        yield row


def build_index(root, output, config=None):
    """Scan label and input eligibility separately; reject duplicate IDs."""
    root, output = Path(root), Path(output)
    manifest = verify_snapshot(root)
    config = config or SpectrumPreprocessConfig()
    if output.exists() or output.with_suffix(".json").exists():
        raise ValueError("Use a new index destination")
    output.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(output)
    db.execute("CREATE TABLE objects (object_id TEXT PRIMARY KEY, file TEXT, row_index INTEGER, row_group INTEGER, in_group INTEGER, ordinal INTEGER UNIQUE, z REAL, zwarn INTEGER, label_eligible INTEGER, input_eligible INTEGER, cause TEXT, valid_fraction REAL)")
    counts, causes = Counter(), Counter()
    ordinal = 0
    groups = []
    try:
        for file in manifest["files"]:
            parquet = pq.ParquetFile(root/file["path"])
            group_fields = [name for name in ("TARGETID", "targetid", "source_id", "parent_object_id") if name in parquet.schema_arrow.names]
            groups.extend(group_fields)
            file_row = 0
            for group in range(parquet.num_row_groups):
                table = parquet.read_row_group(group)
                for in_group, example in enumerate(spectrum_rows(table)):
                    oid = object_id(example["object_id"])
                    flux, ivar, wave, mask, z = extract_mmu_desi_example(example)
                    label_ok = bool(np.isfinite(z) and z >= 0)
                    cause, fraction = None, None
                    try:
                        processed = preprocess_spectrum(flux, ivar, wave, mask, config)
                        fraction = processed["coverage"]["model_valid_fraction"]
                        input_ok = True
                    except SpectrumInputError as exc:
                        input_ok = False
                        cause = str(exc)
                        causes[cause] += 1
                    counts["rows"] += 1
                    counts["label_eligible"] += int(label_ok)
                    counts["input_eligible"] += int(input_ok)
                    counts["label_and_input_eligible"] += int(label_ok and input_ok)
                    db.execute("INSERT INTO objects VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (oid, file["path"], file_row, group, in_group, ordinal,
                        z if np.isfinite(z) else None, int(example["ZWARN"]) if "ZWARN" in example else None,
                        int(label_ok), int(input_ok), cause, fraction))
                    file_row += 1
                    ordinal += 1
                db.commit()
            print(json.dumps({"indexed":file["path"], "counts":dict(counts)}), flush=True)
        # Record any relationship keys available; never infer independence from unique IDs.
        label_rejections = {"nonfinite": db.execute("SELECT count(*) FROM objects WHERE z IS NULL").fetchone()[0],
                            "negative": db.execute("SELECT count(*) FROM objects WHERE z < 0").fetchone()[0]}
        input_exclusions = [dict(zip(("object_id", "cause"), row)) for row in db.execute("SELECT object_id,cause FROM objects WHERE input_eligible=0 ORDER BY ordinal")]
        db.close()
        report = {"status":"complete", "snapshot_sha256":sha256(root/"snapshot.json"), "index_sha256":sha256(output),
                  "counts":dict(counts), "input_rejection_causes":dict(causes), "preprocess":asdict(config),
                  "contract":"spectra-input-v2", "id_storage":"TEXT; integer/string conversion without floating point",
                  "duplicate_ids":0, "label_rejections":label_rejections, "input_exclusions":input_exclusions,
                  "relationship_fields":sorted(set(groups)),
                  "relationships":"No explicit relationship identifier available" if not groups else "Relationship fields require grouping before future splits"}
        output.with_suffix(".json").write_text(json.dumps(report,indent=2)+"\n")
        return report
    except BaseException:
        db.close()
        # Retain partial index for diagnosis, but it cannot be opened without a complete manifest.
        raise


class LocalDESIStore:
    def __init__(self, root, index, verify=True):
        self.root, self.index = Path(root), Path(index)
        self.manifest = verify_snapshot(root) if verify else json.loads((self.root/"snapshot.json").read_text())
        report = json.loads(self.index.with_suffix(".json").read_text())
        if report["status"] != "complete" or report["snapshot_sha256"] != sha256(self.root/"snapshot.json"):
            raise ValueError("Index does not belong to this complete snapshot")
        if report["index_sha256"] != sha256(self.index):
            raise ValueError("Index identity mismatch")
        self._cache_key, self._cache = None, None

    def rows_for_ids(self, ids):
        ids = [object_id(x) for x in ids]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate requested IDs")
        with sqlite3.connect(f"file:{self.index}?mode=ro", uri=True) as db:
            db.row_factory = sqlite3.Row
            rows = []
            for oid in ids:
                row = db.execute("SELECT * FROM objects WHERE object_id=?", (oid,)).fetchone()
                if row is None:
                    raise KeyError(f"Missing object_id: {oid}")
                rows.append(dict(row))
        return rows

    def raw(self, oid):
        row = self.rows_for_ids([oid])[0]
        key = (row["file"], row["row_group"])
        if key != self._cache_key:
            table = pq.ParquetFile(self.root/row["file"]).read_row_group(row["row_group"])
            self._cache = list(spectrum_rows(table))
            self._cache_key = key
        example = self._cache[row["in_group"]]
        if object_id(example["object_id"]) != row["object_id"]:
            raise ValueError("Raw row does not match requested ID")
        return example


class LocalDESISpectra(Dataset):
    """Map-style membership is fixed in caller order before workers are started."""
    def __init__(self, root, index, ids, config=None):
        self.store = LocalDESIStore(root, index)
        self.rows = self.store.rows_for_ids(ids)
        self.config = config or SpectrumPreprocessConfig()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, position):
        oid = self.rows[position]["object_id"]
        example = self.store.raw(oid)
        flux, ivar, wave, mask, z = extract_mmu_desi_example(example)
        processed = preprocess_spectrum(flux, ivar, wave, mask, self.config)
        return {"object_id":oid, "flux":processed["flux"], "valid":processed["valid"], "z":np.float32(z)}
