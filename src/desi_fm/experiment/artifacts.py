"""Small manifests, durable atomic files, and identities used by batch 4."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile

from ..assets import sha256


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def atomic_write(path, writer):
    path = Path(path)
    fd, tmp = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_json(path, value):
    atomic_write(path, lambda stream: stream.write((json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()))


def read_json(path):
    return json.loads(Path(path).read_text())


def file_record(path):
    path = Path(path)
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def verify_record(path, record):
    if file_record(path) != record:
        raise ValueError(f"Artifact identity mismatch: {path}")


def source_identity():
    root = Path(__file__).resolve().parents[3]
    paths = sorted((root / "src/desi_fm").rglob("*.py"))
    paths += [root/'scripts'/name for name in ('batch4.py','prepare_batch4.py','check_batch4_data.py','check_batch4_resume.py') if (root/'scripts'/name).exists()]
    hashes = {str(p.relative_to(root)): sha256(p) for p in paths}
    return {"head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "files": hashes, "sha256": digest(hashes)}


def environment():
    packages = {}
    for name in ("numpy", "torch", "pyarrow", "scipy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}
