"""Immutable published asset identities, checked against batch-2 local bytes.

The index and metadata share a release; this does not establish which encoder
or preprocessing generated that historical index.
"""
from pathlib import Path
import hashlib

MODEL_REPO = "jirustaroure/desi-spectra-fm"
MODEL_REVISION = "40ff2ed09e5e1584303052e1a5dfe218a4a396b3"
ASSETS = {
    "checkpoint_last.pt": (103964688, "40247cbe0e706a66084d001b36424081554c26d3eb75c1f8b0ebec89cf5afe5a"),
    "faiss/spectra.faiss": (30720045, "fc13007f951e2d1c1dc7e30338f6de4203b65d5c9c44fddc1d39ce4da921968b"),
    "faiss/spectra_meta.npz": (60256, "83e4bdc035328618636b1e86ddad49367df8b2d70dd9ac90a2bb8bd4f503e9ee"),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path, size, digest):
    path = Path(path)
    if path.stat().st_size != size or sha256(path) != digest:
        raise ValueError(f"Asset identity mismatch: {path}")
    return str(path)


def download_asset(filename):
    """No mutable revision or unverified remote fallback is permitted."""
    from huggingface_hub import hf_hub_download
    size, digest = ASSETS[filename]
    path = hf_hub_download(MODEL_REPO, filename, revision=MODEL_REVISION)
    return verify_file(path, size, digest)
