"""Build an object-ID index and contract eligibility inventory from local data."""
import argparse
from desi_fm.local_data import build_index

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    build_index(a.root, a.output)
