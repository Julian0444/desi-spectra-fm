"""Shared deterministic fixtures; consumed by cross-repository integration too."""
import numpy as np

def spectra():
    w = np.geomspace(3600, 9800, 64).astype(np.float32)
    f = np.sin(w/500).astype(np.float32)
    m = np.zeros(64, bool); m[20:28] = True
    bad = f.copy(); bad[20:24] = np.nan; bad[24:28] = np.inf
    return {"valid": dict(flux=f, wavelength=w),
            "nonfinite": dict(flux=bad, wavelength=w),
            "masked": dict(flux=f, wavelength=w, mask=m),
            "gap": dict(flux=f, wavelength=w, mask=m),
            "partial": dict(flux=f, wavelength=w+1000),
            "reordered": dict(flux=f[::-1], wavelength=w[::-1]),
            "no_coverage": dict(flux=f, wavelength=w+20000),
            "bad_mask": dict(flux=f, wavelength=w, mask=[False]*3),
            "bad_ivar": dict(flux=f, wavelength=w, ivar=[[1]*64]*2),
            "empty": dict(flux=[], wavelength=[]),
            "strings": dict(flux=["1"]*64, wavelength=w),
            "container": dict(flux={"flux":f}, wavelength=w)}
