"""Input contract v2: real arrays, declared units, quality before sanitizing."""
from dataclasses import dataclass
import zipfile
import numpy as np

CONTRACT_VERSION = "spectra-input-v2"
MIN_VALID_PIXELS = 8  # numerical requirement, not a calibrated quality cut

class SpectrumInputError(ValueError):
    """Malformed or unusable spectrum supplied by a caller."""


def real_array(value, name):
    try:
        a = np.asarray(value)
    except (ValueError, TypeError) as exc:
        raise SpectrumInputError(f"{name}: expected a rectangular real numeric array") from exc
    if a.dtype.kind not in "iuf":
        raise SpectrumInputError(f"{name}: expected real numbers (no strings, objects, bool or complex)")
    with np.errstate(over="ignore", invalid="ignore"):
        return a.astype(np.float32)


def mask_array(value):
    try:
        a = np.asarray(value)
    except (ValueError, TypeError) as exc:
        raise SpectrumInputError("mask: expected a rectangular boolean or integer array") from exc
    if a.dtype.kind not in "biu":
        raise SpectrumInputError("mask: expected bool or integer bad-pixel flags; nonzero means bad")
    return a != 0


def bounded_real(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise SpectrumInputError(f"{name}: expected a real number in [{low}, {high}]")
    if not np.isfinite(value) or not low <= value <= high:
        raise SpectrumInputError(f"{name}: expected a finite value in [{low}, {high}]")
    return float(value)


def positive_int(value, name, maximum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not 1 <= value <= maximum:
        raise SpectrumInputError(f"{name}: expected an integer in [1, {maximum}]")
    return int(value)


@dataclass
class SpectrumBatch:
    flux: np.ndarray
    wavelength: np.ndarray
    ivar: np.ndarray
    mask: np.ndarray

    def single(self):
        if self.flux.shape[0] != 1:
            raise SpectrumInputError("This interface requires exactly one spectrum; use the batch Python/REST/CLI interface")
        return self.flux[0], self.wavelength[0], self.ivar[0], self.mask[0]


def parse_arrays(flux, wavelength, ivar=None, mask=None, *, wavelength_unit="angstrom"):
    f = real_array(flux, "flux")
    if f.ndim not in (1, 2) or f.size == 0:
        raise SpectrumInputError("flux: expected nonempty (P,) or (N,P) array")
    if f.ndim == 1:
        f = f[None, :]
    n, p = f.shape
    if p < MIN_VALID_PIXELS:
        raise SpectrumInputError(f"At least {MIN_VALID_PIXELS} input pixels required")
    def aligned(value, name, parser):
        a = parser(value)
        if a.shape == (p,):
            return np.broadcast_to(a, (n, p)).copy()
        if a.shape != (n, p):
            raise SpectrumInputError(f"{name}: expected ({p},) shared or {(n, p)}, got {a.shape}")
        return a.copy()
    w = aligned(wavelength, "wavelength", lambda x: real_array(x, "wavelength"))
    if wavelength_unit not in ("angstrom", "nm"):
        raise SpectrumInputError("wavelength_unit: use 'angstrom' or 'nm'; units are never guessed")
    if wavelength_unit == "nm":
        w *= 10
    v = np.ones_like(f) if ivar is None else aligned(ivar, "ivar", lambda x: real_array(x, "ivar"))
    m = np.zeros_like(f, dtype=bool) if mask is None else aligned(mask, "mask", mask_array)
    for i in range(n):
        finite_w = np.isfinite(w[i]) & (w[i] > 0)
        if np.unique(w[i, finite_w]).size != finite_w.sum():
            raise SpectrumInputError("wavelength: duplicate finite positive coordinates are ambiguous")
        good = finite_w & np.isfinite(f[i]) & np.isfinite(v[i]) & (v[i] > 0) & ~m[i]
        if good.sum() < MIN_VALID_PIXELS:
            raise SpectrumInputError(f"Spectrum {i}: fewer than {MIN_VALID_PIXELS} usable input pixels")
    return SpectrumBatch(f, w, v, m)


def read_npz(source):
    """Read only scientific inputs; ignore labels and arbitrary metadata."""
    try:
        data = np.load(source, allow_pickle=False)
        if not isinstance(data, np.lib.npyio.NpzFile):
            raise SpectrumInputError("Expected a .npz archive")
        with data:
            if not {"flux", "wavelength"} <= set(data.files):
                raise SpectrumInputError("The .npz must contain flux and wavelength")
            unit = "angstrom"
            if "wavelength_unit" in data.files:
                a = data["wavelength_unit"]
                if a.shape != () or a.dtype.kind not in "US":
                    raise SpectrumInputError("wavelength_unit: expected a scalar string")
                unit = a.item().decode("ascii") if a.dtype.kind == "S" else str(a.item())
            return parse_arrays(data["flux"], data["wavelength"],
                                data["ivar"] if "ivar" in data.files else None,
                                data["mask"] if "mask" in data.files else None,
                                wavelength_unit=unit)
    except (ValueError, TypeError, OSError, EOFError, zipfile.BadZipFile) as exc:
        if isinstance(exc, SpectrumInputError):
            raise
        raise SpectrumInputError("Could not read scientific inputs as a .npz archive: " + str(exc)) from exc
