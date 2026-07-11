# 07 · spectra-copilot: repo nuevo + herramientas

> **Bloque:** Nivel 2 · **Tiempo:** 3–4 h · **Depende de:** 04 — o de **ningún plan** si exportás `DESI_FM_CKPT` apuntando a un checkpoint local · **Entregable:** `tools.py` testeado + CLI de demo que imprime JSON

## Objetivo

Crear el segundo repo (`spectra-copilot`) con las herramientas determinísticas que después usarán el agente (08) y el servidor MCP (09). La estrella es `identify_spectral_lines`: le da al agente una forma de **verificar** físicamente la predicción del modelo, no solo repetirla.

Regla de oro: las tools devuelven **JSON compacto** (conclusiones), jamás arrays de 7081 floats. Los plots van a disco y se devuelve la ruta.

## Pasos

### 1. Esqueleto del repo

```bash
mkdir -p ~/proyectos/spectra-copilot/{copilot,eval/cases,examples,docs/img}
cd ~/proyectos/spectra-copilot && git init -b main
```

`pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "spectra-copilot"
version = "0.1.0"
description = "LLM agent that analyzes astronomical spectra using the desi-fm foundation model as a tool"
requires-python = ">=3.10"
dependencies = [
  "numpy", "scipy", "torch", "huggingface_hub",
  "desi-fm @ git+https://github.com/TU_USUARIO/desi-spectra-fm",
  "anthropic>=0.5", "mcp[cli]",
]

[project.optional-dependencies]
ui = ["gradio", "matplotlib"]
dev = ["pytest"]
```

Copiar 2–3 `.npz` de ejemplo desde el repo principal a `examples/`.

### 2. `copilot/tools.py`

Estructura (implementar cada `_impl` devolviendo dicts):

```python
import json
import numpy as np
from pathlib import Path
from functools import lru_cache
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import torch
from huggingface_hub import hf_hub_download
from desi_fm.predict import load_model_from_checkpoint, predict_spectrum

# catálogo rest-frame (Å) — galaxias + cuásares
LINES = {
    "Lyα 1216": 1215.7, "CIV 1549": 1549.1, "CIII] 1909": 1908.7,
    "MgII 2799": 2798.8, "[OII] 3727": 3727.1, "Ca K 3934": 3933.7,
    "Ca H 3969": 3968.5, "Hδ 4102": 4101.7, "Hγ 4340": 4340.5,
    "Hβ 4861": 4861.3, "[OIII] 4959": 4958.9, "[OIII] 5007": 5006.8,
    "Hα 6563": 6562.8, "[NII] 6583": 6583.5, "[SII] 6716": 6716.4,
}

import os

@lru_cache(maxsize=1)
def _model():
    # DESI_FM_CKPT permite trabajar sin depender del Hub (clave para el sprint):
    #   export DESI_FM_CKPT="/ruta/al/repo/runs/desi_150k_classhead/checkpoint_last.pt"
    ckpt = os.environ.get("DESI_FM_CKPT") or hf_hub_download(
        "TU_USUARIO/desi-spectra-fm", "checkpoint_last.pt")
    return load_model_from_checkpoint(ckpt, torch.device("cpu"))

def _load(npz_path):
    d = np.load(npz_path)
    return d["flux"].astype(np.float32).ravel(), d["wavelength"].astype(np.float32).ravel()

def predict_redshift_impl(npz_path: str) -> dict:
    flux, wave = _load(npz_path)
    r = predict_spectrum(flux=flux, wavelength=wave, model=_model())
    out = {"z_pred": round(float(r["z_pred"]), 4)}
    # si el checkpoint es v2 (cabeza de clasificación), exponer confianza:
    # out["z_pred_map"], out["confidence"] = ...
    return out

def reconstruct_spectrum_impl(npz_path: str, mask_ratio: float = 0.5) -> dict:
    flux, wave = _load(npz_path)
    r = predict_spectrum(flux=flux, wavelength=wave, model=_model(), mask_ratio=mask_ratio)
    masked = r["spectrum_mask"]
    return {"mask_ratio": mask_ratio, "n_tokens_masked": int(masked.sum()),
            "z_pred_under_masking": round(float(r["z_pred"]), 4)}

def identify_spectral_lines_impl(npz_path: str, z: float, tol_angstrom: float = 12.0) -> dict:
    flux, wave = _load(npz_path)
    smooth = gaussian_filter1d(flux.astype(float), sigma=3.0)
    prominence = 0.8 * float(np.std(flux - smooth))
    peaks, _ = find_peaks(smooth, prominence=prominence, distance=10)
    peak_wl = wave[peaks]
    expected, matched = [], []
    for name, lam in LINES.items():
        obs = lam * (1.0 + z)
        if not (wave[0] <= obs <= wave[-1]):
            continue
        entry = {"line": name, "lambda_expected": round(obs, 1)}
        if peak_wl.size:
            d = float(np.abs(peak_wl - obs).min())
            if d <= tol_angstrom:
                entry["matched_peak_at"] = round(float(peak_wl[np.abs(peak_wl - obs).argmin()]), 1)
                entry["delta"] = round(d, 1)
                matched.append(entry)
        expected.append(entry)
    frac = len(matched) / max(len(expected), 1)
    return {"z_tested": z, "n_expected_in_coverage": len(expected),
            "n_matched": len(matched), "match_fraction": round(frac, 2),
            "matched_lines": matched,
            "verdict": "consistent" if frac >= 0.4 else "weak_or_inconsistent"}
```

Detalles que importan:
- `identify_spectral_lines` detecta **picos de emisión**; para espectros dominados por absorción (Ca H&K) el `verdict` puede ser débil aun con z correcto — documentarlo en el docstring (el agente lo tendrá en cuenta).
- La tolerancia de 12 Å es ~5 píxeles DESI; ajustable por parámetro.

### 3. CLI de demo (`copilot/__main__.py`)

```python
import json, sys
from copilot import tools

npz = sys.argv[1]
z = tools.predict_redshift_impl(npz)["z_pred"]
print(json.dumps({
    "predict_redshift": tools.predict_redshift_impl(npz),
    "identify_spectral_lines": tools.identify_spectral_lines_impl(npz, z),
}, indent=2))
```

### 4. Tests (`tests/test_tools.py`, completo)

```python
import numpy as np
import pytest

from copilot import tools


def _synth(z, seed=1, n=6000):
    """Espectro sintético con líneas fuertes en z conocido."""
    rng = np.random.default_rng(seed)
    w = np.linspace(3600.0, 9800.0, n).astype(np.float32)
    f = 0.5 + 0.15 * np.sin(w / 700.0) + rng.normal(0, 0.05, n)
    for lam in (3727.1, 4861.3, 5006.8, 6562.8):
        obs = lam * (1 + z)
        if w[0] <= obs <= w[-1]:
            f += 0.9 * np.exp(-0.5 * ((w - obs) / 5.0) ** 2)
    return w, f.astype(np.float32)


@pytest.fixture()
def case(tmp_path):
    w, f = _synth(0.42)
    p = tmp_path / "s.npz"
    np.savez(p, flux=f, wavelength=w)
    return str(p)


def test_lines_match_at_true_z(case):
    r = tools.identify_spectral_lines_impl(case, 0.42)
    assert r["match_fraction"] >= 0.5
    assert r["verdict"] == "consistent"


def test_lines_fail_at_wrong_z(case):
    good = tools.identify_spectral_lines_impl(case, 0.42)["match_fraction"]
    bad = tools.identify_spectral_lines_impl(case, 0.85)["match_fraction"]
    assert bad < good          # esto es lo que el agente explota para validar


def test_predict_redshift_range(case):
    # baja el checkpoint la 1ª vez (~104 MB) — o export DESI_FM_CKPT=<ruta local>
    r = tools.predict_redshift_impl(case)
    assert 0.0 <= r["z_pred"] <= 6.0


def test_reconstruct_reports_masking(case):
    r = tools.reconstruct_spectrum_impl(case, mask_ratio=0.5)
    assert 100 <= r["n_tokens_masked"] <= 180   # ~50 % de 273
```

```bash
pip install -e ".[dev]"
export DESI_FM_CKPT="/ruta/al/repo/runs/desi_150k_classhead/checkpoint_last.pt"  # opcional, evita descargas
pytest -q
python -m copilot examples/galaxy_z042.npz   # imprime el JSON combinado
```

### 5. Subir

```bash
gh repo create spectra-copilot --public --source . --push
```

README corto por ahora (una frase + el JSON de ejemplo); el README bueno llega con el agente (08) y el GIF.

## Definición de hecho

- [ ] `pytest` verde (≥3 tests de tools).
- [ ] `python -m copilot examples/galaxy_z042.npz` imprime el JSON con z y líneas matcheadas.
- [ ] El caso "z equivocado" da `match_fraction` visiblemente menor que el correcto (esto es lo que el agente va a explotar).
- [ ] Repo `spectra-copilot` público.
- [ ] Tracker actualizado.

## Si algo falla

- **`find_peaks` matchea ruido:** subir `prominence` (multiplicador 0.8 → 1.5) o `distance`.
- **Espectros reales con píxeles inválidos:** interpolar/enmascarar NaNs antes del suavizado (`np.interp` sobre los índices buenos).
- **Descarga del checkpoint lenta en cada test:** `lru_cache` ya lo evita por proceso; para CI, cachear `~/.cache/huggingface`.
