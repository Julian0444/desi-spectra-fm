# 07 · spectra-copilot: repo nuevo + herramientas

> **Bloque:** Nivel 2 · **Tiempo:** 3–4 h · **Depende de:** 04 — o de **ningún plan** si exportás `DESI_FM_CKPT` apuntando a un checkpoint local · **Entregable:** `tools.py` testeado + CLI de demo que imprime JSON
>
> **Namespaces (regla permanente):** GitHub **`Julian0444`** para código y enlaces; Hugging Face **`jirustaroure`** para modelo y Spaces.

## Objetivo

Crear el segundo repo (`spectra-copilot`) con las herramientas determinísticas que después usarán el agente (08) y el servidor MCP (09). La estrella es `identify_spectral_lines`: le da al agente una forma de **verificar** físicamente la predicción del modelo, no solo repetirla.

Regla de oro: las tools devuelven **JSON compacto** (conclusiones), jamás arrays de 7081 floats. Los plots van a disco y se devuelve la ruta.

> **Adaptaciones clave (2026-08-03, ejecutado):** checkpoint oficial = **v2.1** (`jirustaroure/desi-spectra-fm` en el Hub, `runs/desi_80k_classhead_v21/checkpoint_last.pt` local vía `DESI_FM_CKPT`), así que `predict_redshift` expone la salida honesta del proyecto — `z_pred_map` (oficial) + `z_confidence` + `z_pred` secundario — no el `z_pred` pelado del plan original. Los ejemplos son **espectros DESI held-out reales** (los mismos `.npz` de la demo, con `z_true` de referencia), no sintéticos. Se agregó CI propio al repo nuevo.

## Cómo se ejecutó (runbook real)

### 1. Esqueleto del repo — `~/proyectos/spectra-copilot`

`git init -b main`; `pyproject.toml` como el plan original pero con `desi-fm @ git+https://github.com/Julian0444/desi-spectra-fm` (la instalación desde el repo público funciona — de paso lo prueba). Ejemplos copiados del repo principal y **commiteados** (~103 KB c/u): `heldout_z020.npz` (galaxia, z_true 0.204), `heldout_z287.npz` (QSO, z_true 2.866) y `heldout_lowconf_z157.npz` (el outlier catastrófico honesto, z_true 1.574 — oro para el agente del 08).

### 2. `copilot/tools.py`

Tres tools sobre `desi_fm.predict` (mismo catálogo de 15 líneas rest-frame del plan):

- `predict_redshift_impl` — determinista (`mask_ratio=0`), pasa `ivar`/`mask` del `.npz` al modelo (protocolo de la demo/API); devuelve `{z_pred_map, z_confidence, z_pred}` (fallback a `z_pred` solo si el checkpoint no tiene cabeza). Helper `official_z()` para que agente/CLI actúen sobre la predicción oficial.
- `identify_spectral_lines_impl` — `find_peaks` sobre continuo suavizado (σ=3, prominence 0.8σ, tolerancia 12 Å ≈ 5 px DESI); docstring avisa que detecta **emisión** (veredicto débil ≠ refutación en espectros de absorción).
- `reconstruct_spectrum_impl` — enmascara tokens al azar y reporta `n_tokens_masked` + z bajo masking (sonda de estabilidad).
- `_load` robusto: 2-D → primer espectro; NaN/Inf → zereados y marcados en `mask`. `_model()` con `lru_cache`: `DESI_FM_CKPT` o `hf_hub_download("jirustaroure/desi-spectra-fm", "checkpoint_last.pt")`.

### 3. CLI de demo (`copilot/__main__.py`)

`python -m copilot examples/heldout_z020.npz` → JSON con `predict_redshift`, `identify_spectral_lines` (al `official_z`) y `z_true_reference` si el `.npz` lo trae. Una sola pasada de modelo (no dos como el snippet original).

### 4. Tests — 7 en `tests/test_tools.py`

Los 4 del plan + 3 nuevos: `z_pred_map`/`z_confidence` expuestos y en rango, `_load` con 2-D+NaN, y **discriminación sobre el espectro real** (sin modelo): en `heldout_z020`, líneas a z_true 0.2036 → **8/11, `consistent`**; a z=0.85 → débil. **Ojo:** el sintético del plan original inyectaba solo 4 líneas y el catálogo espera 12 en cobertura a z=0.42 → `match_fraction` 0.42 < 0.5 y el test del plan **fallaba tal cual estaba escrito**; el fix correcto fue inyectar 9 líneas del catálogo en `_synth` (no aflojar el umbral).

```bash
cd ~/proyectos/spectra-copilot && python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]"
export DESI_FM_CKPT=".../runs/desi_80k_classhead_v21/checkpoint_last.pt"
.venv/bin/python -m pytest -q          # 7 passed
.venv/bin/python -m copilot examples/heldout_z020.npz
```

### 5. Subir

`gh repo create spectra-copilot --public --source . --push` → <https://github.com/Julian0444/spectra-copilot>. CI propio (`ci.yml`: Python 3.11 + torch CPU + `actions/cache` de `~/.cache/huggingface` para el checkpoint, como anticipaba "Si algo falla"). README corto con el JSON real de `heldout_z020` y el contraste 2/11 vs 8/11; el README bueno llega con el agente (08).

## Hallazgo que importa para el 08

En la galaxia real `heldout_z020` la tool de líneas **discrimina de verdad**: a `z_pred_map` 0.2267 matchea 2/11 (Δz 0.023 ≈ 150 Å en Hα, fuera de la tolerancia), a z_true 0.2036 matchea 8/11 (`consistent`), a z=0.85 queda débil. O sea: el agente puede **detectar y refinar** una predicción corrida, exactamente la historia que el 08 tiene que contar.

## Definición de hecho

- [x] `pytest` verde (≥3 tests de tools) — **7/7** local (venv 3.12) y en CI.
- [x] `python -m copilot examples/heldout_z020.npz` imprime el JSON con z y líneas matcheadas — `z_pred_map` **0.2267** / conf 0.6376 (idéntico al valor verificado de demo y API; z_true 0.204), 2 líneas matcheadas; `heldout_z287` → 2.4406, `lowconf` → 0.9569/conf 0.18.
- [x] El caso "z equivocado" da `match_fraction` visiblemente menor que el correcto — sintético 0.75 vs wrong-z; real 0.73 (z_true) vs 0.18–0.22 (z corrido / z=0.85).
- [x] Repo `spectra-copilot` público — <https://github.com/Julian0444/spectra-copilot> (Actions verde).
- [x] Tracker actualizado.

## Si algo falla

- **`find_peaks` matchea ruido:** subir `prominence` (multiplicador 0.8 → 1.5) o `distance`.
- **`test_lines_match_at_true_z` < 0.5 con el sintético:** el sintético tiene que inyectar la mayoría de las líneas del catálogo en cobertura (9 de 12 a z=0.42), no 4 — si no, la fracción esperada máxima es ~0.4.
- **Espectros reales con píxeles inválidos:** `_load` ya los zerea y los suma a `mask` — no hace falta interpolar.
- **Descarga del checkpoint lenta en cada test:** `lru_cache` lo evita por proceso; en CI, `actions/cache` de `~/.cache/huggingface` (implementado).
- **`UserWarning: enable_nested_tensor ...` al cargar el modelo:** benigno, viene de `desi_fm.model` con PyTorch ≥2.x; ignorar.
