# 04 · Checkpoint en Hugging Face Hub + model card

> **Bloque:** Nivel 1 · **Tiempo:** ~1 h · **Depende de:** 03 (ideal: también 02, para subir la v2 directamente) · **Entregable:** model card pública + descarga funcionando desde cualquier máquina

## Objetivo

Sacar el checkpoint de tu disco y ponerlo en HF Hub con una model card honesta. Desbloquea la demo (05), la API (06) y que cualquiera reproduzca tu quick start.

**Modo sprint:** no esperes a la v2 — subí la **v1 hoy mismo** apenas cierres el plan 03 (desbloquea 05, 06 y 07). Cuando la v2 esté evaluada, repetí los mismos `hf upload` con `RUN=runs/desi_150k_classhead`: el Hub pisa el archivo y guarda historial de versiones. Actualizar la model card en ese mismo momento (columna v2 de la tabla).

## Pasos

### 1. Crear el repo de modelo y subir

```bash
pip install -U huggingface_hub
hf auth login                      # token con permiso "write" (en CLIs viejos: huggingface-cli login)
hf repo create desi-spectra-fm --repo-type model

RUN=runs/desi_150k_classhead       # o runs/desi_50k_big si aún no hay v2
hf upload TU_USUARIO/desi-spectra-fm $RUN/checkpoint_last.pt checkpoint_last.pt
hf upload TU_USUARIO/desi-spectra-fm $RUN/config.json config.json
hf upload TU_USUARIO/desi-spectra-fm $RUN/metrics.jsonl metrics.jsonl
```

### 2. Model card (README.md del repo de HF)

Crear localmente `model_card.md` y subirlo como `README.md`:

```markdown
---
license: mit
tags: [astronomy, spectroscopy, transformer, masked-modeling, redshift, pytorch]
---

# DESI Spectra Foundation Model (26M)

Encoder-only transformer (8 layers, d_model 512) trained with masked-token
prediction on DESI EDR/SV3 spectra. The redshift token is **always masked** and a
prediction head is trained **jointly** with reconstruction, so redshift enters the
representation space from step one (the redesign of AION-1's redshift handling).

## Validation metrics (held-out DESI spectra)

| metric | v1 (50k, regression head) | v2 (150k, classification head) |
|---|---|---|
| MAE_norm ⟨|Δz|/(1+z)⟩ | 0.124 | X |
| σ_NMAD | 0.101 | X |
| catastrophic outliers η₀.₁₅ | 25.1 % | X |

## Usage

    pip install "desi-fm @ git+https://github.com/TU_USUARIO/desi-spectra-fm"

    from huggingface_hub import hf_hub_download
    from desi_fm.predict import predict_spectrum
    ckpt = hf_hub_download("TU_USUARIO/desi-spectra-fm", "checkpoint_last.pt")
    result = predict_spectrum(flux=flux, wavelength=wavelength_angstrom,
                              checkpoint_path=ckpt)
    result["z_pred"], result["reconstruction_input_grid"]

Accepts spectra from **any instrument** (interpolates onto an internal log-λ grid;
sinusoidal positional embedding over physical log-wavelength).

## Limitations

Trained on ~150k spectra on a laptop — not for production science. v1 had a strong
regression-to-the-mean bias at z > 1.5 (η₀.₁₅ = 25 %); v2's classification head
over log(1+z) bins addresses it (see the evaluation notebook in the GitHub repo).

## Links

Code + evaluation notebook: https://github.com/TU_USUARIO/desi-spectra-fm
```

```bash
hf upload TU_USUARIO/desi-spectra-fm model_card.md README.md
```

### 3. Actualizar el quick start del repo de GitHub

En `README.md` del proyecto, reemplazar la ruta local del checkpoint por:

```python
from huggingface_hub import hf_hub_download
ckpt = hf_hub_download("TU_USUARIO/desi-spectra-fm", "checkpoint_last.pt")
```

y en la sección CLI: `--checkpoint $(python3 -c "from huggingface_hub import hf_hub_download as d; print(d('TU_USUARIO/desi-spectra-fm','checkpoint_last.pt'))")` o simplemente documentar la descarga previa. Agregar `huggingface_hub>=0.23` a `requirements.txt`.

### 4. Prueba desde cero

```bash
cd /tmp/desi-spectra-fm && source .venv/bin/activate    # el clon del plan 03
pip install huggingface_hub
python3 -c "
from huggingface_hub import hf_hub_download
from desi_fm.predict import predict_spectrum
import numpy as np
ckpt = hf_hub_download('TU_USUARIO/desi-spectra-fm', 'checkpoint_last.pt')
w = np.linspace(3600, 9800, 5000, dtype=np.float32)
r = predict_spectrum(flux=np.random.randn(5000).astype(np.float32), wavelength=w,
                     checkpoint_path=ckpt, device='cpu')
print('OK z_pred =', r['z_pred'])
"
```

## Definición de hecho

- [ ] `hf.co/TU_USUARIO/desi-spectra-fm` público con checkpoint + config + model card renderizada.
- [ ] La prueba desde el clon limpio descarga y predice.
- [ ] README de GitHub actualizado (quick start ya no asume archivo local).
- [ ] Commit + tracker actualizado.

## Si algo falla

- **`hf: command not found`:** el CLI nuevo viene con `huggingface_hub>=0.24`; usar `huggingface-cli` (mismos subcomandos) o `python3 -m huggingface_hub.commands.huggingface_cli`.
- **403 al subir:** el token es "read" — crear uno "write" en hf.co/settings/tokens y `hf auth login` de nuevo.
- **El upload de 104 MB corta:** reintentar; `hf upload` reanuda. Alternativa: `HF_HUB_ENABLE_HF_TRANSFER=1` con `pip install hf_transfer`.
