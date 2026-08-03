# 04 · Checkpoint en Hugging Face Hub + model card

> **Bloque:** Nivel 1 · **Tiempo:** ~1 h · **Depende de:** 03 (y 02/02R: se publica la v2.1 promovida) · **Entregable:** model card pública + descarga funcionando desde cualquier máquina

## Objetivo

Sacar el checkpoint de tu disco y ponerlo en HF Hub con una model card honesta. Desbloquea la demo (05), la API (06) y que cualquiera reproduzca tu quick start.

## Namespaces (no mezclar)

- **Código, CI y enlaces** → GitHub **`Julian0444`**: <https://github.com/Julian0444/desi-spectra-fm>
- **Modelo, uploads y `hf_hub_download`** → Hugging Face **`jirustaroure`**: <https://huggingface.co/jirustaroure/desi-spectra-fm>

> **Handoff del plan 02R (2026-08-01) — la v2.1 quedó promovida; subir ESTA:**
>
> - **Decisión:** `promote_v2_1` por gates conjuntos — reproducible en `runs/desi_80k_classhead_v21/comparison.json`.
> - **Checkpoint exacto:** `runs/desi_80k_classhead_v21/checkpoint_last.pt` (ganador; pesos idénticos a `checkpoint_best.pt` porque el best disparó en la validación final, step 30000).
> - **Métricas held-out canónicas** (2.000 labels válidos post-80k, `--skip-examples 80000`, predicción oficial **`z_pred_map`**): η₀.₁₅ **14.95 %** (v1 22.6 %) · σ_NMAD **0.0303** (v1 0.083) · MAE_norm **0.0959** (v1 0.107) · η₀.₁₅ z∈[1.5,2.5) **23.47 %** (v1 82.7 %) · techo z_pred **3.52** (v1 2.00) · RMSE recon **0.8174**.
> - **Model card:** `model_card.md` (v2.1) se sube como `README.md` del repo de HF.
> - **Narrativa obligatoria:** v2.1 es *fine-tuning de v1 con una cabeza nueva de clasificación de redshift* — nunca "entrenada desde cero".

## Pasos

### 0. Autenticación

El CLI no está en PATH; usarlo por ruta completa:

```bash
HF_CLI="$HOME/Library/Python/3.9/bin/hf"
"$HF_CLI" auth whoami        # debe devolver: user: jirustaroure
```

Si falla, repetir `"$HF_CLI" auth login` interactivamente (token con permiso "write"). Nunca imprimir el token ni pasarlo por argumentos.

### 1. Crear el repo de modelo y subir la v2.1

```bash
"$HF_CLI" repo create jirustaroure/desi-spectra-fm --repo-type model --exist-ok

RUN=runs/desi_80k_classhead_v21
"$HF_CLI" upload jirustaroure/desi-spectra-fm "$RUN/checkpoint_last.pt" checkpoint_last.pt
"$HF_CLI" upload jirustaroure/desi-spectra-fm "$RUN/config.json" config.json
"$HF_CLI" upload jirustaroure/desi-spectra-fm "$RUN/training_args.json" training_args.json
"$HF_CLI" upload jirustaroure/desi-spectra-fm "$RUN/metrics.jsonl" metrics.jsonl
```

No subir `checkpoint_best.pt` (pesos idénticos a `last`), `checkpoint_step_*.pt` intermedios, logs, CSVs ni NPZs.

### 2. Model card (README.md del repo de HF)

`model_card.md` (mantenida en la raíz del repo de GitHub) se sube como `README.md` del repo de HF:

```bash
"$HF_CLI" upload jirustaroure/desi-spectra-fm model_card.md README.md
```

Verificar en <https://huggingface.co/jirustaroure/desi-spectra-fm> que renderiza, que la tabla v1→v2.1 usa los números held-out canónicos con `z_pred_map` y que describe la v2.1 como fine-tuning de v1.

### 3. Actualizar el quick start del repo de GitHub

En `README.md` del proyecto, el checkpoint ya no se asume local:

```python
from huggingface_hub import hf_hub_download

ckpt = hf_hub_download(
    "jirustaroure/desi-spectra-fm",
    "checkpoint_last.pt",
)
```

y en la sección CLI, `--checkpoint "$CKPT"` con la descarga previa documentada. `huggingface_hub>=0.23` ya está en `requirements.txt`.

### 4. Prueba desde cero

```bash
TMP=$(mktemp -d) && cd "$TMP"
git clone https://github.com/Julian0444/desi-spectra-fm && cd desi-spectra-fm
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt && pip install -e .
python3 - <<'PY'
from huggingface_hub import hf_hub_download
from desi_fm.predict import predict_spectrum
import numpy as np
ckpt = hf_hub_download("jirustaroure/desi-spectra-fm", "checkpoint_last.pt")
w = np.linspace(3600, 9800, 5000, dtype=np.float32)
r = predict_spectrum(flux=np.random.randn(5000).astype(np.float32), wavelength=w,
                     checkpoint_path=ckpt, device="cpu")
print("OK z_pred_map =", r["z_pred_map"])
PY
```

## Definición de hecho

- [x] <https://huggingface.co/jirustaroure/desi-spectra-fm> público con `checkpoint_last.pt` + `config.json` + `training_args.json` + `metrics.jsonl` + model card renderizada. *(2026-08-02: verificado por API anónima — `private: False`, 5 archivos en `main`, card con η₀.₁₅ 14.95 %.)*
- [x] La prueba desde el clon limpio descarga y predice (`z_pred_map` finito y en rango). *(2026-08-02: `CLEAN_TEST_OK z_pred_map=0.5195 z_confidence=0.2835`, CPU, caché HF fresca, sin dependencia del workspace original.)*
- [x] README de GitHub actualizado (quick start ya no asume archivo local). *(Commit `42c94e6`, CI verde.)*
- [x] Commit + tracker actualizado.

## Si algo falla

- **`hf: command not found`:** usar la ruta completa `~/Library/Python/3.9/bin/hf` (CLI de `huggingface_hub` 0.36).
- **403 al subir:** el token es "read" — crear uno "write" en hf.co/settings/tokens y repetir `hf auth login`.
- **El upload de ~99 MB corta:** reintentar; `hf upload` reanuda solo.
