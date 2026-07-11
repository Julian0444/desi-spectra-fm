# 06 · API FastAPI + Docker

> **Bloque:** Nivel 1 · **Tiempo:** 2–3 h · **Depende de:** 04 · **Entregable:** endpoint público `/predict` respondiendo a `curl`

## Objetivo

Servir el modelo como API REST containerizada — la pata "serving/MLOps" del stack. Deploy gratis como segundo HF Space (SDK Docker).

## Pasos

### 1. `api/main.py` en el repo principal

```python
import io
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
from desi_fm.predict import load_model_from_checkpoint, predict_spectra_batch

app = FastAPI(title="desi-fm API", version="1.0",
              description="Redshift prediction + masked reconstruction for spectra")
_model = None

def model():
    global _model
    if _model is None:
        ckpt = hf_hub_download("TU_USUARIO/desi-spectra-fm", "checkpoint_last.pt")
        _model = load_model_from_checkpoint(ckpt, torch.device("cpu"))
    return _model

class SpectrumJSON(BaseModel):
    flux: list[float]
    wavelength: list[float]

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/predict")
async def predict_npz(file: UploadFile, mask_ratio: float = 0.0):
    d = np.load(io.BytesIO(await file.read()))
    if "flux" not in d.files or "wavelength" not in d.files:
        raise HTTPException(422, "el .npz debe tener 'flux' y 'wavelength'")
    r = predict_spectra_batch(fluxes=d["flux"], wavelengths=d["wavelength"],
                              model=model(), mask_ratio=mask_ratio)
    return {"n": int(len(r["z_pred"])), "z_pred": r["z_pred"].tolist()}

@app.post("/predict_json")
def predict_json(s: SpectrumJSON):
    r = predict_spectra_batch(fluxes=np.asarray(s.flux, dtype=np.float32),
                              wavelengths=np.asarray(s.wavelength, dtype=np.float32),
                              model=model())
    return {"z_pred": float(r["z_pred"][0])}
```

Nota Python 3.9 local: si `list[float]` molesta, `from typing import List` + `List[float]` (el contenedor usa 3.11, no le importa).

### 2. `Dockerfile`

```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt fastapi "uvicorn[standard]" python-multipart
COPY . .
RUN pip install --no-cache-dir -e .
EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

(El checkpoint se baja lazy en el primer request y queda cacheado en el contenedor — evita meter 104 MB en la imagen.)

### 3. Probar local

```bash
docker build -t desi-fm-api .
docker run -p 7860:7860 desi-fm-api
# en otra terminal:
curl -s localhost:7860/healthz
curl -s -F "file=@examples/galaxy_z042.npz" "localhost:7860/predict?mask_ratio=0.0"
```

(Sin Docker Desktop: probar primero con `uvicorn api.main:app --port 7860` a secas; Docker puede quedar para el deploy.)

### 4. Deploy — HF Space con SDK Docker (gratis)

```bash
hf repo create desi-fm-api --repo-type space --space-sdk docker
```

README del Space:

```markdown
---
title: desi-fm API
emoji: 📡
sdk: docker
app_port: 7860
pinned: false
---
```

Push del contenido del repo (Dockerfile incluido) al Space. Verificar:

```bash
curl -s https://TU_USUARIO-desi-fm-api.hf.space/healthz
curl -s -F "file=@examples/galaxy_z042.npz" https://TU_USUARIO-desi-fm-api.hf.space/predict
```

### 5. Documentar

Sección "API" en el README de GitHub: los dos `curl` de arriba + link a `/docs` (Swagger UI que FastAPI genera solo — muy vistoso para el portfolio).

## Definición de hecho

- [ ] `curl` local y público devuelven `z_pred`.
- [ ] `/docs` (Swagger) accesible en el Space.
- [ ] Sección API en el README con ejemplos copy-paste.
- [ ] Commit + tracker.

## Si algo falla

- **El Space Docker excede memoria (16 GB no, pero 2 vCPU/16GB free tier):** el modelo son ~100 MB en fp32, sobra; si OOM, revisar que no se carguen N modelos (patrón `_model` global ya lo evita).
- **Cold start lento (~30 s):** aceptable en free tier; documentarlo ("primer request despierta el Space").
- **CORS para llamarlo desde un frontend:** `from fastapi.middleware.cors import CORSMiddleware` + `app.add_middleware(CORSMiddleware, allow_origins=["*"])`.
