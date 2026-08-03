# 06 · API FastAPI + Docker

> **Bloque:** Nivel 1 · **Tiempo:** 2–3 h · **Depende de:** 04 · **Entregable:** endpoint público `/predict` respondiendo a `curl`
>
> **Namespaces (regla permanente):** GitHub **`Julian0444`** para código y enlaces; Hugging Face **`jirustaroure`** para modelo y Spaces.

## Objetivo

Servir el modelo como API REST containerizada — la pata "serving/MLOps" del stack.

> **Adaptación clave (2026-08-03):** el deploy original ("segundo HF Space con SDK Docker gratis") ya no existe — los **Docker Spaces requieren PRO** (402, misma política que paywalleó cpu-basic, ver plan 05). El deploy real es un **segundo Space ZeroGPU con SDK Gradio** (el free tier permite 2): gradio se lanza normal para satisfacer a la plataforma y la API FastAPI se injerta como sub-app en **`/api`**, con la inferencia REST en **CPU** — `curl` anónimo funciona sin gastar cuota ZeroGPU. El Dockerfile quedó en el repo (self-hosting) y está probado localmente.

## Cómo se ejecutó (runbook real)

### 1. La API es un módulo del paquete: `src/desi_fm/api.py`

No un `api/main.py` suelto (el plan original): como módulo es importable idéntico desde uvicorn local, el Dockerfile, los tests y el Space, sin duplicar código.

- `POST /predict` — multipart `.npz` con `flux` (P,) o (N, P) y `wavelength`; `ivar`/`mask` opcionales; query `mask_ratio` (0–0.9, validado). `POST /predict_json` — un espectro como listas JSON. `GET /healthz`.
- **Salida honesta como en todo el proyecto:** `z_pred_map` primero (+ `z_confidence`), `z_pred` (esperanza) secundario.
- Checkpoint **lazy** del Hub en el primer request (`hf_hub_download`), override con `DESI_FM_CKPT` (path local) y `DESI_FM_DEVICE` (default `cpu`).
- Límites anti-abuso para free tier: 32 espectros / 50 MB por request (413); errores de validación → 422 con mensaje útil.
- CORS abierto (llamable desde frontends).
- Packaging: extra `[api]` en `pyproject.toml` (`fastapi`, `uvicorn[standard]`, `python-multipart`, `huggingface_hub`); `httpx` en `[dev]` para el TestClient. CI instala `.[api,dev]`.

### 2. Tests (8 nuevos, suite 16→24)

`tests/test_api.py` con `TestClient` y un modelo sintético chico inyectado en `api._model` — **sin red ni checkpoint** (CI-friendly): healthz, predict single/batch, claves faltantes → 422, archivo corrupto → 422, `mask_ratio` fuera de rango → 422, predict_json OK y con shapes inconsistentes → 422.

### 3. Prueba local (uvicorn y Docker)

```bash
DESI_FM_CKPT=runs/desi_80k_classhead_v21/checkpoint_last.pt \
  python3 -m uvicorn desi_fm.api:app --port 7861
curl -s -F "file=@examples/heldout_z020.npz" "localhost:7861/predict?mask_ratio=0.0"
# → {"n":1,"z_pred_map":[0.2267],"z_confidence":[0.6376],"z_pred":[0.2314]}  (z_true 0.204)

docker build -t desi-fm-api .    # python:3.11-slim + torch CPU; .dockerignore deja fuera runs/ y .git
docker run -p 7860:7860 desi-fm-api   # sin DESI_FM_CKPT baja el checkpoint del Hub al primer request
```

Ambos devuelven **exactamente** el `z_pred_map` 0.2267 verificado en la demo para `heldout_z020` (protocolo `mask_ratio=0`, determinista). `/docs` responde 200 en los dos.

### 4. Deploy — Space ZeroGPU `jirustaroure/desi-fm-api`

Fuente versionada en `api/` (`app.py` + `requirements.txt` + `README.md` con el YAML del Space). Creación y uploads **solo por API Python** (el CLI `hf` 402ea con Spaces):

```python
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("jirustaroure/desi-fm-api", repo_type="space",
                space_sdk="gradio", space_hardware="zero-a10g", exist_ok=True)
api.upload_folder(folder_path="api", repo_id="jirustaroure/desi-fm-api", repo_type="space")
api.restart_space("jirustaroure/desi-fm-api")   # un commit NO reinicia el contenedor
```

Dos intentos fallidos que definieron la arquitectura final de `api/app.py` (ver "Si algo falla"): SSR de gradio robándose el puerto 7860, y `mount_gradio_app` + uvicorn propio muriendo por SIGTERM en ZeroGPU. Lo que funciona: `GRADIO_SSR_MODE=false` + `demo.launch(prevent_thread_lock=True)` + `server_app.mount("/api", fastapi_app)` + `block_thread()`. La UI mínima (tester del mismo modelo) lleva el `@spaces.GPU(duration=8)` obligatorio en `zero-a10g`; las rutas REST corren en CPU fuera de la cuota.

### 5. Verificación pública (anónima, `curl` desde afuera)

```bash
curl -s https://jirustaroure-desi-fm-api.hf.space/api/healthz
curl -s -F "file=@examples/heldout_z020.npz" \
  "https://jirustaroure-desi-fm-api.hf.space/api/predict?mask_ratio=0.0"
```

### 6. Documentar

Sección "REST API (FastAPI)" en el README de GitHub (curl público + `/api/docs` + uvicorn/Docker/`DESI_FM_CKPT`), línea 📡 arriba del README, layout con `api/` y `Dockerfile`, link en la model card (re-subida al Hub).

## Definición de hecho

- [x] `curl` local y público devuelven `z_pred` — local uvicorn y Docker: `z_pred_map` 0.2267 en `heldout_z020`; público anónimo: 0.2267 (0.63 s) y 2.4406 en `heldout_z287` (0.63 s), `predict_json` 0.2267. Idénticos a los valores verificados de la demo.
- [x] `/docs` (Swagger) accesible en el Space — <https://jirustaroure-desi-fm-api.hf.space/api/docs> (HTTP 200; también local y Docker).
- [x] Sección API en el README con ejemplos copy-paste (+ model card re-subida al Hub con el link).
- [x] Commit + tracker (06 ✅ con la URL del endpoint; suite 24/24; CI verde).

**Extra verificado:** el panel Gradio del Space (ruta `@spaces.GPU`) responde 0.2267 en 7.9 s autenticado; anónimo desde una IP con la cuota diaria gastada da "exceeded your ZeroGPU runs limit" — limitación de plataforma documentada en la card, no bug (idéntico al plan 05).

## Si algo falla

- **`hf repo create`/`hf upload` (CLI) → 402 con Spaces:** usar siempre `HfApi.create_repo/upload_folder/upload_file` (API Python).
- **`RUNTIME_ERROR` con "address already in use" en el puerto 7860:** en Spaces `GRADIO_SSR_MODE=true` por default — el frontend Node ocupa el 7860 y cualquier server propio choca. `os.environ["GRADIO_SSR_MODE"] = "false"` **antes** de `import gradio`.
- **SIGTERM limpio segundos después de "Uvicorn running" (con `mount_gradio_app` + uvicorn propio):** en ZeroGPU el handshake de la lib `spaces` con el scheduler pasa por `demo.launch()`; si gradio no se lanza por esa vía, la plataforma mata el pod. Usar el patrón launch + `server_app.mount("/api", ...)` de `api/app.py`.
- **Un commit al Space no reinicia el contenedor:** `restart_space()` explícito y confirmar "Application Startup" nuevo en los logs (`/api/spaces/<id>/logs/run`, SSE autenticado).
- **El panel UI da "Error" pelado con la API REST sana:** casi seguro cuota ZeroGPU anónima agotada (el mensaje real está en el SSE de `queue/data`); autenticarse o esperar al reset diario. Si TODAS las llamadas GPU fallan con `AcceleratorError` (ECC), `restart_space()` para caer en otro host.
- **OOM / lentitud en el contenedor Docker:** el modelo son ~100 MB fp32 y el patrón `_model` global evita cargas múltiples; el cold start público (~30 s si el Space durmió) es del free tier, documentado en el README.
