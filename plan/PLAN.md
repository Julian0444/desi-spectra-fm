# Plan de ejecución — de proyecto de curso a stack completo de AI Engineer

> ⚡ **Este documento es la visión general (contexto, diagnóstico, narrativa).** La ejecución está subdividida en 13 planes concretos y auto-contenidos, uno por sesión de trabajo: ver el tracker en [`plan/README.md`](README.md).

> **Proyecto base:** `desi_fm` — foundation model unimodal para espectros DESI (PHYS303/CS486, USF).
> **Objetivo:** cerrar la corrección del TA y convertir el proyecto en una pieza de portfolio que cubra las cuatro capas: research ML → MLOps/serving → AI engineering (LLM + agentes) → producto.
> **Fecha del plan:** 2026-07-04.

---

## Índice

- [Contexto](#contexto)
- [Fase 0 — Aplicar la corrección del TA](#fase-0--aplicar-la-corrección-del-ta)
- [Nivel 1 — Hacerlo público y usable](#nivel-1--hacerlo-público-y-usable)
- [Nivel 2 — Spectra Copilot (la capa agéntica)](#nivel-2--spectra-copilot-la-capa-agéntica)
- [Nivel 3 — Profundidad técnica](#nivel-3--profundidad-técnica)
- [La narrativa final](#la-narrativa-final)
- [Cronograma sugerido](#cronograma-sugerido)
- [Apéndice: checklist global](#apéndice-checklist-global)

---

## Contexto

El feedback del TA sobre la entrega original tuvo dos observaciones:

1. *"The final redshift performance is relatively weak. You have a high bias and a large catastrophic outlier fraction."*
2. *"Your submission also lacked a ready-to-run evaluation notebook and plots, which made grading less direct."*

**Estado al escribir este plan:**

| ítem | estado |
|---|---|
| Notebook de evaluación con plots (`notebooks/evaluation.ipynb`) | ✅ **Hecho** (2026-07-04) — ejecutado, 8 gráficos embebidos |
| Diagnóstico cuantificado del bias/outliers | ✅ **Hecho** — está en el notebook §1 y en la Fase 0.2 de este plan |
| Mejora del modelo (cabeza de clasificación + reentrenamiento) | ⬜ Pendiente — receta en Fase 0.3 |
| Niveles 1, 2 y 3 | ⬜ Pendientes |

La regla general de todo el plan: **cada fase termina con algo que se puede mostrar con un link** (un repo, una demo, una tabla de evals). Si una tarea no produce algo mostrable, es infraestructura para una que sí.

---

## Fase 0 — Aplicar la corrección del TA

### 0.1 Notebook de evaluación ✅ (hecho)

`notebooks/evaluation.ipynb` responde directamente la segunda observación:

- **§1** métricas de redshift con las convenciones de la literatura (MAE, bias, σ_NMAD, fracción de outliers catastróficos η₀.₁₅), scatter z_pred vs z_true, residuales normalizados, histograma y desglose por bins de z.
- **§2** reconstrucción de regiones enmascaradas: RMSE + galería de 3 espectros (mejor/mediano/peor).
- **§3** curvas de entrenamiento desde `metrics.jsonl`.
- **§4** inferencia en vivo con el checkpoint real sobre espectros sintéticos OOD (CPU).
- **§5** resumen que mapea cada frase del feedback a la evidencia.

Las secciones 1–3 corren **offline** desde los artefactos ya commiteados (`predictions.csv`, `reconstructions.npz`, `metrics.jsonl`) — un grader lo abre y ve todo sin bajar datos. Se entrega **pre-ejecutado** (los plots se ven en GitHub sin correr nada).

Para regenerarlo tras cualquier cambio:

```bash
python3 -m nbconvert --to notebook --execute --inplace notebooks/evaluation.ipynb
```

### 0.2 El diagnóstico (números reales, validación de 1000 espectros)

| métrica | valor |
|---|---|
| bias global ⟨Δz⟩ | **−0.090** (la mediana es solo −0.024 → lo domina la cola de alto z) |
| σ_NMAD | 0.101 |
| η₀.₁₅ (outliers catastróficos, \|Δz\|/(1+z) > 0.15) | **25.1 %** |
| techo de predicción | max z_pred ≈ **1.95**, con z_true hasta **4.71** |

Por bins de z_true:

| bin | n | bias ⟨Δz⟩ | η₀.₁₅ |
|---|---|---|---|
| 0.0–0.1 | 110 | **+0.163** | 23 % |
| 0.1–0.3 | 206 | +0.070 | 14 % |
| 0.3–0.6 | 129 | +0.037 | 22 % |
| 0.6–1.0 | 294 | −0.022 | 12 % |
| 1.0–1.5 | 201 | **−0.323** | 42 % |
| 1.5–2.5 | 55 | **−0.816** | 84 % |
| 2.5+ | 5 | **−2.149** | 100 % |

Lectura: es **regresión a la media condicional**. Dos causas que se refuerzan:

1. **Distribución de z desbalanceada** (SV3 es mayormente z ≲ 1.5; los cuásares de alto z son raros) + una cabeza de **regresión escalar** (SmoothL1): el óptimo de esa loss colapsa hacia la media condicional → sube las predicciones de bajo z, aplasta las de alto z.
2. **El problema es multimodal**: confundir Hα ↔ [OIII] ↔ [OII] produce soluciones de z *discretas* y equivocadas. Una cabeza de regresión unimodal **promedia los modos** en vez de comprometerse con uno — eso genera exactamente "bias alto + outliers catastróficos".

### 0.3 La receta de mejora (3 cambios, en orden de impacto)

#### Cambio A — Cabeza de clasificación sobre bins de log(1+z)

Es el fix estándar en la literatura de photo-z para matar outliers por multimodalidad: en vez de predecir un escalar, el modelo predice una **distribución** sobre ~200 bins de log(1+z) y puede expresar "o es z=0.4 o es z=1.8". Cambios localizados en `src/desi_fm/model.py` (~40 líneas):

```python
# En DESIFoundationModelConfig:
n_z_bins: int = 200
z_max: float = 6.0

# En __init__ — la última capa de redshift_head pasa de →1 a →n_z_bins:
self.redshift_head = nn.Sequential(
    nn.LayerNorm(config.d_model),
    nn.Linear(config.d_model, config.d_model),
    nn.GELU(),
    nn.Dropout(config.dropout),
    nn.Linear(config.d_model, config.n_z_bins),
)
edges = torch.linspace(0.0, math.log1p(config.z_max), config.n_z_bins + 1)
self.register_buffer("z_bin_edges", edges, persistent=False)
self.register_buffer("z_bin_centers", 0.5 * (edges[:-1] + edges[1:]), persistent=False)

# En forward():
z_logits = self.redshift_head(z_hidden)                      # (B, n_bins)
p = z_logits.softmax(-1)
z_encoded_pred = (p * self.z_bin_centers).sum(-1)            # esperanza en log(1+z)
z_pred = self.decode_redshift(z_encoded_pred)

# Loss (reemplaza el SmoothL1):
target_bin = torch.bucketize(self.encode_redshift(z), self.z_bin_edges[1:-1])
z_loss = F.cross_entropy(z_logits, target_bin, label_smoothing=0.05)
```

Notas:
- Guardar también `z_pred_map = expm1(z_bin_centers[z_logits.argmax(-1)])`: cuando la posterior es bimodal, el **argmax** evita el promedio de modos; la esperanza sirve como métrica suave. Reportar ambas en `evaluate.py`.
- Bonus enorme para el Nivel 2: la **entropía de la posterior es una medida de confianza** gratis — el agente podrá decir "predicción ambigua, dos modos posibles".
- Los tests existentes siguen pasando (la firma de `z_pred` no cambia); agregar un test de que `z_pred ∈ [0, z_max]`.
- El diseño sigue siendo foundation model (enfoques A + B intactos: token de z siempre enmascarado + entrenamiento conjunto); solo cambia la parametrización de la salida.

#### Cambio B — Rebalancear la distribución de z

Pesos por bin inverso-frecuencia (capados para no explotar la varianza), pasados a la cross-entropy:

```python
# Precomputar una vez (p. ej. desde predictions.csv o un pase por el stream):
#   hist = histograma de log1p(z) en los n_z_bins
#   w_bin = clip((hist.sum()/n_bins) / hist, 0.3, 10.0)
z_loss = F.cross_entropy(z_logits, target_bin, weight=w_bin, label_smoothing=0.05)
```

Alternativa (más trabajo, mejor señal): oversampling de alto z en el stream con un buffer que re-emite ejemplos con z > 1.5. Empezar por los pesos; es una línea.

#### Cambio C — Más datos y más épocas

El run entregado fue 50k ejemplos × 1 época (~30 min en MPS). La tabla de progresión del `DELIVERABLE.md` ya muestra que escalar datos mejora. Comando de reentrenamiento sugerido (~3–4 h en MPS):

```bash
python3 -m desi_fm.train \
  --dataset MultimodalUniverse/desi --data-dir edr_sv3 \
  --output-dir runs/desi_150k_classhead \
  --batch-size 8 --max-train-examples 150000 --val-examples 2000 --epochs 2 \
  --redshift-loss-weight 10 --mask-ratio 0.5 --wavelength-grid log \
  --d-model 512 --n-layers 8 --n-heads 8 \
  --log-every-steps 50 --save-every-steps 5000
```

#### Criterios de éxito y cierre de la fase

| métrica | hoy | objetivo |
|---|---|---|
| η₀.₁₅ | 25.1 % | **< 10 %** |
| \|bias_norm\| = \|⟨Δz/(1+z)⟩\| | 0.018 (global, esconde ±0.16 por bin) | **< 0.02 en cada bin poblado** |
| σ_NMAD | 0.101 | **< 0.05** |
| techo de z_pred | 1.95 | predicciones reales hasta z ≳ 3 |

Al terminar: re-ejecutar el notebook con el checkpoint nuevo → los gráficos **antes/después** son material de portfolio de primera ("recibí feedback del TA, diagnostiqué, rediseñé la cabeza, mejoré η de 25 % a X %"). Documentarlo en el README como "v2".

---

## Nivel 1 — Hacerlo público y usable

**Meta:** repo público limpio + checkpoint en Hugging Face Hub + demo clickeable + API + CI. Todo gratis.

### 1.1 Git + limpieza del repo

```bash
cd "/Users/jirustaroure/Desktop/FINAL PROJECT DEEP LEARNING"
git init -b main
```

Crear `.gitignore`:

```gitignore
__pycache__/
*.egg-info/
.pytest_cache/
.ipynb_checkpoints/
.DS_Store
external/
*.zip
*.docx
# checkpoints: solo artefactos livianos del run final
runs/*
!runs/desi_50k_big/
runs/desi_50k_big/*.pt
```

Decisiones de limpieza:

- **`external/`** (clones de AION y MultimodalUniverse) — fuera del repo; ya están referenciados como links en el README.
- **Checkpoints `.pt`** — nunca a GitHub (104 MB c/u). El final va a HF Hub (1.2); los intermedios quedan solo locales.
- **De `runs/desi_50k_big/` sí se commitea:** `config.json`, `metrics.jsonl`, `predictions.csv`, `reconstructions.npz` (~2.7 MB total — son los insumos del notebook).
- **`PHYS303_Final-Project_20266.pdf` / `.docx`** — fuera del repo público (material del profesor); la consigna queda resumida en el README propio.
- Los runs intermedios viejos (`desi_500`, `desi_tiny*`, `smoke*`, etc.) pueden borrarse localmente o quedar ignorados; la progresión ya está documentada en la tabla del `DELIVERABLE.md`.

Crear el repo (nombre sugerido: `desi-spectra-fm`; sin espacios):

```bash
git add -A && git commit -m "DESI spectra foundation model: training, evaluation notebook, inference"
gh repo create desi-spectra-fm --public --source . --push
```

Topics sugeridos en GitHub: `deep-learning`, `transformers`, `astronomy`, `foundation-models`, `pytorch`, `self-supervised-learning`.

### 1.2 Checkpoint en Hugging Face Hub + model card

```bash
pip install -U huggingface_hub
hf auth login                       # (en versiones viejas del CLI: huggingface-cli login)
hf repo create desi-spectra-fm --repo-type model
hf upload TU_USUARIO/desi-spectra-fm runs/desi_50k_big/checkpoint_last.pt checkpoint_last.pt
hf upload TU_USUARIO/desi-spectra-fm runs/desi_50k_big/config.json config.json
```

Model card (`README.md` del repo de HF) — esqueleto:

```markdown
---
license: mit
tags: [astronomy, spectroscopy, transformer, masked-modeling, redshift]
---
# DESI Spectra Foundation Model (26M)
Encoder-only transformer entrenado con masked-token prediction sobre 50k espectros
DESI EDR/SV3, con el token de redshift siempre enmascarado y una cabeza de z
entrenada conjuntamente.

## Métricas (validación, espectros DESI held-out)
MAE_norm 0.124 · σ_NMAD 0.101 · η₀.₁₅ 25.1 % (v1 — ver Limitations)

## Uso
    from huggingface_hub import hf_hub_download
    from desi_fm.predict import predict_spectrum
    ckpt = hf_hub_download("TU_USUARIO/desi-spectra-fm", "checkpoint_last.pt")
    result = predict_spectrum(flux=flux, wavelength=wavelength, checkpoint_path=ckpt)

## Limitations
Bias negativo y outliers a z > 1.5 (regresión a la media; fix en curso — v2 con
cabeza de clasificación). No apto para uso científico de producción.
```

Ser honesto con las limitaciones **suma** en un portfolio — demuestra que sabés evaluar modelos.

Actualizar el README del repo de GitHub para que el quick start descargue el checkpoint de HF en vez de asumir el archivo local.

### 1.3 Demo interactiva en HF Spaces (Gradio)

Crear un Space (SDK: Gradio, hardware: CPU basic — gratis). Estructura:

```
spaces/desi-spectra-fm-demo/
  app.py
  requirements.txt      # torch, gradio, numpy, matplotlib, huggingface_hub, desi-fm@git+...
  examples/             # 4-5 espectros .npz de ejemplo (sintéticos + 1-2 reales exportados)
```

`app.py` — esqueleto (~50 líneas):

```python
import gradio as gr, numpy as np, matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from desi_fm.predict import load_model_from_checkpoint, predict_spectrum
import torch

ckpt = hf_hub_download("TU_USUARIO/desi-spectra-fm", "checkpoint_last.pt")
model = load_model_from_checkpoint(ckpt, torch.device("cpu"))

def analyze(npz_file, mask_ratio):
    d = np.load(npz_file.name)
    r = predict_spectrum(flux=d["flux"], wavelength=d["wavelength"],
                         model=model, mask_ratio=mask_ratio)
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(d["wavelength"], d["flux"], lw=0.7, color="0.45", label="input")
    ax.plot(d["wavelength"], r["reconstruction_input_grid"], lw=1.0,
            color="#3D6FD6", label="reconstruction")
    ax.legend(); ax.set_xlabel("wavelength [Å]")
    return f"z predicho = {r['z_pred']:.4f}", fig

demo = gr.Interface(
    analyze,
    inputs=[gr.File(label="Espectro (.npz con flux + wavelength)"),
            gr.Slider(0.0, 0.9, value=0.0, label="mask ratio")],
    outputs=[gr.Textbox(label="Redshift"), gr.Plot()],
    examples=[["examples/galaxy_z04.npz", 0.0], ["examples/qso_z21.npz", 0.5]],
    title="DESI Spectra Foundation Model",
)
demo.launch()
```

Inferencia en CPU ≈ 1 s/espectro → el tier gratuito alcanza de sobra. **Este link es la pieza más valiosa del Nivel 1** — va arriba de todo en el README y en el CV.

### 1.4 API con FastAPI + Docker

Dentro del repo principal, `api/main.py`:

```python
from fastapi import FastAPI, UploadFile
import numpy as np, torch, io
from desi_fm.predict import load_model_from_checkpoint, predict_spectra_batch

app = FastAPI(title="desi-fm API")
model = load_model_from_checkpoint("checkpoint_last.pt", torch.device("cpu"))

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile, mask_ratio: float = 0.0):
    d = np.load(io.BytesIO(await file.read()))
    r = predict_spectra_batch(fluxes=d["flux"], wavelengths=d["wavelength"],
                              model=model, mask_ratio=mask_ratio)
    return {"z_pred": r["z_pred"].tolist(), "n": len(r["z_pred"])}
```

`Dockerfile`:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt fastapi uvicorn python-multipart huggingface_hub
COPY . .
RUN pip install -e . && python -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download('TU_USUARIO/desi-spectra-fm','checkpoint_last.pt', local_dir='.')"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

Deploy más simple: un segundo HF Space con SDK "Docker" (gratis). Alternativa: Fly.io. Verificación:

```bash
curl -F "file=@examples/galaxy_z04.npz" https://TU-SPACE.hf.space/predict
```

### 1.5 CI con GitHub Actions

`.github/workflows/ci.yml`:

```yaml
name: ci
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.11"}
      - run: pip install torch --index-url https://download.pytorch.org/whl/cpu
      - run: pip install -e . pytest
      - run: pytest -q
```

Los 4 tests no necesitan datos ni red → CI verde en ~2 min. Agregar el badge al README.

### Criterios de aceptación del Nivel 1

- [ ] Repo público sin zips/docx/external/checkpoints, con historia limpia y topics.
- [ ] `pip install` + `pytest` funcionan en una máquina limpia (lo prueba el CI).
- [ ] Checkpoint descargable de HF Hub con model card honesta.
- [ ] Demo Gradio viva (link en README y en el CV).
- [ ] Endpoint `/predict` respondiendo con `curl`.

---

## Nivel 2 — Spectra Copilot (la capa agéntica)

**Meta:** un agente LLM que usa **tu** modelo como herramienta: recibe un espectro, lo analiza llamando a `desi_fm`, **valida** la predicción físicamente, y escribe un reporte de observación. Expuesto como chat y como **servidor MCP**.

**Repo nuevo** (`spectra-copilot`) que instala `desi-fm` como dependencia — dos piezas de portfolio y disciplina de packaging:

```
spectra-copilot/
  copilot/
    tools.py          # wrappers de desi_fm + física (líneas espectrales)
    agent.py          # loop de tool-use con la Claude API
    mcp_server.py     # mismo toolset expuesto vía MCP (FastMCP)
    report.py         # system prompt + formato del reporte
  eval/
    cases/            # espectros etiquetados para evals (Nivel 3)
    run_evals.py
  app.py              # UI de chat (Gradio)
  pyproject.toml      # deps: anthropic, mcp[cli], gradio, scipy, numpy,
                      #       desi-fm @ git+https://github.com/TU_USUARIO/desi-spectra-fm
```

### 2.1 Las herramientas (`tools.py`)

Regla de oro: **los resultados de las tools son JSON compacto, nunca arrays de 7081 floats**. El LLM necesita conclusiones, no píxeles; los plots se guardan a disco y se devuelve la ruta.

| tool | qué hace | qué devuelve |
|---|---|---|
| `predict_redshift(npz_path)` | corre el foundation model | `{z_pred, z_map, confidence}` (la entropía de la posterior de la Fase 0 se vuelve `confidence` gratis) |
| `reconstruct_spectrum(npz_path, mask_ratio)` | reconstrucción enmascarada | `{rmse_masked, plot_path}` |
| `identify_spectral_lines(npz_path, z)` | **la herramienta de validación**: calcula dónde deberían caer las líneas conocidas dado z (`λ_obs = λ_rest·(1+z)`) y las cruza con picos reales (`scipy.signal.find_peaks` sobre el flujo suavizado) | `{expected: [...], matched: [...], match_fraction}` |
| `find_similar_spectra(npz_path, k)` | vecinos en el índice FAISS (Nivel 3) | `[{id, z, similarity}, ...]` |

Las líneas rest-frame ya están en el repo (`SyntheticSpectra.rest_lines`: OII 3727, Ca K/H 3934/3969, Hβ 4861, OIII 5007, Hα 6563) — extender con NII/SII/MgII/CIV para cubrir cuásares.

`identify_spectral_lines` es lo que convierte al agente en algo más que un wrapper: **puede verificar al modelo en vez de repetirlo** ("el modelo dice z=0.42; a ese z, Hα debería estar en 9320 Å y hay un pico en 9318 Å → consistente").

### 2.2 El agente (`agent.py`) — Claude API con tool use

- **Modelo por defecto:** `claude-opus-4-8` (US$ 5 / 25 por millón de tokens in/out). Para demos públicas de bajo costo o barridos de evals: `claude-haiku-4-5` (US$ 1 / 5); punto medio: `claude-sonnet-5` (US$ 3 / 15; precio intro 2/10 hasta el 31-08-2026). Costo típico de un análisis completo (~10k tokens in, ~2k out): **~US$ 0.10 con Opus, ~US$ 0.02 con Haiku**.
- **Patrón recomendado:** el *tool runner* del SDK (beta) — define las tools como funciones tipadas y maneja el loop solo:

```python
import json
import anthropic
from anthropic import beta_tool

client = anthropic.Anthropic()   # ANTHROPIC_API_KEY en el entorno

@beta_tool
def predict_redshift(npz_path: str) -> str:
    """Run the DESI foundation model on a spectrum file.

    Args:
        npz_path: path to a .npz file with 'flux' and 'wavelength' arrays.
    """
    return json.dumps(_predict_redshift_impl(npz_path))

# ... @beta_tool para reconstruct_spectrum, identify_spectral_lines ...

runner = client.beta.messages.tool_runner(
    model="claude-opus-4-8",
    max_tokens=16000,
    thinking={"type": "adaptive"},
    system=REPORT_SYSTEM_PROMPT,
    tools=[predict_redshift, reconstruct_spectrum, identify_spectral_lines],
    messages=[{"role": "user", "content": f"Analiza el espectro en {path} y escribí el reporte."}],
)
for message in runner:
    ...   # el último message es el reporte final
```

- **System prompt** (esqueleto para `report.py`): rol de astrónomo asistente; flujo obligatorio (predecir → validar líneas → reconstruir si hay huecos → reportar); **cada afirmación del reporte debe citar qué herramienta la respalda**; si `match_fraction` es baja o `confidence` es baja, decirlo explícitamente en vez de afirmar el z.
- Para la UI en streaming, usar `client.messages.stream(...)` con el loop manual (el tool runner devuelve mensajes completos).

### 2.3 El servidor MCP (`mcp_server.py`) — FastMCP

El mismo toolset, expuesto al ecosistema de agentes. Con FastMCP son ~40 líneas:

```python
# pip install "mcp[cli]"
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("desi-fm")

@mcp.tool()
def predict_redshift(npz_path: str) -> dict:
    """Predict the redshift of spectra in a .npz file (flux + wavelength)."""
    return _predict_redshift_impl(npz_path)

@mcp.tool()
def identify_spectral_lines(npz_path: str, z: float) -> dict:
    """Check which known emission/absorption lines match peaks at redshift z."""
    return _identify_lines_impl(npz_path, z)

if __name__ == "__main__":
    mcp.run()   # transporte stdio
```

Registrarlo:

```bash
# En Claude Code:
claude mcp add desi-fm -- python3 /ruta/a/spectra-copilot/copilot/mcp_server.py
```

```json
// En Claude Desktop (claude_desktop_config.json):
{"mcpServers": {"desi-fm": {
  "command": "python3",
  "args": ["/ruta/a/spectra-copilot/copilot/mcp_server.py"]
}}}
```

Con eso, cualquier cliente MCP puede usar tu foundation model como herramienta. *"Escribí un servidor MCP para mi propio modelo"* es la línea de CV más actual de todo el plan.

### 2.4 UI de chat + demo

`app.py` con `gr.ChatInterface` envolviendo el agente (subís un `.npz`, conversás sobre el espectro). Se puede publicar como tercer HF Space — atención al costo: poner el API key como secret del Space, limitar a Haiku y/o cap de requests, o dejarlo solo como video/GIF en el README si no querés pagar el uso público.

### Criterios de aceptación del Nivel 2

- [ ] `python -m copilot.agent examples/galaxy.npz` produce un reporte que cita ≥ 2 herramientas.
- [ ] El agente **detecta** una inconsistencia sembrada (espectro con z ambiguo → el reporte menciona la baja confianza / doble modo).
- [ ] `claude mcp add` + una conversación en Claude Code usando las tools funciona (screenshot para el README).
- [ ] README del repo con GIF del flujo completo.

---

## Nivel 3 — Profundidad técnica

### 3.1 Búsqueda semántica de espectros (embeddings + FAISS)

Demuestra que entendés *para qué sirve* un foundation model: representaciones reutilizables.

1. **`encode()` en `desi_fm`** (~15 líneas nuevas en `model.py`): devolver el mean-pooling de `spectrum_hidden` sobre tokens válidos (y opcionalmente el hidden del token de z como segunda variante). Exponerlo en `predict.py` como `embed_spectrum(...) -> (d_model,)`.
2. **`scripts/build_index.py`**: streamear 10–20k espectros de MMU/DESI → embeddings float32 L2-normalizados → `faiss.IndexFlatIP` + metadata (`targetid`, `z`) en un `.npz`. Una sola corrida, ~30–60 min.
3. **Tool `find_similar_spectra`** para el agente + comando CLI de demo.
4. **Visual de portfolio:** UMAP 2D de los embeddings coloreado por z — si el modelo aprendió física, el gradiente de z se ve a ojo. Esa imagen va al README.

### 3.2 Evals del agente (la habilidad que más separa candidatos)

1. **`eval/cases/`**: exportar una vez ~150 espectros reales con z de pipeline (streaming → `.npz` + `labels.csv`).
2. **Salida estructurada del reporte**: el agente termina llamando una tool `submit_report(z: float, lines: list[str], confidence: str)` — así el eval parsea campos, no prosa. (Alternativa: `client.messages.parse()` con un schema Pydantic.)
3. **`run_evals.py`**: corre el agente end-to-end por caso y mide:
   - tasa de acierto en z: |Δz|/(1+z) < 0.15 (y < 0.05),
   - precisión/recall de líneas identificadas vs las esperables dado z_true,
   - % de reportes que citan herramientas (regla anti-alucinación).
4. Para abaratar barridos: `claude-haiku-4-5`, o la **Batches API** (50 % de descuento, ideal para 150 casos).
5. Resultados en `eval/results.csv` + tabla en el README. Correr el eval **antes y después** de cambiar el system prompt = demostración de metodología, no vibes.

### 3.3 Mini-RAG enfocado

Chico y honesto: un catálogo de líneas espectrales (JSON con λ_rest, nombre, tipo de objeto típico) + 10–20 abstracts/fragmentos de papers de DESI EDR. Retrieval simple (BM25 con `rank_bm25`, o sentence-transformers local si querés embeddings) detrás de una tool `lookup_reference(query)`. El reporte del agente pasa a **citar fuentes**. No hace falta más escala — el punto es mostrar el patrón completo: chunking → índice → retrieval → cita.

---

## La narrativa final

**Pitch de una línea (EN, para CV/LinkedIn):**

> "Trained a 26M-parameter foundation model for astronomical spectra from scratch (PyTorch), shipped it as a live demo and containerized API with CI, and built an LLM agent that uses the model as a tool — exposed via MCP — to generate validated observation reports, with an end-to-end eval harness."

**Bullets de CV (elegir 3–4):**

- Diseñé y entrené un transformer de 26M parámetros con masked-token prediction sobre 50k espectros DESI; rediseñé la cabeza de redshift (clasificación sobre bins de log(1+z)) reduciendo la fracción de outliers catastróficos de 25 % a X %.
- Publiqué el modelo en Hugging Face Hub con demo Gradio en vivo y API FastAPI dockerizada con CI.
- Construí un agente con la Claude API que usa el modelo como herramienta y valida sus predicciones contra líneas espectrales físicas; expuesto como servidor MCP.
- Armé un harness de evals end-to-end (150 casos etiquetados) para medir la tasa de acierto del sistema agéntico completo.

**Cómo contar el feedback del TA** (¡a favor!): "la v1 tenía bias alto y 25 % de outliers; lo diagnostiqué con métricas estándar (σ_NMAD, η₀.₁₅), identifiqué regresión-a-la-media + multimodalidad por confusión de líneas, y la v2 con cabeza de clasificación lo llevó a X %". Iterar sobre feedback con números es exactamente lo que hace un ingeniero senior.

**Material visual para el README/LinkedIn:** GIF de la demo Gradio · scatter antes/después de la Fase 0 · screenshot del agente citando herramientas en Claude Code vía MCP · UMAP coloreado por z.

---

## Cronograma sugerido

| cuándo | qué | entregable visible |
|---|---|---|
| Finde 1 | Fase 0.3 (cabeza de clasificación + reentrenar) + 1.1–1.2 | repo público + checkpoint v2 en HF Hub + notebook antes/después |
| Finde 2 | 1.3–1.5 | demo Gradio viva + API + badge de CI |
| Finde 3 | Nivel 2 completo | agente + MCP + GIF del flujo |
| Finde 4 | Nivel 3 + pulido | tabla de evals + UMAP + post de LinkedIn |

Si solo hubiera tiempo para la mitad: **Fase 0 + Nivel 1 + el MCP server del Nivel 2** son el 80 % del valor.

---

## Apéndice: checklist global

- [ ] **Fase 0** — ✅ notebook · ⬜ cabeza de clasificación · ⬜ rebalanceo · ⬜ reentrenar · ⬜ notebook antes/después
- [ ] **Nivel 1** — ⬜ git init + limpieza + GitHub · ⬜ HF Hub + model card · ⬜ Space Gradio · ⬜ FastAPI + Docker · ⬜ CI
- [ ] **Nivel 2** — ⬜ repo spectra-copilot · ⬜ tools.py · ⬜ agente (tool runner) · ⬜ MCP server · ⬜ UI/GIF
- [ ] **Nivel 3** — ⬜ encode() + FAISS · ⬜ evals (150 casos) · ⬜ mini-RAG
- [ ] **Narrativa** — ⬜ README principal reescrito para portfolio · ⬜ CV/LinkedIn actualizados

*Precios y patrones de la Claude API verificados al 2026-07-04 (model ids: `claude-opus-4-8`, `claude-sonnet-5`, `claude-haiku-4-5`; tool runner beta del SDK Python; FastMCP del paquete `mcp`).*
