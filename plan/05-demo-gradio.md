# 05 · Demo Gradio en HF Spaces

> **Bloque:** Nivel 1 · **Tiempo:** 2–3 h · **Depende de:** 04 · **Entregable:** **link de demo en vivo** — la pieza más valiosa del Nivel 1
>
> **Namespaces (regla permanente):** GitHub **`Julian0444`** para código y enlaces; Hugging Face **`jirustaroure`** para el modelo y el Space. CLI `hf` fuera de PATH: `~/Library/Python/3.9/bin/hf`.

## Objetivo

Un Space público donde cualquiera (recruiter incluido) elige un espectro de ejemplo o sube el suyo, mueve un slider de masking, y ve la reconstrucción + el z predicho (**`z_pred_map` oficial**, con `z_confidence`). El modelo es de 26M y tarda ~1–3 s por espectro.

> **Cambio de política de HF (2026):** los Spaces Gradio/Docker en cpu-basic ya **no** son gratis (requieren PRO). El camino gratis para cuentas personales en regla es **ZeroGPU** (hasta 2 Spaces Gradio): se crea el Space con `space_hardware="zero-a10g"` y se decora la función de inferencia con `@spaces.GPU` (no-op fuera de Spaces). Torch soportado 2.8+ (acá pineado `torch==2.11.0`, el último listado en la doc).

## Cómo se ejecutó (runbook real)

### 1. Espectros de ejemplo — reales del held-out, no sintéticos

El plan original proponía espectros sintéticos con z conocido. **Se probaron primero contra el checkpoint v2.1 y el modelo colapsa a z≈3.4 en los 4** (son demasiado out-of-distribution: continuo plano + gaussianas de 5 Å no se parecen a un espectro DESI). Una demo honesta no puede mostrar eso como ejemplos canónicos, así que los ejemplos son **4 espectros DESI reales del held-out canónico** (stream `filter(z válido) → skip(80000) → take(...)`, el mismo orden que `predictions.csv`), exportados **crudos** (flux/ivar/wavelength/mask + `z_true` del pipeline) antes de cualquier preprocesado:

La selección final se hizo con predicciones a **protocolo de demo** (`mask_ratio=0`, determinista) sobre los primeros 41 held-out — no con los números del CSV, que vienen de forwards con el masking interno del 50 % y difieren:

| archivo | índice held-out | z_true | z_pred_map (demo) | z_confidence | por qué |
|---|---|---|---|---|---|
| `heldout_z020.npz` | 12 | 0.204 | 0.227 | 0.64 | z bajo bien predicho, confianza alta |
| `heldout_z083.npz` | 27 | 0.835 | 0.810 | 0.61 | z medio bien predicho (preset del slider en 0.35 para mostrar el masking) |
| `heldout_z287.npz` | 10 | 2.866 | 2.441 | 0.35 | z alto — arriba del techo z≈2 de la v1, con el delta honesto a la vista |
| `heldout_lowconf_z157.npz` | 40 | 1.574 | 0.957 | **0.18** | **outlier catastrófico honesto** en la banda débil z∈[1.5,2.5); la confianza baja lo delata |

```bash
python3 scripts/make_demo_examples.py          # streamea el held-out y escribe examples/*.npz
python3 scripts/make_demo_examples.py --synthetic  # fallback sintético offline (no lo usa la demo)
```

`examples/` no se commitea (`.gitignore`); las copias públicas viven en el Space. El skip de 80k tarda 15 min–2.5 h según el día del CDN de HF.

### 2. La app vive en `demo/` del repo principal

`demo/app.py` + `demo/requirements.txt` + `demo/README.md` (header YAML del Space) están versionados acá — el Space es una copia publicada. Puntos clave de `app.py`:

- Descarga el checkpoint con `hf_hub_download("jirustaroure/desi-spectra-fm", "checkpoint_last.pt")` al arrancar y corre en CPU.
- **Salida honesta:** `z_pred_map` como titular + `z_confidence` + `z_pred` (esperanza, secundario) + `z_true` si viene en el `.npz`, con `|Δz|/(1+z)` y el flag de outlier catastrófico (>0.15).
- El plot sombrea las regiones enmascaradas (tokens ocultos agrupados en spans contiguos) sobre input + reconstrucción.
- Robusto a `.npz` ajenos: `flux`/`wavelength` 1-D o `(N, P)` (usa el primero y lo avisa), `ivar`/`mask` opcionales, grillas de cualquier instrumento, errores amigables con `gr.Error` (claves faltantes, shapes inconsistentes, archivos ilegibles).
- `demo.queue(max_size=8)` para cargas concurrentes.

`requirements.txt` del Space (gradio lo pone el SDK; `sdk_version: 6.22.0` pineado = el probado localmente; `spaces` + `torch==2.11.0` por ZeroGPU):

```
spaces
numpy
matplotlib
huggingface_hub
torch==2.11.0
desi-fm @ git+https://github.com/Julian0444/desi-spectra-fm
```

En `app.py`: `import spaces` **antes** de torch y `@spaces.GPU(duration=15)` sobre `analyze` — localmente es no-op (verificado re-corriendo el test de usuario con el decorador puesto).

### 3. Prueba local ANTES de publicar (obligatoria)

- `pip install 'desi-fm @ git+https://github.com/Julian0444/desi-spectra-fm'` en un venv 3.12 limpio (lo que hará el build del Space) — OK.
- Server local (`GRADIO_SERVER_PORT=7860 python app.py` en un staging con `examples/`) + test de usuario con Playwright: los 4 ejemplos con un click, slider 0.5 recalcula, `.npz` subido a mano `(3, 4000)` en grilla ajena (aviso "showing the first one"), archivo roto → toast de error amigable y la app sigue viva.
- **Ojo con servers huérfanos:** si el proceso de prueba anterior no murió, el nuevo cae en otro puerto y terminás testeando una instancia vieja (gradio salta de puerto en silencio). Matar listeners (`lsof -nP -iTCP:7860-7875 -sTCP:LISTEN`) antes de re-testear; con `with_server.py` usar `exec` en el comando del server.
- Chequeo de honestidad con el checkpoint local: correr los 4 ejemplos por `predict_spectrum` con el `mask_ratio` que usa cada fila de ejemplos y verificar que los valores mostrados cuadren con lo que se afirma en la descripción.

### 4. Crear el Space y publicar

**Ojo:** `hf repo create ... --repo-type space --space_sdk gradio` (sin hardware) devuelve **402 Payment Required** en cuentas gratis, y `hf upload` también (hace un `create_repo(exist_ok=True)` interno que 402ea aunque el repo exista). Se usa la API Python:

```python
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("jirustaroure/desi-spectra-fm-demo", repo_type="space",
                space_sdk="gradio", space_hardware="zero-a10g")   # ZeroGPU = gratis
api.upload_folder(folder_path="<staging>", repo_id="jirustaroure/desi-spectra-fm-demo",
                  repo_type="space", commit_message="...")
```

El staging = `demo/*` + `examples/*.npz` en `examples/`. El build tarda ~5–10 min la primera vez (instala torch + clona el repo de GitHub).

### 5. Probar como usuario

- Runtime `RUNNING` vía API pública anónima (`https://huggingface.co/api/spaces/jirustaroure/desi-spectra-fm-demo`).
- Abrir la página, click en cada ejemplo, mover el slider, subir un `.npz` propio.
- Link arriba de todo del README de GitHub y en la model card (re-subir la card al Hub).

## Definición de hecho

- [x] Space público, build verde, análisis en < 5 s por click — <https://huggingface.co/spaces/jirustaroure/desi-spectra-fm-demo>, `RUNNING` en `zero-a10g`; latencias medidas contra el Space en vivo: 0.5–4.3 s (UI anónima 1.8/0.5/3.4 s; API autenticada 1.5–3.1 s; upload externo 3.45–4.3 s).
- [x] 4 ejemplos funcionan con un click (UI local Playwright `USER_TEST_OK` + UI pública con valores honestos + API en vivo: z020→0.2267, z083@0.35→0.8822, z287→2.4406, lowconf_z157→0.9569); el slider cambia visiblemente la reconstrucción (sombreado de tokens ocultos; z020 a mask 0.6 → 0.1571 ≠ 0.2267 a mask 0).
- [x] Link agregado al README de GitHub (arriba de todo + notas + layout) y a la model card (local y re-subida al Hub, commit `32463db`).
- [x] Tracker actualizado (05 ✅ con URL del Space).

**Limitación conocida (plataforma, no bug):** los visitantes anónimos de cualquier Space ZeroGPU tienen una cuota diaria chica de runs — al agotarla ven "ZeroGPU quota exceeded" hasta el día siguiente (logueado en hf.co la cuota es mayor). La card del Space lo avisa.

## Si algo falla

- **Build del Space falla instalando `desi-fm`:** el repo de GitHub debe ser público y el `pyproject.toml` instalable — probado localmente primero con el mismo `pip install "desi-fm @ git+..."`.
- **Timeout al arrancar (descarga del checkpoint):** es normal la primera vez; si persiste, lazy-load (cargar el modelo dentro de `analyze` con `@lru_cache`).
- **La demo queda lenta en cargas concurrentes:** ya está `demo.queue(max_size=8)`.
- **Los ejemplos sintéticos predicen z≈3.4:** no es un bug de la demo — el modelo nunca vio espectros así; usar espectros DESI reales (sección 1).
- **"Error" pelado al clickear en el Space con la API/local OK:** mirar el SSE de `queue/data` en el navegador (o los logs de run como owner). Dos causas vistas: (a) **"You have exceeded your ZeroGPU runs limit"** — cuota diaria de visitantes anónimos agotada (p. ej. por probar mucho desde la misma IP); logueado en hf.co la cuota es mayor, y al día siguiente se resetea. No es un bug de la app. (b) `InvalidPathError` con paths de `examples/` — cubierto con `allowed_paths` en `launch()`.
- **RUNTIME_ERROR con `ValueError: Invalid file descriptor: -1` justo tras el launch:** en hardware `zero-a10g` es **obligatorio** tener al menos una función decorada con `@spaces.GPU` — la variante "CPU sin decorador para no gastar cuota" no arranca.
- **Un commit al Space NO reinicia el contenedor solo:** después de subir archivos, `HfApi().restart_space(...)` explícito; verificar en los logs que aparezca un nuevo "Application Startup".
