# 13 · Narrativa final del portfolio

> **Bloque:** Cierre · **Tiempo:** ~2 h · **Depende de:** lo que exista de 01–12 (se puede hacer con la ruta mínima completada) · **Entregable:** README maestro + pitch + bullets de CV + perfil ordenado

## Objetivo

Que un reclutador (o un tech lead haciendo screening) entienda en **60 segundos** qué construiste, qué demuestra, y dónde clickear. Todo lo técnico ya existe; este plan es puro empaquetado — y el empaquetado decide si lo demás se ve.

## Pasos

### 1. README maestro de `desi-spectra-fm` (reescritura)

Estructura objetivo (el contenido grader-facing actual se conserva más abajo o en `docs/`):

```markdown
# DESI Spectra Foundation Model
[badge CI] [badge HF model] [badge demo]

Transformer de 26M entrenado desde cero con masked-token prediction sobre espectros
del telescopio DESI. Predice redshift y reconstruye regiones enmascaradas de
espectros de cualquier instrumento.

**[▶ Demo en vivo]** · **[🤗 Modelo]** · **[📡 API]** · **[🤖 Agente (spectra-copilot)]** · **[📓 Notebook de evaluación]**

![gif de la demo](docs/img/demo.gif)

## Resultados
(tabla v1 → v2 con η₀.₁₅, bias, σ_NMAD + los dos scatter lado a lado)
La v1 tenía 25 % de outliers catastróficos por regresión-a-la-media; la v2
(cabeza de clasificación sobre bins de log(1+z)) lo bajó a X %. [Diagnóstico completo →notebook]

## Arquitectura (diagrama de 6 líneas: espectro → log-λ grid → 273 tokens → transformer → 2 cabezas)
## Quick start (3 comandos)
## Cómo se entrenó / Reproducir
## Stack: PyTorch · HF Hub/Spaces · FastAPI · Docker · GitHub Actions
```

Reglas: los links viven arriba del fold; cada afirmación de resultados tiene número; cero jerga sin explicar en las primeras 10 líneas.

### 2. README de `spectra-copilot`

Mismo tratamiento: qué es (1 párrafo), GIF del agente analizando un espectro, reporte de ejemplo real (el del caso ambiguo del plan 08 — muestra juicio, no solo éxito), tabla de evals (plan 11), sección MCP con screenshot (plan 09), UMAP (plan 10). Stack: Claude API (tool use) · MCP · FAISS · evals.

### 3. Los assets visuales (checklist)

- [ ] `docs/img/demo.gif` — grabar la demo Gradio (QuickTime → File > New Screen Recording; convertir con `ffmpeg -i in.mov -vf "fps=10,scale=900:-1" demo.gif` o usar Kap). 10–15 s: click en ejemplo → z + reconstrucción → mover slider.
- [ ] `docs/img/scatter_v1.png` + `scatter_v2.png` lado a lado (del plan 02).
- [ ] `docs/img/mcp-session.png` (del plan 09).
- [ ] `docs/img/umap_z.png` (del plan 10).
- [ ] GIF corto del agente en terminal (reporte generándose) para spectra-copilot.

### 4. El pitch (memorizable, 2 versiones)

**EN (CV/LinkedIn):**
> Trained a 26M-parameter foundation model for astronomical spectra from scratch (PyTorch), shipped it as a live demo and containerized API with CI, and built an LLM agent that uses the model as a tool — exposed via MCP — to generate physically-validated observation reports, with an end-to-end eval harness.

**ES (entrevista, 30 segundos):**
> Entrené un foundation model para espectros astronómicos y, en vez de dejarlo en un notebook, lo convertí en producto: demo pública, API dockerizada con CI, y un agente LLM que usa el modelo como herramienta vía MCP y *verifica* sus predicciones contra la física de líneas espectrales. Cuando el TA me marcó bias y outliers, lo diagnostiqué con métricas estándar, rediseñé la cabeza de salida y documenté la mejora con evals.

### 5. Bullets de CV (elegir 3–4, con los números reales)

- Designed and trained a 26M-parameter transformer with masked-token pretraining on 150k DESI spectra; redesigned the redshift head (classification over log(1+z) bins), cutting catastrophic outliers from 25% to X%.
- Shipped the model publicly: Hugging Face Hub + live Gradio demo + Dockerized FastAPI service with CI (GitHub Actions).
- Built an LLM agent (Claude API, tool use) that validates model predictions against spectral-line physics; exposed the toolset as an MCP server usable from any MCP client.
- Evaluated the full agentic system on 150 labeled spectra with a structured-output eval harness (agent recovered X% of the model's catastrophic outliers).

### 6. Higiene de perfiles

- GitHub: pin de `desi-spectra-fm` y `spectra-copilot`; bio con una línea; el resto de repos de curso en privado si hacen ruido.
- HF: perfil con link a GitHub; el Space y el modelo con thumbnails decentes.
- LinkedIn: post corto (el pitch EN + el GIF + 2 links). Publicarlo un martes/miércoles a la mañana, etiqueta #buildinpublic si te cabe.
- El screenshot del feedback del TA **no** se publica; la historia se cuenta con tus números ("v1 → v2").

### 7. El test final

Pasale los dos repos a alguien que no sea del palo y tomale el tiempo: ¿en 60 segundos puede decir qué hace el proyecto y encontrar la demo? Si no, el problema está en el primer párrafo o en el orden de los links — iterar ahí.

## Definición de hecho

- [x] Ambos READMEs reescritos con links arriba, números reales y ≥ 3 assets visuales.
- [x] Pitch EN/ES escritos y los bullets con números finales.
- [~] Perfiles: GitHub repos con descripción/homepage/topics ✅ vía `gh repo edit`; **pins, HF y LinkedIn quedan manuales** (ver checklist abajo).
- [ ] Test de los 60 segundos pasado con una persona real (manual).
- [x] Tracker: fila 13 actualizada (✅ con pendientes humanos anotados).

---

## Ejecución — 2026-08-18

### Qué se hizo (automatizable, sin API)

1. **README maestro de `desi-spectra-fm` reescrito** con la estructura del plan: badges (CI + HF model + Space + API), párrafo sin jerga, 5 links arriba del fold, screenshot real de la demo, sección Results con la historia diagnóstico→rediseño y la tabla v1→v2.1, UMAP como "foundation-model claim", sección "From model to product" (con el titular honesto de las evals), diagrama de arquitectura de 6 líneas, quick start de 3 comandos, cómo se entrenó, y todo el contenido grader-facing conservado bajo "Reference" + "Academic context".
2. **README de `spectra-copilot`**: badge de CI, links al ecosistema del modelo y bloque "The 60-second version" (5 tools + MCP; evals honestas 92.7 vs 79.3 → híbrido 90.0; RAG 6/6 citas respaldadas; 35 tests offline). El cuerpo ya cumplía el plan (reporte del caso ambiguo, tabla de evals, MCP, UMAP).
3. **Assets nuevos**:
   - `docs/img/demo.png` — screenshot real de la demo viva (Playwright: click en ejemplo + Submit + predicción renderizada: z_pred_map 0.2267, z_true 0.2036).
   - `docs/img/scatter_v1_v2.png` — scatter held-out lado a lado v1 (η 22.6 %, techo z=2) vs v2.1 (η 14.9 %); regenerable con `scripts/plot_scatter_v1_v2.py` (nuevo, offline, usa los CSVs commiteados).
   - `docs/img/umap_z.png` ya existía. Total ≥ 3 ✅.
4. **GitHub repo metadata** (vía `gh repo edit`): descripción con números + homepage (demo) + topics nuevos en ambos repos (`huggingface`, `astrophysics`, `fastapi`, `redshift`, `spectroscopy` / `llm-agents`, `claude`, `mcp`, `model-context-protocol`, `rag`, `faiss`, `evals`, `astronomy`, `anthropic` — spectra-copilot no tenía ninguno).

### Pitch (versiones finales)

**EN (CV/LinkedIn, 1–2 oraciones):**

> Trained a 26M-parameter foundation model for astronomical spectra from scratch (PyTorch), shipped it as a live demo and containerized API with CI, and built an LLM agent that uses the model as a tool — exposed via MCP — to write physically-verified observation reports. An end-to-end eval on 150 held-out spectra caught the cheap-LLM agent underperforming the bare model (79.3 % vs 92.7 %) — and its calibrated confidence signal yields a hybrid policy that recovers 90 %.

**ES (entrevista, ~30 s):**

> Entrené un foundation model de 26M de parámetros para espectros astronómicos y, en vez de dejarlo en un notebook, lo convertí en producto: demo pública, API dockerizada con CI, y un agente LLM que usa el modelo como herramienta vía MCP y *verifica* sus predicciones contra la física de líneas espectrales. Cuando el TA me marcó bias y outliers, lo diagnostiqué con métricas estándar de surveys, rediseñé la cabeza de salida y bajé los outliers catastróficos de 22.6 % a 15 %. Y lo que más me enorgullece: escribí evals de punta a punta sobre 150 espectros que demostraron que mi agente barato *empeoraba* al modelo — y encontré en su señal de confianza la política híbrida que lo arregla. Las demos se veían perfectas; la eval dijo la verdad.

### Bullets de CV (elegir 3–4)

- Designed and trained a 26M-parameter transformer for galaxy spectra with masked-token pretraining on 80k DESI spectra (PyTorch, single laptop); diagnosed regression-to-the-mean in the redshift head with survey metrics (η, σ_NMAD) and redesigned it as 100-bin classification over log(1+z), cutting catastrophic outliers from 22.6 % to 14.95 % (82.7 % → 23.5 % in the z ∈ [1.5, 2.5) quasar band).
- Shipped the model publicly: Hugging Face Hub checkpoint + live Gradio demo + Dockerized FastAPI service, all CI-tested (GitHub Actions, 26 tests).
- Built an LLM agent (Claude API, tool use) that verifies model predictions against spectral-line physics, FAISS nearest-neighbors over a 15k-spectrum index, and a cited BM25 mini-RAG (0 hallucinated citations across all verified runs); exposed the 5-tool set as an MCP server usable from any MCP client.
- Built an end-to-end eval harness (150 labeled held-out spectra, structured output, US$ 4.06 measured): it caught the cheap-LLM agent underperforming the bare model (79.3 % vs 92.7 %) and showed its self-reported confidence tracks accuracy monotonically, yielding a confidence-gated hybrid that recovers 90.0 %.

### Post de LinkedIn (borrador listo para pegar)

> I trained a 26M-parameter foundation model for astronomical spectra from scratch — and then treated it like a product, not a notebook.
>
> 🔭 It reads a galaxy/quasar spectrum from any instrument, predicts its redshift, and reconstructs masked regions.
> 🚀 Shipped: live demo (HF Spaces) + Dockerized FastAPI + CI.
> 🤖 Then I built an LLM agent (Claude tool use + MCP) that verifies the model's predictions against spectral-line physics, FAISS neighbors, and a cited reference corpus.
> 📊 The best part: end-to-end evals on 150 held-out spectra showed my cheap-LLM agent actually *underperformed* the bare model (79.3 % vs 92.7 %) — but its confidence is calibrated, and a "trust the agent only when it's confident" hybrid recovers 90 %. The demos looked great; the eval told the truth.
>
> Demo: https://huggingface.co/spaces/jirustaroure/desi-spectra-fm-demo
> Model: https://github.com/Julian0444/desi-spectra-fm
> Agent: https://github.com/Julian0444/spectra-copilot
>
> #buildinpublic #machinelearning #astronomy

(Publicar martes/miércoles a la mañana; idealmente con el GIF de la demo cuando exista.)

### Checklist manual para Julián (lo que ninguna API puede hacer)

- [ ] **Pinear repos en GitHub** — hoy los 6 pins son repos viejos y NO incluyen los dos del proyecto. En <https://github.com/Julian0444> → "Customize your pins" → agregar `desi-spectra-fm` y `spectra-copilot` primeros (sacar dos de los actuales).
- [ ] **Bio de GitHub** de una línea (sugerida): "Building ML systems end-to-end — foundation models, LLM agents, evals. USF."
- [ ] **Perfil de HF** (<https://huggingface.co/jirustaroure>): agregar link a GitHub en el perfil.
- [ ] **GIF de la demo** (`docs/img/demo.gif`, 10–15 s: click en ejemplo → z + reconstrucción → mover slider): QuickTime → New Screen Recording, después `ffmpeg -i in.mov -vf "fps=10,scale=900:-1" demo.gif` (o Kap). Reemplaza el `demo.png` del README (o se agrega debajo).
- [ ] **GIF corto del agente en terminal** para spectra-copilot (reporte generándose) — mismo método.
- [ ] **Screenshot de la sesión MCP** (`docs/img/mcp-session.png`) — las tools `desi-fm` corriendo dentro de Claude Code.
- [ ] **Post de LinkedIn** (borrador arriba).
- [ ] **Test de los 60 segundos** con una persona no técnica: ¿puede decir qué hace el proyecto y encontrar la demo en 60 s? Si no → iterar el primer párrafo / orden de links.

### Recordatorio

El screenshot del feedback del TA **no** se publica; la historia se cuenta con los números propios (v1 → v2.1), y así quedó escrita en ambos READMEs.
