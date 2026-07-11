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

- [ ] Ambos READMEs reescritos con links arriba, números reales y ≥ 3 assets visuales.
- [ ] Pitch EN/ES escritos y los bullets con números finales.
- [ ] Perfiles (GitHub/HF/LinkedIn) actualizados; repos pineados.
- [ ] Test de los 60 segundos pasado con una persona real.
- [ ] Tracker: fila 13 en ✅ — portfolio listo.
