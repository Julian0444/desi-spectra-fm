# Tracker de ejecución — portfolio desi_fm

> Cada plan numerado es **auto-contenido y completable en una sentada** (2–4 h). Se hacen en orden salvo que la columna "depende de" diga otra cosa. Al terminar uno: marcar acá, commitear, y recién ahí abrir el siguiente.
>
> La visión general (contexto, diagnóstico, narrativa) está en [`PLAN.md`](PLAN.md). Este archivo es el tablero.

## Estado

**Ya completado antes de este tracker (2026-07-04):**
- ✅ Notebook de evaluación ready-to-run con plots (`notebooks/evaluation.ipynb`, pre-ejecutado) — corrección del TA, parte 2.
- ✅ Diagnóstico cuantificado del bias/outliers (notebook §1 + `PLAN.md` Fase 0.2) — corrección del TA, parte 1 (diagnóstico).

| # | Plan | Bloque | Tiempo | Depende de | Entregable visible | Estado |
|---|---|---|---|---|---|---|
| 01 | [Cabeza de clasificación para el redshift](01-cabeza-clasificacion.md) | Fase 0 | 2–3 h | — | tests verdes + smoke run con la cabeza nueva | ✅ |
| 02 | [Reentrenamiento v2 + antes/después](02-reentrenamiento-v2.md) → cerrado vía [02R](02R-reentrenamiento-v2-calibrado.md) | Fase 0 | 1 h activa + 3–4 h cómputo | 01 | checkpoint v2.1 promovido (`runs/desi_80k_classhead_v21`) + notebook re-ejecutado + tabla v1→v2.1 | ✅ |
| 03 | [Repo público en GitHub + CI](03-repo-publico-ci.md) | Nivel 1 | 1.5–2 h | — | repo público con Actions en verde — <https://github.com/Julian0444/desi-spectra-fm> | ✅ |
| 04 | [Checkpoint en Hugging Face Hub](04-checkpoint-hf-hub.md) | Nivel 1 | 1 h | 03 (ideal: 02) | model card pública + descarga funcionando — <https://huggingface.co/jirustaroure/desi-spectra-fm> | ✅ |
| 05 | [Demo Gradio en HF Spaces](05-demo-gradio.md) | Nivel 1 | 2–3 h | 04 | **link de demo en vivo** — <https://huggingface.co/spaces/jirustaroure/desi-spectra-fm-demo> | ✅ |
| 06 | [API FastAPI + Docker](06-api-fastapi-docker.md) | Nivel 1 | 2–3 h | 04 | endpoint público respondiendo a `curl` — <https://jirustaroure-desi-fm-api.hf.space/api/docs> | ✅ |
| 07 | [spectra-copilot: repo + herramientas](07-spectra-copilot-tools.md) | Nivel 2 | 3–4 h | 04 | `tools.py` testeado + CLI de demo — <https://github.com/Julian0444/spectra-copilot> | ✅ |
| 08 | [El agente con la Claude API](08-agente-claude.md) | Nivel 2 | 2–3 h | 07 | reporte de observación generado por el agente — [reportes en el README de spectra-copilot](https://github.com/Julian0444/spectra-copilot#the-agent-claude-api) | ✅ |
| 09 | [Servidor MCP](09-mcp-server.md) | Nivel 2 | 1–2 h | 07 | tus tools corriendo dentro de Claude Code/Desktop | ⬜ |
| 10 | [Embeddings + búsqueda semántica (FAISS)](10-embeddings-faiss.md) | Nivel 3 | 3–4 h | 02, 07 | búsqueda de espectros similares + UMAP coloreado por z | ⬜ |
| 11 | [Evals del agente](11-evals-agente.md) | Nivel 3 | 3–4 h | 08 | tabla de métricas end-to-end sobre ≥100 casos | ⬜ |
| 12 | [Mini-RAG de referencias](12-mini-rag.md) | Nivel 3 | 2–3 h | 08 | reportes del agente con citas a fuentes | ⬜ |
| 13 | [Narrativa final del portfolio](13-narrativa-portfolio.md) | Cierre | 2 h | todo lo anterior que exista | README maestro + pitch + bullets de CV | ⬜ |

## Ruta mínima (si hay poco tiempo)

`01 → 02 → 03 → 04 → 05 → 09 → 13` — con eso ya tenés: modelo mejorado tras feedback, repo público con CI, demo clickeable y "escribí un servidor MCP para mi propio modelo". Es el 80 % del valor.

## Sprint de 2 días (hoy y mañana)

Son ~30 h de trabajo estimado, así que la clave es **pipelinear lo desatendido**: el entrenamiento (02), el indexado (10) y las evals (11) corren solos — se lanzan y se sigue con otro plan mientras tanto. Nunca quedarse mirando una barra de progreso: si algo espera (build de un Space, descarga, entrenamiento), se abre el siguiente plan.

### Hoy

| orden | qué | tiempo activo |
|---|---|---|
| 1 | **01** completo (código + tests + smoke) | 2–3 h |
| 2 | **Lanzar 02** — el entrenamiento queda en background (~2.5–4 h de cómputo) | 10 min |
| 3 | **03** repo público + CI, mientras entrena | 1.5–2 h |
| 4 | **04** con la **v1** — no esperar la v2, después se pisa (ver nota de sprint en el plan 04) | 45 min |
| 5 | **07** tools de spectra-copilot (con `DESI_FM_CKPT` local no depende de nada) | 3–4 h |
| 6 | Al terminar el entrenamiento: cerrar **02** (evaluar v2 + notebook antes/después) y re-subir al Hub (**04**) | 1 h |

### Mañana

| orden | qué | tiempo activo |
|---|---|---|
| 1 | **05** demo Gradio (lanzar el build y seguir) + **06** API | 4–5 h |
| 2 | **08** agente + **09** MCP (el screenshot del MCP es rápido y rinde muchísimo) | 3–4 h |
| 3 | **Lanzar 10** (indexado ~30–60 min) y **lanzar 11** (evals) en background | 30 min |
| 4 | **12** mini-RAG solo si sobra tiempo | 2 h |
| 5 | **13** narrativa con lo que exista — **nunca se recorta** | 1.5–2 h |

**Orden de recorte si mañana se acorta:** primero 12, después 10, después 11. El producto mínimo del sprint es 01–09 + 13.

## Reglas de uso

1. **Un plan en foreground por vez.** Lo desatendido (entrenamiento del 02, indexado del 10, evals del 11) se lanza y se sigue con el próximo plan mientras corre — ver el orden pipelineado del sprint.
2. Cada plan tiene una sección **"Definición de hecho"** — no se marca ✅ hasta cumplirla entera.
3. Al completar: actualizar la columna Estado acá y hacer commit con mensaje `plan-NN: <qué se completó>`.
4. Si un paso falla y la sección "Si algo falla" del plan no lo cubre, anotar el problema al pie del plan y seguir con otro plan no dependiente.
