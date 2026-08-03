# HANDOFF — registro de continuidad entre sesiones

> Log acumulativo (skill `/handoff`): cada sesión agrega una sección datada abajo, sin borrar las anteriores. Un lector sin contexto previo debería poder retomar el trabajo leyendo la última sección + el tracker [`plan/README.md`](README.md).

---

## Session — 2026-07-13 00:04 (trabajo del 2026-07-11)

### Resumen de la sesión

Se ejecutó completo el **plan 03 — Repo público en GitHub + CI**. El proyecto pasó de repo git local a repo público con CI en verde:

1. **Limpieza pre-publicación**: el material del curso (`PHYS303_Final-Project_20266.pdf`/`.docx`, `docs.zip`, `docs to finish omnicursor.zip`) se movió a `~/Documents/PHYS303-privado/`, y los 12 runs intermedios viejos (`desi_500*`, `desi_debug_*`, `desi_tiny*`, `smoke`, etc.) a `~/Documents/PHYS303-privado/runs-viejos/`. **Nada se borró** (el permiso de borrado fue denegado; mover resultó equivalente y reversible). Se conservaron localmente `runs/desi_50k_big` (v1), `runs/desi_50k_zw10_mask50` y `runs/smoke_classhead` (evidencia del plan 01).
2. **Referencias arregladas**: `README.md`, `README.es.md` y `DELIVERABLE.md` ya no citan el PDF de la consigna como archivo local; ahora resumen la consigna en una frase ("course material, not distributed in this repo").
3. **`.gitignore`** según plan 03: excluye checkpoints/zips/docx/`external/`; whitelistea los artefactos livianos de `runs/desi_50k_big/` y (a futuro) `runs/desi_150k_classhead/` — la carpeta del run v2 ya está prevista.
4. **Repo público creado y pusheado**: <https://github.com/Julian0444/desi-spectra-fm> (cuenta `Julian0444`, confirmado explícitamente por Julián), con descripción y topics (deep-learning, transformers, astronomy, foundation-models, pytorch, self-supervised-learning).
5. **CI (GitHub Actions)** en `.github/workflows/ci.yml`: Python 3.11 + torch CPU + `pytest -q` → **verde en ~46 s**, badge al tope del `README.md`.
6. **Prueba de clon limpio verificada**: clon fresco + venv + `pip install -r requirements.txt && pip install -e .` + `pytest` → **6 tests pasan**.
7. Tracker `plan/README.md` actualizado (03 ✅) y memoria persistente del proyecto actualizada.

### Estado actual

- **Branch**: `main`, sincronizada con `origin/main`, working tree limpio (solo `.claude/` sin trackear — ver Avisos).
- **Commits recientes**:
  - `68bdb75` (2026-07-11) docs: CI badge + mark plan 03 done in tracker
  - `96826fd` (2026-07-11) ci: run unit tests on push
  - `bee483b` (2026-07-11) DESI spectra foundation model: training, evaluation notebook, inference ← snapshot inicial público
  - `a4fa790` (2026-07-06) plan-01: classification head over log(1+z) bins (backward-compatible)
- **Tests**: 6/6 pasando (local, clon limpio y CI). Último run de Actions: success.
- **Tracker** (`plan/README.md`): 01 ✅ · 03 ✅ · resto ⬜. **No hay checkpoint v2 todavía** — el modelo público sigue siendo la v1 (`runs/desi_50k_big`, η₀.₁₅ = 25.1 %).

### Tareas pendientes (en orden)

1. **Plan 02 — Reentrenamiento v2** (`plan/02-reentrenamiento-v2.md`): lanzar el entrenamiento con la cabeza de clasificación (~10 min activos + 3–4 h de cómputo MPS desatendido; output esperado en `runs/desi_150k_classhead`, ya whitelisteado en `.gitignore`). Al terminar: evaluar, re-ejecutar `notebooks/evaluation.ipynb`, tabla antes/después v1→v2.
2. **Plan 04 — Checkpoint en HF Hub** (`plan/04-checkpoint-hf-hub.md`): se puede hacer con la **v1 sin esperar la v2** (nota de sprint del propio plan; después se pisa el checkpoint). Requiere `hf auth login`.
3. **Plan 05 — Demo Gradio en HF Spaces** (depende de 04).
4. **Plan 06 — API FastAPI + Docker** (depende de 04).
5. Planes 07–09 (spectra-copilot: tools → agente → MCP), luego 10–12 (FAISS, evals, RAG) y 13 (narrativa final — nunca se recorta).

### Avisos / bloqueos

- **Ningún bloqueo duro.** Cosas a saber:
- **pip viejo con Python 3.9 de sistema**: `pip install -e .` falla con pip 21 (no soporta editable desde `pyproject.toml`). En venvs nuevos: `pip install --upgrade pip` primero. CI usa Python 3.11 y no lo sufre.
- **`plan/` es público**: todo lo que se commitee ahí (incluido este archivo) se publica en GitHub al pushear. Este HANDOFF quedó **sin commitear** a propósito — decidir si commitearlo o agregarlo al `.gitignore`.
- **`.claude/` está sin trackear** (contiene el skill `handoff` creado el 2026-07-12): decidir si commitearlo (útil si se clona en otra máquina) o ignorarlo.
- **Material privado**: todo en `~/Documents/PHYS303-privado/` (consigna del profesor + runs viejos). No re-agregarlo al repo.
- **Permisos**: el clasificador de auto-mode deniega `rm -rf` de directorios no nombrados por el usuario y la creación de superficies públicas sin autorización explícita — pedir confirmación explícita a Julián antes de crear repos/Spaces públicos (para el 04/05 va a volver a pasar con HF Hub).
- Anotación cosmética en Actions: deprecación de Node 20 en `actions/checkout@v4`/`setup-python@v5` (corren forzadas en Node 24; no afecta, se puede ignorar o subir versiones de actions más adelante).

### Primera acción sugerida para la próxima sesión

Abrir `plan/02-reentrenamiento-v2.md` y **lanzar el entrenamiento v2 en background** (comando en PLAN.md §0.3 Cambio C: `python3 -m desi_fm.train ... --output-dir runs/desi_150k_classhead`); mientras corre, avanzar con el plan 04 usando el checkpoint v1.

---

## Session — 2026-07-30 11:59 (trabajo del 2026-07-29/30)

### Resumen de la sesión

Se ejecutó el **plan 02 (reentrenamiento v2)** de punta a punta — entrenamiento + evaluación honesta — con dos descubrimientos importantes que cambian los números de referencia del proyecto. **La v2 entrenó bien pero NO cumple los criterios de éxito**; queda una iteración pendiente ya diagnosticada.

**Descubrimiento 1 — el dataset es más chico de lo que asumía el plan.** `MultimodalUniverse/desi` con `data_dir=edr_sv3` rinde **~84.760 ejemplos**, no 150k. El primer intento (comando literal del plan, `--max-train-examples 150000`) consumió el dataset entero por época y dejó el split de validación vacío (skip 150000 > tamaño del stream → métricas de validación todas en cero, detectado por el monitor al fin de la época 1). Ese run se descartó y se relanzó con `--max-train-examples 80000 --val-examples 2000`: entrena con los primeros 80k y valida con los ejemplos **80k–82k, nunca vistos ni por la v1 ni por la v2** (la v1 entrenó con los primeros 50k).

**Descubrimiento 2 — la métrica histórica de la v1 estaba medida sobre datos vistos.** `evaluate.py` no tenía forma de saltar ejemplos: evaluaba desde el inicio del stream, es decir sobre espectros que el modelo vio en entrenamiento. Se le agregó `--skip-examples` (pasa `skip_examples` a `HFDESISpectra`; los 6 tests siguen verdes) y se re-evaluó la v1 sobre el held-out real: su η₀.₁₅ honesto es **27.3 %**, no 25.1 %.

**Entrenamiento v2 ejecutado** (run `runs/desi_150k_classhead` — el nombre se mantiene por consistencia con `.gitignore` y los planes, pero son 80k ejemplos): 2 épocas, 20.000 steps, ~1.7 h en MPS (~166 steps/min), sin NaNs, `checkpoint_last.pt` de 99 MB guardado. Lanzado con `nohup` + `disown` (lección de la sesión anterior: un run lanzado como hijo de la sesión de Claude Code muere al cerrarla).

### Resultados v1 → v2 (mismos 2000 espectros held-out, ejemplos 80k–82k)

| métrica | v1 (regresión) | v2 esperanza | v2 argmax (MAP) | objetivo plan 02 |
|---|---|---|---|---|
| η₀.₁₅ | **27.3 %** | 63.8 % | 31.1 % | < 10 % ❌ |
| σ_NMAD | **0.093** | 0.220 | 0.111 | < 0.05 ❌ |
| MAE_norm | **0.125** | 0.301 | 0.182 | — |
| techo z_pred | 1.85 | 1.89 | **2.36** | ≳ 3 ❌ |
| η₀.₁₅ en z∈[1.5,2.5] | 87 % | — | **32 %** | — |
| bias ⟨Δz⟩ en z∈[1.5,2.5] | −0.754 | — | **−0.263** | — |

**Diagnóstico:** el rediseño funciona direccionalmente — en z alto (el régimen que motivó la corrección del TA) la v2-MAP aplasta a la v1 (η 87 %→32 %, bias −0.75→−0.26) — pero el **rebalanceo con cap 10 sobrecorrige el régimen fácil**: en z<0.5 (938/2000 ejemplos) mete bias +0.165 donde la v1 tenía +0.07, y eso arrastra el η global por encima de la v1. Además los posteriors están chatos (confianza media 0.19): a 2 épocas de CE sobre 200 bins le faltó cocción. La variante MAP le gana a la esperanza en todo (la esperanza colapsa al techo z≈1.9 con posteriors multimodales).

**Siguiente iteración propuesta** (fallback previsto en plan 02 "Si algo falla", pendiente de OK de Julián): bajar el cap de pesos del rebalanceo de 10 a 5 en `train.py` + `--epochs 3` (~2.5 h). Predicción oficial de la v2: `z_pred_map`.

### Estado actual

- **Branch** `main` al día con `origin/main`; commits sin cambios desde `68bdb75`. **Nada de esta sesión está commiteado todavía.**
- **Modificados:** `src/desi_fm/evaluate.py` (nuevo `--skip-examples`), `requirements.txt` (+`huggingface_hub>=0.23`), `notebooks/evaluation.ipynb` (RUN_DIR parametrizado vía env `DESI_FM_RUN`, default `desi_150k_classhead`; **aún no re-ejecutado** — esperar al checkpoint definitivo).
- **Nuevos:** `LICENSE` (MIT), `model_card.md` (model card honesta v1 para HF Hub), `notebooks/evaluation_v1_baseline.ipynb` (copia ejecutada del "antes"), `runs/desi_150k_classhead/` (checkpoint v2 + `predictions.csv` + `reconstructions.npz` + métricas), `runs/desi_50k_big/predictions_heldout.csv` (v1 re-evaluada en held-out).
- **Tests:** 6/6 pasando tras el cambio de `evaluate.py`.
- **Tracker:** plan 02 sigue ⬜ (no cumple la definición de hecho: η₀.₁₅ ≥ 10 %). Plan 04 preparado pero bloqueado (ver abajo).
- Los `checkpoint_step_*.pt` intermedios de `runs/desi_150k_classhead/` (4 × 99 MB) se pueden borrar; solo importa `checkpoint_last.pt`.

### Tareas pendientes (en orden)

1. **Iteración v2.1**: con OK de Julián, editar cap de rebalanceo 10→5 en `train.py`, relanzar con `--epochs 3` (mismo comando corregido de 80k, `nohup`), evaluar con `--skip-examples 80000`, y documentar el intento en la tabla al pie de `plan/02-reentrenamiento-v2.md`.
2. **Plan 04 (HF Hub)**: sigue bloqueado en `hf auth login` de Julián (CLI en `~/Library/Python/3.9/bin/hf`, token write). Todo lo demás está listo (model card, autorización explícita para crear el repo público `desi-spectra-fm`). Se puede subir la v1 sin esperar la v2.1.
3. **Cierre plan 02**: cuando una vX cumpla (o se decida aceptar la mejor disponible): re-ejecutar `notebooks/evaluation.ipynb` (`DESI_FM_RUN` ya cableado), tabla v1→v2 en README/DELIVERABLE **con los números held-out honestos** (corregir el 25.1 % histórico → 27.3 %), scatters a `docs/img/`, commit `plan-02: ...`, tracker ✅.

### Avisos / bloqueos

- **La comparación válida es solo sobre el held-out** (ejemplos 80k–82k, `--skip-examples 80000`): cualquier evaluación sin skip pisa datos de entrenamiento. Los CSVs viejos de la v1 (`predictions.csv` de 1000 filas) tienen ese sesgo.
- La red con HF estuvo intermitente (resets del CDN); los streams reintentan solos y terminan — no confundir los `Retrying...` del log con fallas.
- `hf` CLI no está en PATH: usar `~/Library/Python/3.9/bin/hf`.
- Sin bloqueos de disco ni de entorno nuevos.

### Primera acción sugerida para la próxima sesión

Si Julián dio el OK: aplicar cap 5 en `train.py`, relanzar v2.1 (3 épocas, `nohup`) y, mientras entrena, cerrar el plan 04 con la v1 (solo falta el `hf auth login`).

---

## Session — 2026-07-30 16:20 (plan 02R en ejecución — pausado a pedido hasta la noche)

### Resumen de la sesión

Se está ejecutando **`plan/02R-reentrenamiento-v2-calibrado.md`** (la revisión calibrada del plan 02; el fallback viejo "cap 5 + 3 épocas" quedó descartado). **Fases 0–4.4 completas con TDD estricto**; la sesión se pausó a pedido de Julián justo antes de la Fase 4.5 (re-evaluación v1 sobre el held-out canónico). Ningún entrenamiento se lanzó todavía.

**Hecho y verificado:**

1. **Fase 0** ✅ — sin entrenamiento activo; checkpoints de referencia intactos (`runs/desi_50k_big/checkpoint_last.pt` v1 y `runs/desi_150k_classhead/checkpoint_last.pt` v2.0 con `done step=20000`); auditoría 0.4 en el entorno real: `SIGNATURE_AUDIT_OK` y `DATASETS_SEMANTICS_OK version=4.5.0` (`filter → take → shuffle` NO incorpora miembros del held-out).
2. **Fase 1** ✅ (commit `b8f83ee`) — `normalize_redshift_ce` en `DESIFoundationModelConfig` (default `False`, v1/v2.0 cargan igual); CE ÷ `log(n_bins)` cuando está activo; `redshift_loss_raw` expuesto en `forward()`. Test fallido primero → verde.
3. **Fase 2** ✅ (commit `a46a33e`) — split aislado en `data.py`: `filter(has_valid_redshift) → skip → take → shuffle(solo train, seed + epoch)` con `set_epoch()`; helpers en `train.py`: `compute_z_histogram`, `build_z_bin_weights` (sqrt-inverse, bins vacíos→0), `load_compatible_checkpoint` (warm start por shape), `is_better_checkpoint` (exige `examples>0` y `eta15_map` finito). 6 tests nuevos fallaron por la razón esperada antes de implementar; disjointness/frontera/membresía/orden-por-época/repetibilidad probados.
4. **Fase 3** ✅ (commit `8334906`) — flags `--z-label-smoothing --normalize-redshift-ce --z-weighting --z-histogram --z-weight-min --z-weight-cap --init-checkpoint`; `training_args.json` se escribe al inicio del run; evaluador interno de `train.py` ahora reporta `examples`, `eta15` y (`redshift_mae_map`, `redshift_mae_norm_map`, `eta15_map`); `checkpoint_best.pt` por `eta15_map`; `write_metrics_json()` atómico con `allow_nan=False` + flag `--metrics-json` en `evaluate.py`; `z_pred_map`/`z_confidence` en `predict_spectrum`/`predict_spectra_batch`. **Este commit incluye también el `--skip-examples` de la sesión anterior** (dependencia directa del flujo held-out). Suite: **16/16 verdes**; v1 carga con `n_z_bins=0`.
5. **Fase 4.1–4.4** ✅ (commit `475e30e` para el script) — `scripts/estimate_z_histogram.py`; `runs/calibration/z_hist_80k_100bins.npz` generado y **revalidado**: `n_examples=80000`, `counted=80000`, `100` bins, `91` no vacíos; pesos sqrt-inverse resultantes: observados en `[0.5, 3.0]`, mediana ≈ 1.157, bins vacíos con peso 0. El `.npz` NO se commitea (artefacto reproducible, decisión del plan).

**Interrumpido a pedido (no es un fallo):** la evaluación v1 canónica (Fase 4.5) se mató a mitad del streaming con SIGTERM. `runs/calibration/predictions_v1_heldout_canonical.csv` quedó en 0 bytes (el modo `"w"` lo sobreescribe al relanzar) y `metrics_v1_heldout_canonical.json` no existe (la escritura atómica solo ocurre al final — comportamiento diseñado). `runs/calibration/eval_v1_canonical.log` solo tiene warnings conocidos.

### Estado actual

- **Branch** `main`, 4 commits locales por encima de `origin/main` (`b8f83ee`, `a46a33e`, `8334906`, `475e30e`) — **sin pushear**.
- **Tests:** 16/16.
- **Sin procesos**: verificado `pgrep -af 'desi_fm\.(evaluate|train)'` → vacío.
- `runs/desi_80k_classhead_v21_preflight/` y `runs/desi_80k_classhead_v21/` **no existen** (libres para las Fases 5–6).
- v1 y v2.0 intactos; nada borrado ni sobreescrito.
- Working tree restante (igual que antes): `notebooks/evaluation.ipynb`, `plan/02-reentrenamiento-v2.md`, `requirements.txt` modificados sin commitear; `plan/HANDOFF.md`, `plan/02R-...md`, `LICENSE`, `model_card.md`, etc. sin trackear.

### Primera acción para esta noche — reanudar en Fase 4.5 (comando exacto)

```bash
cd "/Users/jirustaroure/Desktop/FINAL PROJECT DEEP LEARNING"
python3 -m desi_fm.evaluate \
  --checkpoint runs/desi_50k_big/checkpoint_last.pt \
  --data-dir edr_sv3 --max-examples 2000 --skip-examples 80000 \
  --predictions-csv runs/calibration/predictions_v1_heldout_canonical.csv \
  --metrics-json runs/calibration/metrics_v1_heldout_canonical.json
python3 -m json.tool runs/calibration/metrics_v1_heldout_canonical.json >/dev/null
```

Esperado: 2.000 predicciones; comparar η con el 27.3 % conocido y registrar cualquier diferencia. Ese JSON/CSV es el baseline autoritativo de la Fase 7. Después: **Fase 5** (preflight 1.000 steps + gate `PREFLIGHT_OK`), **Fase 6** (full run 30k steps con `nohup` + `caffeinate`, PID guardado), **Fase 7** (evaluar best/last, `comparison.json`, decisión `promote_v2_1`/`keep_v1`), **Fase 8** (cierre documental). Todos los comandos están literales en `plan/02R-reentrenamiento-v2-calibrado.md`.

### Avisos

- **Throughput de streaming HOY ~550 filas/min** (el pase de 80k del histograma tardó ~2.4 h, 13:13→15:39, con `Retrying` del CDN de HF). La sesión v2.0 sostuvo ~1330 ej/min con cómputo incluido. Si esta noche sigue lento, el full run (3 épocas × 80k + 3 validaciones × 82k de skip+take) puede superar largamente las 2.5–3 h orientativas — medir con los primeros 500 steps antes de estimar, y dejar el proceso desacoplado (`nohup` + `disown`, PID en `train.pid`) como ya prevé el plan.
- `scripts/estimate_z_histogram.py` termina el trabajo y puede quedar con el intérprete vivo por threads de fsspec (no tiene el `os._exit(0)` de train/evaluate): si pasa, `pkill -f estimate_z_histogram` después de ver el JSON final en el log. El NPZ ya está generado y validado — **no regenerarlo**.
- La comparación válida sigue siendo únicamente sobre los 2.000 labels válidos post-80k (`--skip-examples 80000`).

---

## Session — 2026-08-01 (plan 02R COMPLETADO: decisión `promote_v2_1`)

### Resumen ejecutivo

El plan **02R se ejecutó de punta a punta** (Fases 4.5–8) y terminó con la decisión reproducible **`promote_v2_1`**: la v2.1 (fine-tuning de v1 con cabeza de clasificación de 100 bins y loss calibrada) **pasa los 5 gates conjuntos** contra el baseline v1 re-medido. El checkpoint entregable del proyecto pasa a ser `runs/desi_80k_classhead_v21/checkpoint_last.pt`.

| métrica (held-out canónico, `z_pred_map`) | v1 | **v2.1** |
|---|---:|---:|
| η₀.₁₅ global | 22.6 % | **15.0 %** |
| σ_NMAD | 0.083 | **0.030** |
| MAE_norm | 0.107 | **0.096** |
| η₀.₁₅ z∈[1.5,2.5) | 82.7 % | **23.5 %** |
| techo z_pred | 2.00 | **3.52** |
| RMSE recon (arcsinh) | 0.819 | **0.817** |

### Cronología y verificaciones

1. **Fase 4.5** — baseline v1 re-evaluado sobre el held-out canónico con la semántica nueva de skip (cuenta labels **válidos**, no filas crudas): **η₀.₁₅ = 22.6 %**, no 27.3 % (el 27.3 % venía de la ventana vieja desplazada por inválidos). `runs/calibration/metrics_v1_heldout_canonical.json` + CSV de 2.000 filas validados con `json.tool`. 6 `Retrying` recuperables, sin tracebacks.
2. **Fase 5** — preflight de 1.000 steps con la config exacta del full: **`PREFLIGHT_OK`** con todos los asserts (primera loss 1.83 < 5 gracias a la CE normalizada — v2.0 arrancaba en ~50; mediana 1.720→1.539; validación `examples=2000`, `eta15_map=0.2315` < 0.60; args registrados; `checkpoint_best.pt` presente). Warm start: **115 claves cargadas, solo `redshift_head.4.{weight,bias}` reinicializadas**. Duración 24.2 min.
3. **Fase 6** — full run autorizado por Julián, lanzado 01:55 con `nohup caffeinate -i`, `train.pid=4526` verificado como el python real (`-m desi_fm.train`). A los 500 steps: 139 steps/min, loss 1.716, `redshift_loss_raw` 4.12↓, `pmset -g therm` limpio. **Incidente**: la batería se agotó ~03:0x → macOS hizo *safe-sleep* (no reboot); al enchufar, el proceso despertó, los streams reintentaron (`Retrying 1-2/5`) y recuperaron solos. Época 0 completa en 47:58 (~208 steps/min). Fin: **`done step=30000 elapsed_sec=13818.9`** (3.84 h con la siesta incluida), seeds por época **42/43/44** impresas, **`FINITE_METRICS_OK rows=604`**, sin Traceback/RuntimeError/OOM/Killed. Validaciones internas (MAP): 18.2 % → 17.25 % → **14.3 %** — mejora monótona; `new_best` en las tres → `checkpoint_best.pt` = step 30000 = mismos pesos que `last`.
4. **Fase 7** — `best` y `last` evaluados con el evaluador externo sobre los mismos 2.000 held-out (`--metrics-json`, ambos validados con `json.tool`). Script de gates 7.2 literal → `runs/desi_80k_classhead_v21/comparison.json`: **ambos candidatos pasan los 5 gates**; ganador `checkpoint_last.pt` (η 14.95 % vs 15.15 % — diferencia dentro del ruido del masking aleatorio; pesos idénticos). **`decision: promote_v2_1`**. Confianza media de posterior 0.52 (v2.0: 0.19).
5. **Fase 8** — `notebooks/evaluation.ipynb` re-ejecutado contra `DESI_FM_RUN=desi_80k_classhead_v21` sin errores (usa **`z_pred_map` oficial**, 2.000 predicciones, header/diagnóstico/§5 reescritos: v2.1 descrita siempre como *fine-tuning de v1*, nunca from-scratch; el "antes" sigue en `evaluation_v1_baseline.ipynb`). Actualizados: `README.md` (16 tests, rutas v2.1, tabla v1→v2.1, `z_pred_map` en API/outputs, layout), `DELIVERABLE.md` §1/§5 (tabla held-out + progresión con v2.0 y v2.1), `model_card.md` (v2.1 recomendada, usage con `z_pred_map`, limitaciones honestas), `plan/04` (nota de handoff con checkpoint exacto, métricas y comandos `hf upload` para v2.1), `.gitignore` (whitelist de artefactos livianos de `desi_80k_classhead_v21/` y `runs/calibration/`; checkpoints/logs/npz siguen fuera), registro de intentos del plan 02 y tracker (02 ✅ vía 02R).

### Artefactos nuevos clave

- `runs/desi_80k_classhead_v21/`: `checkpoint_best.pt` + `checkpoint_last.pt` (99 MB c/u, mismos pesos), `comparison.json` (decisión), `metrics_best.json`, `metrics_last.json`, `predictions_{best,last}.csv`, `predictions.csv` (= last, contrato del notebook), `reconstructions{,_best}.npz`, `metrics.jsonl` (604 filas finitas), `training_args.json`, `train.log`.
- `runs/calibration/`: `metrics_v1_heldout_canonical.json` + `predictions_v1_heldout_canonical.csv` (baseline autoritativo), `z_hist_80k_100bins.npz` (no se commitea).

### Pendiente (en orden)

1. **Plan 04 — HF Hub con la v2.1**: todo listo (model card v2.1, comandos en la nota de handoff del plan 04). Solo falta `hf auth login` de Julián (CLI en `~/Library/Python/3.9/bin/hf`) y la autorización explícita para crear el repo público. **Nada se subió en esta sesión** (02R lo prohíbe).
2. Exportar scatters v1/v2.1 a `docs/img/` para el plan 13 (opcional, quedó fuera del alcance de 02R).
3. Planes 05+ según tracker.

### Avisos

- Los `checkpoint_step_*.pt` intermedios de `runs/desi_80k_classhead_v21/` (6 × 99 MB) son prescindibles; solo importan `best`/`last`. No se borró nada.
- El throughput de streaming de HF varió 550–5400 filas/min entre sesiones; cualquier ETA futura debe medirse en el momento.
- La política de esta serie de sesiones sigue vigente: comparaciones SOLO sobre el held-out canónico; `z_pred_map` es la predicción oficial de cualquier checkpoint de clasificación.

---

## Session — 2026-08-02 (plan 04 COMPLETADO: checkpoint v2.1 público en HF Hub)

### Resumen ejecutivo

El **plan 04 se ejecutó de punta a punta** y quedó ✅ en el tracker: la v2.1 promovida por el 02R está publicada en Hugging Face Hub con model card renderizada, y la prueba de clon limpio (GitHub → venv nuevo → descarga del Hub → inferencia CPU) pasa sin depender del workspace original. **El plan 05 NO se inició.**

- **Modelo público:** <https://huggingface.co/jirustaroure/desi-spectra-fm> (verificado por API anónima: `private: False`, `gated: False`, licencia MIT detectada por el Hub).
- **Namespaces (regla permanente — no mezclar):** GitHub **`Julian0444`** para código, CI y enlaces (<https://github.com/Julian0444/desi-spectra-fm>); Hugging Face **`jirustaroure`** para modelo, uploads y `hf_hub_download`.
- **Archivos subidos** (verificados en el tree de `main` por API anónima): `checkpoint_last.pt` (103.964.688 bytes — idéntico al local de `runs/desi_80k_classhead_v21/`), `config.json` (459 B), `training_args.json` (1.259 B), `metrics.jsonl` (161.172 B) y `README.md` (4.349 B — es `model_card.md` subido con ese nombre). **NO** se subieron `checkpoint_best.pt` (pesos idénticos a `last`), steps intermedios, logs, CSVs ni NPZs.
- **Métricas publicadas** (held-out canónico, predicción oficial `z_pred_map`): η₀.₁₅ **14.95 %** (v1 22.6 %) · σ_NMAD **0.0303** · MAE_norm **0.0959** · η₀.₁₅ z∈[1.5,2.5) **23.47 %** · techo z_pred **3.52** · RMSE recon **0.8174**. La card describe la v2.1 como *fine-tuning de v1 con cabeza nueva de clasificación* y no afirma haber alcanzado el objetivo `<10 %`.

### Qué se hizo (cronología)

1. **Fase A — prepublicación** (commit `42c94e6`; push `68bdb75..42c94e6` publicó también los 6 commits locales del 02R):
   - `LICENSE` MIT agregada (estaba sin trackear); `requirements.txt` con `huggingface_hub>=0.23` commiteado.
   - `plan/04-checkpoint-hf-hub.md` reescrito: namespaces correctos, comandos reales de la v2.1, sin `TU_USUARIO` ni referencias operativas a v1/v2.0.
   - `README.md`: quick start/CLI/Python API/layout descargan el checkpoint del Hub con `hf_hub_download("jirustaroure/desi-spectra-fm", "checkpoint_last.pt")`; ya no se asume archivo local; η exacta 14.95 %.
   - `model_card.md`: namespace HF corregido (`Julian0444`→`jirustaroure` solo en `hf_hub_download`; los enlaces de GitHub siguen con `Julian0444`), fila `training_args.json`, η exacta 14.95 %.
   - Suite **16/16** antes del commit; **CI verde**: <https://github.com/Julian0444/desi-spectra-fm/actions/runs/30774254046>.
2. **Fase B — publicación:** `hf auth whoami` → `user: jirustaroure` (token nunca impreso ni pasado por argumentos); `"$HF_CLI" repo create jirustaroure/desi-spectra-fm --repo-type model --exist-ok` + los 5 `hf upload` exactos del runbook. Verificación anónima: página HTTP 200 con la card renderizada (título y 14.95 % presentes) y checkpoint descargable (HEAD 200).
3. **Fase C — prueba limpia:** `mktemp -d` → clon de GitHub (HEAD `42c94e6`) → venv nuevo (pip 26.0.1, torch 2.8.0, huggingface_hub 1.8.0, `pip install -r requirements.txt && pip install -e .`) → descarga del Hub con `HF_HOME` fresco dentro del tempdir → inferencia CPU sobre espectro sintético: **`CLEAN_TEST_OK z_pred_map=0.5195 z_pred=0.5461 z_confidence=0.2835`**, reconstrucción `(5000,)` toda finita, asserts de rango `[0, 4]` y de no-dependencia del workspace original pasados.
4. **Fase D — cierre documental:** tracker 04 ✅ (con URL del Hub en el entregable), definición de hecho del plan 04 marcada con evidencia, esta sección, suite re-ejecutada, commit documental enfocado + push + CI verificada.

### Estado actual

- Branch `main` sincronizada con `origin/main`; CI verde. Tests **16/16**.
- Working tree: solo residuos sin trackear **preservados a propósito** (`.agents/`, `.claude/`, `runs/desi_150k_classhead/`, `runs/desi_50k_big/predictions_heldout.csv`, `runs/desi_80k_classhead_v21/reconstructions_best.npz`). Nada se borró; checkpoints y runs intactos.
- El clon de la prueba limpia quedó en un tempdir del sistema (`mktemp -d`, venv + caché HF) — borrable cuando se quiera; no pertenece al repo.

### Siguiente paso — Plan 05 (Demo Gradio en HF Spaces)

`plan/05-demo-gradio.md` queda desbloqueado: el Space descargará el checkpoint con `hf_hub_download("jirustaroure/desi-spectra-fm", "checkpoint_last.pt")`. Ojo al ejecutarlo: ese plan todavía dice `TU_USUARIO` — usar `jirustaroure` para Space/modelo y `Julian0444` para los enlaces a GitHub (misma regla de namespaces de arriba). **No se inició en esta sesión.**

### Avisos

- El CLI `hf` (huggingface_hub 0.36.2) sigue fuera de PATH: `~/Library/Python/3.9/bin/hf`.
- El Hub avisa rate limits menores para descargas anónimas (la prueba limpia funcionó sin token); para trabajo intensivo, autenticarse.
- Los planes 05/06/07/10 y `PLAN.md` aún contienen `TU_USUARIO` — corregirlos al ejecutar cada uno (no se tocaron en esta sesión para no abrir planes ajenos al 04).

---

## Session — 2026-08-03 (plan 05 COMPLETADO: demo Gradio en vivo en HF Spaces)

### Resumen ejecutivo

El **plan 05 se ejecutó de punta a punta** y quedó ✅ en el tracker: demo Gradio pública en <https://huggingface.co/spaces/jirustaroure/desi-spectra-fm-demo> (`RUNNING` en hardware `zero-a10g`), con **espectros DESI reales del held-out canónico** como ejemplos y salida honesta centrada en **`z_pred_map`** (+ `z_confidence`, `z_pred` secundario, `z_true` y flag de outlier catastrófico cuando el `.npz` trae ground truth). **El plan 06 NO se inició.**

- **App:** fuente versionada en `demo/` (app.py + requirements.txt + README con YAML del Space); el Space es una copia publicada. Gradio **6.22.0** pineado (= el probado localmente en venv 3.12), `torch==2.11.0` (último listado como soportado por ZeroGPU), `desi-fm` instalado desde la URL git de GitHub, checkpoint bajado del Hub al arrancar. Robusta a `.npz` ajenos (2-D → primer espectro con aviso, `ivar`/`mask` opcionales, grillas de cualquier instrumento, `gr.Error` amigables) y con las regiones enmascaradas sombreadas en el plot (bug corregido: ancho de token = `config.patch_size` 26 px, no `n_pixels//n_tokens`=25).
- **Ejemplos** (elegidos con predicciones a protocolo de demo, `mask=0` determinista, sobre los primeros 41 held-out exportados crudos; alineación 1:1 contra `predictions.csv` verificada): `heldout_z020` (0.204→0.227, conf 0.64), `heldout_z083` (0.835→0.810; preset del slider 0.35), `heldout_z287` (2.866→2.441 — arriba del techo z≈2 de v1, delta a la vista), `heldout_lowconf_z157` (**1.574→0.957, conf 0.18** — outlier catastrófico honesto de la banda débil). Los sintéticos del plan original colapsan a z≈3.4 (OOD) y quedaron solo como `--synthetic` en `scripts/make_demo_examples.py`.
- **Verificación:** local con Playwright (`USER_TEST_OK`: 4 ejemplos, slider, upload 2-D en grilla ajena, archivo roto → toast y app viva); pública anónima por UI (valores honestos, 1.8/0.5/3.4 s) y autenticada por API en vivo (z083@0.35→0.8822 en 3.1 s, z287→2.4406 en 1.5 s, lowconf→0.9569 en 1.6 s, slider z020@0.6→0.1571 ≠ 0.2267\@0; verificación externa adicional: upload z020 0.2267 en ~4.3 s y 2-D en ~3.45 s). Todas < 5 s.
- **Docs:** link de demo arriba del `README.md` de GitHub (+ notas + layout con `demo/` y `scripts/`), `model_card.md` con el link y **re-subida al Hub** (commit `32463db`, verificada anónima), `plan/05` reescrito como runbook real con DoD marcada, tracker 05 ✅.

### Realidades de plataforma descubiertas (importan para los planes 06+)

1. **HF paywalleó los Spaces Gradio/Docker en cpu-basic** (402 "requires a PRO subscription"). El camino gratis para cuentas personales en regla es **ZeroGPU** (hasta 2 Spaces Gradio): `create_repo(..., space_sdk="gradio", space_hardware="zero-a10g")` por API Python. **El plan 06 (Docker Space para la API FastAPI) va a chocar con el mismo 402** — evaluar alternativas al ejecutarlo.
2. `hf repo create` sin hardware y `hf upload` (CLI) devuelven **402** para Spaces (el upload hace un `create_repo(exist_ok=True)` interno que 402ea aunque el repo exista) — usar `HfApi.upload_folder/upload_file`.
3. En `zero-a10g` es **obligatorio** al menos una función `@spaces.GPU` — la variante "CPU sin decorador para esquivar la cuota" muere al arrancar (`ValueError: Invalid file descriptor: -1` tras el launch). Quedó `@spaces.GPU(duration=8)`.
4. **Cuota anónima de ZeroGPU por visitante/IP**: pocas runs diarias — al agotarla, "You have exceeded your ZeroGPU runs limit" (el pill "Error" pelado de la UI; el mensaje real está en el SSE de `queue/data`). Mis pruebas quemaron la del equipo ese día; usuarios logueados tienen cuota mayor y se resetea a las 24 h. **Documentado en la card del Space; no es un bug.**
5. **Un commit al Space no reinicia el contenedor** — siempre `HfApi.restart_space()` y confirmar "Application Startup" nuevo en los logs (un fix estuvo ~35 min sin aplicar por esto).
6. `allowed_paths=[<dir>/examples]` en `launch()` para que los clicks de ejemplos no mueran con `InvalidPathError` bajo SSR (visto server-side; la UI real además re-sube el ejemplo como upload de usuario).
7. **ZeroGPU puede caer en una GPU física rota** (`torch.AcceleratorError: uncorrectable ECC error` en el `worker_init` de la lib `spaces`, todas las llamadas fallan con "AcceleratorError") — se arregla con `restart_space()` para caer en otro host. Si todas las llamadas fallan así, es eso.
8. El primer render de ejemplos en la UI los sirve desde la caché de gradio; **probar el Space con sesiones frescas y mirar el wire** (`queue/join`/`queue/data`) antes de culpar a la app.

### Estado actual

- Working tree con los cambios del plan 05 listos para commit enfocado (`.gitignore` con `examples/`, `README.md`, `model_card.md`, `demo/*`, `scripts/make_demo_examples.py`, `plan/05`, `plan/README.md`, `plan/HANDOFF.md`). `examples/*.npz` NO se commitean (gitignored; reproducibles con `scripts/make_demo_examples.py`, copias públicas en el Space).
- Residuos sin trackear previos **preservados**: `.agents/`, `.claude/`, `runs/desi_150k_classhead/`, `runs/desi_50k_big/predictions_heldout.csv`, `runs/desi_80k_classhead_v21/reconstructions_best.npz`.
- Espectros crudos exportados (41) en el scratchpad de la sesión (temporal, prescindible — el script del repo regenera los 4 elegidos).

### Siguiente paso — Plan 06 (API FastAPI + Docker)

**No iniciado.** Aviso clave: los Docker Spaces también requieren PRO (realidad nº 1) — al ejecutar el plan 06 habrá que decidir alternativa (p. ej. servir la API desde otro host gratuito, o documentar el endpoint local con Docker). `plan/06` todavía contiene `TU_USUARIO`.

---

## Session — 2026-08-03 (plan 06 COMPLETADO: API REST pública FastAPI)

### Resumen ejecutivo

El **plan 06 se ejecutó de punta a punta** y quedó ✅ en el tracker: API REST pública en <https://jirustaroure-desi-fm-api.hf.space/api/docs> (Space `jirustaroure/desi-fm-api`, `RUNNING` en `zero-a10g`), respondiendo a `curl` anónimo en ~0.6 s con la salida honesta del proyecto (`z_pred_map` + `z_confidence` primero, `z_pred` secundario). **El plan 07 NO se inició.**

- **Adaptación central (autorizada por el marco de la sesión: nada pago, nada fuera de HF):** el deploy original del plan (Docker Space gratis) ya no existe — Docker Spaces requieren PRO (realidad nº 1 del plan 05). La API vive en el **segundo Space ZeroGPU Gradio** del free tier (límite: 2): gradio se lanza normal (handshake ZeroGPU) y FastAPI va injertada como sub-app en **`/api`**; las rutas REST corren en **CPU**, así que `curl` anónimo nunca toca la cuota ZeroGPU. El Dockerfile igual quedó en el repo, probado local.
- **Código:** la API es `src/desi_fm/api.py` (módulo del paquete, no `api/main.py`): `/predict` (.npz multipart, batch hasta 32 espectros / 50 MB), `/predict_json`, `/healthz`; checkpoint lazy del Hub con overrides `DESI_FM_CKPT` / `DESI_FM_DEVICE`; validación → 422/413; CORS abierto. Extra `[api]` en pyproject; CI instala `.[api,dev]`.
- **Tests 16→24** (8 nuevos en `tests/test_api.py`, TestClient + modelo sintético inyectado, sin red). Suite 24/24 local; CI verde en ambos commits.
- **Verificación end-to-end** (mismos números en las 4 vías, protocolo determinista `mask_ratio=0`): uvicorn local, Docker (python:3.11-slim + torch CPU), y Space público anónimo devuelven `z_pred_map` **0.2267** para `heldout_z020` (z_true 0.204) — idéntico al valor verificado de la demo — y **2.4406** para `heldout_z287`; `/api/docs` (Swagger) HTTP 200 público. El panel Gradio del Space (ruta `@spaces.GPU(duration=8)`) responde 0.2267 en 7.9 s autenticado.

### Realidades de plataforma NUEVAS (se suman a las 8 del plan 05)

1. **`GRADIO_SSR_MODE=true` es el default en Spaces**: el frontend Node se bindea al 7860 — cualquier server propio (uvicorn) muere con "address already in use". Fix: `os.environ["GRADIO_SSR_MODE"] = "false"` **antes** de `import gradio`.
2. **En ZeroGPU no se puede servir gradio vía `mount_gradio_app` + uvicorn propio**: arranca, bindea y ~segundos después recibe un SIGTERM limpio (el handshake de la lib `spaces` con el scheduler pasa por `demo.launch()`). El patrón que funciona (está en `api/app.py`): `demo.launch(prevent_thread_lock=True)` → `demo.server_app.mount("/api", fastapi_app)` → `demo.block_thread()`. Consecuencia: los endpoints públicos cuelgan de **`/api/...`**, no de la raíz.
3. Los logs de runtime de un Space se leen por SSE autenticado: `GET https://huggingface.co/api/spaces/<id>/logs/run` con Bearer token (con `hf` CLI no hay comando).
4. La cuota ZeroGPU anónima de esta IP ya estaba agotada hoy al probar el panel UI ("exceeded your ZeroGPU runs limit" instantáneo); autenticado funcionó a la primera. Para verificar el Space de la API alcanza el REST (CPU, sin cuota).

### Qué se hizo (cronología)

1. **Código + tests + Docker** (commit `5cfd178`, push, CI verde): `src/desi_fm/api.py`, `tests/test_api.py`, extras `[api]`/`[dev]`, `requirements.txt` ampliado, `ci.yml` → `.[api,dev]`, `Dockerfile` (imagen CPU, checkpoint lazy) + `.dockerignore` (whitelist: sin `runs/` ni `.git` en el build context), `api/` (fuente del Space). Smoke local uvicorn y Docker con el checkpoint v2.1 real → 0.2267 ✓.
2. **Space:** `create_repo(..., space_sdk="gradio", space_hardware="zero-a10g")` + `upload_folder("api/")` por API Python. Dos iteraciones de arranque (SSR / SIGTERM, arriba) con `restart_space()` + logs SSE entre medio; tercera variante `RUNNING`.
3. **Verificación pública** anónima con curl (healthz, predict × 2 ejemplos held-out, predict_json, docs, raíz) + panel GPU autenticado.
4. **Docs:** README (línea 📡, sección "REST API (FastAPI)" con curls copy-paste, layout con `api.py`/`api/`/`Dockerfile`, "16 passed"→"24 passed"), model card con el link de la API **re-subida al Hub** (commit `455390e`, verificada anónima), `plan/06` reescrito como runbook real con DoD marcada, tracker 06 ✅.

### Estado actual

- Branch `main` sincronizada; CI verde; suite 24/24. Working tree limpio salvo los residuos sin trackear de siempre, **preservados**: `.agents/`, `.claude/`, `runs/desi_150k_classhead/`, `runs/desi_50k_big/predictions_heldout.csv`, `runs/desi_80k_classhead_v21/reconstructions_best.npz`.
- Spaces activos: demo (`desi-spectra-fm-demo`) + API (`desi-fm-api`) — **los 2 slots ZeroGPU del free tier ocupados**; un tercer Space gratis no va a poder crearse.
- La imagen Docker local `desi-fm-api` quedó en el daemon local (borrable con `docker rmi desi-fm-api`).

### Siguiente paso — Plan 07 (spectra-copilot: repo + herramientas)

**No iniciado.** `plan/07-spectra-copilot-tools.md` todavía contiene `TU_USUARIO` — corregir namespaces al ejecutarlo (GitHub `Julian0444`, HF `jirustaroure`). Con `DESI_FM_CKPT` local no depende de servicios nuevos.

---

## Session — 2026-08-03 (plan 07 COMPLETADO: spectra-copilot repo + tools)

### Resumen ejecutivo

El **plan 07 se ejecutó de punta a punta** y quedó ✅ en el tracker: segundo repo público <https://github.com/Julian0444/spectra-copilot> con las tools determinísticas que usarán el agente (08) y el MCP server (09), **7/7 tests** locales y **CI verde propia**, y la demo CLI imprimiendo el JSON honesto sobre espectros held-out reales. **El plan 08 NO se inició.**

- **Repo local:** `~/proyectos/spectra-copilot` (venv `.venv` con Python 3.12, gitignored). Estructura: `copilot/{__init__,tools,__main__}.py`, `tests/test_tools.py`, `examples/` (3 `.npz` held-out reales **commiteados**, ~103 KB c/u), `eval/cases/` y `docs/img/` (placeholder para 08/11), `.github/workflows/ci.yml`.
- **Adaptaciones vs el plan original:** namespaces reales (pyproject instala `desi-fm @ git+https://github.com/Julian0444/desi-spectra-fm`; `_model()` usa `DESI_FM_CKPT` o `hf_hub_download("jirustaroure/desi-spectra-fm", "checkpoint_last.pt")`); `predict_redshift` devuelve la salida honesta del proyecto (`z_pred_map` oficial + `z_confidence` + `z_pred` secundario, helper `official_z()`); ejemplos reales en vez de `galaxy_z042.npz` sintético; CLI hace una sola pasada de modelo y agrega `z_true_reference` si el `.npz` lo trae.
- **Números de control reproducidos exactos** (protocolo determinista `mask_ratio=0`, pasando `ivar`/`mask` del `.npz`): `heldout_z020` → `z_pred_map` **0.2267** / conf 0.6376 (idéntico a demo y API, z_true 0.204); `heldout_z287` → **2.4406**; `heldout_lowconf_z157` → **0.9569** / conf 0.18.
- **Hallazgo clave para el 08:** en la galaxia real `heldout_z020`, `identify_spectral_lines` discrimina de verdad — a z_true 0.2036 matchea **8/11 (`consistent`)**, al `z_pred_map` 0.2267 solo 2/11 (Δz 0.023 ≈ 150 Å en Hα, fuera de la tolerancia de 12 Å) y a z=0.85 queda débil (0.22). El agente puede **detectar y refinar** una predicción corrida — esa es la historia del plan 08. Documentado en el README del repo nuevo.

### Qué se hizo (cronología)

1. **Esqueleto + código:** `~/proyectos/spectra-copilot`, `git init -b main`; `pyproject.toml` (deps del plan: numpy/scipy/torch/huggingface_hub/desi-fm\@git/anthropic/mcp[cli]; extras `ui`/`dev`); las 3 tools + `_load` robusto (2-D → primer espectro, NaN/Inf → zereados y sumados a `mask`); catálogo de 15 líneas rest-frame del plan sin cambios.
2. **Tests (7):** los 4 del plan + `z_pred_map`/`z_confidence` en rango, `_load` 2-D+NaN, y discriminación sobre el espectro real (sin modelo). **Bug del plan original encontrado:** `_synth` inyectaba solo 4 líneas pero el catálogo espera 12 en cobertura a z=0.42 → `match_fraction` máx ~0.42 y `test_lines_match_at_true_z` fallaba tal cual estaba escrito; fix correcto = inyectar 9 líneas del catálogo en el sintético (no aflojar el umbral 0.5).
3. **Verificación local:** venv 3.12 + `pip install -e ".[dev]"` (instala `desi-fm` desde el repo público de GitHub — de paso lo prueba); `pytest` **7/7** con `DESI_FM_CKPT` → checkpoint v2.1 local; CLI sobre los 3 ejemplos con los números de arriba.
4. **Publicación:** commit inicial + `gh repo create spectra-copilot --public --source . --push` (cuenta `Julian0444`, autorizado explícitamente en esta sesión). CI propia (Python 3.11 + torch CPU + `actions/cache` de `~/.cache/huggingface` para el checkpoint de 104 MB) → **verde**.
5. **Cierre documental (repo principal):** `plan/07` reescrito como runbook real con DoD marcada, tracker 07 ✅ con la URL, esta sección, commit `plan-07` + push + CI principal verde.

### Estado actual

- **Repo nuevo:** `Julian0444/spectra-copilot` público, `main` pusheada, Actions verde, 7/7 tests. Los `.npz` de ejemplo están commiteados ahí (en el repo principal siguen gitignored).
- **Repo principal:** solo cambios documentales (plan/07, tracker, HANDOFF); suite sigue 24/24; residuos sin trackear de siempre **preservados** (`.agents/`, `.claude/`, `runs/desi_150k_classhead/`, `runs/desi_50k_big/predictions_heldout.csv`, `runs/desi_80k_classhead_v21/reconstructions_best.npz`).
- Sin servicios nuevos: no se tocaron los Spaces ni el Hub (los 2 slots ZeroGPU siguen ocupados por demo + API).

### Siguiente paso — Plan 08 (agente con la Claude API)

**No iniciado.** `plan/08-agente-claude.md` construye sobre `copilot/tools.py` — las tools ya exponen todo lo que el agente necesita (`official_z()`, confidence para detectar outliers, la tool de líneas para verificar/refinar). El caso `heldout_lowconf_z157` (conf 0.18, z_pred_map 0.957 vs z_true 1.574) y el corrimiento de `heldout_z020` son el material narrativo ideal para el reporte del agente. Ojo: `anthropic` y `mcp` ya están instalados en el venv del repo nuevo.

### Avisos

- El venv `~/proyectos/spectra-copilot/.venv` usa el Homebrew Python 3.12; `requires-python >=3.10` (por `mcp`). El sistema tiene 3.9 como default — no usarlo para este repo.
- La CI del repo nuevo descarga el checkpoint del Hub en el primer run y lo cachea (`actions/cache`); si el Hub rate-limitea descargas anónimas, re-lanzar el job.
- `UserWarning: enable_nested_tensor ...` al cargar el modelo es benigno (viene de `desi_fm.model` con PyTorch ≥2.x).
