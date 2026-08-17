# 11 · Evals del agente — ⏳ ARNÉS COMPLETO; CORRIDA API BLOQUEADA POR CRÉDITO (2026-08-17)

> **Bloque:** Nivel 3 · **Tiempo real (arnés):** ~2 h activas + 20 min de export desatendido · **Dependía de:** 08 · **Entregable:** tabla de métricas end-to-end sobre ≥ 100 casos etiquetados — **pendiente solo de la corrida paga** (crédito API en cero, verificado 2026-08-17).

## Qué quedó hecho (commit `8fb4a0f` en spectra-copilot, CI verde, 26/26 tests)

- **`eval/export_cases.py`** — 150 espectros DESI reales del **lado held-out** del split (stream canónico `filter(valid z) → skip(80000)`, el mismo de `desi_fm.evaluate` y `make_demo_examples.py`), estratificados sobre la distribución de z_true conocida del CSV de predicciones v2.1: **45 casos con z > 1.5** (la banda donde v2.1 todavía falla; el plan pedía ≥ 20). Cada z_true exportado se **asserta contra la fila del CSV** — si MultimodalUniverse reordena el dataset, el export aborta en vez de etiquetar mal. Corrió en ~20 min (150/150, 15 MB en `eval/cases/`, npz gitignoreados, `labels.csv` commiteado con `case,heldout_index,z_true`).
- **`submit_report` + `run_structured()`** en `copilot/agent.py` — salida estructurada vía tool (la eval lee campos tipados del dict `SUBMITTED`, cero parsing de prosa), con `tokens_in/tokens_out` + dict `usage` completo para estimar costo. El system suma `STRUCTURED_RULE` (submit_report exactamente una vez, ≤ 6 tool calls).
- **`eval/run_evals.py`** — por caso: baseline modelo-solo (`tools.official_z(predict_redshift_impl(...))`, la misma llamada que hace la primera tool del agente) + loop completo del agente. El CSV de resultados se **reescribe tras cada caso** y `--resume` retoma una corrida muerta (crédito, red) sin perder nada; cada caso fallido se reintenta 1 vez antes de marcarse `error`. El summary incluye ambos sistemas, el **desglose confianza↔acierto** (`by_confidence`, lo pide la DoD) y el costo estimado de la corrida.
- **`eval/heldout_predictions_v21.csv`** commiteado (copia de `runs/desi_80k_classhead_v21/predictions.csv`): fuente de la selección y del η = 14.95 % oficial.
- **`eval/README.md`** — metodología + comandos de reproducción.
- **Tests 20→26**, todos offline: contrato de `submit_report` (tipado, "recorded"), `run_structured` contra un tool runner falso monkeypatcheado (payload + contabilidad de tokens + submit_report ofrecida + regla en el system + sin `thinking` en Haiku), estado stale no se filtra, estratificación determinista del export sobre el CSV commiteado, métricas de `summarize` (ambos sistemas + by_confidence), convención de dz_norm.

## Baseline modelo-solo ya corrido (sin API) — `eval/results_baseline.csv`

| sistema | tasa < 0.15 | tasa < 0.05 | MAE_norm | n |
|---|---|---|---|---|
| modelo v2.1 solo (protocolo de las tools) | **92.7 %** | **77.3 %** | **0.061** | 150 |
| **agente (v2.1 + verificación)** | pendiente de crédito | — | — | — |

Son **11 outliers catastróficos** (7.3 %) para que el agente intente recuperar — en ambas direcciones y por todo el rango (z_true 0.22→z_model 1.29, z_true 2.35→1.08, z_true 3.14→2.44, ...). La lista exacta sale de `results_baseline.csv`.

**Hallazgo de protocolo:** sobre los mismos 150 índices, el CSV de `evaluate` da 84 % / 68 % / 0.090 — el protocolo de evaluate mide con masking, el de las tools usa contexto completo determinista. La fila "modelo solo" de la tabla final **debe** usar el protocolo de las tools (es literalmente lo que ve el agente); el 14.95 % de η oficial es del otro protocolo y no es comparable directo.

## Adaptaciones vs el plan original

1. **Casos del lado held-out, no del stream shuffleado**: el sketch (`shuffle(buffer_size=4096)` desde el inicio) muestreaba casi todo del lado de ENTRENAMIENTO — el modelo y el índice FAISS ya vieron esos espectros; las métricas saldrían infladas. Se reemplazó por selección estratificada de índices contra el CSV held-out + aserción de alineación.
2. **`official_z` en vez de `["z_pred"]`**: el sketch leía la cabeza de regresión secundaria; la predicción oficial de v2.1 es `z_pred_map`.
3. **`run_structured` incluye `find_similar_spectra`** (el sketch era anterior al plan 10 y el SYSTEM la exige) y usa `request_kwargs()` — Haiku 4.5 rechaza `thinking: adaptive` con 400.
4. **Robustez de corrida**: CSV reescrito por caso + `--resume` + retry×2 (el sketch escribía el CSV recién al final — una corrida muerta a mitad perdía todo, letal con el crédito justo).
5. **`--baseline-only`**: la fila modelo-solo no necesita API — ya está corrida y commiteada.

## Para retomar cuando haya crédito (los dos comandos)

```bash
cd ~/proyectos/spectra-copilot
ANTHROPIC_API_KEY="$(cat ~/.anthropic_key)" .venv/bin/python eval/run_evals.py --limit 20   # validación (~US$ 0.4)
ANTHROPIC_API_KEY="$(cat ~/.anthropic_key)" .venv/bin/python eval/run_evals.py --resume     # 150 casos (~US$ 3–4.5 Haiku)
```

La key SOLO al proceso, nunca exportada global. Costo total estimado: **~US$ 4 solo Haiku; +US$ 2.5–3.5 si se agrega Opus 4.8 sobre los ~25 difíciles (opcional). Recarga recomendada: ≥ US$ 10.** Nota (plan 12, 2026-08-17): desde `3eea698` el agente ofrece además `lookup_reference` (mini-RAG) — la corrida medirá agente **con** priors citables; documentarlo en la fila de metodología. Después de la corrida: tabla + párrafo de metodología en el README de spectra-copilot (n=150, semilla 7, estratos, modelo LLM, costo real medido) + `results.csv` commiteado + cerrar la DoD acá y el tracker.

## Definición de hecho

- [ ] ≥ 100 casos corridos end-to-end con salida estructurada (0 reportes sin `submit_report`). — **bloqueado por crédito**; arnés validado offline con runner falso + baseline real de 150 casos.
- [ ] Tabla modelo-solo vs agente en el README, con metodología y costo. — fila modelo-solo lista (92.7 % / 77.3 % / 0.061); fila agente pendiente.
- [ ] La correlación confianza↔acierto reportada. — implementada y testeada (`by_confidence` en el summary); números pendientes de la corrida.
- [x] Commit + tracker. — arnés en `8fb4a0f` (CI verde); tracker en ⏳ con nota.

## Si algo falla (aprendido)

- **El export aborta con "stream z != csv z"**: el dataset upstream cambió de orden — NO usar esas etiquetas; regenerar la selección contra un CSV nuevo de predicciones.
- **Corrida muerta a mitad (crédito/red)**: `--resume` retoma; los casos `error` se reintentan, los `ok` se conservan.
- **El agente no llama `submit_report`**: `run_structured` devuelve `None` → run_evals reintenta 1 vez y después marca `error` (se reporta el % en `n_error`).
- **Los stdout del export no aparecen en el log**: Python bufferea al redirigir a archivo — correr con `python -u` si se quiere progreso en vivo (los npz aparecen en disco igual).
