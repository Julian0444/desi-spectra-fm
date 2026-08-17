# 11 · Evals del agente — ✅ COMPLETADO (2026-08-17)

> **Bloque:** Nivel 3 · **Tiempo real:** ~2 h de arnés + 20 min de export + ~2.5 h de corrida desatendida · **Dependía de:** 08 · **Entregable:** tabla de métricas end-to-end sobre 150 casos etiquetados — **cumplido** (commits `8fb4a0f` arnés + `6efbdfd` resultados en spectra-copilot, CI verde, 35/35 tests). El arnés se construyó con el crédito en cero; la corrida paga se hizo el mismo día tras la recarga, con go-ahead explícito de Julián por lote.

## El resultado (corrida 2026-08-17, `eval/results.csv` commiteado)

**150/150 casos con `submit_report` estructurado, 0 errores, costo medido US$ 4.06** (`claude-haiku-4-5`, 3.27 M tokens ≈ $0.027/caso; validación de 20 casos $0.57 + corrida `--resume` de 130 $3.49 — dentro del estimado de $3–4.5).

| sistema (protocolo de las tools) | tasa < 0.15 | tasa < 0.05 | MAE_norm |
|---|---|---|---|
| modelo v2.1 solo | **92.7 %** | **77.3 %** | **0.061** |
| agente (Haiku + 5 tools incl. RAG del plan 12) | 79.3 % | 72.0 % | 0.104 |
| híbrido (agente solo si reporta confianza `high`) | 90.0 % | — | — |

**El titular honesto: con un LLM barato, el loop de verificación empeora al modelo.** Transiciones: 118 ambos-bien · **21 modelo-bien/agente-mal** · **1 outlier recuperado** (de 11) · 10 ambos-mal. El daño se concentra en la banda estratificada z > 1.5 (agente 68.9 % vs modelo 91.1 %): con tan pocas líneas del catálogo en cobertura DESI, el z *verdadero* también "se ve débil" al matching de emisión, y Haiku pisa una predicción correcta con una hipótesis de línea única a z bajo — exactamente la degeneración sobre la que el SYSTEM advierte. Los éxitos demo del plan 08 (Opus recuperando outliers elegidos a mano) **no generalizan hacia abajo** a Haiku a 1/25 del precio.

**Lo que sí funciona: la correlación confianza↔acierto** (la métrica que pedía la DoD) es monótona y accionable — high 88.9 % (n=63) / medium 77.1 % (n=70) / low 52.9 % (n=17); 17 de los 21 casos rotos venían auto-marcados medium/low, por eso el híbrido recupera a 90.0 %. **La eval hizo su trabajo: cazó un fallo relevante para deployment (no dejar que un LLM barato pise a un modelo bien calibrado) que 4 demos elegidas a mano escondían.** Tablas y discusión completa en el README de spectra-copilot (sección "End-to-end evals") y en `eval/README.md`.

## Qué quedó hecho (arnés: `8fb4a0f` · resultados: `6efbdfd`)

- **`eval/export_cases.py`** — 150 espectros del lado held-out (stream canónico `filter(valid z) → skip(80000)`), estratificados con 45 en z > 1.5, cada z_true asserted contra `eval/heldout_predictions_v21.csv`; npz gitignoreados (regenerables deterministas, semilla 7), `labels.csv` commiteado.
- **`submit_report` + `run_structured()`** en `copilot/agent.py` — salida estructurada tipada vía tool, tokens + usage para costo; incluye `find_similar_spectra` y (desde el plan 12) `lookup_reference`.
- **`eval/run_evals.py`** — baseline modelo-solo + agente por caso; CSV reescrito tras cada caso + `--resume` (la corrida real usó exactamente ese camino: 20 validados primero, 130 después) + retry×2 + summary con `by_confidence` y costo. `--baseline-only` sin API.
- **Tests 20→26** (después 35 con el plan 12), todos offline.

## Adaptaciones vs el plan original

1. **Casos del lado held-out, no del stream shuffleado** (el sketch muestreaba del lado de entrenamiento → métricas infladas).
2. **`official_z` en vez de `["z_pred"]`** (la predicción oficial de v2.1 es `z_pred_map`).
3. **`run_structured` con toolset completo** y `request_kwargs()` (Haiku 4.5 rechaza `thinking: adaptive`).
4. **Robustez de corrida**: CSV por caso + `--resume` + retry — clave con presupuesto justo.
5. **Corrida en dos lotes con go-ahead por lote** (validación n=20 → OK de Julián → 130 restantes), tras el susto de contabilidad del panel (ver HANDOFF: los $22.71 previos eran del 16-ago, no de la sesión).

## Hallazgo de protocolo (mantener en toda tabla futura)

La fila "modelo solo" usa el protocolo de las tools (contexto completo determinista — lo que el agente ve): 92.7 % sobre estos 150. El CSV de `evaluate` (masking) da 84 % sobre los mismos índices y el η oficial 14.95 % es de ese otro protocolo — **no mezclar** sin decirlo.

## Definición de hecho

- [x] ≥ 100 casos corridos end-to-end con salida estructurada (0 reportes sin `submit_report`). — 150/150, 0 errores.
- [x] Tabla modelo-solo vs agente en el README, con metodología y costo. — README de spectra-copilot + `eval/README.md` (`6efbdfd`), costo medido US$ 4.06.
- [x] La correlación confianza↔acierto reportada. — monótona: 88.9/77.1/52.9 %; base de la política híbrida (90.0 %).
- [x] Commit + tracker. — `8fb4a0f` + `6efbdfd`, CI verde; tracker ✅.

## Si algo falla (aprendido)

- **El export aborta con "stream z != csv z"**: el dataset upstream cambió de orden — regenerar la selección contra un CSV nuevo.
- **Corrida muerta a mitad**: `--resume` retoma; los `ok` se conservan (probado en la corrida real).
- **El agente no llama `submit_report`**: `run_structured` → `None` → retry → `error` (esta corrida: 0 casos).
- **Números del agente peores que el modelo**: no es bug — es el resultado (ver arriba). Antes de "arreglarlo", recordar que la corrida de referencia con Opus (plan 08) sí recuperaba outliers: el siguiente experimento natural es Opus sobre los 21+10 casos que Haiku falló (~US$ 3), no tocar el arnés.
