# 12 · Mini-RAG de referencias — ✅ COMPLETADO (2026-08-17)

> **Bloque:** Nivel 3 · **Tiempo real:** ~2.5 h · **Dependía de:** 08 · **Entregable:** reportes del agente con citas a fuentes — **cumplido**: 3 corridas reales Haiku (~US$ 0.07), 6/6 citas respaldadas, reporte verbatim en el README de spectra-copilot. La parte offline se construyó con el crédito en cero; la verificación paga se corrió el mismo día tras la recarga de Julián.

## Qué quedó hecho (commit `3eea698` en spectra-copilot, 35/35 tests offline)

- **Corpus commiteado en `refs/`** (30 documentos): `refs/lines.json` — el catálogo de 15 líneas extendido con contexto de retrieval (ventanas de visibilidad en la cobertura DESI 3600–9824 Å, confusiones conocidas, pares confirmatorios) — + **15 notas** `refs/*.md` redactadas desde la doc pública de DESI (overview, EDR, SV3, los 4 tipos de target con sus rangos de z, Redrock, cobertura del espectrógrafo, degeneración de línea única, doblete [OII], outliers catastróficos, espectros de absorción, serie de Balmer, residuos de cielo), cada una con `source:` (arXiv/NIST/GitHub) en la primera línea.
- **`copilot/rag.py`**: BM25 (`rank_bm25`, agregado a `pyproject.toml`) con tokenización alfanumérica (así "[OII]" matchea "OII"), corpus lazy con `lru_cache`, clamp de `k`, filtro score > 0 (query fuera de corpus → lista vacía, no ruido) y **`valid_ids()`** — el ground truth para chequear citas alucinadas (insumo directo si el plan 11 agrega la métrica "% citas válidas").
- **`lookup_reference` integrada en las 3 capas**: tool `@beta_tool` en el agente (ofrecida en `run()` y en `run_structured()` → la corrida pendiente del 11 medirá agente **con** RAG), tool MCP (el server pasa de 4 a 5 tools, `instructions` actualizadas) y regla nueva en el SYSTEM: paso 5 del workflow (priors de tipo de target / degeneraciones) + **contrato de citas** — citar ids entre corchetes y *solo* ids devueltos por lookup_reference en esa conversación, nunca inventar una fuente (el endurecimiento de "Si algo falla" se aplicó desde el día 1).
- **Tests 26→35**, todos offline: corpus bien formado (≥10 notas con source https), la query de la DoD "Halpha OII confusion" → nota de degeneración + `Halpha_6563` (top-3) + `OII_3727` (top-5), priors recuperables ("ELG redshift range" → `desi-targets-elg` con "0.6" en el snippet), query fuera de corpus vacía, clamp de k, tool del agente ≡ impl, contrato de citas en el SYSTEM, capa MCP real (`call_tool`) ≡ impl, 5 tools exactas con schemas.
- **README de spectra-copilot**: sección "Mini-RAG: cited references (BM25)" con defensa de BM25 (corpus de ~30 docs técnicos → embeddings no aportan y agregan dependencia; BM25 es determinista, offline y testeable en CI), salida real de `lookup_reference("ELG redshift range")`, y el contrato de citas. Nota visible de que el reporte del agente con cita queda pendiente de crédito.

## Adaptaciones vs el plan original

1. **Corpus en `refs/`, no en `data/`**: el plan lo ponía en `data/lines.json` + `data/refs/`, pero `data/` está gitignoreado en spectra-copilot (ahí vive el índice FAISS de 30 MB). La DoD exige corpus *commiteado* → `refs/` en la raíz.
2. **Tokenización regex en vez de `split()`**: el sketch tokenizaba con `.lower().split()`, con lo cual "[oii]" nunca matchearía una query "OII". `re.findall(r"[a-z0-9]+")` resuelve eso y de paso el truncado de puntuación.
3. **Corpus lazy (`lru_cache`)** en vez de índice a nivel módulo: mismo patrón que `tools._model()`/`_index()`; importar `copilot.rag` no lee disco.
4. **La regla anti-alucinación de "Si algo falla" se incorporó de entrada** al SYSTEM ("solo ids devueltos en esta conversación") en lugar de esperar a observar el fallo.

## Verificación con el agente (3 corridas Haiku, 2026-08-17, ~US$ 0.07 — commit `ac04121`)

Tras la recarga de crédito (key `sk-ant-api03-mr...DgAA` confirmada con llamada mínima), se corrieron los 3 análisis del plan con `claude-haiku-4-5` y transcript commiteado (`eval/transcripts/{lowconf,trap,z287}_rag.json`):

| caso | consultó lookup_reference | citas en el reporte | ¿respaldadas? |
|---|---|---|---|
| `heldout_lowconf_z157` ($0.023) | ✓ (2 queries: rangos BGS/LRG, diagnóstico) | `[desi-targets-lrg]` `[absorption-dominated-spectra]` | 2/2 ✓ |
| `trap_single_line` ($0.024) | ✓ ("BGS redshift range Halpha") | `[desi-targets-bgs]` | 1/1 ✓ |
| `heldout_z287` ($0.023) | ✓ ("QSO redshift range … MgII") | `[MgII_2799]` `[Lya_1216]` `[desi-targets-qso]` | 3/3 ✓ |

- (a) el agente consultó la referencia **sin que se lo pidieran** en los 3 casos ✓; (b) citas presentes en los 3 reportes ✓; (c) **6/6 citas corresponden a ids devueltos por lookup_reference en esa misma conversación** (verificado programáticamente contra `rag.valid_ids()` + los transcripts) — 0 fuentes inventadas ✓.
- El reporte del caso lowconf (indeterminado honesto, ambas citas trabajando como priors: ventana LRG 0.4–1.1 + caveat de absorción) quedó **verbatim en el README** reemplazando el aviso "Pending".
- **Hallazgo honesto (Haiku vs Opus, no es fallo del RAG):** en el trap sintético Haiku concluyó z=0.219 con confianza "High" apoyándose en vecinos sin sentido (el trap está fuera del manifold), donde la corrida de referencia con Opus (plan 08) decía "Indeterminate". La mecánica de citas funcionó igual; para reportes de calidad usar Opus, para evals masivas Haiku.

## Definición de hecho

- [x] Corpus commiteado (`lines.json` + ≥ 10 refs con fuente). — 15 + 15 en `refs/`, cada nota con `source:`.
- [x] `lookup_reference` integrada (agente + MCP), con test unitario (query "Halpha OII confusion" → devuelve los docs correctos). — 3 capas + 9 tests nuevos.
- [x] Un reporte real del agente con ≥ 1 cita `[fuente]` pegado en el README. — reporte lowconf con 2 citas, verbatim (`ac04121`).
- [x] Commit + tracker. — `3eea698` + `ac04121` en spectra-copilot; tracker ✅.

## Si algo falla (actualizado)

- **El agente cita fuentes inventadas**: NO pasó en las 3 corridas (6/6 respaldadas) — la regla dura del SYSTEM alcanzó. Si reaparece en corridas futuras, agregar al plan 11 la métrica % de citas válidas usando `rag.valid_ids()`.
- **BM25 devuelve basura con queries largas**: mitigado por la tokenización regex y el filtro score > 0; si reaparece, truncar la query a los top-10 tokens en la tool.
- **`rank_bm25` falta en un entorno viejo**: está en `dependencies` de `pyproject.toml`; `pip install -e .` lo trae.
