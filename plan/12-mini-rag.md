# 12 · Mini-RAG de referencias — ⏳ RAG COMPLETO Y COMMITEADO; VERIFICACIÓN CON EL AGENTE BLOQUEADA POR CRÉDITO (2026-08-17)

> **Bloque:** Nivel 3 · **Tiempo real:** ~2 h · **Dependía de:** 08 · **Entregable:** reportes del agente con citas a fuentes — **pendiente solo la corrida paga** (crédito API en cero, verificado 2026-08-17 con una llamada mínima a Haiku).

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

## Para cerrar cuando haya crédito (una corrida, ~US$ 0.02–0.15)

```bash
cd ~/proyectos/spectra-copilot
ANTHROPIC_API_KEY="$(cat ~/.anthropic_key)" .venv/bin/python -m copilot.agent \
    examples/heldout_lowconf_z157.npz --model claude-haiku-4-5   # ~$0.02
```

(El caso lowconf es el que más invita a consultar referencias; con Opus 4.8 sin `--model` ≈ $0.10–0.15.) Verificar los 3 puntos del plan: (a) consulta la referencia cuando el z es raro para el tipo de espectro, (b) las citas `[id]` aparecen en el reporte, (c) los ids citados ∈ `rag.valid_ids()` — si alucina, correr 2 casos más y anotar el %. Pegar el reporte en la sección Mini-RAG del README (reemplaza el aviso "Pending"), commit, y marcar acá la DoD 3.

## Definición de hecho

- [x] Corpus commiteado (`lines.json` + ≥ 10 refs con fuente). — 15 + 15 en `refs/`, cada nota con `source:`.
- [x] `lookup_reference` integrada (agente + MCP), con test unitario (query "Halpha OII confusion" → devuelve los docs correctos). — 3 capas + 9 tests nuevos.
- [ ] Un reporte real del agente con ≥ 1 cita `[fuente]` pegado en el README. — **bloqueado por crédito API**; el README ya tiene la salida real del retrieval y el aviso.
- [x] Commit + tracker. — `3eea698` en spectra-copilot; tracker en ⏳ con nota.

## Si algo falla (actualizado)

- **El agente cita fuentes inventadas** (a verificar en la corrida): la regla dura ya está en el SYSTEM; si igual pasa, agregar al plan 11 la métrica % de citas válidas usando `rag.valid_ids()`.
- **BM25 devuelve basura con queries largas**: mitigado por la tokenización regex y el filtro score > 0; si reaparece, truncar la query a los top-10 tokens en la tool.
- **`rank_bm25` falta en un entorno viejo**: está en `dependencies` de `pyproject.toml`; `pip install -e .` lo trae.
