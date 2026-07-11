# 12 · Mini-RAG de referencias

> **Bloque:** Nivel 3 · **Tiempo:** 2–3 h · **Depende de:** 08 · **Entregable:** reportes del agente con citas a fuentes

## Objetivo

Un RAG chico y honesto: catálogo de líneas espectrales + fragmentos de documentación de DESI, detrás de una tool `lookup_reference`. El punto no es la escala — es mostrar el patrón completo (corpus → índice → retrieval → **cita en la respuesta**) sin inflarlo.

## Pasos

### 1. El corpus — `data/refs/`

- `data/lines.json`: el catálogo de líneas extendido, con contexto útil para retrieval:

```json
[
  {"name": "Halpha", "rest_angstrom": 6562.8,
   "text": "Hydrogen Balmer alpha line at 6562.8 A. Strongest optical emission line in star-forming galaxies. At z>0.49 it leaves the DESI optical range (9800 A ceiling)."},
  {"name": "OII_3727", "rest_angstrom": 3727.1,
   "text": "[OII] doublet at 3727 A. Primary star-formation tracer for 0.6<z<1.6 in DESI when Halpha is out of range. Commonly confused with Halpha in single-line spectra."},
  ...
]
```

- `data/refs/*.md`: 10–20 fragmentos cortos (2–5 oraciones c/u) sobre DESI EDR/SV3: qué es SV3, tipos de target (BGS/LRG/ELG/QSO) y sus rangos de z típicos, cómo mide z la pipeline (Redrock), degeneraciones clásicas de identificación de líneas. Redactalos vos a partir de la doc pública de DESI, con el link de origen en cada archivo (`source:` en la primera línea).

Ese contenido no es relleno: los rangos de z por tipo de target le dan al agente **priors** ("los ELG viven en 0.6–1.6; un ELG con z_pred 3.5 es sospechoso").

### 2. Retrieval — BM25 (sin infraestructura)

```python
# pip install rank_bm25
# copilot/rag.py
import json
from pathlib import Path
from rank_bm25 import BM25Okapi

def _load_corpus():
    docs = []
    for e in json.loads(Path("data/lines.json").read_text()):
        docs.append({"id": e["name"], "source": "line-catalog", "text": e["text"]})
    for p in sorted(Path("data/refs").glob("*.md")):
        text = p.read_text()
        docs.append({"id": p.stem, "source": text.splitlines()[0].removeprefix("source:").strip(),
                     "text": text})
    return docs

_DOCS = _load_corpus()
_BM25 = BM25Okapi([d["text"].lower().split() for d in _DOCS])

def lookup_reference_impl(query: str, k: int = 3) -> dict:
    scores = _BM25.get_scores(query.lower().split())
    top = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    return {"results": [
        {"id": _DOCS[i]["id"], "source": _DOCS[i]["source"],
         "snippet": _DOCS[i]["text"][:400]}
        for i in top if scores[i] > 0
    ]}
```

BM25 es la elección correcta acá y hay que poder defenderla: corpus de ~40 docs cortos y técnicos, queries con vocabulario controlado → embeddings no aportan y agregan una dependencia. (Si querés la versión con embeddings para comparar, `sentence-transformers` + coseno en 15 líneas — como *apéndice*, midiendo si mejora el hit rate.)

### 3. Integrar al agente y al MCP server

- Tool `lookup_reference(query)` con docstring que diga cuándo usarla: *"Consult the line catalog and DESI reference notes; use it to sanity-check target-type vs redshift and to explain line degeneracies."*
- Regla nueva en el system prompt: *"Cuando uses información de lookup_reference, citá la fuente entre corchetes: [line-catalog], [desi-sv3-overview]."*

### 4. Verificar con 3 análisis

Correr el agente sobre 3 casos y chequear que: (a) consulta la referencia cuando el z es raro para el tipo de espectro, (b) las citas aparecen en el reporte, (c) no alucina fuentes que no existen (comparar contra los ids reales del corpus).

## Definición de hecho

- [ ] Corpus commiteado (`lines.json` + ≥ 10 refs con fuente).
- [ ] `lookup_reference` integrada (agente + MCP), con test unitario (query "Halpha OII confusion" → devuelve los docs correctos).
- [ ] Un reporte real del agente con ≥ 1 cita `[fuente]` pegado en el README.
- [ ] Commit + tracker.

## Si algo falla

- **El agente cita fuentes inventadas:** endurecer la regla ("solo podés citar ids devueltos por lookup_reference en esta conversación") y validarlo en el eval del plan 11 (agregar métrica: % de citas válidas).
- **BM25 devuelve basura con queries largas:** truncar la query a los términos clave en la tool (top-10 tokens) o indexar también los títulos con peso extra.
