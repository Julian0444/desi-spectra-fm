# 11 · Evals del agente

> **Bloque:** Nivel 3 · **Tiempo:** 3–4 h (+ corrida desatendida) · **Depende de:** 08 · **Entregable:** tabla de métricas end-to-end sobre ≥ 100 casos etiquetados

## Objetivo

Medir el **sistema completo** (agente + tools + modelo) contra ground truth, con salida estructurada — la habilidad que hoy más separa candidatos de AI engineering. Pregunta que responde: ¿el agente con verificación de líneas le gana al modelo solo (recupera outliers)?

## Pasos

### 1. Exportar los casos — `eval/export_cases.py`

```python
"""Guarda ~150 espectros DESI crudos + z de pipeline como casos de eval."""
import csv, numpy as np
from datasets import load_dataset
from desi_fm.data import extract_mmu_desi_example

stream = load_dataset("MultimodalUniverse/desi", data_dir="edr_sv3",
                      split="train", streaming=True)
stream = stream.shuffle(buffer_size=4096, seed=7)
rows, n = [], 0
for ex in stream:
    flux, ivar, wave, mask, z = extract_mmu_desi_example(ex)
    if not np.isfinite(z) or z < 0:
        continue
    np.savez(f"eval/cases/case_{n:03d}.npz", flux=flux, wavelength=wave,
             ivar=ivar, mask=mask)
    rows.append({"case": f"case_{n:03d}", "z_true": z})
    n += 1
    if n >= 150:
        break
with open("eval/cases/labels.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["case", "z_true"]); w.writeheader(); w.writerows(rows)
```

Importante: samplear con shuffle para que entren casos de z alto (los difíciles). Verificar el histograma de `z_true` y, si hace falta, forzar ≥ 20 casos con z > 1.5.

### 2. Salida estructurada — `submit_report` + `run_structured()` (agregar a `copilot/agent.py`)

La tool guarda sus argumentos en un dict de módulo — el eval lee campos, cero parsing de prosa:

```python
SUBMITTED: dict = {}

@beta_tool
def submit_report(z_final: float, confidence: str, lines_matched: int,
                  summary: str) -> str:
    """Submit the final structured result of the analysis. Call exactly once, at the end.

    Args:
        z_final: your final redshift estimate after verification.
        confidence: one of "high", "medium", "low".
        lines_matched: how many expected lines matched peaks at z_final.
        summary: 2-3 sentence justification citing the tools used.
    """
    SUBMITTED["last"] = {"z_final": z_final, "confidence": confidence,
                         "lines_matched": lines_matched, "summary": summary}
    return "recorded"


def run_structured(npz_path: str, model: str = "claude-haiku-4-5") -> dict | None:
    """Corre el agente y devuelve el dict de submit_report (None si no lo llamó)."""
    SUBMITTED.pop("last", None)
    client = anthropic.Anthropic()
    runner = client.beta.messages.tool_runner(
        model=model,
        max_tokens=8000,
        thinking={"type": "adaptive"},
        system=SYSTEM + "\nRegla adicional: terminá SIEMPRE llamando submit_report "
                        "exactamente una vez, después de verificar con las líneas.",
        tools=[predict_redshift, identify_spectral_lines,
               reconstruct_spectrum, submit_report],
        messages=[{"role": "user",
                   "content": f"Analizá el espectro en {npz_path} y enviá submit_report."}],
    )
    tin = tout = 0
    for message in runner:
        tin += message.usage.input_tokens
        tout += message.usage.output_tokens
    result = SUBMITTED.get("last")
    if result is not None:
        result["tokens_in"], result["tokens_out"] = tin, tout
    return result
```

### 3. `eval/run_evals.py` (completo)

```python
import argparse
import csv
import json
from pathlib import Path

from copilot.agent import run_structured
from copilot import tools

FIELDS = ["case", "z_true", "z_model", "z_final", "confidence",
          "dzn", "ok15", "ok05", "tokens", "status"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--limit", type=int, default=0, help="0 = todos los casos")
    ap.add_argument("--cases", default="eval/cases")
    ap.add_argument("--out", default="eval/results.csv")
    a = ap.parse_args()

    labels = {r["case"]: float(r["z_true"])
              for r in csv.DictReader(open(Path(a.cases) / "labels.csv"))}
    items = sorted(labels)[: a.limit or None]
    rows = []
    for i, case in enumerate(items):
        z_true = labels[case]
        npz = str(Path(a.cases) / f"{case}.npz")
        z_model = tools.predict_redshift_impl(npz)["z_pred"]   # baseline modelo-solo
        try:
            r = run_structured(npz, model=a.model)
        except Exception as e:
            print(f"[{case}] ERROR {e!r}")
            r = None
        if r is None:
            rows.append({"case": case, "z_true": z_true,
                         "z_model": z_model, "status": "error"})
            continue
        dzn = abs(r["z_final"] - z_true) / (1 + z_true)
        rows.append({"case": case, "z_true": z_true, "z_model": z_model,
                     "z_final": r["z_final"], "confidence": r["confidence"],
                     "dzn": round(dzn, 4), "ok15": int(dzn < 0.15),
                     "ok05": int(dzn < 0.05),
                     "tokens": r["tokens_in"] + r["tokens_out"], "status": "ok"})
        print(f"[{i + 1}/{len(items)}] {case}  z_true={z_true:.3f}  "
              f"z_model={z_model:.3f}  z_final={r['z_final']:.3f}  ok15={dzn < 0.15}")

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, restval="")
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r["status"] == "ok"]
    dzn_model = [abs(r["z_model"] - r["z_true"]) / (1 + r["z_true"]) for r in rows]
    print(json.dumps({
        "n": len(rows), "n_ok": len(ok), "n_error": len(rows) - len(ok),
        "agent_rate_015": round(sum(r["ok15"] for r in ok) / max(len(ok), 1), 3),
        "agent_rate_005": round(sum(r["ok05"] for r in ok) / max(len(ok), 1), 3),
        "agent_mae_norm": round(sum(r["dzn"] for r in ok) / max(len(ok), 1), 4),
        "model_only_rate_015": round(sum(d < 0.15 for d in dzn_model) / max(len(dzn_model), 1), 3),
        "total_tokens": sum(r.get("tokens", 0) for r in ok),
    }, indent=2))


if __name__ == "__main__":
    main()
```

Correr en tandas (el script ya calcula el baseline modelo-solo por caso, así la tabla comparativa sale del mismo CSV):

```bash
python eval/run_evals.py --limit 20 --model claude-haiku-4-5     # validar el arnés (~US$ 0.50)
python eval/run_evals.py --model claude-haiku-4-5                # corrida completa
python eval/run_evals.py --limit 25 --model claude-opus-4-8 --out eval/results_opus.csv
```

Costos estimados 150 casos: Haiku ≈ US$ 3–5, Opus 4.8 ≈ US$ 15–25. Para abaratar Opus: **Batches API** (50 % de descuento) — requiere reescribir el loop como lotes; alternativa pragmática del sprint: Haiku para el barrido completo + Opus solo en los casos difíciles (z_true > 1.2).

### 4. La comparación que importa: agente vs modelo solo

Con `predictions.csv` ya tenés el modelo solo. La tabla final del README:

| sistema | tasa < 0.15 | tasa < 0.05 | MAE_norm |
|---|---|---|---|
| modelo v1 solo | 75 % | 40 % | 0.120 |
| modelo v2 solo | X | X | X |
| **agente (v2 + verificación de líneas)** | **X** | **X** | **X** |

Si el agente recupera aunque sea una parte de los outliers vía hipótesis alternativas, tenés la frase de oro: *"la capa agéntica redujo los outliers catastróficos del X % al Y % sobre el mismo modelo"*.

### 5. Documentar

`eval/results.csv` commiteado + tabla en el README de spectra-copilot + 1 párrafo de metodología (n, modelo LLM usado, costo total, semilla).

## Definición de hecho

- [ ] ≥ 100 casos corridos end-to-end con salida estructurada (0 reportes sin `submit_report`).
- [ ] Tabla modelo-solo vs agente en el README, con metodología y costo.
- [ ] La correlación confianza↔acierto reportada (aunque sea mala — honestidad > marketing).
- [ ] Commit + tracker.

## Si algo falla

- **El agente no llama `submit_report`:** forzar con `tool_choice` en el último turno o reintentar el caso con un recordatorio; si un caso falla 2 veces, marcarlo `error` y seguir (reportar el %).
- **Costo se dispara:** cap de `max_tokens`, Haiku, y limitar a 4 tool calls por caso vía system prompt.
- **Errores transitorios de API en el lote:** el SDK reintenta 429/5xx solo; envolver cada caso en try/except y loggear — nunca perder la corrida entera por un caso.
