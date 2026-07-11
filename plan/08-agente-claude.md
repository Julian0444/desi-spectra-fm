# 08 · El agente con la Claude API

> **Bloque:** Nivel 2 · **Tiempo:** 2–3 h · **Depende de:** 07 · **Entregable:** reporte de observación generado end-to-end por el agente, citando herramientas

## Objetivo

El corazón del portfolio de AI engineer: un agente (Claude + tool use) que recibe un espectro, llama a **tu** modelo, **valida** el resultado contra la física de líneas, y escribe un reporte donde cada afirmación cita la herramienta que la respalda.

**Modelo:** `claude-opus-4-8` por defecto (US$ 5/25 por MTok; un análisis ≈ US$ 0.10). Con flag `--model claude-haiku-4-5` para pruebas baratas (≈ US$ 0.02).

## Pasos

### 1. Credenciales

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # console.anthropic.com → API keys
# (si usás el CLI `ant`: `ant auth login` y el SDK lo toma solo, sin env var)
```

### 2. `copilot/report.py` — el system prompt

```python
SYSTEM = """Sos un asistente de análisis espectroscópico. Analizás espectros \
astronómicos usando un foundation model (desi-fm) y herramientas de verificación física.

Flujo obligatorio para cada espectro:
1. predict_redshift para obtener z.
2. identify_spectral_lines con ese z para VERIFICARLO contra picos reales.
3. Si match_fraction < 0.4, probá hipótesis alternativas (p. ej. z asumiendo que el \
pico más fuerte es Halpha, o [OII]) y compará match_fractions antes de concluir.
4. Opcional: reconstruct_spectrum si te piden evaluar regiones enmascaradas.

Reglas del reporte final:
- Formato: ## Reporte de observación / Objeto / Redshift / Evidencia / Confianza / Notas.
- CADA afirmación cita la herramienta que la respalda, p. ej. "z = 0.42 \
(predict_redshift), consistente: 4/6 líneas esperadas matcheadas (identify_spectral_lines)".
- Si la evidencia es débil o ambigua, decilo explícitamente y presentá las hipótesis \
en competencia. Nunca afirmes un z sin verificación de líneas.
- El modelo puede fallar a z alto (sesgo conocido de la v1): si el espectro parece \
de cuásar y z_pred < 2, tratalo como sospechoso.
"""
```

(La regla 3 es la que convierte al agente en algo mejor que el modelo solo: puede **recuperar** outliers catastróficos probando hipótesis.)

### 3. `copilot/agent.py` — tool runner del SDK

```python
import argparse, json
import anthropic
from anthropic import beta_tool
from copilot import tools
from copilot.report import SYSTEM

@beta_tool
def predict_redshift(npz_path: str) -> str:
    """Run the desi-fm foundation model on a spectrum file.

    Args:
        npz_path: path to a .npz file containing 'flux' and 'wavelength' (Angstrom).
    """
    return json.dumps(tools.predict_redshift_impl(npz_path))

@beta_tool
def identify_spectral_lines(npz_path: str, z: float) -> str:
    """Check which known emission/absorption lines match real peaks if the spectrum is at redshift z.
    Detects emission peaks; absorption-dominated spectra can look 'weak' even at the right z.

    Args:
        npz_path: path to the spectrum .npz file.
        z: redshift hypothesis to test.
    """
    return json.dumps(tools.identify_spectral_lines_impl(npz_path, z))

@beta_tool
def reconstruct_spectrum(npz_path: str, mask_ratio: float = 0.5) -> str:
    """Mask a fraction of spectral tokens and reconstruct them with the foundation model.

    Args:
        npz_path: path to the spectrum .npz file.
        mask_ratio: fraction of the 273 tokens to hide (0-0.9).
    """
    return json.dumps(tools.reconstruct_spectrum_impl(npz_path, mask_ratio))

def run(npz_path: str, model: str = "claude-opus-4-8") -> str:
    client = anthropic.Anthropic()
    runner = client.beta.messages.tool_runner(
        model=model,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        system=SYSTEM,
        tools=[predict_redshift, identify_spectral_lines, reconstruct_spectrum],
        messages=[{"role": "user",
                   "content": f"Analizá el espectro en {npz_path} y escribí el reporte."}],
    )
    final = None
    for message in runner:      # el runner ejecuta las tools y sigue el loop solo
        final = message
    return "".join(b.text for b in final.content if b.type == "text")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("npz_path")
    p.add_argument("--model", default="claude-opus-4-8")
    args = p.parse_args()
    print(run(args.npz_path, args.model))
```

### 4. Probar con 3 casos + 1 trampa

```bash
python -m copilot.agent examples/galaxy_z042.npz
python -m copilot.agent examples/emission_z080.npz --model claude-haiku-4-5
python -m copilot.agent examples/noisy_z025.npz
```

Caso trampa (mide el valor real del agente): generar un espectro con **una sola línea de emisión** (ambigüedad Hα/[OIII] genuina) y verificar que el reporte presenta las dos hipótesis con sus match_fractions en vez de afirmar una. Guardar ese reporte: es material de README.

### 5. Transcripts + costo por análisis (reemplazar `run()` por esta versión)

```python
import sys
from pathlib import Path

def run(npz_path: str, model: str = "claude-opus-4-8",
        transcript: str | None = None) -> str:
    client = anthropic.Anthropic()
    runner = client.beta.messages.tool_runner(
        model=model,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        system=SYSTEM,
        tools=[predict_redshift, identify_spectral_lines, reconstruct_spectrum],
        messages=[{"role": "user",
                   "content": f"Analizá el espectro en {npz_path} y escribí el reporte."}],
    )
    msgs, final, tin, tout = [], None, 0, 0
    for message in runner:
        msgs.append(message.to_dict())
        tin += message.usage.input_tokens
        tout += message.usage.output_tokens
        final = message
    if transcript:
        Path(transcript).write_text(json.dumps(msgs, indent=2, default=str))
    # costo aproximado (opus 4.8: $5/$25 por MTok; haiku 4.5: $1/$5)
    rate = {"claude-opus-4-8": (5, 25), "claude-haiku-4-5": (1, 5)}.get(model, (5, 25))
    print(f"[usage] in={tin} out={tout} ≈ ${tin / 1e6 * rate[0] + tout / 1e6 * rate[1]:.3f}",
          file=sys.stderr)
    return "".join(b.text for b in final.content if b.type == "text")
```

Y en el `argparse`: `p.add_argument("--save-transcript", default=None)` → llamar `run(args.npz_path, args.model, args.save_transcript)`. El transcript (con tool calls incluidos) sirve para debug y como insumo del plan 11. El `[usage]` impreso por análisis es el número que va en la "Definición de hecho".

## Definición de hecho

- [ ] `python -m copilot.agent <caso>` produce un reporte con el formato pedido y ≥ 2 citas a herramientas.
- [ ] En el caso trampa, el reporte reconoce la ambigüedad (dos hipótesis comparadas) en vez de inventar certeza.
- [ ] Reporte de ejemplo pegado en el README de spectra-copilot.
- [ ] Costo por análisis medido y anotado (tokens del último `message.usage`).
- [ ] Commit + tracker.

## Si algo falla

- **El agente "olvida" verificar:** el system prompt manda — reforzar la regla 2 ("SIEMPRE verificá con identify_spectral_lines antes del reporte") y bajar la temperatura del prompt, no del modelo (los modelos actuales siguen instrucciones de sistema al pie de la letra).
- **Tool runner no disponible en tu versión del SDK:** `pip install -U anthropic`; alternativa: loop manual con `client.messages.create` + `stop_reason == "tool_use"` (patrón documentado en el SDK).
- **Respuestas larguísimas / caras:** cap con `max_tokens=8000` y una regla de brevedad en el system prompt.
- **429 / rate limit:** el SDK reintenta solo; para lotes usar el plan 11 (Batches).
