# 08 · El agente con la Claude API

> **Bloque:** Nivel 2 · **Tiempo:** 2–3 h · **Depende de:** 07 · **Entregable:** reporte de observación generado end-to-end por el agente, citando herramientas — [reportes de ejemplo en el README de spectra-copilot](https://github.com/Julian0444/spectra-copilot#the-agent-claude-api)
>
> **Namespaces (regla permanente):** GitHub **`Julian0444`** para código y enlaces; Hugging Face **`jirustaroure`** para modelo y Spaces.

## Objetivo

El corazón del portfolio de AI engineer: un agente (Claude + tool use) que recibe un espectro, llama a **tu** modelo, **valida** el resultado contra la física de líneas, y escribe un reporte donde cada afirmación cita la herramienta que la respalda.

**Modelo:** `claude-opus-4-8` por defecto; `--model claude-haiku-4-5` para iterar barato.

> **Adaptaciones clave (2026-08-16, ejecutado):** el system prompt se escribió contra la **v2.1** (no contra el sesgo v1 del plan original): salida oficial `z_pred_map` + `z_confidence`, conf < 0.3 = sospechoso, techo del grid z≈3.5, veredicto débil = falta de confirmación (no refutación). `claude-haiku-4-5` **no acepta** `thinking: {"type": "adaptive"}` (400) → `request_kwargs(model)` condicional. `identify_spectral_lines` tuvo que **exponer los picos detectados** (ver hallazgo abajo). El repo público va en inglés (mismo criterio del 07).

## Cómo se ejecutó (runbook real)

### 1. Credenciales — sin exportar nunca

La key vive en `~/.anthropic_key` (chmod 600). **NUNCA exportarla global ni ponerla en `~/.zshrc`**: Claude Code la captura al arrancar y factura la sesión entera al crédito de la API (pasó el 2026-08-16 y quemó ~US$ 5.60). Patrón seguro — la key viaja solo al proceso del agente:

```bash
KEY=$(tr -d '\n' < ~/.anthropic_key)
ANTHROPIC_API_KEY="$KEY" .venv/bin/python -m copilot.agent examples/heldout_z020.npz
```

### 2. `copilot/report.py` — system prompt v2.1

Flujo obligatorio: predict → verificar con líneas al `z_pred_map` → si conf < 0.3 o match débil, derivar hipótesis alternativas desde los picos más fuertes (Hα 6563 / [OIII] 5007 / [OII] 3727), testear cada una con `identify_spectral_lines` y comparar match_fractions antes de concluir. Reporte `## Observation report` (Object/Redshift/Evidence/Confidence/Notes), cada afirmación cita su tool, ~300 palabras. El umbral 0.4 del prompt está sincronizado con el verdict de la tool.

### 3. `copilot/agent.py` — tool runner del SDK

`anthropic` 0.120.2: 3 `@beta_tool` que envuelven `tools.*_impl` devolviendo `json.dumps(...)`; `client.beta.messages.tool_runner(...)` con `max_tokens=16000` + `request_kwargs(model)`; acumulación de `usage` por turno (incluye cache tokens: write 1.25×, read 0.1× input); `--save-transcript` guarda turnos assistant + tool_results vía `runner.generate_tool_call_response()` (**cacheado — no re-ejecuta las tools**); línea `[usage] model=... turns=... in=... out=... ~= $...` a stderr. Precios: opus-4-8 $5/$25 por MTok, haiku-4-5 $1/$5.

### 4. Hallazgo que cambió las tools: exponer los picos

**El prompt le pedía al agente derivar hipótesis desde "el pico más fuerte", pero `identify_spectral_lines` no devolvía picos** — solo líneas matcheadas. En el caso trampa el agente quedaba ciego: el modelo da z=2.79 (conf 0.19, OOD), a ese z matchean 0 líneas, y el JSON no traía **ninguna** longitud de onda para anclar las hipótesis. Fix aditivo en `tools.py`: `identify_spectral_lines` ahora reporta `n_peaks_detected` + `strongest_peaks_angstrom` (top 5 por prominencia). Con eso, en `heldout_z020` el pico 7900.8 Å leído como Hα da z=0.2039 ≈ z_true — la historia de "detectar y refinar" se vuelve mecánicamente alcanzable.

### 5. El caso trampa — continuo sin ruido

`scripts/make_trap_example.py` → `examples/trap_single_line.npz`: **una sola gaussiana en 8000 Å sobre continuo suave sin ruido** (sin `z_true` a propósito). La v1 con ruido σ=0.03 estaba ROTA: el umbral de prominencia (`0.8·std(flux−smooth)`) escala con el propio ruido → 81 picos espurios, [OIII]/[OII] salían "consistent" por matchear ruido (bajar σ no ayuda: invariante de escala). Sin ruido: **1 pico exacto**, y Hα (z=0.219) / [OIII] (z=0.598) / [OII] (z=1.146) empatan en 1 línea matcheada cada una, todas `weak_or_inconsistent` — ambigüedad genuina.

### 6. Tests offline — suite 7 → 13

`tests/test_agent.py`, sin API key (CI no la necesita): hechos v2.1 en el `SYSTEM`; `agent.identify_spectral_lines.call({...})` (los `@beta_tool` exponen `.call(dict)` — el wrapper exacto se testea sin API) ≡ impl sobre `heldout_z020`; tarifas (`1M in + 1M out` → haiku $6, opus $30, desconocido → `None`; cache tokens); `request_kwargs` (haiku → `{}`, opus → adaptive); ambigüedad del trap por contrato (1 pico, 3 empates débiles).

### 7. Corridas y costos medidos

Iteración con haiku (~**US$ 0.013**/análisis) sobre z020 y el trap; finales con `claude-opus-4-8` sobre los 4 casos con `--save-transcript eval/transcripts/<caso>.json` (commiteados — insumo del plan 11). **Gasto total de la sesión ≈ US$ 0.49.**

| caso | z_true | modelo solo (`z_pred_map`, conf) | agente (opus) | costo |
|---|---|---|---|---|
| `heldout_z020` | 0.2036 | 0.2267 (0.64) — outlier | rechaza 2/11, reancla Hα → **z=0.204** (8/11) | $0.094 |
| `heldout_z287` | 2.866 | 2.441 (0.35) — outlier | rechaza 1/4, reancla Lyα → **z=2.874** + hipótesis competidora declarada | $0.131 |
| `heldout_lowconf_z157` | 1.574 | 0.957 (**0.18**) — outlier | lo descarta por conf < 0.3 ✓; su recuperación da z≈1.98 (las líneas UV a z_true caen fuera de cobertura — limitación honesta, documentada) | $0.125 |
| `trap_single_line` | — | 2.79 (0.19) | "**Indeterminate** — single-line trap": 3 hipótesis con match_fractions, "no redshift is defensible" | $0.100 |

En z020 y z287 **el agente le gana a su propio modelo** (recupera el z verdadero donde el modelo era outlier catastrófico). Haiku también completa el loop detect→refine (z020 → 0.204 por $0.013).

### 8. Publicación

Commit `c9de49d` en `Julian0444/spectra-copilot` (agent + report + tools con picos + trap + 6 tests + 4 transcripts + README con la sección del agente y los 2 reportes verbatim), push, **CI verde**.

## Definición de hecho

- [x] `python -m copilot.agent <caso>` produce un reporte con el formato pedido y ≥ 2 citas a herramientas — los 4 reportes citan `predict_redshift` + `identify_spectral_lines` en cada afirmación.
- [x] En el caso trampa, el reporte reconoce la ambigüedad (hipótesis comparadas) en vez de inventar certeza — "Indeterminate", 3 hipótesis con sus match_fractions, transcript en `eval/transcripts/trap_single_line.json`.
- [x] Reporte de ejemplo pegado en el README de spectra-copilot — 2 reportes verbatim (z020 y trap) + resumen de los otros 2.
- [x] Costo por análisis medido y anotado — opus **$0.09–0.13**, haiku **~$0.013** (línea `[usage]` por corrida; total sesión ≈ $0.49).
- [x] Commit + tracker — `c9de49d` (spectra-copilot, CI verde) + tracker 08 ✅.

## Si algo falla

- **El agente "olvida" verificar:** el system prompt manda — reforzar la regla 2; los modelos actuales siguen instrucciones de sistema al pie de la letra.
- **El agente no puede derivar hipótesis (queda ciego):** la tool de líneas tiene que exponer los picos detectados (`strongest_peaks_angstrom`) — sin eso, con 0 matches no hay ninguna λ para anclar (fue el bloqueo real de esta sesión).
- **El trap matchea ruido:** el continuo del trap va SIN ruido — el umbral de prominencia escala con el ruido y planta picos espurios que rompen la ambigüedad (bajar σ no sirve).
- **400 con haiku:** `claude-haiku-4-5` no soporta `thinking adaptive` — `request_kwargs()` condicional.
- **Costo se dispara:** iterar siempre con `--model claude-haiku-4-5`; opus solo para las corridas finales.
- **429 / rate limit:** el SDK reintenta solo; para lotes usar el plan 11 (Batches).
