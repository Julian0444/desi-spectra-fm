# 09 · Servidor MCP — ✅ COMPLETADO (2026-08-16)

> **Bloque:** Nivel 2 · **Tiempo real:** ~1.5 h (en dos sesiones) · **Dependía de:** 07 (no necesitó el 08, pero heredó sus picos) · **Entregable:** las tools del foundation model corriendo dentro de Claude Code vía MCP — verificado con una conversación real que ejecutó las 3 tools y completó el loop detect→refine.

## Qué quedó hecho

- **`copilot/mcp_server.py`** en [`Julian0444/spectra-copilot`](https://github.com/Julian0444/spectra-copilot) (commit `5340bae`, CI verde): servidor MCP stdio que expone `predict_redshift`, `identify_spectral_lines` y `reconstruct_spectrum` delegando en `copilot/tools.py`. Sin lógica propia: el LLM del cliente razona, las tools solo miden.
- **`tests/test_mcp_server.py`** (4 tests offline, suite 13→**17/17**): ejercitan la capa MCP real (`list_tools`/`call_tool` sobre la instancia `MCPServer`), no las impls directamente — una regresión de schema o de delegación falla en pytest, no adentro de Claude Code. Cubren: exactamente 3 tools expuestas; descripciones que dicen *cuándo* usarlas + schemas (`required` correctos, default de `mask_ratio` aplicado por el server); `call_tool` ≡ impl sobre el espectro held-out real (verdict `consistent` a z_true); delegación con impls monkeypatcheadas.
- **Registrado en Claude Code** (scope usuario, `~/.claude-personal/.claude.json`): `claude mcp list` → `desi-fm: ... - ✔ Connected`.
- **Sección "Use it from any MCP client"** en el README de spectra-copilot (commit `843ac92`): snippets de registro para Claude Code y Claude Desktop + la sesión verificada transcripta.

## Adaptaciones vs el plan original

1. **`mcp` 2.x renombró la API**: `mcp.server.fastmcp.FastMCP` ya no existe — el server usa `from mcp.server.mcpserver import MCPServer`. Mismo patrón de decoradores `@mcp.tool()`.
2. **`instructions` del server**: además de las descripciones por tool, el server declara el loop típico (predict primero; si `z_confidence < 0.3` o se pide verificación → `identify_spectral_lines` + hipótesis alternativas desde `strongest_peaks_angstrom`). Claude Code las muestra como instrucciones del servidor MCP.
3. **Descripciones por tool reescritas según el consejo del propio plan** ("decir cuándo usarlas, no solo qué hacen"), incluyendo los campos de picos que el plan 08 agregó a `identify_spectral_lines` — el server MCP los hereda gratis y son los que permiten al cliente derivar hipótesis.
4. **Registro con env y venv**: `claude mcp add desi-fm -e DESI_FM_CKPT=<checkpoint local> -- <venv>/bin/python <ruta>/copilot/mcp_server.py` (python del venv 3.12 del repo, no el del sistema; `DESI_FM_CKPT` evita la descarga del Hub en la primera llamada).
5. **Evidencia como transcript, no screenshot**: la sesión verificada quedó transcripta verbatim en el README (verificable y accesible); el screenshot `docs/img/mcp-session.png` queda como mejora opcional que solo puede capturar Julián desde la UI.
6. **`mcp dev` (inspector) no se corrió**: quedó superado por evidencia más fuerte — los 4 tests offline sobre la capa MCP real + la conversación real en Claude Code.

## Verificación en cliente real (Claude Code, 2026-08-16)

Sobre `examples/heldout_z020.npz` (galaxia real, z_true = 0.204), **5 tool calls, las 3 tools**, sin código del agente involucrado:

1. `predict_redshift` → `z_pred_map = 0.2267`, `z_confidence = 0.64`.
2. `identify_spectral_lines(z=0.2267)` → **débil** (2/11 líneas, 0.18): la física no confirma al modelo.
3. `reconstruct_spectrum(mask_ratio=0.5)` → `z = 0.2031` — swing notable, evidencia frágil.
4. El cliente lee el pico más fuerte (7900.8 Å) como Hα y re-testea `z = 0.204` → **`consistent`** (8/11 líneas, 0.73, deltas < 2 Å).

La misma historia detect→refine del agente del plan 08, reproducida por un cliente MCP genérico guiado solo por las descripciones de las tools.

## Definición de hecho

- [x] Las 3 tools listadas y ejecutadas vía la capa MCP real — cubierto por `tests/test_mcp_server.py` (4/4) en lugar del inspector `mcp dev` (adaptación nº 6).
- [x] `claude mcp list` muestra `desi-fm` (✔ Connected) y una conversación real usó ≥ 2 tools — usó las 3, loop completo (sección de arriba).
- [x] Sección "Use it from any MCP client" en el README con la evidencia — transcript verbatim en vez de screenshot (adaptación nº 5; `docs/img/mcp-session.png` opcional pendiente).
- [x] Commits (`5340bae` código + tests, `843ac92` README) + push + CI verde + tracker ✅.

## Si algo falla (aprendido)

- **El cliente no ve el server:** rutas absolutas y el python del venv (el 3.9 del sistema no tiene `mcp`). Probar el comando exacto en la terminal primero.
- **Primera tool call lenta:** es la carga del checkpoint (o su descarga del Hub sin `DESI_FM_CKPT`); las descripciones ya lo avisan ("the first call loads the model checkpoint").
- **Permisos de Claude Code:** la primera llamada a cada tool MCP pide permiso; si el prompt falla por un error transitorio del clasificador, reintentar la misma llamada.
