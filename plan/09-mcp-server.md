# 09 · Servidor MCP

> **Bloque:** Nivel 2 · **Tiempo:** 1–2 h · **Depende de:** 07 (no necesita el 08) · **Entregable:** tus herramientas corriendo dentro de Claude Code / Claude Desktop — screenshot para el README

## Objetivo

Exponer el toolset como servidor MCP (Model Context Protocol): cualquier cliente del ecosistema (Claude Code, Claude Desktop, etc.) puede usar tu foundation model como herramienta. Es la línea de CV más actual del plan y cuesta ~40 líneas gracias a FastMCP.

## Pasos

### 1. `copilot/mcp_server.py`

```python
"""MCP server exposing the desi-fm foundation model as tools.

Run standalone:  python -m copilot.mcp_server
Dev inspector:   mcp dev copilot/mcp_server.py
"""
from mcp.server.fastmcp import FastMCP
from copilot import tools

mcp = FastMCP("desi-fm")

@mcp.tool()
def predict_redshift(npz_path: str) -> dict:
    """Predict the redshift of a spectrum stored in a .npz file (flux + wavelength in Angstrom)."""
    return tools.predict_redshift_impl(npz_path)

@mcp.tool()
def identify_spectral_lines(npz_path: str, z: float) -> dict:
    """Test a redshift hypothesis: which known emission lines match real peaks at redshift z.
    Returns matched lines and a match_fraction; low values mean the hypothesis is weak."""
    return tools.identify_spectral_lines_impl(npz_path, z)

@mcp.tool()
def reconstruct_spectrum(npz_path: str, mask_ratio: float = 0.5) -> dict:
    """Mask a fraction of the spectrum's tokens and reconstruct them with the foundation model."""
    return tools.reconstruct_spectrum_impl(npz_path, mask_ratio)

if __name__ == "__main__":
    mcp.run()   # transporte stdio
```

Las descripciones de las tools importan: son lo único que ve el cliente para decidir cuándo llamarlas — decir *cuándo usarlas*, no solo qué hacen.

### 2. Probar con el inspector

```bash
pip install "mcp[cli]"
mcp dev copilot/mcp_server.py
```

Abre un inspector web: listar tools, ejecutar `predict_redshift` con un path de ejemplo, verificar el JSON de respuesta.

### 3. Registrarlo en Claude Code

```bash
claude mcp add desi-fm -- python3 /ruta/absoluta/a/spectra-copilot/copilot/mcp_server.py
claude mcp list      # debe aparecer "desi-fm"
```

Después, en una sesión de Claude Code:

> "Usando las herramientas de desi-fm, analizá el espectro en /ruta/a/examples/galaxy_z042.npz y verificá el redshift contra las líneas."

### 4. Registrarlo en Claude Desktop (opcional)

`~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "desi-fm": {
      "command": "python3",
      "args": ["/ruta/absoluta/a/spectra-copilot/copilot/mcp_server.py"]
    }
  }
}
```

Reiniciar Claude Desktop; el ícono de herramientas debe listar las 3 tools.

Ojo: usar el python del venv donde está instalado todo (`/ruta/al/venv/bin/python3`), no el del sistema.

### 5. Capturar la evidencia

Screenshot de Claude Code/Desktop llamando `predict_redshift` + `identify_spectral_lines` y razonando sobre los resultados → `docs/img/mcp-session.png`. Sección "Use it from any MCP client" en el README de spectra-copilot con el snippet de config y el screenshot.

## Definición de hecho

- [ ] `mcp dev` lista y ejecuta las 3 tools.
- [ ] `claude mcp list` muestra `desi-fm` y una conversación real usa ≥ 2 tools.
- [ ] Screenshot en `docs/img/` + sección en el README.
- [ ] Commit + tracker.

## Si algo falla

- **El cliente no ve el server:** casi siempre es el path del python o del script (usar rutas absolutas; probar el comando exacto en la terminal primero).
- **Timeout en la primera tool call:** es la descarga del checkpoint; pre-calentar corriendo `python -m copilot examples/...` una vez antes de registrar, o loggear a stderr "downloading checkpoint…".
- **`mcp dev` no abre el inspector:** actualizar `pip install -U "mcp[cli]"`; requiere Node para la UI del inspector (`npx` disponible).
