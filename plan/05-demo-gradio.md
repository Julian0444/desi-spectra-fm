# 05 · Demo Gradio en HF Spaces

> **Bloque:** Nivel 1 · **Tiempo:** 2–3 h · **Depende de:** 04 · **Entregable:** **link de demo en vivo** — la pieza más valiosa del Nivel 1

## Objetivo

Un Space público donde cualquiera (recruiter incluido) elige un espectro de ejemplo o sube el suyo, mueve un slider de masking, y ve la reconstrucción + el z predicho. CPU del tier gratuito alcanza (~1 s por espectro).

## Pasos

### 1. Generar los espectros de ejemplo

`scripts/make_demo_examples.py` en el repo principal:

```python
"""Genera examples/*.npz para la demo (sintéticos con z conocido)."""
import numpy as np
from pathlib import Path

REST = {"OII": 3727.0, "CaK": 3934.0, "CaH": 3969.0, "Hb": 4861.0,
        "OIII": 5007.0, "Ha": 6563.0}

def synth(z, seed, n=6000):
    rng = np.random.default_rng(seed)
    w = np.linspace(3600.0, 9800.0, n).astype(np.float32)
    f = 0.5 + 0.15 * np.sin(w / 700.0) + rng.normal(0, 0.05, n)
    for lam in REST.values():
        obs = lam * (1 + z)
        if w[0] <= obs <= w[-1]:
            f += rng.uniform(0.4, 1.2) * np.exp(-0.5 * ((w - obs) / 5.0) ** 2)
    return w, f.astype(np.float32)

out = Path("examples"); out.mkdir(exist_ok=True)
for name, z, seed in [("galaxy_z010", 0.10, 1), ("galaxy_z042", 0.42, 2),
                      ("emission_z080", 0.80, 3), ("noisy_z025", 0.25, 4)]:
    w, f = synth(z, seed)
    np.savez(out / f"{name}.npz", flux=f, wavelength=w, z_true=np.float32(z))
print("ok")
```

Opcional (mejor aún): agregar 1–2 espectros reales de DESI exportados una vez con streaming (`HFDESISpectra` → guardar `flux/wavelength/ivar/mask` crudos antes del preprocesado).

### 2. Crear el Space

```bash
hf repo create desi-spectra-fm-demo --repo-type space --space-sdk gradio
git clone https://huggingface.co/spaces/TU_USUARIO/desi-spectra-fm-demo && cd desi-spectra-fm-demo
```

`requirements.txt` del Space:

```
numpy
matplotlib
huggingface_hub
torch
gradio
desi-fm @ git+https://github.com/TU_USUARIO/desi-spectra-fm
```

`README.md` del Space (encabezado YAML obligatorio):

```markdown
---
title: DESI Spectra Foundation Model
emoji: 🔭
colorFrom: indigo
colorTo: gray
sdk: gradio
app_file: app.py
pinned: false
---
Demo of a 26M-parameter foundation model for astronomical spectra:
redshift prediction + masked-region reconstruction. Code: <link GitHub>.
```

### 3. `app.py`

```python
import numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr
from huggingface_hub import hf_hub_download
from desi_fm.predict import load_model_from_checkpoint, predict_spectrum

CKPT = hf_hub_download("TU_USUARIO/desi-spectra-fm", "checkpoint_last.pt")
MODEL = load_model_from_checkpoint(CKPT, torch.device("cpu"))

def analyze(npz_file, mask_ratio):
    d = np.load(npz_file if isinstance(npz_file, str) else npz_file.name)
    flux, wave = d["flux"].astype(np.float32), d["wavelength"].astype(np.float32)
    r = predict_spectrum(flux=flux, wavelength=wave, model=MODEL,
                         mask_ratio=float(mask_ratio))
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    ax.plot(wave, flux, lw=0.7, color="0.45", label="input")
    ax.plot(wave, r["reconstruction_input_grid"], lw=1.0, color="#3D6FD6",
            label="reconstruction")
    ax.set_xlabel("wavelength [Å]"); ax.legend(frameon=False)
    fig.tight_layout()
    z_line = f"z predicho = {r['z_pred']:.4f}"
    if "z_true" in d.files:
        z_line += f"   (z real del ejemplo = {float(d['z_true']):.2f})"
    return z_line, fig

demo = gr.Interface(
    analyze,
    inputs=[gr.File(label="Espectro .npz (flux + wavelength en Å)"),
            gr.Slider(0.0, 0.9, value=0.0, step=0.05, label="fracción de tokens enmascarados")],
    outputs=[gr.Textbox(label="Redshift"), gr.Plot(label="Espectro")],
    examples=[["examples/galaxy_z010.npz", 0.0], ["examples/galaxy_z042.npz", 0.0],
              ["examples/emission_z080.npz", 0.5], ["examples/noisy_z025.npz", 0.35]],
    title="DESI Spectra Foundation Model",
    description="Transformer de 26M entrenado con masked-token prediction sobre espectros DESI. "
                "Acepta espectros de cualquier instrumento. "
                "[Código](https://github.com/TU_USUARIO/desi-spectra-fm)",
)
demo.launch()
```

Copiar `examples/` al Space, commit y push — el Space buildea solo (~5–10 min la primera vez).

### 4. Probar como usuario

- Abrir el link en incógnito, click en cada ejemplo, mover el slider.
- Subir un `.npz` propio y verificar que no explota con shapes raras (1-D vs 2-D).
- Poner el link arriba de todo del README de GitHub: `**[▶ Demo en vivo](https://huggingface.co/spaces/...)**`.

## Definición de hecho

- [ ] Space público, build verde, análisis en < 5 s por click.
- [ ] 4 ejemplos funcionan con un click; el slider de masking cambia visiblemente la reconstrucción.
- [ ] Link agregado al README de GitHub y a la model card.
- [ ] Tracker actualizado.

## Si algo falla

- **Build del Space falla instalando `desi-fm`:** el repo de GitHub debe ser público (plan 03) y el `pyproject.toml` instalable (`pip install "desi-fm @ git+..."` probado localmente primero).
- **Timeout al arrancar (descarga del checkpoint):** es normal la primera vez; si persiste, `HF_HUB_ENABLE_HF_TRANSFER` o reducir a lazy-load (cargar el modelo dentro de `analyze` con un `@lru_cache`).
- **La demo queda lenta en cargas concurrentes:** `demo.queue(max_size=8)` antes de `launch()`.
