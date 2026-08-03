"""HF Space wrapper for the desi-fm REST API (plan 06).

Docker Spaces need a PRO subscription (HTTP 402 on free accounts), so the free
deploy path for this API is a ZeroGPU *Gradio* Space that serves FastAPI:

  * the REST endpoints (`/predict`, `/predict_json`, `/healthz`, `/docs`) come
    from `desi_fm.api` and run inference on **CPU** — `curl` works for anyone,
    anonymously, without touching the ZeroGPU visitor quota;
  * a minimal Gradio UI is mounted on the same app for browser visitors; its
    handler carries the `@spaces.GPU` decorator that zero-a10g hardware
    requires at startup (UI clicks spend a little ZeroGPU quota, the REST API
    does not).
"""

import json
import os

# Con SSR (default en Spaces) el frontend Node ocupa el puerto 7860 y las
# rutas REST inyectadas abajo no quedan expuestas; servidor python clasico.
os.environ["GRADIO_SSR_MODE"] = "false"

import spaces  # must be imported before torch on ZeroGPU
import numpy as np
import gradio as gr

from desi_fm.api import app as fastapi_app, get_model
from desi_fm.predict import predict_spectrum

CODE_URL = "https://github.com/Julian0444/desi-spectra-fm"
MODEL_URL = "https://huggingface.co/jirustaroure/desi-spectra-fm"

MODEL = get_model()  # eager CPU load at startup: fail fast, fast first request


def _first_row(arr):
    arr = np.asarray(arr)
    return arr if arr.ndim == 1 else arr[0]


@spaces.GPU(duration=8)
def predict_ui(npz_file, mask_ratio: float):
    path = npz_file if isinstance(npz_file, str) else getattr(npz_file, "name", None)
    if not path:
        raise gr.Error("Upload a .npz file with 'flux' and 'wavelength' arrays.")
    try:
        data = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise gr.Error(f"Could not read the file as a .npz archive: {exc}")
    if "flux" not in data.files or "wavelength" not in data.files:
        raise gr.Error(
            f"The .npz must contain 'flux' and 'wavelength'; found {sorted(data.files)}."
        )
    flux = _first_row(data["flux"]).astype(np.float32)
    wavelength = _first_row(data["wavelength"]).astype(np.float32)
    if flux.shape != wavelength.shape:
        raise gr.Error(f"flux {flux.shape} and wavelength {wavelength.shape} differ.")
    result = predict_spectrum(
        flux=flux, wavelength=wavelength, model=MODEL, mask_ratio=float(mask_ratio)
    )
    payload = {}
    if "z_pred_map" in result:
        payload["z_pred_map"] = round(result["z_pred_map"], 4)
        payload["z_confidence"] = round(result["z_confidence"], 4)
    payload["z_pred"] = round(result["z_pred"], 4)
    return json.dumps(payload, indent=2)


demo = gr.Interface(
    fn=predict_ui,
    inputs=[
        gr.File(label="Spectrum .npz (flux + wavelength in Å)", file_types=[".npz"]),
        gr.Slider(0.0, 0.9, value=0.0, step=0.05,
                  label="fraction of input tokens masked"),
    ],
    outputs=gr.Textbox(label="API response", lines=5),
    title="desi-fm API — REST endpoint for spectrum redshifts",
    description=(
        "This Space is primarily a **REST API** (FastAPI): interactive Swagger "
        "docs at [/api/docs](/api/docs), health check at `/api/healthz`. "
        "`POST /api/predict` takes a `.npz` upload, `POST /api/predict_json` "
        "takes JSON lists — the official prediction is `z_pred_map` "
        "(+ `z_confidence`); `z_pred` is secondary. REST calls run on CPU and "
        "spend **no** ZeroGPU quota.\n\n"
        "```bash\n"
        "curl -s https://jirustaroure-desi-fm-api.hf.space/api/healthz\n"
        "curl -s -F \"file=@spectrum.npz\" "
        "\"https://jirustaroure-desi-fm-api.hf.space/api/predict?mask_ratio=0.0\"\n"
        "```\n\n"
        "This panel is a small tester for the same model call "
        f"(interactive demo with plots: [desi-spectra-fm-demo]"
        f"(https://huggingface.co/spaces/jirustaroure/desi-spectra-fm-demo)). "
        f"[Code + tests]({CODE_URL}) · [Model checkpoint]({MODEL_URL})"
    ),
)

demo.queue(max_size=8)

if __name__ == "__main__":
    # En ZeroGPU el handshake de la lib `spaces` con el scheduler pasa por
    # demo.launch() (con mount_gradio_app el pod recibe SIGTERM a los pocos
    # segundos), asi que se lanza gradio normal y se injerta la API REST como
    # sub-app en /api del FastAPI que gradio ya sirve: /api/docs, /api/predict.
    launched = demo.launch(prevent_thread_lock=True)
    gradio_app = getattr(demo, "server_app", None)
    if gradio_app is None and isinstance(launched, tuple):
        gradio_app = launched[0]
    gradio_app.mount("/api", fastapi_app)
    demo.block_thread()
