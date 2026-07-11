# 10 · Embeddings + búsqueda semántica (FAISS)

> **Bloque:** Nivel 3 · **Tiempo:** 3–4 h (+ ~40 min de indexado desatendido) · **Depende de:** 02 (checkpoint bueno), 07 · **Entregable:** "espectros similares a este" funcionando + UMAP coloreado por z

## Objetivo

Usar el encoder como **modelo de embeddings** — la demostración conceptual de que entendés para qué sirve un foundation model: representaciones reutilizables downstream. Además le da al agente la tool `find_similar_spectra` y produce la imagen más linda del portfolio (UMAP con gradiente de z).

## Pasos

### 1. `encode()` en `desi-spectra-fm`

En `model.py` (método nuevo de `DESIFoundationModel`):

```python
@torch.no_grad()
def encode(self, flux: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Embedding por espectro: mean-pooling de los tokens espectrales válidos."""
    tokens, _, valid_patches = self.tokenizer(flux, valid)
    z_token = self.z_mask_token.expand(flux.shape[0], 1, -1)
    seq = torch.cat([tokens, z_token], dim=1)          # sin masking: contexto completo
    seq = seq + self.wavelength_pos_embed.to(seq.dtype)
    if self.learned_pos_embed is not None:
        seq = seq + self.learned_pos_embed
    hidden = self.norm(self.encoder(seq))
    spec_hidden = hidden[:, : self.config.n_tokens]     # (B, 273, d)
    w = (valid_patches.mean(-1) > 0.0).float()          # token válido si tiene píxeles válidos
    pooled = (spec_hidden * w.unsqueeze(-1)).sum(1) / w.sum(1, keepdim=True).clamp_min(1.0)
    return pooled                                       # (B, d_model)
```

En `predict.py`, wrapper público `embed_spectrum(flux, wavelength, ..., model=...) -> np.ndarray` que reusa el mismo preprocesado de `predict_spectrum`. Test: dos espectros sintéticos con el mismo z tienen mayor similitud coseno entre sí que contra uno de z muy distinto.

### 2. `scripts/build_index.py` (en spectra-copilot)

```python
"""Indexa N espectros DESI: embeddings L2-normalizados → FAISS IndexFlatIP."""
# pip install faiss-cpu
import faiss, numpy as np, torch
from desi_fm.data import HFDESISpectra, SpectrumPreprocessConfig, collate_spectra
from torch.utils.data import DataLoader

import os
from huggingface_hub import hf_hub_download
from desi_fm.predict import load_model_from_checkpoint

N = 15_000
ds = HFDESISpectra(max_examples=N, shuffle_buffer=4096)
loader = DataLoader(ds, batch_size=32, collate_fn=collate_spectra)

ckpt = os.environ.get("DESI_FM_CKPT") or hf_hub_download(
    "TU_USUARIO/desi-spectra-fm", "checkpoint_last.pt")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = load_model_from_checkpoint(ckpt, device)

embs, zs, done = [], [], 0
for batch in loader:
    e = model.encode(batch["flux"].to(device), batch["valid"].to(device))
    embs.append(e.cpu().numpy()); zs.append(batch["z"].numpy())
    done += len(batch["z"])
    if done % 1600 == 0:
        print(f"{done}/{N}")
X = np.concatenate(embs).astype(np.float32)
X /= np.linalg.norm(X, axis=1, keepdims=True)
index = faiss.IndexFlatIP(X.shape[1])
index.add(X)
faiss.write_index(index, "data/spectra.faiss")
np.savez("data/spectra_meta.npz", z=np.concatenate(zs))
```

~15k espectros en streaming ≈ 30–60 min (una sola vez). Guardar también los `flux/valid` de un subset chico si querés mostrar los vecinos gráficamente.

### 3. Tool `find_similar_spectra` (agregar a `tools.py` + al agente + al MCP server)

```python
def find_similar_spectra_impl(npz_path: str, k: int = 5) -> dict:
    flux, wave = _load(npz_path)
    e = embed_spectrum(flux=flux, wavelength=wave, model=_model())
    e = (e / np.linalg.norm(e)).astype(np.float32)[None, :]
    index = faiss.read_index("data/spectra.faiss")
    meta = np.load("data/spectra_meta.npz")
    sims, ids = index.search(e, k)
    return {"neighbors": [
        {"rank": i + 1, "similarity": round(float(s), 3), "z": round(float(meta["z"][j]), 3)}
        for i, (s, j) in enumerate(zip(sims[0], ids[0]))
    ]}
```

Uso por el agente: "los 5 vecinos más cercanos tienen z entre 0.40 y 0.45 (find_similar_spectra) — consistente con z_pred = 0.42". **Otra vía de validación independiente.**

### 4. El visual: UMAP coloreado por z

```python
# pip install umap-learn
import umap, numpy as np, matplotlib.pyplot as plt
X = ...  # embeddings ya normalizados
z = np.load("data/spectra_meta.npz")["z"]
xy = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine").fit_transform(X)
plt.figure(figsize=(7, 6))
sc = plt.scatter(xy[:, 0], xy[:, 1], c=np.log1p(z), s=2, cmap="viridis", alpha=0.6)
plt.colorbar(sc, label="log(1+z)"); plt.axis("off")
plt.title("Espacio de embeddings del foundation model, coloreado por redshift")
plt.savefig("docs/img/umap_z.png", dpi=180, bbox_inches="tight")
```

Si el modelo aprendió física, el gradiente de z se ve **a simple vista** sin que nadie le haya enseñado a ordenarse así. Esa imagen + una frase van al README de ambos repos.

## Definición de hecho

- [ ] `embed_spectrum` con test de sanidad (mismo z → más similar).
- [ ] Índice de ≥ 10k espectros construido y guardado.
- [ ] `find_similar_spectra` integrada al agente y al MCP server; una consulta devuelve vecinos con z coherentes.
- [ ] `docs/img/umap_z.png` con gradiente de z visible, linkeado en el README.
- [ ] Commit + tracker.

## Si algo falla

- **faiss-cpu no instala en Python 3.9:** usar el venv 3.10+ de spectra-copilot (el índice vive ahí).
- **El UMAP sale sin estructura:** verificar que los embeddings vienen del checkpoint v2 entrenado (no de pesos random), y probar `n_neighbors=50`; también puede ayudar poolear solo tokens con `w>0`.
- **Streaming lento:** bajar a N=8k; el punto se demuestra igual.
