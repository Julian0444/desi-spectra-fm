# 01 · Cabeza de clasificación para el redshift

> **Bloque:** Fase 0 (corrección del TA) · **Tiempo:** 2–3 h · **Depende de:** — · **Entregable:** tests verdes + smoke run sintético con la cabeza nueva

## Objetivo

Reemplazar la cabeza de regresión escalar (SmoothL1) por una **clasificación sobre bins de log(1+z)**. Es el fix directo al feedback del TA: la regresión escalar promedia los modos cuando la identificación de líneas es ambigua (Hα↔[OIII]↔[OII]) → bias + outliers catastróficos. Una distribución sobre bins puede expresar la multimodalidad, y el argmax se compromete con un modo.

Diseño clave: **retrocompatible por configuración**. `n_z_bins = 0` (default) mantiene la cabeza escalar v1 — así los checkpoints viejos siguen cargando y los tests existentes no cambian de semántica. `n_z_bins > 0` activa la clasificación.

En este plan **no se reentrena** (eso es el plan 02); acá se deja el código listo y probado.

## Pasos

### 1. `src/desi_fm/model.py` — config

```python
# agregar a DESIFoundationModelConfig:
n_z_bins: int = 0        # 0 = cabeza escalar (v1); >0 = clasificación sobre bins de log(1+z)
z_max: float = 6.0       # techo del rango de bins
z_label_smoothing: float = 0.05
```

(agregar `import math` arriba)

### 2. `model.py` — `__init__`

```python
head_out_dim = config.n_z_bins if config.n_z_bins > 0 else 1
self.redshift_head = nn.Sequential(
    nn.LayerNorm(config.d_model),
    nn.Linear(config.d_model, config.d_model),
    nn.GELU(),
    nn.Dropout(config.dropout),
    nn.Linear(config.d_model, head_out_dim),
)
if config.n_z_bins > 0:
    edges = torch.linspace(0.0, math.log1p(config.z_max), config.n_z_bins + 1)
    self.register_buffer("z_bin_edges", edges, persistent=False)
    self.register_buffer("z_bin_centers", 0.5 * (edges[:-1] + edges[1:]), persistent=False)
    # pesos de rebalanceo (los setea train.py; ones = sin rebalanceo)
    self.register_buffer("z_bin_weights", torch.ones(config.n_z_bins), persistent=False)
```

### 3. `model.py` — `forward()`

Reemplazar el bloque que produce `z_encoded_pred` / `z_pred`:

```python
head_out = self.redshift_head(z_hidden)
if self.config.n_z_bins > 0:
    z_logits = head_out                                   # (B, n_bins)
    p = z_logits.softmax(-1)
    z_encoded_pred = (p * self.z_bin_centers).sum(-1)     # esperanza en log(1+z)
    z_pred = self.decode_redshift(z_encoded_pred)
    z_pred_map = self.decode_redshift(self.z_bin_centers[z_logits.argmax(-1)])
    entropy = -(p.clamp_min(1e-9).log() * p).sum(-1) / math.log(self.config.n_z_bins)
    out["z_logits"] = z_logits
    out["z_pred_map"] = z_pred_map                        # argmax: el anti-outliers
    out["z_confidence"] = 1.0 - entropy                   # 1 = posterior concentrada
else:
    z_encoded_pred = head_out.squeeze(-1)
    z_pred = self.decode_redshift(z_encoded_pred)
```

Y el bloque de loss:

```python
if z is not None:
    if self.config.n_z_bins > 0:
        target_bin = torch.bucketize(self.encode_redshift(z), self.z_bin_edges[1:-1])
        z_loss = F.cross_entropy(
            z_logits, target_bin,
            weight=self.z_bin_weights,
            label_smoothing=self.config.z_label_smoothing,
        )
    else:
        z_loss = F.smooth_l1_loss(z_encoded_pred, self.encode_redshift(z))
    ...  # combinación con recon_loss igual que antes
```

### 4. `src/desi_fm/train.py` — flags + rebalanceo

```python
parser.add_argument("--n-z-bins", type=int, default=0)
parser.add_argument("--z-max", type=float, default=6.0)
parser.add_argument("--z-rebalance", action="store_true",
                    help="pesos inverso-frecuencia por bin, estimados de runs/desi_50k_big/predictions.csv")
```

Pasar `n_z_bins=args.n_z_bins, z_max=args.z_max` al config. Después de crear el modelo:

```python
if args.z_rebalance and args.n_z_bins > 0:
    import csv as _csv
    zs = [float(r["z_true"]) for r in _csv.DictReader(open("runs/desi_50k_big/predictions.csv"))]
    enc = torch.log1p(torch.tensor(zs).clamp(min=0.0))
    hist = torch.histc(enc, bins=args.n_z_bins, min=0.0, max=math.log1p(args.z_max)) + 1.0
    w = (hist.sum() / args.n_z_bins) / hist
    model.z_bin_weights.copy_(w.clamp(0.3, 10.0).to(model.z_bin_weights.device))
```

(Es una aproximación de la distribución de SV3 con 1000 muestras — suficiente porque los pesos van capados. Si más adelante querés la distribución exacta, un pase de streaming solo-z lo reemplaza.)

### 5. `src/desi_fm/evaluate.py` — reportar ambas variantes

Solo se toca `evaluate_and_write_outputs` (el `evaluate()` interno de `train.py` queda como está — las métricas nuevas viven en el evaluador standalone). Tres cambios:

**(a)** junto a los acumuladores existentes (`total_loss = 0.0`, …):

```python
    total_abs_z_map = 0.0
    total_abs_z_norm_map = 0.0
    total_conf = 0.0
    n_out15 = 0
    n_out15_map = 0
    has_map = False
```

**(b)** dentro del loop `for batch in loader:`, justo después de `dz = (out["z_pred"] - z).abs()`:

```python
        n_out15 += int(((dz / (1.0 + z)) > 0.15).sum().cpu())
        if "z_pred_map" in out:
            has_map = True
            dz_map = (out["z_pred_map"] - z).abs()
            total_abs_z_map += float(dz_map.sum().cpu())
            total_abs_z_norm_map += float((dz_map / (1.0 + z)).sum().cpu())
            n_out15_map += int(((dz_map / (1.0 + z)) > 0.15).sum().cpu())
            total_conf += float(out["z_confidence"].sum().cpu())
```

Para el CSV: cambiar el `DictWriter` a
`fieldnames=["z_true", "z_pred", "abs_dz", "abs_dz_norm", "z_pred_map", "z_confidence"], restval=""`
y reemplazar el bloque que escribe filas por:

```python
        if writer is not None:
            z_map_cpu = out["z_pred_map"].cpu().tolist() if "z_pred_map" in out else [None] * bsz
            conf_cpu = out["z_confidence"].cpu().tolist() if "z_confidence" in out else [None] * bsz
            for z_true_v, z_pred_v, z_map_v, conf_v in zip(z_cpu, z_pred_cpu, z_map_cpu, conf_cpu):
                row = {
                    "z_true": z_true_v,
                    "z_pred": z_pred_v,
                    "abs_dz": abs(z_pred_v - z_true_v),
                    "abs_dz_norm": abs(z_pred_v - z_true_v) / (1.0 + z_true_v),
                }
                if z_map_v is not None:
                    row["z_pred_map"] = z_map_v
                    row["z_confidence"] = conf_v
                writer.writerow(row)
```

**(c)** en el dict de retorno, agregar al final:

```python
        "eta15": n_out15 / ex_denom,
        **({
            "redshift_mae_map": total_abs_z_map / ex_denom,
            "redshift_mae_norm_map": total_abs_z_norm_map / ex_denom,
            "eta15_map": n_out15_map / ex_denom,
            "mean_z_confidence": total_conf / ex_denom,
        } if has_map else {}),
```

### 6. Tests — `tests/test_model.py`

```python
def test_classification_head_shapes_and_ranges():
    config = DESIFoundationModelConfig(n_pixels=128, n_tokens=16, d_model=48,
                                       n_layers=2, n_heads=4, dropout=0.0,
                                       n_z_bins=32, z_max=6.0)
    model = DESIFoundationModel(config)
    flux = torch.randn(3, config.n_pixels); valid = torch.ones_like(flux)
    out = model(flux, valid, z=torch.tensor([0.1, 0.8, 3.0]), mask_ratio=0.5)
    assert out["z_logits"].shape == (3, 32)
    assert torch.isfinite(out["loss"])
    for key in ("z_pred", "z_pred_map"):
        assert (out[key] >= 0).all() and (out[key] <= 6.0 + 1e-4).all()
    assert (out["z_confidence"] >= 0).all() and (out["z_confidence"] <= 1).all()

def test_scalar_head_remains_default():
    config = DESIFoundationModelConfig(n_pixels=64, n_tokens=8, d_model=32,
                                       n_layers=1, n_heads=4, dropout=0.0)
    out = DESIFoundationModel(config)(torch.randn(2, 64), torch.ones(2, 64),
                                      z=torch.tensor([0.1, 0.5]))
    assert "z_logits" not in out           # v1 intacta
```

El test de no-fuga existente (`test_redshift_is_not_input_dependent_on_true_z`) debe seguir verde — la cabeza nueva no cambia la entrada.

### 7. Verificación

```bash
python3 -m pytest tests/ -v                       # 6 tests verdes

# retrocompat: el checkpoint v1 sigue cargando y prediciendo
python3 - <<'EOF'
from desi_fm.predict import predict_spectrum
import numpy as np
w = np.linspace(3600, 9800, 5000, dtype=np.float32)
r = predict_spectrum(flux=np.random.randn(5000).astype(np.float32), wavelength=w,
                     checkpoint_path="runs/desi_50k_big/checkpoint_last.pt", device="cpu")
print("v1 OK, z_pred =", r["z_pred"])
EOF

# smoke sintético con la cabeza nueva (~3 min en CPU/MPS):
python3 -m desi_fm.train --synthetic --output-dir runs/smoke_classhead \
  --n-pixels 512 --n-tokens 32 --d-model 64 --n-layers 2 --n-heads 4 \
  --batch-size 8 --max-train-examples 512 --val-examples 64 --epochs 3 \
  --n-z-bins 64 --redshift-loss-weight 5
```

## Definición de hecho

- [ ] `pytest` verde con los 2 tests nuevos incluidos.
- [ ] El checkpoint v1 (`runs/desi_50k_big`) sigue cargando y prediciendo (script de arriba).
- [ ] El smoke sintético con `--n-z-bins 64` termina con `redshift_mae` claramente menor que al inicio del run (verlo en `runs/smoke_classhead/metrics.jsonl`).
- [ ] `evaluate.py` imprime las métricas nuevas (`*_map`, `eta15`, `mean_z_confidence`) en el run smoke.
- [ ] Commit: `plan-01: classification head over log(1+z) bins (backward-compatible)`.

## Si algo falla

- **`bucketize` fuera de rango con z > z_max:** clampear el target: `self.encode_redshift(z).clamp(max=self.z_bin_edges[-1] - 1e-6)`.
- **Device mismatch en `z_bin_weights`:** los buffers viajan con `.to(device)` del modelo; asegurarse de copiar los pesos *después* de `model.to(device)`.
- **El smoke no aprende:** subir `--redshift-loss-weight` a 10 y `--epochs` a 5; con 64 bins y datos sintéticos limpios tiene que bajar sí o sí.
