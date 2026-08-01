# 02R · Fine-tuning v2.1: loss calibrada, split aislado y checkpoint seleccionable

> **Bloque:** Fase 0 (recuperación del plan 02) · **Tiempo:** 1.5–2 h activas + 2.5–3 h de cómputo · **Depende de:** 01 y del diagnóstico del intento v2.0 registrado en `plan/HANDOFF.md` · **Entregable:** un checkpoint fine-tune candidato evaluado honestamente contra la v1, o una decisión explícita de publicar la v1

## Objetivo

Producir una v2.1 científicamente defendible mediante **fine-tuning del encoder v1**
con una cabeza de clasificación nueva, corrigiendo cuatro problemas del intento v2.0
antes de gastar otra corrida completa:

1. La `cross_entropy` de 200 clases conservó el peso `w_z=10` de la loss SmoothL1
   escalar. En el último validation de v2.0, `0.769 + 10 × 5.027 = 51.04`; la
   clasificación representó aproximadamente 98 % de la loss total.
2. Los pesos se estimaron con solo 1.000 redshifts para 200 bins: 83 bins quedaron
   vacíos. Combinados con `label_smoothing=0.05`, esos bins igualmente reciben
   gradiente.
3. El entrenamiento v2.0 aplicó `shuffle_buffer=2048` antes de cortar train, por lo
   que el buffer pudo leer hasta ~2.047 posiciones posteriores e introducir parte
   del supuesto held-out 80k–82k. A la vez, `skip_examples` cuenta filas crudas antes
   de descartar labels inválidos. La corrección es:
   `filter(valid) → skip → take(window) → shuffle(train only)`. Así la membresía
   queda fija y los batches siguen decorrelacionados.
4. El loop guarda `checkpoint_last.pt` y valida la esperanza de la posterior, pero el
   intento v2.0 demostró que la predicción útil es `z_pred_map`. La selección del mejor
   checkpoint debe usar `eta15_map`.

La estrategia elegida es conservadora: reutilizar el encoder y las capas compatibles
de v1, reinicializar únicamente la salida incompatible de redshift, usar 100 bins,
normalizar la CE por `log(n_bins)`, calcular pesos raíz-inversa desde los 80k labels
de entrenamiento, desactivar smoothing en la primera corrida y seleccionar el
checkpoint por MAP. Cada época conserva los mismos 80k miembros pero usa un orden
distinto (`seed + epoch`). No se sobreescribe ningún artefacto anterior.

**Narrativa obligatoria:** v2.1 no es un modelo entrenado desde cero. Debe describirse
en README, model card, notebook y portfolio como “fine-tuning de v1 con una nueva
cabeza de clasificación de redshift”.

## Arquitectura

El stream se filtra por labels válidos, se divide determinísticamente en train
`[0, 80000)` y held-out `[80000, 82000)`, y solo la ventana de train recibe buffer
shuffle con una semilla distinta por época. El modelo carga todas las capas v1 con
shape compatible, reinicializa la salida de clasificación y optimiza reconstrucción
más CE calibrada. Cada época valida por `z_pred_map`; se conservan `best` y `last`, y
un evaluador externo decide contra la v1 canónica mediante gates conjuntos.

## Stack y archivos

- Python 3.9+, PyTorch, NumPy, Hugging Face `datasets` (versión y semántica auditadas
  en Fase 0).
- Modificar: `src/desi_fm/model.py`, `src/desi_fm/data.py`, `src/desi_fm/train.py`,
  `src/desi_fm/predict.py`, `src/desi_fm/evaluate.py`.
- Crear: `scripts/estimate_z_histogram.py`, `tests/test_data.py`,
  `tests/test_train.py`, `tests/test_predict.py`, `tests/test_evaluate.py`.
- Modificar tests: `tests/test_model.py`.
- Artefactos nuevos:
  - `runs/calibration/z_hist_80k_100bins.npz`
  - `runs/desi_80k_classhead_v21_preflight/`
  - `runs/desi_80k_classhead_v21/`
- Preservar sin modificar: `runs/desi_150k_classhead/` (v2.0 fallida) y
  `runs/desi_50k_big/checkpoint_last.pt` (v1).

## Baseline y gates

La decisión v1↔v2.1 usa exactamente los labels válidos 80k–82k:

| métrica | v1 held-out | v2.0 MAP* | gate mínimo v2.1 | objetivo fuerte | stretch |
|---|---:|---:|---:|---:|---:|
| η₀.₁₅ global | 27.3 % | 31.1 % | **< 27.3 %** | < 20 % | < 10 % |
| σ_NMAD | 0.093 | 0.111 | **< 0.093** | < 0.07 | < 0.05 |
| MAE_norm | 0.125 | 0.182 | **≤ 0.125** | < 0.10 | — |
| η₀.₁₅, z∈[1.5,2.5] | 87 % | 32 % | **< 50 %** | < 30 % | — |
| RMSE reconstrucción | 0.854 | 0.877 | **≤ 0.90** | ≤ 0.86 | — |

La v2.1 solo reemplaza a la v1 en Hugging Face si cumple **todos** los gates mínimos.
Si no los cumple, la v1 sigue siendo el checkpoint recomendado y v2.0/v2.1 se
documentan como experimentos.

Los valores de la tabla son la referencia conocida. Después de corregir la semántica
de `skip_examples`, se vuelve a medir la v1 y los gates globales se calculan desde ese
artefacto, no copiando números a mano.

\* La evaluación v2.0 usó esas posiciones, pero su train fue barajado antes del corte
y pudo haber visto hasta ~2.047 miembros de esa ventana. El 31.1 % es diagnóstico y
posiblemente optimista; no participa como baseline ni como candidato de publicación.
Como aun con esa ventaja perdió contra v1, la conclusión de que v2.0 no se promueve
permanece válida.

---

## Fase 0 · Auditar y preservar el estado

### Paso 0.1 — Confirmar que no hay entrenamiento real activo

```bash
cd "/Users/jirustaroure/Desktop/FINAL PROJECT DEEP LEARNING"
pgrep -af 'python.*-m desi_fm\.train' || true
```

Esperado: ninguna línea. Un `tail -f metrics.jsonl` no es entrenamiento.

### Paso 0.2 — Registrar el working tree y no mezclar cambios previos

```bash
git status --short --branch
git diff -- src/desi_fm/evaluate.py plan/02-reentrenamiento-v2.md
```

No ejecutar `git reset`, `git checkout --`, `git clean` ni `git add -A`. Los cambios
de la sesión anterior pertenecen al usuario.

### Paso 0.3 — Verificar los dos checkpoints de referencia

```bash
test -f runs/desi_50k_big/checkpoint_last.pt
test -f runs/desi_150k_classhead/checkpoint_last.pt
tail -1 runs/desi_150k_classhead/metrics.jsonl
rg 'done step=20000' runs/desi_150k_classhead/train.log
```

Esperado: la última métrica es validation del step 20.000 y el log contiene
`done step=20000`.

### Paso 0.4 — Revalidar firmas antes de escribir tests

```bash
python3 - <<'PY'
import inspect
import math
from pathlib import Path

import datasets
import torch
from datasets import IterableDataset

from desi_fm.model import DESIFoundationModel, DESIFoundationModelConfig
from desi_fm.train import save_checkpoint

config = DESIFoundationModelConfig(
    n_pixels=64,
    n_tokens=8,
    d_model=32,
    n_layers=1,
    n_heads=4,
    n_z_bins=16,
)
model = DESIFoundationModel(config)
params = inspect.signature(save_checkpoint).parameters
assert "scheduler" in params
assert model.redshift_head[-1] is model.redshift_head[4]
assert model.redshift_head[-1].out_features == 16

v1 = torch.load(
    Path("runs/desi_50k_big/checkpoint_last.pt"),
    map_location="cpu",
)
for key, expected in {
    "d_model": 512,
    "n_layers": 8,
    "n_heads": 8,
    "n_tokens": 273,
}.items():
    assert v1["config"][key] == expected, (key, v1["config"][key])

rows = [
    {"Z": float("nan")},
    {"Z": -1.0},
    *[{"Z": float(z)} for z in range(6)],
]


def valid_redshift(row):
    z = float(row["Z"])
    return math.isfinite(z) and z >= 0.0


def stream():
    return IterableDataset.from_generator(lambda: iter(rows)).filter(
        valid_redshift
    )


train = stream().take(4).shuffle(buffer_size=4, seed=42)
heldout = stream().skip(4).take(2)
train_z = {float(row["Z"]) for row in train}
heldout_z = [float(row["Z"]) for row in heldout]
assert train_z == {0.0, 1.0, 2.0, 3.0}, train_z
assert heldout_z == [4.0, 5.0], heldout_z
assert train_z.isdisjoint(heldout_z)

print("SIGNATURE_AUDIT_OK")
print(f"DATASETS_SEMANTICS_OK version={datasets.__version__}")
PY
```

Esperado: `SIGNATURE_AUDIT_OK` y `DATASETS_SEMANTICS_OK version=...`. La prueba de
semántica importa más que un número de versión: demuestra en el entorno real que
`filter → take → shuffle` no incorpora miembros del held-out. Si falla, no continuar
con entrenamiento y adaptar el pipeline antes del TDD. Si una firma cambió por
trabajo posterior, adaptar los tests preservando la intención —carga compatible,
salida MAP y selección de best— en vez de forzar índices obsoletos.

---

## Fase 1 · TDD para normalizar la CE

### Paso 1.1 — Escribir primero el test que debe fallar

Agregar a `tests/test_model.py`:

```python
import math


def test_classification_loss_can_be_normalized_by_log_bins():
    config = DESIFoundationModelConfig(
        n_pixels=64,
        n_tokens=8,
        d_model=32,
        n_layers=1,
        n_heads=4,
        dropout=0.0,
        n_z_bins=16,
        z_max=6.0,
        z_label_smoothing=0.0,
        normalize_redshift_ce=True,
        redshift_loss_weight=1.0,
    )
    model = DESIFoundationModel(config)
    with torch.no_grad():
        model.redshift_head[-1].weight.zero_()
        model.redshift_head[-1].bias.zero_()

    flux = torch.randn(4, config.n_pixels)
    valid = torch.ones_like(flux)
    mask = torch.zeros(4, config.n_tokens, dtype=torch.bool)
    out = model(
        flux,
        valid,
        z=torch.tensor([0.1, 0.4, 1.0, 2.0]),
        spectrum_mask=mask,
    )

    assert torch.allclose(
        out["redshift_loss_raw"],
        torch.tensor(math.log(config.n_z_bins)),
        atol=1e-5,
    )
    assert torch.allclose(out["redshift_loss"], torch.tensor(1.0), atol=1e-5)
```

### Paso 1.2 — Ejecutar el test y confirmar el fallo correcto

```bash
python3 -m pytest tests/test_model.py::test_classification_loss_can_be_normalized_by_log_bins -q
```

Esperado: falla porque `DESIFoundationModelConfig` todavía no acepta
`normalize_redshift_ce`.

### Paso 1.3 — Implementar la normalización de manera retrocompatible

En `DESIFoundationModelConfig`, dentro de `src/desi_fm/model.py`, agregar:

```python
normalize_redshift_ce: bool = False
```

El default `False` permite cargar v1 y v2.0 sin cambiar su semántica. En el bloque de
loss de clasificación de `forward()`, reemplazar la asignación directa por:

```python
z_loss_raw = F.cross_entropy(
    z_logits,
    target_bin,
    weight=self.z_bin_weights,
    label_smoothing=self.config.z_label_smoothing,
)
z_loss = (
    z_loss_raw / math.log(self.config.n_z_bins)
    if self.config.normalize_redshift_ce
    else z_loss_raw
)
```

En la rama escalar:

```python
z_loss_raw = F.smooth_l1_loss(z_encoded_pred, self.encode_redshift(z))
z_loss = z_loss_raw
```

Antes de devolver, junto con `out["redshift_loss"]`, agregar:

```python
out["redshift_loss_raw"] = z_loss_raw
```

### Paso 1.4 — Verificar test estrecho y suite completa

```bash
python3 -m pytest tests/test_model.py::test_classification_loss_can_be_normalized_by_log_bins -q
python3 -m pytest tests/ -q
```

Esperado: test nuevo verde y suite completa verde.

### Paso 1.5 — Commit enfocado

```bash
git add src/desi_fm/model.py tests/test_model.py
git commit -m "fix: normalize classification redshift loss"
```

No incluir cambios previos no relacionados.

---

## Fase 2 · TDD para split, pesos, histograma y warm start

### Paso 2.1 — Crear los tests fallidos

Crear `tests/test_data.py`:

```python
import numpy as np
from datasets import IterableDataset

from desi_fm.data import HFDESISpectra, SpectrumPreprocessConfig


def _example(z: float) -> dict:
    wavelength = np.linspace(3600.0, 9800.0, 16, dtype=np.float32)
    return {
        "Z": z,
        "spectrum": {
            "flux": np.ones(16, dtype=np.float32),
            "ivar": np.ones(16, dtype=np.float32),
            "lambda": wavelength,
            "mask": np.zeros(16, dtype=bool),
        },
    }


def _attach_stream(monkeypatch, dataset, rows):
    def load_base_stream():
        return IterableDataset.from_generator(lambda: iter(rows))

    monkeypatch.setattr(dataset, "_load_base_stream", load_base_stream)


def test_take_then_shuffle_keeps_train_and_heldout_disjoint(monkeypatch):
    rows = [
        _example(float("nan")),
        _example(-1.0),
        *[_example(float(z)) for z in range(6)],
    ]
    train = HFDESISpectra(
        max_examples=4,
        skip_examples=0,
        shuffle_buffer=4,
        seed=17,
        preprocess=SpectrumPreprocessConfig(n_pixels=16),
    )
    heldout = HFDESISpectra(
        max_examples=2,
        skip_examples=4,
        shuffle_buffer=0,
        preprocess=SpectrumPreprocessConfig(n_pixels=16),
    )
    _attach_stream(monkeypatch, train, rows)
    _attach_stream(monkeypatch, heldout, rows)

    train_z = [float(row["z"]) for row in train]
    heldout_z = [float(row["z"]) for row in heldout]

    assert set(train_z) == {0.0, 1.0, 2.0, 3.0}
    assert heldout_z == [4.0, 5.0]
    assert set(train_z).isdisjoint(heldout_z)


def test_training_membership_is_fixed_but_order_changes_by_epoch(monkeypatch):
    rows = [_example(float(z)) for z in range(10)]
    dataset = HFDESISpectra(
        max_examples=10,
        skip_examples=0,
        shuffle_buffer=10,
        seed=42,
        preprocess=SpectrumPreprocessConfig(n_pixels=16),
    )
    _attach_stream(monkeypatch, dataset, rows)

    dataset.set_epoch(0)
    epoch_0 = [float(row["z"]) for row in dataset]
    dataset.set_epoch(1)
    epoch_1 = [float(row["z"]) for row in dataset]
    dataset.set_epoch(0)
    epoch_0_repeat = [float(row["z"]) for row in dataset]

    assert set(epoch_0) == set(range(10))
    assert set(epoch_1) == set(range(10))
    assert epoch_0 != epoch_1
    assert epoch_0_repeat == epoch_0
```

Crear `tests/test_train.py`:

```python
from pathlib import Path

import numpy as np
import torch

from desi_fm.model import DESIFoundationModel, DESIFoundationModelConfig
from desi_fm.train import (
    build_z_bin_weights,
    compute_z_histogram,
    is_better_checkpoint,
    load_compatible_checkpoint,
)


def _tiny_config(n_z_bins: int) -> DESIFoundationModelConfig:
    return DESIFoundationModelConfig(
        n_pixels=64,
        n_tokens=8,
        d_model=32,
        n_layers=1,
        n_heads=4,
        dropout=0.0,
        n_z_bins=n_z_bins,
        z_max=6.0,
    )


def test_compute_z_histogram_counts_every_valid_label_once():
    counts, edges = compute_z_histogram(
        np.asarray([0.0, 0.1, 0.5, 1.0, 3.0, np.nan, -1.0]),
        n_bins=10,
        z_max=6.0,
    )
    assert counts.shape == (10,)
    assert edges.shape == (11,)
    assert int(counts.sum()) == 5


def test_sqrt_inverse_weights_ignore_empty_bins():
    counts = torch.tensor([100.0, 25.0, 0.0])
    weights = build_z_bin_weights(
        counts,
        mode="sqrt_inverse",
        min_weight=0.5,
        max_weight=3.0,
    )
    assert weights[2].item() == 0.0
    assert 0.5 <= weights[0].item() < weights[1].item() <= 3.0


def test_warm_start_loads_encoder_but_skips_incompatible_head(tmp_path: Path):
    source = DESIFoundationModel(_tiny_config(n_z_bins=0))
    with torch.no_grad():
        first_encoder_param = next(source.encoder.parameters())
        first_encoder_param.fill_(0.123)
    checkpoint = tmp_path / "v1.pt"
    torch.save(
        {"model": source.state_dict(), "config": source.config.to_dict()},
        checkpoint,
    )

    target = DESIFoundationModel(_tiny_config(n_z_bins=16))
    report = load_compatible_checkpoint(target, checkpoint, torch.device("cpu"))

    assert torch.allclose(
        next(target.encoder.parameters()),
        torch.full_like(next(target.encoder.parameters()), 0.123),
    )
    assert "redshift_head.4.weight" in report["skipped"]
    assert target.redshift_head[-1].out_features == 16


def test_best_checkpoint_requires_nonempty_map_improvement():
    assert not is_better_checkpoint(
        {"examples": 0, "eta15_map": 0.0},
        best_score=0.5,
    )
    assert is_better_checkpoint(
        {"examples": 2000, "eta15_map": 0.25},
        best_score=0.5,
    )
    assert not is_better_checkpoint(
        {"examples": 2000, "eta15_map": 0.60},
        best_score=0.5,
    )
```

### Paso 2.2 — Confirmar los fallos esperados

```bash
python3 -m pytest tests/test_data.py -q
python3 -m pytest tests/test_train.py -q
```

Esperado: `tests/test_data.py` falla porque `_load_base_stream`/`set_epoch` todavía
no existen; `tests/test_train.py` falla por imports ausentes.

### Paso 2.3 — Corregir el split e implementar helpers

En `src/desi_fm/data.py`, extraer la validación del label para compartirla entre el
pipeline de Hugging Face y `extract_mmu_desi_example()`:

```python
def extract_redshift(example: dict[str, Any]) -> float:
    redshift_key = next(
        (key for key in ("Z", "redshift", "z") if key in example),
        None,
    )
    if redshift_key is None:
        raise KeyError("Expected a redshift field named 'Z', 'redshift', or 'z'.")
    return float(example[redshift_key])


def has_valid_redshift(example: dict[str, Any]) -> bool:
    redshift = extract_redshift(example)
    return math.isfinite(redshift) and redshift >= 0.0
```

En `extract_mmu_desi_example()`, reemplazar la búsqueda duplicada por:

```python
redshift = extract_redshift(example)
```

En `HFDESISpectra.__init__()`, agregar:

```python
self.epoch = 0
```

Reemplazar `_load_stream()` y `__iter__()`, y agregar `set_epoch()`:

```python
def set_epoch(self, epoch: int) -> None:
    self.epoch = int(epoch)


def _load_base_stream(self):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Install the data dependencies first: pip install -r requirements.txt"
        ) from exc

    kwargs: dict[str, Any] = {"split": self.split, "streaming": True}
    if self.data_dir:
        kwargs["data_dir"] = self.data_dir
    return load_dataset(self.dataset_name, **kwargs)


def _load_stream(self):
    stream = self._load_base_stream()
    stream = stream.filter(has_valid_redshift)
    if self.skip_examples > 0:
        stream = stream.skip(self.skip_examples)
    if self.max_examples is not None:
        stream = stream.take(self.max_examples)
    if self.shuffle_buffer > 0:
        stream = stream.shuffle(
            buffer_size=self.shuffle_buffer,
            seed=self.seed + self.epoch,
        )
    return stream


def __iter__(self):
    for example in self._load_stream():
        flux, ivar, wavelength, mask, redshift = extract_mmu_desi_example(
            example
        )
        processed = preprocess_spectrum(
            flux,
            ivar,
            wavelength,
            mask,
            self.preprocess,
        )
        yield {
            "flux": processed["flux"],
            "valid": processed["valid"],
            "z": np.float32(redshift),
        }
```

El orden es deliberado: primero se filtra, después se fija la ventana con
`skip/take`, y solo entonces se aplica el buffer shuffle. `shuffle()` no puede leer
fuera de los 80k miembros tomados.

En el loop de `train()` dentro de `src/desi_fm/train.py`, antes de construir
`progress` para cada época:

```python
if hasattr(train_loader.dataset, "set_epoch"):
    train_loader.dataset.set_epoch(epoch)
print(f"dataset_epoch={epoch} shuffle_seed={args.seed + epoch}")
```

Así las tres épocas conservan la misma membresía y usan semillas 42, 43 y 44.

Luego agregar `import numpy as np` y estos helpers antes de `train()`:

```python
def compute_z_histogram(
    redshifts: np.ndarray,
    *,
    n_bins: int,
    z_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(redshifts, dtype=np.float64)
    valid = np.isfinite(values) & (values >= 0.0)
    encoded = np.log1p(np.minimum(values[valid], z_max))
    edges = np.linspace(0.0, math.log1p(z_max), n_bins + 1)
    counts, _ = np.histogram(encoded, bins=edges)
    return counts.astype(np.int64), edges.astype(np.float64)


def build_z_bin_weights(
    counts: torch.Tensor,
    *,
    mode: str,
    min_weight: float,
    max_weight: float,
) -> torch.Tensor:
    counts = counts.float()
    if counts.ndim != 1:
        raise ValueError("counts must be one-dimensional")
    if mode == "none":
        return torch.ones_like(counts)
    if mode != "sqrt_inverse":
        raise ValueError(f"unknown z weighting mode: {mode}")
    observed = counts > 0
    if not bool(observed.any()):
        raise ValueError("histogram has no observed bins")
    mean_count = counts[observed].mean()
    weights = torch.zeros_like(counts)
    weights[observed] = torch.sqrt(mean_count / counts[observed]).clamp(
        min=min_weight,
        max=max_weight,
    )
    return weights


def load_compatible_checkpoint(
    model: DESIFoundationModel,
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    source = dict(checkpoint["model"])
    if "pos_embed" in source and "learned_pos_embed" not in source:
        source["learned_pos_embed"] = source.pop("pos_embed")
    target = model.state_dict()
    compatible = {
        key: value
        for key, value in source.items()
        if key in target and target[key].shape == value.shape
    }
    skipped = sorted(key for key in source if key not in compatible)
    model.load_state_dict(compatible, strict=False)
    return {"loaded": len(compatible), "skipped": skipped}


def is_better_checkpoint(
    metrics: dict[str, float],
    *,
    best_score: float,
) -> bool:
    return (
        metrics.get("examples", 0.0) > 0
        and "eta15_map" in metrics
        and math.isfinite(metrics["eta15_map"])
        and metrics["eta15_map"] < best_score
    )
```

### Paso 2.4 — Verificar los helpers

```bash
python3 -m pytest tests/test_data.py -q
python3 -m pytest tests/test_train.py -q
python3 -m pytest tests/ -q
```

Esperado: todos verdes.

### Paso 2.5 — Commit enfocado

```bash
git add src/desi_fm/data.py src/desi_fm/train.py tests/test_data.py tests/test_train.py
git commit -m "feat: isolate split and calibrate redshift fine-tuning"
```

---

## Fase 3 · Cablear CLI, métricas MAP y mejor checkpoint

### Paso 3.1 — Agregar flags reproducibles

En `parse_args()` de `src/desi_fm/train.py`, agregar:

```python
parser.add_argument("--z-label-smoothing", type=float, default=0.05)
parser.add_argument("--normalize-redshift-ce", action="store_true")
parser.add_argument(
    "--z-weighting",
    choices=["none", "sqrt_inverse"],
    default="none",
)
parser.add_argument("--z-histogram", default="")
parser.add_argument("--z-weight-min", type=float, default=0.5)
parser.add_argument("--z-weight-cap", type=float, default=3.0)
parser.add_argument("--init-checkpoint", default="")
```

Pasar al config:

```python
z_label_smoothing=args.z_label_smoothing,
normalize_redshift_ce=args.normalize_redshift_ce,
```

Inmediatamente después de crear el modelo:

```python
if args.init_checkpoint:
    report = load_compatible_checkpoint(
        model,
        Path(args.init_checkpoint),
        device,
    )
    print("warm_start", json.dumps(report))

if args.z_weighting != "none":
    if not args.z_histogram:
        raise ValueError("--z-histogram is required when --z-weighting is enabled")
    payload = np.load(args.z_histogram, allow_pickle=False)
    counts = torch.from_numpy(payload["counts"])
    if counts.numel() != args.n_z_bins:
        raise ValueError(
            f"histogram has {counts.numel()} bins; model uses {args.n_z_bins}"
        )
    weights = build_z_bin_weights(
        counts,
        mode=args.z_weighting,
        min_weight=args.z_weight_min,
        max_weight=args.z_weight_cap,
    )
    model.z_bin_weights.copy_(weights.to(model.z_bin_weights.device))
```

No usar simultáneamente el flag legacy `--z-rebalance` en v2.1.

Guardar los argumentos de entrenamiento además del config del modelo:

```python
(out_dir / "training_args.json").write_text(
    json.dumps(vars(args), indent=2, sort_keys=True)
)
```

Agregar al JSONL de entrenamiento:

```python
"redshift_loss_raw": float(out["redshift_loss_raw"].detach().cpu()),
```

### Paso 3.2 — Extender el evaluador interno de `train.py`

En `evaluate()`, acumular `eta15` y, cuando exista, métricas MAP:

```python
total_abs_z_map = 0.0
total_abs_z_norm_map = 0.0
n_out15 = 0
n_out15_map = 0
has_map = False
```

Dentro del loop:

```python
dzn = dz / (1.0 + z)
n_out15 += int((dzn > 0.15).sum().cpu())
if "z_pred_map" in out:
    has_map = True
    dz_map = (out["z_pred_map"] - z).abs()
    total_abs_z_map += float(dz_map.sum().cpu())
    total_abs_z_norm_map += float((dz_map / (1.0 + z)).sum().cpu())
    n_out15_map += int(((dz_map / (1.0 + z)) > 0.15).sum().cpu())
```

En el retorno:

```python
"examples": float(examples),
"eta15": n_out15 / ex_denom,
**({
    "redshift_mae_map": total_abs_z_map / ex_denom,
    "redshift_mae_norm_map": total_abs_z_norm_map / ex_denom,
    "eta15_map": n_out15_map / ex_denom,
} if has_map else {}),
```

### Paso 3.3 — Guardar `checkpoint_best.pt`

Antes del loop de épocas:

```python
best_eta15_map = math.inf
```

Después de obtener las métricas de validación:

```python
if is_better_checkpoint(metrics, best_score=best_eta15_map):
    best_eta15_map = metrics["eta15_map"]
    save_checkpoint(
        out_dir / "checkpoint_best.pt",
        model,
        optimizer,
        scheduler,
        config,
        step,
        epoch,
        save_optimizer=args.save_optimizer,
    )
    print(f"new_best eta15_map={best_eta15_map:.6f} step={step}")
```

### Paso 3.4 — Escribir métricas de evaluación sin depender de stdout

Crear `tests/test_evaluate.py`:

```python
import json
from pathlib import Path

import pytest

from desi_fm.evaluate import write_metrics_json


def test_write_metrics_json_is_atomic_and_machine_readable(tmp_path: Path):
    output = tmp_path / "nested" / "metrics.json"
    metrics = {
        "reconstruction_rmse_masked": 0.854,
        "eta15_map": 0.273,
    }

    write_metrics_json(metrics, output)

    assert json.loads(output.read_text()) == metrics
    assert not output.with_name(f".{output.name}.tmp").exists()


def test_write_metrics_json_rejects_nonfinite_values(tmp_path: Path):
    output = tmp_path / "metrics.json"

    with pytest.raises(ValueError):
        write_metrics_json({"eta15_map": float("nan")}, output)

    assert not output.exists()
```

Ejecutar primero:

```bash
python3 -m pytest tests/test_evaluate.py -q
```

Esperado: `ImportError` porque `write_metrics_json` aún no existe.

En `src/desi_fm/evaluate.py`, agregar:

```python
def write_metrics_json(metrics: dict[str, float], output: Path) -> None:
    payload = json.dumps(metrics, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(payload)
    temporary.replace(output)
```

Agregar a `parse_args()`:

```python
parser.add_argument(
    "--metrics-json",
    default="",
    help="write the final metrics as standalone JSON; logs remain independent",
)
```

En `main()`, antes del `print()`:

```python
if args.metrics_json:
    write_metrics_json(metrics, Path(args.metrics_json))
```

Verificar:

```bash
python3 -m pytest tests/test_evaluate.py -q
```

Esperado: verde. La Fase 7 consumirá únicamente `--metrics-json`; mensajes,
reintentos o warnings en stdout/stderr no pueden corromper el artefacto.

### Paso 3.5 — Verificar CLI, suite y compatibilidad v1

Antes de la verificación, exponer la salida oficial MAP en la API pública. Crear
`tests/test_predict.py`:

```python
import numpy as np

from desi_fm.model import DESIFoundationModel, DESIFoundationModelConfig
from desi_fm.predict import predict_spectrum


def test_predict_spectrum_exposes_map_and_confidence_for_classification():
    config = DESIFoundationModelConfig(
        n_pixels=64,
        n_tokens=8,
        lambda_min=3600.0,
        lambda_max=9800.0,
        d_model=32,
        n_layers=1,
        n_heads=4,
        dropout=0.0,
        n_z_bins=16,
        z_max=6.0,
    )
    model = DESIFoundationModel(config)
    wavelength = np.geomspace(3600.0, 9800.0, 64).astype(np.float32)
    flux = np.sin(wavelength / 500.0).astype(np.float32)
    result = predict_spectrum(flux=flux, wavelength=wavelength, model=model)

    assert 0.0 <= result["z_pred_map"] <= 6.0
    assert 0.0 <= result["z_confidence"] <= 1.0
    assert "z_pred" in result  # la esperanza se conserva por retrocompatibilidad
```

Ejecutarlo primero:

```bash
python3 -m pytest \
  tests/test_predict.py::test_predict_spectrum_exposes_map_and_confidence_for_classification \
  -q
```

Esperado: `KeyError: 'z_pred_map'`.

En `predict_spectrum()` de `src/desi_fm/predict.py`, construir el retorno en una
variable y agregar condicionalmente:

```python
result = {
    "z_pred": z_pred,
    "reconstruction_input_grid": recon_input_grid,
    "reconstruction_model_grid": recon_native,
    "normalized_reconstruction": recon_normalized,
    "normalized_input": processed["flux"].astype(np.float32),
    "model_wavelength": model_wavelength,
    "spectrum_mask": spectrum_mask.squeeze(0).cpu().numpy(),
    "mask_ratio_used": float(mask_ratio),
    "center": center,
    "scale": scale,
}
if "z_pred_map" in out:
    result["z_pred_map"] = float(out["z_pred_map"].item())
    result["z_confidence"] = float(out["z_confidence"].item())
return result
```

En `predict_spectra_batch()`, si `model.config.n_z_bins > 0`, reservar y llenar
`z_pred_map` y `z_confidence`, y agregarlos al dict final:

```python
has_map = model.config.n_z_bins > 0
z_pred_map = np.zeros(n, dtype=np.float32) if has_map else None
z_confidence = np.zeros(n, dtype=np.float32) if has_map else None
```

Dentro del loop:

```python
if has_map:
    z_pred_map[i] = r["z_pred_map"]
    z_confidence[i] = r["z_confidence"]
```

Antes del retorno, construir `result`; luego:

```python
if has_map:
    result["z_pred_map"] = z_pred_map
    result["z_confidence"] = z_confidence
return result
```

Verificar:

```bash
python3 -m desi_fm.train --help | rg \
  'normalize-redshift-ce|z-weighting|z-histogram|init-checkpoint'
python3 -m desi_fm.evaluate --help | rg 'metrics-json'
python3 -m pytest tests/ -q
python3 - <<'PY'
from pathlib import Path
import torch
from desi_fm.evaluate import load_model

model = load_model(Path("runs/desi_50k_big/checkpoint_last.pt"), torch.device("cpu"))
print(type(model).__name__, model.config.n_z_bins)
PY
```

Esperado: flags visibles, suite verde y checkpoint v1 cargado con `n_z_bins=0`.

### Paso 3.6 — Commit enfocado

```bash
git add \
  src/desi_fm/train.py \
  src/desi_fm/predict.py \
  src/desi_fm/evaluate.py \
  tests/test_predict.py \
  tests/test_evaluate.py
git commit -m "feat: add robust MAP checkpoint selection and metrics"
```

---

## Fase 4 · Histograma real de los 80k labels

### Paso 4.1 — Crear el script

Crear `scripts/estimate_z_histogram.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset

from desi_fm.data import extract_redshift, has_valid_redshift
from desi_fm.train import compute_z_histogram


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="MultimodalUniverse/desi")
    parser.add_argument("--data-dir", default="edr_sv3")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-examples", type=int, default=80000)
    parser.add_argument("--n-z-bins", type=int, default=100)
    parser.add_argument("--z-max", type=float, default=6.0)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stream = load_dataset(
        args.dataset,
        data_dir=args.data_dir,
        split=args.split,
        streaming=True,
    )
    train_window = stream.filter(has_valid_redshift).take(args.max_examples)
    redshifts = [extract_redshift(example) for example in train_window]

    counts, edges = compute_z_histogram(
        np.asarray(redshifts),
        n_bins=args.n_z_bins,
        z_max=args.z_max,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        counts=counts,
        edges=edges,
        n_examples=np.int64(len(redshifts)),
        n_z_bins=np.int64(args.n_z_bins),
        z_max=np.float64(args.z_max),
    )
    print(json.dumps({
        "output": str(output),
        "n_examples": len(redshifts),
        "n_bins": args.n_z_bins,
        "nonempty_bins": int((counts > 0).sum()),
        "counted": int(counts.sum()),
    }, indent=2))


if __name__ == "__main__":
    main()
```

### Paso 4.2 — Verificar sintaxis

```bash
python3 -m py_compile scripts/estimate_z_histogram.py
```

### Paso 4.3 — Generar el histograma sin incluir el held-out

```bash
python3 scripts/estimate_z_histogram.py \
  --dataset MultimodalUniverse/desi \
  --data-dir edr_sv3 \
  --max-examples 80000 \
  --n-z-bins 100 \
  --z-max 6.0 \
  --output runs/calibration/z_hist_80k_100bins.npz
```

Esperado: `n_examples=80000` y `counted=80000`. Los labels negativos/no finitos se
filtran antes de `take`, usando exactamente la misma función que `HFDESISpectra`.
Los valores por encima de `z_max` entran en el último bin, igual que
`torch.bucketize` en el modelo.

### Paso 4.4 — Inspeccionar pesos antes de entrenar

```bash
python3 - <<'PY'
import numpy as np
import torch
from desi_fm.train import build_z_bin_weights

p = np.load("runs/calibration/z_hist_80k_100bins.npz")
w = build_z_bin_weights(
    torch.from_numpy(p["counts"]),
    mode="sqrt_inverse",
    min_weight=0.5,
    max_weight=3.0,
)
print({
    "bins": len(w),
    "nonzero": int((w > 0).sum()),
    "min_observed": float(w[w > 0].min()),
    "median_observed": float(w[w > 0].median()),
    "max_observed": float(w.max()),
    "empty_weight_nonzero": int(((p["counts"] == 0) & (w.numpy() != 0)).sum()),
})
PY
```

Esperado: pesos observados en `[0.5, 3.0]` y peso cero para bins vacíos.

### Paso 4.5 — Recalcular la v1 sobre el held-out canónico

```bash
python3 -m desi_fm.evaluate \
  --checkpoint runs/desi_50k_big/checkpoint_last.pt \
  --data-dir edr_sv3 --max-examples 2000 --skip-examples 80000 \
  --predictions-csv runs/calibration/predictions_v1_heldout_canonical.csv \
  --metrics-json runs/calibration/metrics_v1_heldout_canonical.json
python3 -m json.tool \
  runs/calibration/metrics_v1_heldout_canonical.json >/dev/null
```

Esperado: 2.000 predicciones. Comparar el η reportado con 27.3 %; si cambia, registrar
la diferencia en el handoff. Este JSON/CSV es el baseline autoritativo de la Fase 7.

### Paso 4.6 — Commit del script

```bash
git add scripts/estimate_z_histogram.py
git commit -m "feat: estimate redshift histogram from the training split"
```

El `.npz` es un artefacto reproducible; no commitearlo salvo decisión explícita.

---

## Fase 5 · Preflight de 1.000 steps

### Paso 5.1 — Asegurar un output nuevo

```bash
test ! -e runs/desi_80k_classhead_v21_preflight
```

Si existe, usar un nombre nuevo con sufijo; no borrar ni mezclar métricas.

### Paso 5.2 — Ejecutar el preflight

```bash
python3 -m desi_fm.train \
  --dataset MultimodalUniverse/desi --data-dir edr_sv3 \
  --output-dir runs/desi_80k_classhead_v21_preflight \
  --batch-size 8 --num-workers 0 \
  --max-train-examples 80000 --val-examples 2000 \
  --epochs 1 --max-steps 1000 --shuffle-buffer 2048 \
  --n-z-bins 100 --z-max 6.0 \
  --z-weighting sqrt_inverse \
  --z-histogram runs/calibration/z_hist_80k_100bins.npz \
  --z-weight-min 0.5 --z-weight-cap 3.0 \
  --z-label-smoothing 0.0 --normalize-redshift-ce \
  --init-checkpoint runs/desi_50k_big/checkpoint_last.pt \
  --redshift-loss-weight 1.0 --reconstruction-loss-weight 1.0 \
  --mask-ratio 0.5 --wavelength-grid log \
  --d-model 512 --n-layers 8 --n-heads 8 \
  --lr 1e-4 --log-every-steps 50 --save-every-steps 0
```

### Paso 5.3 — Gate automático del preflight

```bash
python3 - <<'PY'
import json
from pathlib import Path

import numpy as np

p = Path("runs/desi_80k_classhead_v21_preflight/metrics.jsonl")
rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
train = [row for row in rows if row["phase"] == "train"]
val = [row for row in rows if row["phase"] == "validation"][-1]
args = json.loads(
    Path(
        "runs/desi_80k_classhead_v21_preflight/training_args.json"
    ).read_text()
)
assert all(row["loss"] == row["loss"] for row in rows), "NaN detected"
assert train[0]["loss"] < 5.0, train[0]
assert len(train) >= 20, len(train)
first_loss = float(np.median([row["loss"] for row in train[:5]]))
last_loss = float(np.median([row["loss"] for row in train[-5:]]))
assert last_loss <= first_loss * 1.5, (first_loss, last_loss)
assert val["examples"] == 2000.0, val
assert "eta15_map" in val, val
assert val["eta15_map"] < 0.60, val
assert args["shuffle_buffer"] == 2048, args["shuffle_buffer"]
assert args["num_workers"] == 0, args["num_workers"]
assert args["init_checkpoint"].endswith(
    "runs/desi_50k_big/checkpoint_last.pt"
)
assert Path("runs/desi_80k_classhead_v21_preflight/checkpoint_best.pt").exists()
print(
    "PREFLIGHT_OK",
    {"first_loss": first_loss, "last_loss": last_loss, "validation": val},
)
PY
```

Si falla cualquier assert, **no lanzar la corrida completa**. Diagnosticar primero la
causa; no relajar el gate sin registrar evidencia.

---

## Fase 6 · Corrida completa v2.1

### Paso 6.1 — Preparar un directorio nuevo y lanzar desacoplado

```bash
test ! -e runs/desi_80k_classhead_v21
mkdir -p runs/desi_80k_classhead_v21
nohup caffeinate -i python3 -m desi_fm.train \
  --dataset MultimodalUniverse/desi --data-dir edr_sv3 \
  --output-dir runs/desi_80k_classhead_v21 \
  --batch-size 8 --num-workers 0 \
  --max-train-examples 80000 --val-examples 2000 \
  --epochs 3 --shuffle-buffer 2048 \
  --n-z-bins 100 --z-max 6.0 \
  --z-weighting sqrt_inverse \
  --z-histogram runs/calibration/z_hist_80k_100bins.npz \
  --z-weight-min 0.5 --z-weight-cap 3.0 \
  --z-label-smoothing 0.0 --normalize-redshift-ce \
  --init-checkpoint runs/desi_50k_big/checkpoint_last.pt \
  --redshift-loss-weight 1.0 --reconstruction-loss-weight 1.0 \
  --mask-ratio 0.5 --wavelength-grid log \
  --d-model 512 --n-layers 8 --n-heads 8 \
  --lr 1e-4 --log-every-steps 50 --save-every-steps 5000 \
  > runs/desi_80k_classhead_v21/train.log 2>&1 &
TRAIN_PID=$!
echo "$TRAIN_PID" > runs/desi_80k_classhead_v21/train.pid
disown
echo "TRAIN_PID=$TRAIN_PID"
```

ETA orientativa en M4 Pro: 30.000 steps, aproximadamente 2.5–3 h. No prometer una
hora exacta hasta medir los primeros 500 steps.

### Paso 6.2 — Confirmar que el proceso real arrancó

```bash
TRAIN_PID=$(cat runs/desi_80k_classhead_v21/train.pid)
ps -p "$TRAIN_PID" -o pid,etime,%cpu,%mem,command
while [ ! -f runs/desi_80k_classhead_v21/metrics.jsonl ]; do sleep 10; done
tail -1 runs/desi_80k_classhead_v21/metrics.jsonl
```

El comando debe contener `python3 -m desi_fm.train`; un `tail -f` no cuenta.

### Paso 6.3 — Medir velocidad sin dejar un monitor eterno

Después de al menos 500 steps:

```bash
python3 - <<'PY'
import json
from pathlib import Path

rows = [
    json.loads(line)
    for line in Path("runs/desi_80k_classhead_v21/metrics.jsonl").read_text().splitlines()
    if '"phase": "train"' in line
]
print("latest_step", rows[-1]["step"])
print("latest_loss", rows[-1]["loss"])
print("latest_raw_ce", rows[-1]["redshift_loss_raw"])
PY
```

Revisar periódicamente con comandos que terminan (`tail -n 3`), no con `tail -f`
persistente.

### Paso 6.4 — Verificar terminación

```bash
pgrep -af 'python.*-m desi_fm\.train' || true
rg 'done step=30000' runs/desi_80k_classhead_v21/train.log
for expected in \
  'dataset_epoch=0 shuffle_seed=42' \
  'dataset_epoch=1 shuffle_seed=43' \
  'dataset_epoch=2 shuffle_seed=44'; do
  rg -F "$expected" runs/desi_80k_classhead_v21/train.log
done
if rg 'Traceback|RuntimeError|OutOfMemoryError|Killed: 9' \
  runs/desi_80k_classhead_v21/train.log; then
  echo "training log contains an execution failure" >&2
  exit 1
fi
python3 - <<'PY'
import json
import math
from pathlib import Path

path = Path("runs/desi_80k_classhead_v21/metrics.jsonl")
rows = [
    json.loads(line)
    for line in path.read_text().splitlines()
    if line.strip()
]
assert rows, "metrics.jsonl is empty"
for line_number, row in enumerate(rows, start=1):
    for key, value in row.items():
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and not math.isfinite(float(value))
        ):
            raise AssertionError(
                f"non-finite metric at line {line_number}: {key}={value}"
            )
assert rows[-1]["step"] == 30000, rows[-1]
print(f"FINITE_METRICS_OK rows={len(rows)}")
PY
ls -lh \
  runs/desi_80k_classhead_v21/checkpoint_best.pt \
  runs/desi_80k_classhead_v21/checkpoint_last.pt
```

Esperado: `done step=30000`, tres líneas de epoch/seed (`42`, `43`, `44`),
`FINITE_METRICS_OK`, ningún traceback/runtime error y ambos checkpoints presentes.
No buscar el substring genérico `nan`: la validación estructurada detecta valores no
finitos sin falsos positivos de texto.

---

## Fase 7 · Evaluación honesta y selección

### Paso 7.1 — Evaluar best y last sobre el mismo held-out

```bash
python3 -m desi_fm.evaluate \
  --checkpoint runs/desi_80k_classhead_v21/checkpoint_best.pt \
  --data-dir edr_sv3 --max-examples 2000 --skip-examples 80000 \
  --predictions-csv runs/desi_80k_classhead_v21/predictions_best.csv \
  --reconstructions-npz runs/desi_80k_classhead_v21/reconstructions_best.npz \
  --num-reconstructions 50 \
  --metrics-json runs/desi_80k_classhead_v21/metrics_best.json

python3 -m desi_fm.evaluate \
  --checkpoint runs/desi_80k_classhead_v21/checkpoint_last.pt \
  --data-dir edr_sv3 --max-examples 2000 --skip-examples 80000 \
  --predictions-csv runs/desi_80k_classhead_v21/predictions_last.csv \
  --metrics-json runs/desi_80k_classhead_v21/metrics_last.json

python3 -m json.tool \
  runs/desi_80k_classhead_v21/metrics_best.json >/dev/null
python3 -m json.tool \
  runs/desi_80k_classhead_v21/metrics_last.json >/dev/null
```

### Paso 7.2 — Calcular gates y métricas por bin

```bash
python3 - <<'PY'
import csv
import json
from pathlib import Path

import numpy as np


def metrics(path, eval_path, prediction_column):
    rows = list(csv.DictReader(open(path)))
    eval_metrics = json.loads(Path(eval_path).read_text())
    zt = np.asarray([float(row["z_true"]) for row in rows])
    zp = np.asarray([float(row[prediction_column]) for row in rows])
    dzn = (zp - zt) / (1.0 + zt)
    high = (zt >= 1.5) & (zt < 2.5)
    if high.sum() == 0:
        raise ValueError(f"{path} has no examples in z=[1.5, 2.5)")
    return {
        "path": path,
        "n": len(zt),
        "eta15": float((np.abs(dzn) > 0.15).mean()),
        "sigma_nmad": float(
            1.4826 * np.median(np.abs(dzn - np.median(dzn)))
        ),
        "mae_norm": float(np.abs(dzn).mean()),
        "eta15_high": float((np.abs(dzn[high]) > 0.15).mean()),
        "max_z_pred_map": float(zp.max()),
        "reconstruction_rmse_masked": float(
            eval_metrics["reconstruction_rmse_masked"]
        ),
    }


baseline = metrics(
    "runs/calibration/predictions_v1_heldout_canonical.csv",
    "runs/calibration/metrics_v1_heldout_canonical.json",
    "z_pred",
)
BASE = {
    "eta15": baseline["eta15"],
    "sigma_nmad": baseline["sigma_nmad"],
    "mae_norm": baseline["mae_norm"],
    "eta15_high": 0.50,
    "reconstruction_rmse_masked": 0.90,
}
results = [
    metrics(
        "runs/desi_80k_classhead_v21/predictions_best.csv",
        "runs/desi_80k_classhead_v21/metrics_best.json",
        "z_pred_map",
    ),
    metrics(
        "runs/desi_80k_classhead_v21/predictions_last.csv",
        "runs/desi_80k_classhead_v21/metrics_last.json",
        "z_pred_map",
    ),
]
for item in results:
    item["passes_release_gate"] = (
        item["n"] == 2000
        and item["eta15"] < BASE["eta15"]
        and item["sigma_nmad"] < BASE["sigma_nmad"]
        and item["mae_norm"] <= BASE["mae_norm"]
        and item["eta15_high"] < BASE["eta15_high"]
        and item["reconstruction_rmse_masked"]
        <= BASE["reconstruction_rmse_masked"]
    )

eligible = [item for item in results if item["passes_release_gate"]]
pool = eligible or results
winner = min(
    pool,
    key=lambda item: (
        item["eta15"],
        item["sigma_nmad"],
        item["mae_norm"],
    ),
)
decision = "promote_v2_1" if eligible else "keep_v1"
report = {
    "baseline_v1": baseline,
    "release_thresholds": BASE,
    "candidates": results,
    "winner": winner,
    "decision": decision,
}
Path("runs/desi_80k_classhead_v21/comparison.json").write_text(
    json.dumps(report, indent=2)
)
print(json.dumps(report, indent=2))
PY
```

Esperado: `comparison.json` contiene las métricas completas de ambos checkpoints,
el baseline usado y `decision`. No decidir a partir del texto de consola.

### Paso 7.3 — Decisión obligatoria

- Si todos los gates pasan: declarar v2.1 candidata para Hugging Face.
- Si uno falla: mantener v1 como modelo recomendado. No seguir agregando épocas
  automáticamente; documentar qué gate falló y diseñar otro experimento separado.

---

## Fase 8 · Cierre documental y handoff a Plan 04

### Paso 8.1 — Solo si v2.1 pasa

1. Usar el checkpoint ganador (`best` o `last`) como fuente para Plan 04.
2. Re-ejecutar `notebooks/evaluation.ipynb` con
   `DESI_FM_RUN=desi_80k_classhead_v21`.
3. Actualizar README, DELIVERABLE y model card con números held-out y aclarar que
   `z_pred_map` es la predicción oficial. Describir v2.1 como fine-tuning de v1 con
   una cabeza de clasificación nueva, nunca como entrenamiento desde cero.
4. Dejar en Plan 04 el checkpoint exacto, las métricas y el comando de publicación.
   No subir todavía a Hugging Face desde 02R.

### Paso 8.2 — Si v2.1 no pasa

1. Designar la v1 como fuente para Plan 04 con sus métricas held-out canónicas.
2. Documentar v2.0 y v2.1 como experimentos y sus mejoras en z alto.
3. No afirmar en README/CV que la cabeza nueva mejoró el global.

### Paso 8.3 — Actualizar los registros

- Agregar v2.1 a “Registro de intentos” en `plan/02-reentrenamiento-v2.md`.
- Actualizar `plan/README.md`: 02 solo pasa a ✅ si la definición de hecho acordada se
  cumple o si se cierra explícitamente con v1 como modelo final.
- Añadir una sección datada a `plan/HANDOFF.md` sin borrar las anteriores.

### Paso 8.4 — Commit documental enfocado

```bash
git add \
  plan/02R-reentrenamiento-v2-calibrado.md \
  plan/02-reentrenamiento-v2.md \
  plan/README.md \
  plan/HANDOFF.md
git commit -m "plan-02R: document calibrated v2 training outcome"
```

Revisar `git diff --cached` antes del commit. No incluir checkpoints `.pt`.

---

## Definición de hecho

- [ ] Los tests nuevos fallaron por la razón esperada antes de implementar.
- [ ] Fase 0 confirma firmas y semántica real `take().shuffle()` del entorno.
- [ ] Suite completa verde y v1 sigue cargando.
- [ ] El pipeline ejecuta `filter(valid) → skip → take → shuffle(train only)`.
- [ ] Train y held-out no se solapan; inválidos no desplazan la frontera.
- [ ] Las tres épocas conservan membresía y usan órdenes reproducibles distintos.
- [ ] Histograma construido con exactamente los primeros 80k labels y 100 bins.
- [ ] Baseline v1 recalculado sobre los 2.000 labels válidos siguientes.
- [ ] Preflight de 1.000 steps cumple todos sus asserts.
- [ ] Full run termina en 30.000 steps sin NaNs/tracebacks.
- [ ] Todos los números de `metrics.jsonl` pasan `math.isfinite`.
- [ ] Existen `checkpoint_best.pt` y `checkpoint_last.pt`.
- [ ] Ambos se evalúan sobre 2.000 ejemplos con `--skip-examples 80000`.
- [ ] `metrics_best.json` y `metrics_last.json` son JSON standalone válidos.
- [ ] `comparison.json` contiene la decisión reproducible.
- [ ] Solo se promueve v2.1 si supera todos los gates; de lo contrario se conserva v1.
- [ ] Toda la documentación llama a v2.1 “fine-tuning de v1”, no “from scratch”.
- [ ] Notebook/model card no publican outputs obsoletos.
- [ ] Handoff y registro de intentos quedan actualizados.

## Si algo falla

- **Preflight con `loss > 5`:** comprobar que `--normalize-redshift-ce` llegó al config
  y que `redshift_loss_weight=1`.
- **MAP no aparece en validation:** el evaluador interno no está acumulando
  `z_pred_map`; no lanzar full run.
- **Histograma no suma 80k:** contar labels inválidos y decidir explícitamente si se
  excluyen; no fingir que fueron 80k targets válidos.
- **Train/held-out comparten un label en el test:** comprobar que `shuffle()` ocurre
  después de `take()`, nunca antes; no lanzar ni siquiera el preflight.
- **Las épocas producen el mismo orden:** verificar `set_epoch(epoch)` y que el seed
  efectivo sea `args.seed + epoch`.
- **Un JSON de evaluación no parsea:** comprobar que se usó `--metrics-json`; nunca
  reconstruir métricas desde stdout ni relajar el gate.
- **Warm start carga pocas claves:** verificar compatibilidad de arquitectura
  (`d_model=512`, 8 capas, 8 heads) y listar todas las claves omitidas.
- **La red se corta:** Hugging Face streaming suele reintentar. Distinguir mensajes
  `Retrying` de un traceback final.
- **v2.1 mejora η pero empeora MAE/NMAD:** no promoverla; el gate es conjunto.
- **El full run ya existe:** usar un output nuevo. Nunca anexar métricas de dos runs.

## Revisión adversarial R1–R6

- **R1 — Integridad numérica:** 100 bins, 80k labels válidos de train, los 2k válidos
  siguientes de held-out, buffer 2.048 aplicado dentro de train, 1k preflight y 30k
  steps del full coinciden con los comandos.
- **R2 — Criterios:** todos los gates son comparaciones numéricas contra la v1
  held-out; “mejor” no depende de inspección subjetiva.
- **R3 — Scope:** el plan corrige split, fine-tuning y selección. HF Hub sigue siendo
  Plan 04 y no se publica nada si falla el gate.
- **R4 — Integración:** las rutas y firmas se verificaron contra `model.py`,
  `data.py`, `train.py`, `predict.py`, `evaluate.py` y
  `HFDESISpectra.skip_examples`. La Fase 0 demuestra en la versión instalada que
  `take().shuffle()` conserva la ventana y aplica buffer shuffle dentro de ella.
  `--metrics-json` desacopla artefactos de logs. El default nuevo de normalización es
  retrocompatible.
- **R5 — Idempotencia:** cada corrida usa un directorio nuevo; ningún comando borra o
  sobreescribe v1/v2.0. Seed + epoch hace reproducibles los tres órdenes. Los commits
  usan staging dirigido.
- **R6 — Verificación:** tests unitarios y preflight son fuertes para el código; la
  prueba de membresía/disjointness es fuerte para el split; la evaluación de 2.000
  held-out y JSON standalone son fuertes para promoción; `math.isfinite` es fuerte
  contra NaN/Infinity; la ETA es solo orientativa.

**Siguiente paso:** ejecutar este plan completo y, únicamente después de la decisión
del gate, retomar `plan/04-checkpoint-hf-hub.md`.
