# 02 · Reentrenamiento v2 + antes/después

> **Bloque:** Fase 0 (corrección del TA) · **Tiempo:** ~1 h activa + 3–4 h de cómputo desatendido · **Depende de:** 01 · **Entregable:** checkpoint v2 con métricas que cumplen los criterios, notebook re-ejecutado, tabla v1→v2

## Objetivo

Entrenar el modelo con la cabeza de clasificación + rebalanceo + más datos, y documentar la mejora contra la v1 con números. Este es el cierre de la corrección del TA: *"recibí feedback → diagnostiqué → rediseñé → mejoré"*.

## Criterios de éxito (contra la v1)

| métrica | v1 (hoy) | objetivo v2 |
|---|---|---|
| η₀.₁₅ (outliers catastróficos) | 25.1 % | **< 10 %** |
| bias por bin \|⟨Δz⟩\| en bins poblados | hasta 0.8 | **< 0.05** |
| σ_NMAD | 0.101 | **< 0.05** |
| techo de predicción | z_pred ≤ 1.95 | predicciones reales a z ≳ 3 |

## Pasos

### 1. Lanzar el entrenamiento (dejarlo corriendo de noche)

```bash
caffeinate -i python3 -m desi_fm.train \
  --dataset MultimodalUniverse/desi --data-dir edr_sv3 \
  --output-dir runs/desi_150k_classhead \
  --batch-size 8 --max-train-examples 150000 --val-examples 2000 --epochs 2 \
  --n-z-bins 200 --z-max 6.0 --z-rebalance \
  --redshift-loss-weight 10 --mask-ratio 0.5 --wavelength-grid log \
  --d-model 512 --n-layers 8 --n-heads 8 \
  --log-every-steps 50 --save-every-steps 5000 \
  2>&1 | tee runs/desi_150k_classhead/train.log
```

Notas: `caffeinate -i` evita que la Mac se duerma. Si se corta, `Ctrl+C` guarda `checkpoint_interrupted.pt`.

**ETA:** son 37 500 steps (150k × 2 épocas / batch 8). A los ~5 minutos de lanzado, medí la velocidad real y extrapolá:

```bash
python3 - <<'EOF'
import json, time, os
p = "runs/desi_150k_classhead/metrics.jsonl"
rows = [json.loads(l) for l in open(p) if '"train"' in l]
step = rows[-1]["step"]
age_min = (time.time() - os.path.getmtime("runs/desi_150k_classhead/config.json")) / 60
rate = step / age_min
print(f"step {step}/37500 — {rate:.0f} steps/min — restante ≈ {(37500 - step) / rate / 60:.1f} h")
EOF
```

(referencia: la v1 hizo 12 500 steps de batch 4 en ~25 min en MPS; con batch 8 esperá ~2.5–4 h en total). Monitoreo del loss: `tail -2 runs/desi_150k_classhead/metrics.jsonl | python3 -m json.tool` — `redshift_loss` tiene que bajar sostenido en la primera media hora; si a los 2000 steps está plano, revisar antes de dejarlo horas.

### 2. Evaluar el checkpoint final

```bash
python3 -m desi_fm.evaluate \
  --checkpoint runs/desi_150k_classhead/checkpoint_last.pt \
  --data-dir edr_sv3 --max-examples 2000 \
  --predictions-csv runs/desi_150k_classhead/predictions.csv \
  --reconstructions-npz runs/desi_150k_classhead/reconstructions.npz \
  --num-reconstructions 50
```

Mirar **las dos** variantes: `redshift_mae_norm` (esperanza) y `*_map` (argmax). Contra outliers suele ganar el argmax; decidir cuál es "la" predicción de la v2 y anotarlo.

### 3. Comparar contra la v1 con el notebook

```bash
# preservar la versión v1 ejecutada (el "antes"):
cp notebooks/evaluation.ipynb notebooks/evaluation_v1_baseline.ipynb
```

Editar la celda de setup del notebook — búsqueda y reemplazo exactos. Donde dice:

```python
RUN_DIR = ROOT / "runs" / "desi_50k_big"
```

poner:

```python
import os
RUN_DIR = ROOT / "runs" / os.environ.get("DESI_FM_RUN", "desi_150k_classhead")
```

(el `import os` puede subir junto a los demás imports de la celda). La celda ya imprime de qué run cargó — verificar en la salida que diga `desi_150k_classhead`. Re-ejecutar contra la v2:

```bash
python3 -m nbconvert --to notebook --execute --inplace notebooks/evaluation.ipynb
```

### 4. Documentar la mejora

- Agregar la fila v2 a la tabla de progresión de `DELIVERABLE.md` §5 y actualizar la tabla de métricas del `README.md`.
- Agregar al README una mini-tabla "v1 → v2" (η₀.₁₅, bias, σ_NMAD) con una línea sobre el rediseño de la cabeza.
- Exportar los dos scatter (v1 y v2) a `docs/img/scatter_v1.png` y `docs/img/scatter_v2.png` (desde los notebooks: click derecho → guardar imagen) — se usan en el plan 13.
- Si la v2 cumple los criterios: **pasa a ser el checkpoint entregable** (el que sube al Hub en el plan 04 y usan la demo y la API).

## Definición de hecho

- [ ] Entrenamiento terminado sin NaNs (`train.log` limpio, loss descendente).
- [ ] Métricas v2 sobre 2000 espectros cumpliendo al menos **η₀.₁₅ < 10 %** y sin techo de predicción.
- [ ] `notebooks/evaluation.ipynb` re-ejecutado contra la v2; `evaluation_v1_baseline.ipynb` preservado.
- [ ] README + DELIVERABLE actualizados con la tabla v1→v2.
- [ ] Commit: `plan-02: v2 checkpoint (classification head) — eta15 25%→X%`.

## Si algo falla

- **η₀.₁₅ queda entre 10–15 %:** probar en este orden (una variable por vez): (a) usar `z_pred_map` como predicción oficial; (b) 3ª época (`--epochs 3`); (c) `--n-z-bins 300`; (d) subir datos a 300k. Documentar cada intento en una tabla en este archivo.
- **El rebalanceo desestabiliza el entrenamiento** (loss de z oscila fuerte): bajar el cap de pesos de 10 a 5 en `train.py`, o entrenar 1ª época sin `--z-rebalance` y la 2ª con.
- **Se queda sin disco** (checkpoints cada 5000 pasos ≈ varios GB): borrar los `checkpoint_step_*.pt` intermedios al terminar; solo importa `checkpoint_last.pt`.
- **La v2 empeora la reconstrucción** (`reconstruction_rmse_masked` sube > 0.9): el peso 10 de la loss de z está dominando; probar `--redshift-loss-weight 5`.
