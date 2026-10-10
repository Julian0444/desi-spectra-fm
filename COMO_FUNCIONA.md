# Cómo funciona el código (explicado paso a paso)

Este documento explica qué hace cada parte del proyecto, en orden, sin asumir que ya conocés transformers o astrofísica. Describe la versión **v2.1**, que es la que está publicada en el Hub (`jirustaroure/desi-spectra-fm`, archivo `checkpoint_last.pt`) y en el repo (`runs/desi_80k_classhead_v21/`). Donde importa, se cuenta también cómo era la **v1** y por qué se cambió.

Cada nombre de función o clase que aparece acá existe con ese nombre en `src/desi_fm/`.

---

## La idea de la cadena completa

Todo el proyecto es una sola "máquina" con esta forma:

```
Espectro de entrada (flujo + longitud de onda, crudos)
         ↓
    Limpiar, interpolar a una grilla log-λ y normalizar
         ↓
    Cortar en 273 pedacitos ("tokens") + 1 token de redshift
         ↓
    En train/evaluación histórica: tapar el 50 %; serving: máscara 0
    (el token de redshift está tapado SIEMPRE)
         ↓
    Transformer encoder (8 capas, d_model 512, ~26 M parámetros)
         ↓
    Salidas: (a) qué había en los pedacitos tapados (reconstrucción)
             (b) una distribución de probabilidad sobre 100 bins de z
                 → z_pred_map (oficial), z_confidence, z_pred (compat)
```

El resto del documento explica cada caja.

---

## El paisaje de archivos

```
src/desi_fm/
  data.py            → leer, limpiar, interpolar y normalizar espectros; split train/held-out
  model.py           → el Transformer, la tokenización y las dos cabezas (reconstrucción + redshift)
  train.py           → bucle de entrenamiento, scheduler, pesos de clase, checkpoints
  evaluate.py        → métricas sobre el held-out canónico (--skip-examples) + CSV de predicciones
  predict.py         → inferencia agnóstica al instrumento: predict_spectrum / predict_spectra_batch / embed_spectrum
  api.py             → API REST (FastAPI): /healthz, /predict, /predict_json
  inspect_schema.py  → utilitario para verificar que el dataset llega con el formato esperado

scripts/
  estimate_z_histogram.py  → histograma de z de los 80k de entrenamiento (pesos de clase de v2.1)
  make_demo_examples.py    → exporta los 4 espectros held-out de ejemplo (examples/*.npz)
  plot_scatter_v1_v2.py    → figura v1 vs v2.1 del README

tests/                → 26 tests (shapes, sin fuga de z, split disjunto, loss normalizada, salidas MAP, embeddings, API)
notebooks/            → evaluation.ipynb (v2.1) y evaluation_v1_baseline.ipynb (el "antes")
demo/                 → app Gradio del Space jirustaroure/desi-spectra-fm-demo
api/                  → Space que sirve la API (FastAPI montada en /api dentro de un Space Gradio ZeroGPU)
Dockerfile            → imagen CPU de la API
runs/desi_80k_classhead_v21/   → config.json, training_args.json, metrics.jsonl, comparison.json, CSVs
                                 (los .pt no van en git: se bajan del Hub)
runs/desi_50k_big/             → la v1 (baseline)
```

Hay un solo "cerebro", definido en `model.py`. Los demás archivos lo alimentan, lo entrenan, lo miden o lo sirven.

---

## Paso 1: leer un espectro (`data.py`)

Un espectro de DESI (dataset `MultimodalUniverse/desi`, subconjunto `edr_sv3`, leído en streaming desde Hugging Face) llega con cuatro arrays de 7781 números cada uno, entre 3600 y 9824 Å:

- `flux`: "cuánta luz hay en este color".
- `lambda`: "qué color es este" (longitud de onda en Ångströms).
- `ivar`: inversa de la varianza; "qué tan confiable es esta medición" (0 = inservible).
- `mask`: booleanos; "este píxel está roto".

Más un escalar `Z`: el redshift que la pipeline oficial de DESI ya calculó. Es la etiqueta.

La función `extract_mmu_desi_example` saca esas columnas (tolerando listas, arrays o objetos Arrow), y `has_valid_redshift` descarta ejemplos con `Z` no finito o negativo.

### El split, y por qué está construido así

La clase `HFDESISpectra` arma el stream en este orden fijo: `filter(has_valid_redshift) → skip → take → shuffle`.

- **Train** = los primeros 80.000 ejemplos con etiqueta válida (`skip=0, take=80000`), barajados con un buffer de 2048 y semilla `42 + época`.
- **Held-out canónico** = los 2.000 siguientes (`skip=80000, take=2000`, sin shuffle).

Como la membresía se decide *antes* del shuffle, el buffer nunca puede leer un ejemplo del held-out. Eso está testeado en `tests/test_data.py`.

El cargador actual decide la membresía antes de barajar (con `num_workers=0`
en la corrida). Eso no demuestra independencia del linaje heredado: la
reconstrucción del cargador v1 sostiene fuertemente la exposición de esos
2.000 objetos. Además seleccionaron checkpoints durante v2.1. Por ello son
validación histórica no independiente; no se conoce el signo ni la magnitud
del efecto de esa exposición sobre las métricas.

---

## Paso 2: limpiar, interpolar y normalizar (`preprocess_spectrum` en `data.py`)

El espectro crudo tiene problemas que hay que arreglar antes de que lo vea la red:

1. **Píxeles rotos o con `NaN`.** Se define `good = ~mask & isfinite(flux) & isfinite(ivar) & (ivar > 0)`. Lo que no es `good` queda marcado como inválido y no cuenta ni para normalizar ni para el loss.
2. **La grilla de longitudes de onda no es la misma para todo instrumento.** Se interpola linealmente (`np.interp`) a una grilla común de **7081 puntos espaciados logarítmicamente entre 3600 y 9800 Å** (`SpectrumPreprocessConfig.target_wavelength`, un `np.geomspace`). La máscara de validez se interpola también y se binariza (`> 0.5`); fuera de la cobertura del instrumento queda en 0.
3. **Distintos objetos brillan muy distinto.** Se normaliza de forma robusta: se resta la mediana de los píxeles válidos y se divide por `IQR / 1.349` (una estimación de la desviación estándar que ignora outliers). Después se aplica `arcsinh` (parecido a un logaritmo, pero funciona con valores negativos) y se recorta a ±8. Los píxeles inválidos se ponen en 0.

La función devuelve `flux` normalizado, `valid`, y los dos escalares `center` y `scale` para poder deshacer la normalización más tarde (lo usa `predict.py`).

**¿Por qué grilla logarítmica?** Porque el redshift es un **estiramiento multiplicativo**: `λ_obs = (1+z) · λ_emit`. En `log λ`, multiplicar se vuelve sumar, así que un espectro a redshift `z` es el mismo patrón **desplazado** una cantidad constante `log(1+z)`. Al transformer le resulta mucho más fácil detectar un corrimiento que un estiramiento.

---

## Paso 3: cortar en pedacitos (`SpectrumPatchTokenizer` en `model.py`)

**Problema:** 7081 puntos son demasiados para procesar uno por uno (la atención escala con el cuadrado del número de tokens).

**Solución:** agrupar los píxeles en **273 parches de 26 píxeles** (`patch_size = ceil(7081 / 273) = 26`; los 7081 píxeles se rellenan con ceros hasta 7098 = 273 × 26). Cada parche es un **token espectral**.

Imaginá el espectro como una tira larga con 7081 marcas: la cortás en 273 cuadraditos iguales. Si en esa zona había una línea de emisión, el cuadradito la conserva.

A cada parche se le concatenan sus **26 valores de flujo y sus 26 valores de validez** (52 números en total; así el modelo sabe también qué píxeles del parche son confiables). Ese vector pasa por `LayerNorm + Linear` y sale un vector de **`d_model = 512`** dimensiones: el **embedding del token**.

A los 273 tokens espectrales se les agrega **un token número 274: el token de redshift** (`z_mask_token`). Es un vector aprendido que representa la pregunta "¿cuál es el redshift de este espectro?" y nunca contiene la respuesta. **Está enmascarado siempre**, en todos los ejemplos, en entrenamiento y en inferencia.

---

## Paso 4: decirle al modelo dónde está cada token (embeddings posicionales)

Un transformer puro no sabe en qué orden van los tokens. Acá se le dan dos señales, que se suman al embedding:

1. **Embedding sinusoidal de `log λ` físico** (`build_wavelength_sinusoidal_embedding`). Para cada parche se toma la media de `log λ` de sus píxeles, se normaliza a [0, 1] dentro del rango 3600–9800 Å, y se codifica con senos y cosenos a 256 frecuencias (la mitad de `d_model`). Es fijo, no se aprende (es un `buffer`).
2. **Embedding posicional aprendido** (`learned_pos_embed`, 274 × 512), que el modelo ajusta durante el entrenamiento.

En cristiano: cada token lleva una etiqueta que dice "yo cubro la región de 4861 Å" (por ejemplo, donde cae Hβ en reposo).

**¿Qué permite la grilla?** El código remuestrea otras grillas a la interna.
El test de posiciones físicas comprueba la representación, no la precisión
científica en otros instrumentos. Esa generalización y el uso con cobertura
parcial siguen sin validarse.

El token de redshift tiene embedding sinusoidal exactamente cero: no corresponde a ninguna longitud de onda, es una pregunta sobre todo el espectro.

---

## Paso 5: tapar tokens al azar (masked-token prediction)

Antes de entrar al transformer, `random_spectrum_mask` elige al azar el **50 %** de los tokens espectrales (`mask_ratio 0.5` en v1 y v2.1) y los reemplaza por un vector `mask_token` aprendido.

Es como tachar la mitad de las palabras de una oración y pedirle al modelo que las adivine (así entrena BERT).

**El token de redshift está tapado siempre.** No es aleatorio ni opcional. Esa es una de las dos decisiones centrales del diseño; la otra es que la cabeza de redshift se entrena **junto** con la reconstrucción, no después con el encoder congelado. Así el encoder se ve forzado a representar el redshift desde el primer paso.

En inferencia (`predict.py`, `mask_ratio=0.0` por defecto) no se tapa ningún token espectral y la predicción es determinista; el token de redshift sigue tapado igual.

---

## Paso 6: el Transformer (`DESIFoundationModel`)

Los 274 tokens entran a un `nn.TransformerEncoder` estándar:

- **8 capas**; cada una con atención multi-cabezal (**8 cabezas**) + feed-forward de 2048 (`mlp_ratio 4`), activación GELU, pre-norm, dropout 0.1.
- `d_model = 512`.
- Un `LayerNorm` final.
- **~26 M de parámetros** entrenables.

La atención permite que cada token "mire" a todos los demás. Los visibles dan contexto; los tapados aprenden a inferir su contenido; el token de redshift aprende a resumir la posición de las líneas de todo el espectro.

Salen 274 vectores de 512 dimensiones. Los primeros 273 van a la cabeza de reconstrucción; el número 274 va a la cabeza de redshift.

---

## Paso 7: las dos cabezas de salida

### Cabeza de reconstrucción (`reconstruction_head`)

- Toma los 273 vectores de tokens espectrales.
- Cada uno pasa por un MLP pequeño (`LayerNorm → Linear 512→512 → GELU → Linear 512→26`) que devuelve los 26 píxeles que ese parche contenía (en espacio normalizado `arcsinh`).
- Se compara con los píxeles reales con **MSE**, contando **solo los píxeles que estaban tapados y además eran válidos** (`valid_masked`). Los tokens visibles no cuentan: adivinarlos sería trivial.

### Cabeza de redshift (`redshift_head`): clasificación sobre 100 bins

- Toma el vector 274 (el del token de redshift).
- Pasa por `LayerNorm → Linear 512→512 → GELU → Dropout → Linear 512→100` y devuelve **100 logits**.
- Los 100 bins son intervalos iguales de `log(1+z)` entre 0 y `log(1+6.0)` (`z_max 6.0`). El `softmax` de los logits es un **posterior** `p` sobre esos bins.

De ese posterior salen tres números:

| salida | cómo se calcula | para qué sirve |
|---|---|---|
| **`z_pred_map`** | `expm1(centro del bin argmax(p))` | **la predicción oficial** (MAP: el bin más probable) |
| **`z_confidence`** | `1 − H(p) / log(100)`, con `H` la entropía de `p` | concentración del posterior en [0, 1]; 1 = un solo bin, 0 = plano |
| `z_pred` | `expm1(Σ p · centro_bin)` (esperanza) | compatibilidad con la v1; no es la métrica oficial |

### Por qué se pasó de regresión escalar a clasificación

La **v1** tenía una cabeza que devolvía un solo número, `log(1+z)`, entrenada con `SmoothL1`. Sobre el held-out canónico daba η₀.₁₅ = 22.6 %, un techo de predicción de z ≈ 2.00 y un 82.7 % de outliers catastróficos en z ∈ [1.5, 2.5). Dos motivaciones del rediseño, sin ablación que aísle sus efectos:

1. **Regresión a la media.** Cuando el modelo no está seguro, minimizar un error escalar lo empuja hacia el valor "seguro" del centro de la distribución de entrenamiento. Por eso casi nunca predecía z alto, aunque el espectro lo pidiera.
2. **Multimodalidad por confusión de líneas.** Un espectro con una sola línea de emisión visible es intrínsecamente ambiguo: esa línea puede ser Hα (6563 Å), [OIII] (5007 Å) u [OII] (3727 Å) en reposo, y cada hipótesis implica un redshift distinto. Un número escalar no puede decir "es z = 0.3 o z = 1.2"; promedia y devuelve algo que está mal para las dos hipótesis.

Con clasificación el posterior puede tener **varias jorobas**, `z_pred_map` elige la más alta en vez de promediar, y `z_confidence` avisa cuando la elección fue dudosa. Resultado en el mismo held-out: η₀.₁₅ pasa de 22.6 % a **14.95 %**, σ_NMAD de 0.083 a **0.030**, η en z ∈ [1.5, 2.5) de 82.7 % a **23.5 %**, y el máximo observado sube a **3.52** (rango configurado hasta 6). Son métricas con máscara aleatoria del 50%, sin semilla fijada, sobre validación no independiente. Con `conf ≥ 0.5` quedan 1.175/2.000 y 9 errores (58.75% de cobertura, 0.77% de error); es análisis retrospectivo, no calibración ni política validada. Varias decisiones cambiaron entre versiones.

### El loss total de la v2.1

```
loss = 1.0 × loss_reconstrucción + 1.0 × loss_redshift

loss_redshift = CrossEntropy(logits, bin_verdadero, weight = z_bin_weights) / log(100)
```

Los tres ingredientes, cada uno con su razón:

- **`bin_verdadero`** = `bucketize(log1p(z), z_bin_edges)`; `label_smoothing = 0`.
- **CE normalizada ÷ log(100)** (`--normalize-redshift-ce`): una cross-entropy sobre 100 clases arranca en `log(100) ≈ 4.6`. Dividirla por eso la deja en escala ~[0, 1], comparable con el MSE de reconstrucción, y por eso los pesos pueden ser **1:1** sin calibrar a mano.
- **`z_bin_weights`, sqrt-inverse** (`build_z_bin_weights` en `train.py`): peso = `sqrt(media_de_conteos / conteo_del_bin)`, recortado a **[0.5, 3.0]**. Los conteos salen del histograma **real** de las 80.000 etiquetas de entrenamiento (`scripts/estimate_z_histogram.py` → `runs/calibration/z_hist_80k_100bins.npz`, que está gitignored y se regenera con ese script). La raíz cuadrada suaviza el rebalanceo: los bins raros de z alto pesan más, pero no tanto como para dominar.

Historia corta de los pesos: la v1 usaba `1.0 × recon + 10.0 × SmoothL1`, con el 10 calibrado a mano porque el redshift era un escalar contra 273 targets de reconstrucción. Una **v2.0** intermedia (200 bins, rebalanceo inverso agresivo con tope 10) sobrecorrigió: predecía de más en z < 0.5 y sus posteriors eran chatos; dio η MAP de 31.1 % sobre una ventana distinta, no comparable directamente con el 22.6% canónico de v1, y no se promovió (ese 31.1 % es solo diagnóstico: la v2.0 barajaba antes del corte, así que parte de esos 2.000 espectros pudo verse en su entrenamiento). La v2.1 es la corrección de eso.

---

## Paso 8: entrenamiento (`train.py`)

Es el bucle estándar de deep learning:

1. Tomar un lote (batch) de 8 espectros del stream.
2. Pasarlos por la red con la máscara aleatoria.
3. Calcular el loss.
4. Backpropagation.
5. `AdamW` actualiza los pesos; el scheduler ajusta el learning rate.
6. Repetir.

Detalles que valen la pena:

- **La v2.1 es un fine-tune de la v1**, no un entrenamiento desde cero. `--init-checkpoint runs/desi_50k_big/checkpoint_last.pt` pasa por `load_compatible_checkpoint`, que copia todos los pesos cuya forma coincide (tokenizador, encoder, cabeza de reconstrucción, casi toda la cabeza de redshift) y salta los que no: la última capa lineal de la cabeza pasó de 1 salida a 100, así que esa se inicializa de cero. "From scratch" aplica a la v1 (50.000 ejemplos, 1 época).
- **Learning rate 1e-4 con warmup + coseno** (`build_scheduler`): sube linealmente durante el 5 % inicial de los pasos y baja con una curva coseno hasta el 10 % del pico (`min_lr_ratio 0.1`). `weight_decay 0.05`.
- **Gradient clipping a 1.0.**
- **Streaming desde Hugging Face.** No se descarga el dataset; se lee mientras se entrena. Cada época re-baraja con `seed + época` (`set_epoch`), pero la membresía del train no cambia.
- **3 épocas × 80.000 ejemplos / batch 8 = 30.000 pasos.** En un MacBook (MPS) tomó ≈ 2–4 h (~3.8 h de reloj).
- **Logging cada 50 pasos** a `metrics.jsonl` (loss total, reconstrucción, redshift, lr).
- **Checkpoints cada 5000 pasos** (`checkpoint_step_5000.pt` … `checkpoint_step_30000.pt`), `checkpoint_last.pt` al final de cada época, y `checkpoint_best.pt` cuando mejora `eta15_map` sobre los 2.000 de validación (`is_better_checkpoint`). Si se interrumpe con Ctrl-C, guarda `checkpoint_interrupted.pt`.
- **`checkpoint_best` y `checkpoint_last` de la v2.1 tienen exactamente los mismos pesos** (los 117 tensores son idénticos; ambos son el paso 30.000, porque la mejor validación fue la última — los archivos difieren solo en metadatos). La diferencia entre sus evaluaciones (15.15 % vs 14.95 %) es ruido de la máscara aleatoria de evaluación, porque `evaluate.py` no fija semilla; dos corridas no permiten estimar un intervalo de incertidumbre fiable.

El comando que produjo la v2.1 (flags verificados contra el `argparse` de `train.py` y contra `runs/desi_80k_classhead_v21/training_args.json`; los que no aparecen quedan en su default):

```bash
python3 scripts/estimate_z_histogram.py --max-examples 80000 --n-z-bins 100 --z-max 6.0 \
  --output runs/calibration/z_hist_80k_100bins.npz

python3 -m desi_fm.train \
  --dataset MultimodalUniverse/desi --data-dir edr_sv3 \
  --max-train-examples 80000 --val-examples 2000 \
  --epochs 3 --batch-size 8 --lr 1e-4 --lr-schedule cosine \
  --d-model 512 --n-layers 8 --n-heads 8 \
  --n-z-bins 100 --z-max 6.0 --z-label-smoothing 0.0 \
  --z-weighting sqrt_inverse --z-histogram runs/calibration/z_hist_80k_100bins.npz \
  --z-weight-min 0.5 --z-weight-cap 3.0 --normalize-redshift-ce \
  --init-checkpoint runs/desi_50k_big/checkpoint_last.pt \
  --mask-ratio 0.5 --redshift-loss-weight 1.0 \
  --save-every-steps 5000 --log-every-steps 50 \
  --output-dir runs/desi_80k_classhead_v21
```

Ojo con `--d-model 512 --n-layers 8 --n-heads 8` y `--z-label-smoothing 0.0`: los defaults del script son 384/6/6 y 0.05. Sin esos flags el modelo se construiría con otra forma y el warm-start desde la v1 descartaría casi todos los pesos.

---

## Paso 9: usar el modelo entrenado (`predict.py`)

Una vez entrenado, el modelo ya no necesita gradientes. `predict.py` es el punto de entrada para remuestrear grillas y predecir; la precisión solo tiene evidencia histórica en DESI.

```python
from desi_fm.predict import predict_spectrum

result = predict_spectrum(
    flux=mi_flux,                 # 1-D, cualquier unidad
    wavelength=mi_wavelength,     # 1-D, en Å
    checkpoint_path="runs/desi_80k_classhead_v21/checkpoint_last.pt",  # o el del Hub, ver abajo
)

result["z_pred_map"]                # redshift predicho (oficial: argmax del posterior)
result["z_confidence"]              # concentración del posterior, 0–1
result["z_pred"]                    # expm1(E[log(1+z)]); no media aritmética de z
result["reconstruction_input_grid"] # reconstrucción, en la grilla y las unidades del llamador
```

Para usar el checkpoint publicado en vez del local:

```python
from huggingface_hub import hf_hub_download
ckpt = hf_hub_download("jirustaroure/desi-spectra-fm", "checkpoint_last.pt")
```

Valor de control con un ejemplo held-out real (`examples/heldout_z020.npz`, gitignored en este repo; se regenera con `scripts/make_demo_examples.py`): `z_pred_map 0.2267`, `z_confidence 0.64`, `z_true 0.2036`.

Qué hace por dentro:

1. `load_model_from_checkpoint` reconstruye la config desde el `.pt` (`DESIFoundationModelConfig(**checkpoint["config"])`) y carga los pesos; tolera checkpoints viejos donde el embedding posicional se llamaba `pos_embed`.
2. `_prepare_inputs` llama al mismo `preprocess_spectrum` de `data.py`: si el espectro viene con otra grilla, lo interpola a la interna; los píxeles fuera de la cobertura del instrumento quedan `valid = 0` y el modelo sabe que debe ignorarlos.
3. Corre el modelo con `mask_ratio = 0.0` (nada tapado, salida determinista). Con `mask_ratio > 0` se puede medir reconstrucción de regiones tapadas.
4. Deshace la normalización con `center` y `scale` (`_denormalize_flux`) e interpola la reconstrucción de vuelta a la grilla original del llamador, para comparar píxel a píxel sin conocer la representación interna.

`predict_spectra_batch` hace lo mismo sobre `(N, P)` arrays, y hay CLI:

```bash
python3 -m desi_fm.predict --checkpoint "$CKPT" --input espectros.npz --output predicciones.npz
```

El `.npz` de entrada necesita `flux` y `wavelength` (`(P,)` o `(N, P)`); `ivar` y `mask` son opcionales.

---

## Paso 10: el embedding del espectro (`encode()` / `embed_spectrum`)

Además de predecir, el encoder sirve como **representación reutilizable** del espectro. `DESIFoundationModel.encode` corre el transformer **sin tapar ningún token** (así el embedding es determinista) y promedia los 273 vectores de salida de los tokens espectrales que tienen al menos un píxel válido. El token de redshift participa en la atención pero **no** entra en el promedio: la idea es un embedding del espectro, no de la lectura de z.

`embed_spectrum` en `predict.py` envuelve eso con el mismo preprocesamiento que `predict_spectrum` y devuelve un vector `float32` de 512 dimensiones. Con eso se construyó el índice FAISS de 15.000 espectros del lado de entrenamiento que está en el Hub (`faiss/spectra.faiss`, `faiss/spectra_meta.npz`) y que usa la herramienta `find_similar_spectra` del agente (`spectra-copilot`): vecinos cuyo z de catálogo se agrupa alrededor de un candidato lo respaldan, sin depender de identificar líneas.

---

## Paso 11: evaluación (`evaluate.py`)

Toma un checkpoint, lo corre sobre espectros con `z` verdadero conocido y reporta métricas. Esta es la receta histórica de v2.1; usar un destino nuevo, nunca sobrescribir los artefactos originales:

```bash
python3 -m desi_fm.evaluate \
  --checkpoint runs/desi_80k_classhead_v21/checkpoint_last.pt \
  --skip-examples 80000 --max-examples 2000 \
  --predictions-csv runs/new_evaluation/predictions.csv \
  --metrics-json runs/new_evaluation/metrics.json
```

**Por qué existe `--skip-examples`.** Sin él, el evaluador arranca desde el principio del stream, es decir, sobre los mismos espectros que el modelo vio en entrenamiento, y los números no miden generalización. Los números viejos de la v1 (25.1 % / 27.3 % / 0.124 / 0.864) tenían ese problema: se midieron sobre datos vistos o sobre ventanas mal definidas, y por eso **no** se usan como baseline. Con `--skip-examples 80000 --max-examples 2000` el evaluador reproduce exactamente el held-out canónico (`filter → skip → take`, sin shuffle) y el mismo comando sirve para v1 y v2.1, que es como se armó la tabla del `README.md`.

Qué reporta `evaluate_and_write_outputs`:

- `eta15_map` = fracción de outliers catastróficos, `|z_pred_map − z| / (1+z) > 0.15`. Es **η₀.₁₅**, el headline (14.95 %).
- `redshift_mae_norm_map` = `mean(|z_pred_map − z| / (1+z))` (0.096). Se normaliza por `(1+z)` porque a z grande un error absoluto grande puede ser proporcionalmente pequeño.
- `mean_z_confidence` (0.52).
- `reconstruction_rmse_masked` = RMSE píxel a píxel en espacio `arcsinh`, solo sobre píxeles tapados y válidos (0.817).
- Las versiones sin `_map` (`eta15`, `redshift_mae_norm`) usan `z_pred` (la esperanza) y quedan por compatibilidad.
- Con `--predictions-csv` escribe una fila por espectro (`z_true, z_pred, abs_dz, abs_dz_norm, z_pred_map, z_confidence`); con `--reconstructions-npz` guarda unas pocas reconstrucciones para graficar.

σ_NMAD, η por ventana de z y el techo `max z_pred_map` se calculan a partir de ese CSV; la comparación gate por gate contra la v1 quedó en `runs/desi_80k_classhead_v21/comparison.json` (`decision: promote_v2_1`, 5 gates).

Dos advertencias:

- El evaluador aplica la máscara aleatoria del 50 % y **no fija semilla**, así que dos corridas del mismo checkpoint difieren en décimas de punto (15.15 % vs 14.95 % con pesos idénticos). No conviertas esa diferencia en un intervalo de confianza.
- El 14.95 % de `desi_fm.evaluate` (2.000 held-out, con masking) **no es comparable** con el 92.7 % de aciertos < 0.15 de las evals de `spectra-copilot` (150 casos held-out, herramientas con el espectro completo, sin masking). Son dos protocolos distintos.

---

## Paso 12: la API REST (`api.py`)

`api.py` envuelve `predict.py` en una app FastAPI con tres endpoints:

- `GET /healthz` → `{"status": "ok", "model_loaded": ...}`.
- `POST /predict` → subida multipart de un `.npz` con `flux` y `wavelength` (`(P,)` o `(N, P)`; `ivar`/`mask` opcionales), parámetro de query `mask_ratio`. Límites: 32 espectros y 50 MB por request.
- `POST /predict_json` → un solo espectro como listas JSON.

La respuesta pone primero la predicción oficial: `z_pred_map`, `z_confidence`, y después `z_pred`. El checkpoint se descarga del Hub en la primera request y queda cacheado (`get_model`). Eso vale al correrla localmente (uvicorn/Docker); en el Space publicado, `api/app.py` carga el modelo al arrancar (`get_model()` en el import), así que `/api/healthz` ya reporta `model_loaded: true` desde la primera request. `DESI_FM_CKPT` apunta a un checkpoint local para no descargar y `DESI_FM_DEVICE` cambia el dispositivo (por defecto CPU, para que corra en hardware gratuito).

Dónde está el Swagger, que cambia según cómo la corras:

- **Local** (`uvicorn desi_fm.api:app --port 7860`, o el `Dockerfile`): `http://localhost:7860/docs`, sin prefijo.
- **Space publicado** (`api/app.py` monta esta misma app como sub-app en `/api` dentro de un Space Gradio ZeroGPU): `https://jirustaroure-desi-fm-api.hf.space/api/docs`, y los endpoints son `/api/healthz`, `/api/predict`, `/api/predict_json`.

---

## `inspect_schema.py`

Utilitario de una sola función: baja el primer ejemplo del stream, imprime las claves del dataset, las formas de `flux`/`ivar`/`lambda`/`mask`, el redshift y el rango de longitud de onda. Sirve para confirmar que el dataset sigue teniendo el formato que espera `extract_mmu_desi_example` antes de lanzar un entrenamiento largo.

---

## Mini glosario

| término | qué es |
|---|---|
| **espectro** | curva de intensidad de la luz vs longitud de onda |
| **redshift (z)** | cuánto se estiró la luz por la expansión del universo; `λ_obs = (1+z) · λ_emit` |
| **token** | un pedacito del espectro (26 píxeles) convertido en un vector de 512 |
| **embedding** | un vector que representa algo (un token, una posición, un espectro entero) |
| **masked-token prediction** | tapar pedacitos y pedirle al modelo que los adivine; así entrena BERT |
| **transformer encoder** | la red que usamos: capas de atención + feed-forward, sin decodificador |
| **head (cabeza)** | una pequeña red al final que produce una salida específica |
| **loss** | qué tan equivocado está el modelo, en un solo número |
| **bin** | un intervalo de `log(1+z)`; hay 100, iguales, entre z = 0 y z = 6 |
| **posterior** | la distribución de probabilidad sobre los 100 bins (softmax de los logits) |
| **MAP / `z_pred_map`** | el bin más probable del posterior; la predicción oficial |
| **`z_confidence`** | 1 − entropía normalizada del posterior; 1 = concentrado, 0 = plano; no probabilidad calibrada |
| **η₀.₁₅** | fracción de espectros con `\|Δz\| / (1+z) > 0.15` ("outliers catastróficos") |
| **σ_NMAD** | dispersión robusta de `Δz / (1+z)`: `1.4826 · mediana(\|Δz_n − mediana(Δz_n)\|)`, con `Δz_n = Δz / (1+z)` (calculada en `notebooks/evaluation.ipynb`) |
| **foundation model** | modelo que aprende un dominio en profundidad, reutilizable en varias tareas (acá: z, reconstrucción, embedding) |

---

## Mini glosario de archivos clave

| archivo | qué contiene | en una frase |
|---|---|---|
| `model.py` | tokenización, transformer y sus dos cabezas | el cerebro |
| `data.py` | lectura, limpieza, normalización, split | el sistema digestivo |
| `train.py` | bucle de entrenamiento, pesos de clase, checkpoints | el gimnasio |
| `evaluate.py` | métricas sobre el held-out canónico | la balanza |
| `predict.py` | inferencia agnóstica al instrumento + embeddings | el modelo "en producción" |
| `api.py` | API REST sobre `predict.py` | la ventanilla |
| `inspect_schema.py` | verifica el formato de los datos | el inspector de calidad |

---

## Preguntas que suelen surgir (respuesta corta)

1. **"¿Cómo funciona la tokenización?"**
   → "Interpolo cada espectro a 7081 píxeles en una grilla logarítmica de longitud de onda, lo corto en 273 parches de 26 píxeles, y cada parche (flujo + validez) se proyecta linealmente a 512 dimensiones. Agrego un token 274 que representa el redshift y está siempre enmascarado."

2. **"¿Por qué grilla logarítmica?"**
   → "Porque el redshift es un estiramiento multiplicativo, y en `log λ` se vuelve una traslación. Al transformer le resulta mucho más fácil aprender esa simetría."

3. **"¿Por qué la cabeza de redshift clasifica en vez de regresar un número?"**
   → "Porque la regresión escalar de la v1 sufría regresión a la media —techo de z ≈ 2.00, 82.7 % de outliers en z ∈ [1.5, 2.5)— y no podía representar la ambigüedad Hα / [OIII] / [OII]. Un posterior de 100 bins sobre `log(1+z)` mantiene las hipótesis, `z_pred_map` elige la más probable y `z_confidence` dice cuándo desconfiar. η₀.₁₅ bajó de 22.6 % a 14.95 % en el mismo held-out."

4. **"¿Cómo se entrena el redshift junto con la reconstrucción?"**
   → "El token de redshift está enmascarado en todos los ejemplos y su cabeza se entrena conjuntamente con la reconstrucción, con cross-entropy normalizada por `log(100)` y pesos de clase sqrt-inverse, 1:1 con el MSE de reconstrucción. El encoder representa el redshift desde el primer paso, no se ajusta después con el encoder congelado."

5. **"¿Cómo se comporta con espectros de otro instrumento?"**
   → "Los embeddings posicionales son sinusoidales sobre `log λ` físico, así que cada token está anclado a una longitud de onda, no a un índice. `predict.py` interpola cualquier grilla a la interna y marca como inválidos los píxeles fuera de la cobertura del instrumento. Esto no valida precisión fuera de DESI."

6. **"¿El held-out es realmente held-out?"**
   → "No como test independiente: se excluyeron del fine-tuning v2.1, pero hay evidencia fuerte de exposición en v1 y se usaron para seleccionar checkpoints. El test del cargador actual no repara ese linaje."

7. **"¿Cuántos parámetros tiene?"**
   → "~26 millones: 8 capas, `d_model` 512, 8 cabezas. Se entrenó en un MacBook con MPS; la arquitectura escala directamente en GPU dedicada."

Evidencia y procedencia: [HISTORICAL_EVIDENCE.md](docs/HISTORICAL_EVIDENCE.md).
