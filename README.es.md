# Modelo fundacional unimodal para espectros DESI

Este repo es una implementacion inicial del proyecto final de PHYS303/CS486/CS686.
La idea es construir un modelo tipo BERT/MAE para espectros, no una CNN que solo
predice redshift.

## Resumen sencillo del proyecto

Un espectro es una curva: para cada longitud de onda, mide cuanta luz llega de una
galaxia, estrella o quasar. Las lineas de emision/absorcion son como huellas
quimicas. Cuando el objeto esta lejos, esas huellas aparecen corridas hacia
longitudes de onda mas grandes. Ese corrimiento es el redshift:

```text
lambda_observada = (1 + z) * lambda_emitida
```

La consigna pide entrenar un modelo que aprenda espectros DESI de verdad:

- Entrada principal: espectro DESI solamente.
- Senal auxiliar: redshift `z`.
- Sin imagenes, sin magnitudes, sin Subaru/HSC, sin catalogos.
- Objetivos:
  - reconstruir regiones enmascaradas del espectro;
  - predecir redshift en espectros nuevos.

La critica a AION-1 es que reparte su capacidad entre 39 modalidades y trata el
redshift como un token cualquiera. Aqui hacemos lo contrario: una sola modalidad,
espectros, y el redshift participa en el entrenamiento desde el primer paso.

## Diseno implementado

Esta implementacion combina los dos mecanismos recomendados por la consigna.

Primero, interpola cada espectro a una grilla comun en `log(lambda)` y lo divide
en `273` tokens espectrales continuos. Cada token representa un pequeno parche de
la curva de flujo. Esto es mas simple que el codec discretizado de AION, pero
mantiene la idea importante: el transformer no ve pixeles sueltos, ve una
secuencia compacta de tokens espectrales.

Segundo, agrega un token especial de redshift al final de la secuencia. Ese token
siempre esta enmascarado: el modelo nunca recibe el `z` verdadero como entrada.
Debe inferirlo mirando el contexto del espectro.

Tercero, suma a cada token un embedding sinusoidal construido desde su longitud
de onda fisica. Esto ayuda especialmente para datos fuera de DESI, porque el
modelo no solo sabe "token 17", sino tambien "esta zona corresponde a tal region
del espectro".

Cuarto, entrena dos perdidas juntas:

- `reconstruction_loss`: MSE sobre los parches espectrales enmascarados.
- `redshift_loss`: Smooth L1 sobre `log(1 + z)`.

Asi, el encoder aprende una representacion que sirve simultaneamente para
reconstruir espectros y organizar la informacion fisica del redshift.

Por que `log(lambda)`: el redshift multiplica las longitudes de onda por
`(1 + z)`. En escala logaritmica, multiplicar se vuelve sumar. O sea, una galaxia
con mayor redshift se ve mas como la misma huella corrida horizontalmente, que es
un patron mas facil de aprender para un transformer.

## Instalacion

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

## Verificar schema real de MMU/DESI

Antes de entrenar largo, corre:

```bash
python3 -m desi_fm.inspect_schema \
  --dataset MultimodalUniverse/desi \
  --data-dir edr_sv3
```

Esto imprime las columnas reales, las llaves dentro de `spectrum`, la llave de
redshift encontrada (`Z`, `redshift` o `z`) y el rango de longitudes de onda.

## Smoke test sin descargar DESI

Esto usa espectros sinteticos pequenos para verificar que el entrenamiento corre:

```bash
python3 -m desi_fm.train \
  --synthetic \
  --output-dir runs/smoke \
  --n-pixels 512 \
  --n-tokens 32 \
  --d-model 64 \
  --n-layers 2 \
  --n-heads 4 \
  --batch-size 4 \
  --max-train-examples 32 \
  --val-examples 16 \
  --epochs 1 \
  --wavelength-grid log
```

## Entrenamiento con MultimodalUniverse/DESI

Empieza pequeno, como recomienda la consigna:

```bash
python3 -m desi_fm.train \
  --dataset MultimodalUniverse/desi \
  --data-dir edr_sv3 \
  --output-dir runs/desi_tiny \
  --batch-size 8 \
  --max-train-examples 10000 \
  --val-examples 1000 \
  --epochs 1 \
  --wavelength-grid log \
  --lr-schedule cosine \
  --warmup-ratio 0.05 \
  --redshift-loss-weight 20
```

Para escalar, sube `--max-train-examples`, `--d-model`, `--n-layers` y el tamano
de batch segun tu GPU. La meta realista de la consigna es acercarse a la escala
`aion-base` de 300M parametros, pero primero conviene demostrar que todo el
pipeline funciona con 10k ejemplos.

El entrenamiento escribe `metrics.jsonl` dentro del directorio de salida. Si se
interrumpe con `Ctrl+C`, guarda `checkpoint_interrupted.pt` para no perder todo
el progreso. Por defecto los checkpoints guardan solo pesos del modelo; agrega
`--save-optimizer` si necesitas guardar tambien el estado de AdamW.

## Evaluacion

```bash
python3 -m desi_fm.evaluate \
  --checkpoint runs/desi_tiny/checkpoint_last.pt \
  --dataset MultimodalUniverse/desi \
  --data-dir edr_sv3 \
  --max-examples 1000 \
  --predictions-csv runs/desi_tiny/predictions.csv \
  --reconstructions-npz runs/desi_tiny/reconstructions.npz
```

Metricas principales:

- `redshift_mae`: error absoluto medio en `z`.
- `redshift_mae_norm`: error absoluto medio en `z / (1 + z)`.
- `reconstruction_loss`: error de reconstruccion en parches enmascarados.
- `reconstruction_rmse_masked`: RMSE pixel-wise solo en regiones enmascaradas.

## Que debes poder explicar oralmente

La tokenizacion aqui funciona asi:

1. Interpolamos cada espectro a una grilla comun en `log(lambda)`.
2. Normalizamos robustamente el flujo de cada espectro con mediana y escala tipo
   IQR, para que el modelo no dependa solo del brillo absoluto.
3. Partimos la curva en `273` parches.
4. Cada parche se convierte en un vector con una capa lineal aprendida.
5. Agregamos una senal sinusoidal que dice en que longitud de onda fisica esta
   cada token.
6. El transformer ve esos vectores como "tokens" y aprende a completar los que
   faltan.

La parte fisicamente importante es que las posiciones relativas de las lineas se
conservan. Si una linea como H-alpha aparece desplazada, el transformer puede
relacionar ese patron con el token de redshift siempre enmascarado.

## Fuentes usadas

- Consigna: proyecto final de PHYS303/CS486 (USF) — entrenar y evaluar un foundation model auto-supervisado para espectros DESI con predicción de redshift (material del curso, no se distribuye en este repo)
- MultimodalUniverse: https://github.com/MultimodalUniverse/MultimodalUniverse
- AION: https://github.com/PolymathicAI/AION
- aion-base: https://huggingface.co/polymathic-ai/aion-base
