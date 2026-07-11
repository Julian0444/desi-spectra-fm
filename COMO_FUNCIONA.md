# Cómo funciona el código (explicado simple)

Este documento explica qué hace cada parte del proyecto, paso a paso, sin asumir que ya entendés transformers o astrofísica.

---

## La idea de la cadena completa

Podemos pensar todo el proyecto como una sola "máquina" con esta forma:

```
Espectro de entrada (números crudos)
         ↓
    Limpiar y normalizar
         ↓
    Cortar en pedacitos ("tokens")
         ↓
    Tapar algunos pedacitos al azar
         ↓
    Pasar todo por un Transformer (la red neuronal)
         ↓
    Salidas: (a) qué había en los pedacitos tapados
             (b) cuál es el redshift z
```

Eso es todo. El resto del documento explica cada caja con más detalle.

---

## El paisaje de archivos

```
src/desi_fm/
  data.py         → cómo leer y limpiar espectros
  model.py        → el cerebro: el Transformer
  train.py        → cómo entrenar el cerebro
  evaluate.py     → cómo medir qué tan bien quedó el cerebro entrenado
  predict.py      → cómo usar el cerebro entrenado para predecir cosas nuevas
  inspect_schema.py → utilitario para chequear que los datos llegan bien
```

Hay un solo "cerebro" definido en `model.py`. Los demás archivos son herramientas para alimentarlo, entrenarlo o usarlo.

---

## Paso 1: leer un espectro (`data.py`)

Un espectro de DESI llega como tres arrays de números:

- `flux`: 7,781 números, uno por longitud de onda. "Cuánta luz hay en este color."
- `wavelength`: 7,781 números. "Qué color es este." (en Ångströms, una unidad de longitud)
- `mask`: 7,781 booleanos. "¿Es confiable este pixel o está roto?"

Más un número:
- `Z`: el redshift verdadero, que sabemos porque la pipeline oficial de DESI ya lo calculó.

---

## Paso 2: limpiar y normalizar (`preprocess_spectrum` en `data.py`)

El espectro crudo tiene problemas que hay que arreglar:

1. **Píxeles rotos o con `NaN`.** Los marcamos como inválidos para que el modelo no los tome en cuenta.
2. **Diferentes objetos brillan distinto.** Una estrella brilla mucho, una galaxia lejana brilla poquito. Hay que normalizar para que el modelo no se obsesione con el brillo absoluto.
3. **La grilla de longitudes de onda no es la misma para todos los espectros.** Los interpolamos a una grilla común: 7,081 puntos espaciados logarítmicamente entre 3,600 Å y 9,800 Å.

¿Por qué logarítmico? Porque el redshift es un **estiramiento multiplicativo** (`λ_obs = (1+z) × λ_emit`). En coordenadas logarítmicas, multiplicar se vuelve sumar. Eso hace mucho más fácil para el transformer detectar el redshift: ya no tiene que buscar "líneas estiradas", solo "líneas desplazadas hacia la derecha".

La normalización usa `arcsinh` (similar a tomar logaritmo pero funciona con números negativos). Eso convierte un flujo que puede ir de muy chico a muy grande en algo más manejable, centrado en 0.

---

## Paso 3: cortar en pedacitos (la "tokenización")

Acá está la parte que el profesor te va a preguntar oralmente, así que prestá atención.

**Problema:** un espectro tiene 7,081 puntos. Eso es demasiados para que un transformer lo procese punto por punto (la atención escala con el cuadrado del número de tokens).

**Solución:** agrupamos los 7,081 puntos en **273 pedazos** de 26 puntos cada uno. A cada pedazo lo llamamos un **token espectral**.

Imaginá que el espectro es una tira de papel larga con 7,081 marcas. La cortás en 273 cuadraditos iguales. Cada cuadradito mantiene la información local: si en esa zona había una línea de emisión, ese cuadradito la conserva.

Cada cuadradito (vector de 26 números de flujo) pasa por una capa lineal aprendida que lo convierte en un vector de `d_model=512` dimensiones. Eso es **el embedding del token**. Esa es la "manera de pensar" del modelo sobre ese pedacito del espectro.

A la lista de 273 tokens espectrales le agregamos **un token 274 especial**: el **token de redshift**. Es un vector aprendido que representa "¿cuál es el redshift de este espectro?" sin contener el valor real. **Siempre está enmascarado**, en cada ejemplo.

---

## Paso 4: decirle al modelo dónde está cada token (positional embedding)

Un transformer puro no sabe en qué orden van los tokens. Necesita ayuda.

Le damos a cada token un **embedding posicional sinusoidal indexado por su longitud de onda media en logaritmo**.

En cristiano: cada uno de los 273 tokens tiene un vector "etiqueta" que le dice al modelo "yo soy el token que cubre la región de los 4861 Å" (por ejemplo, ahí está la línea Hβ del hidrógeno).

**¿Por qué esto importa para generalizar?** Si el día de mañana el profesor le da al modelo un espectro de **otro telescopio** (no DESI), el modelo igual sabe qué longitud de onda corresponde a cada token. Eso le permite reconocer la línea Hα aunque el espectro venga de un instrumento que nunca vio.

El token de redshift no tiene embedding posicional físico (su embedding de longitud de onda es exactamente cero), porque no corresponde a ninguna longitud de onda — es una pregunta sobre todo el espectro.

---

## Paso 5: tapar algunos tokens al azar (masked-token prediction)

Antes de pasar los 274 tokens al transformer, **tapamos el 50% de los espectrales** reemplazándolos por un vector "mask" aprendido.

Es como tachar la mitad de las palabras de una oración y pedirle al modelo que las adivine.

**El token de redshift SIEMPRE está tapado.** No es opcional, no se decide al azar — está tapado en cada paso de entrenamiento, en cada ejemplo. Esta es una de las dos contribuciones técnicas del proyecto (la otra es el predictor MLP entrenado conjuntamente).

---

## Paso 6: el Transformer (el "cerebro")

Los 274 tokens (algunos tapados, algunos visibles) entran al transformer encoder. Es la arquitectura estándar:

- 8 capas
- Cada capa: atención multi-cabezal (8 cabezas) + feed-forward
- `d_model = 512` (cada token es un vector de 512 dimensiones)

La atención permite que cada token "mire" a todos los otros tokens y combine información. Los tokens visibles dan contexto. Los tokens tapados aprenden a inferir su contenido a partir del contexto.

Al final salen 274 vectores de 512 dimensiones, uno por cada token de entrada.

---

## Paso 7: las dos cabezas de salida

El cerebro saca dos cosas a la vez:

### Cabeza de reconstrucción
- Toma los 273 vectores correspondientes a tokens espectrales.
- Cada uno pasa por un pequeño MLP que devuelve 26 números: la predicción de los 26 píxeles que ese token contenía.
- Comparamos con los píxeles reales (solo para los tokens que estaban tapados, los visibles no cuentan).
- La diferencia es el **loss de reconstrucción** (`MSE`).

### Cabeza de redshift
- Toma el vector número 274 (el del token de redshift siempre tapado).
- Lo pasa por otro pequeño MLP que devuelve un único número: la predicción de `log(1+z)`.
- Comparamos con el `log(1+z)` real.
- La diferencia es el **loss de redshift** (`SmoothL1`).

### El loss total

```
loss = 1.0 × loss_reconstrucción + 10.0 × loss_redshift
```

El `10.0` está calibrado para que las dos partes contribuyan más o menos igual al gradiente, porque el redshift es un solo escalar mientras que la reconstrucción tiene 273 targets.

---

## Paso 8: entrenamiento (`train.py`)

Es el bucle estándar de deep learning:

1. Tomar un lote (batch) de espectros.
2. Pasarlos por la red.
3. Calcular el loss.
4. Backpropagation: calcular cómo ajustar cada peso para bajar el loss.
5. `AdamW` actualiza los pesos.
6. Repetir.

Detalles que valen la pena:

- **Learning rate con warmup + cosine decay.** El LR empieza en 0, sube linealmente al pico durante el 5% inicial, y después baja siguiendo una curva coseno. Esto evita que el modelo "explote" al inicio.
- **Gradient clipping a 1.0.** Si los gradientes son enormes, los recortamos. Estabiliza el entrenamiento.
- **Streaming desde Hugging Face.** Los datos no se descargan todos a la vez. Se leen mientras se entrena. Eso permite trabajar con muchos espectros sin llenar el disco.
- **Logging cada 25-50 pasos** a `metrics.jsonl`. Si pasa algo raro podemos investigar.
- **Checkpoints intermedios** cada 2500 pasos. Si se corta la luz, no perdimos todo.

---

## Paso 9: usar el modelo entrenado (`predict.py`)

Una vez entrenado, el modelo ya no necesita gradientes. `predict.py` es el "entry point" pensado para el profesor.

El profesor le da un espectro (cualquier formato, cualquier instrumento) y obtiene:

```python
result = predict_spectrum(
    flux=su_flux,
    wavelength=su_wavelength,
    checkpoint_path="runs/desi_50k_big/checkpoint_last.pt",
)

result["z_pred"]                    # redshift predicho
result["reconstruction_input_grid"] # reconstrucción del espectro
```

Por dentro `predict.py` hace lo mismo que `data.py` (limpiar, interpolar, normalizar) y después llama al modelo. La diferencia clave es que **no requiere que el espectro sea de DESI**. Si viene de otro instrumento con otra grilla de longitudes de onda, el código lo interpola a la grilla interna del modelo. Y como el modelo usa posiciones físicas (`log λ`), sabe qué tokens corresponden a qué longitudes de onda del nuevo instrumento.

Después de predecir, `predict.py` interpola la reconstrucción de vuelta a la grilla original del profesor, así él puede comparar pixel a pixel sin tener que entender la representación interna del modelo.

---

## Paso 10: evaluación (`evaluate.py`)

Este archivo es para nosotros, no para el profesor. Toma un checkpoint, lo corre sobre un conjunto de espectros con `z` verdadero conocido, y reporta métricas:

- `redshift_mae` = error absoluto medio en `z`.
- `redshift_mae_norm` = error normalizado por `(1+z)`. Esta es la métrica estándar en astrofísica porque a `z` grande, errores absolutos grandes pueden ser proporcionalmente pequeños.
- `reconstruction_rmse_masked` = RMSE píxel a píxel en las regiones que tapamos.

Estos números son los que aparecen en la tabla del `DELIVERABLE.md`.

---

## Mini glosario

| término | qué es |
|---|---|
| **espectro** | curva de intensidad vs longitud de onda |
| **redshift (z)** | qué tan estirada está la luz por la expansión del universo |
| **token** | un pedacito del espectro convertido en vector |
| **embedding** | un vector que representa algo (un token, una posición) |
| **masked-token prediction** | tapar pedacitos y pedirle al modelo que los adivine; es como entrena BERT |
| **transformer** | el tipo de red neuronal que usamos (igual que GPT pero más chico) |
| **MLP** | "multi-layer perceptron", el tipo de red más simple, dos o tres capas lineales con ReLU/GELU entre medio |
| **encoder** | parte del transformer que lee la entrada y produce representaciones |
| **head (cabeza)** | una pequeña red al final que produce una salida específica |
| **loss** | qué tan equivocado está el modelo, en un solo número |
| **foundation model** | modelo que aprende un dominio en profundidad, no una tarea específica |

---

## Mini glosario de archivos clave

| archivo | qué contiene | en una frase |
|---|---|---|
| `model.py` | el transformer y sus dos cabezas | el cerebro |
| `data.py` | lectura, limpieza, normalización, tokenización | el sistema digestivo |
| `train.py` | bucle de entrenamiento | el gimnasio |
| `evaluate.py` | métricas sobre el conjunto de validación | la balanza |
| `predict.py` | inferencia agnóstica al instrumento | el modelo "en producción" |
| `inspect_schema.py` | verifica el formato de los datos | el inspector de calidad |

---

## Para la conversación oral con el profesor

Los puntos que él casi seguro va a preguntar y la respuesta corta:

1. **"¿Cómo funciona tu tokenización?"**
   → "Interpolo cada espectro a 7,081 píxeles en una grilla logarítmica de longitud de onda, los corto en 273 parches de 26 píxeles, y cada parche se proyecta linealmente a un vector de 512 dimensiones. Agrego un token 274 que representa el redshift y está siempre enmascarado."

2. **"¿Por qué grilla logarítmica?"**
   → "Porque el redshift es un estiramiento multiplicativo, y en coordenadas logarítmicas se vuelve una traslación. Eso hace mucho más fácil para el transformer aprender la simetría del redshift."

3. **"¿Cómo arreglaste el problema de redshift de AION-1?"**
   → "Dos cosas a la vez. Uno: el token de redshift está enmascarado en cada paso de entrenamiento, no aleatoriamente. Dos: una cabeza MLP predice z y se entrena conjuntamente con la reconstrucción, no después con el encoder congelado. Eso fuerza al encoder a representar el redshift desde el primer paso de entrenamiento."

4. **"¿Cómo se va a comportar con espectros de otro instrumento?"**
   → "Los positional embeddings son sinusoidales sobre log-lambda. Eso quiere decir que cada token está anclado a una longitud de onda física, no a un índice arbitrario. Si llega un espectro de otro instrumento con otra grilla, el código lo interpola a mi grilla interna; los tokens que caen fuera de la cobertura del nuevo instrumento se marcan como inválidos. El modelo sabe qué pedacitos del espectro son confiables y cuáles no."

5. **"¿Cuántos parámetros tiene?"**
   → "26 millones. La consigna sugería apuntar a 300M, pero por restricciones de cómputo en una laptop limité el tamaño. La arquitectura escala directamente si se entrenara en GPU dedicada."
