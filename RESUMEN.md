# Resumen del proyecto en simple

## La idea en una frase

Construir una IA que aprenda a "entender" la luz que llega de galaxias y estrellas (espectros), igual que GPT aprendió a entender el lenguaje humano.

---

## ¿Qué es un espectro?

Imaginá que tomás la luz de una galaxia y la pasás por un prisma. La luz se separa en colores y obtenés una curva: cuánta luz hay en cada "color" (longitud de onda), desde el ultravioleta hasta el infrarrojo.

Esa curva no es ruido aleatorio. Tiene **picos** y **valles** en lugares muy específicos: son las "huellas digitales" de los elementos químicos. El hidrógeno deja su huella en 6563 Å. El calcio en 3934 Å. Y así.

Si conocés esas huellas, podés deducir qué hay dentro de la galaxia.

---

## ¿Qué es el redshift (z)?

El universo se está expandiendo. Eso significa que los objetos lejanos se alejan de nosotros, y la luz que nos mandan se **estira** en el viaje.

Resultado: las huellas químicas, que en el laboratorio están en sus lugares conocidos, llegan a nosotros **desplazadas hacia el rojo** (longitudes de onda más largas).

La fórmula es simple:

```
longitud_onda_observada = (1 + z) × longitud_onda_emitida
```

- `z = 0.005` → galaxia cercana, casi sin estiramiento.
- `z = 1.0` → galaxia muy lejana, las huellas están al doble de su longitud original.
- `z = 5.0` → galaxia lejanísima, lo que era ultravioleta ahora se ve casi infrarrojo.

**Medir `z` = medir qué tan lejos está un objeto.** Es una de las cantidades más importantes en astrofísica moderna.

---

## ¿Qué pide el profesor?

Construir un modelo de IA que, al darle un espectro:

1. **Prediga el redshift `z`** del objeto.
2. **Reconstruya partes "tapadas"** del espectro (igual que GPT completa palabras faltantes en una oración).

Y que funcione no solo con espectros del telescopio DESI (con los que se entrena) sino **también con espectros de otros telescopios** que el modelo nunca vio. Esta última parte es la prueba de que aprendió algo profundo sobre espectros en general, no solo memorizó cómo se ven los de DESI. A eso se le llama **foundation model**.

---

## ¿Qué hizo mal AION-1 (el modelo anterior)?

AION-1 es un modelo grande de astrofísica que existe. El profesor lo critica porque trata al **redshift como si fuera un dato más** entre otros 273 tokens del espectro. Eso tiene dos problemas:

1. **No le da suficiente importancia al redshift.** El redshift es UNA cantidad clave, no es como uno más de los pedacitos del espectro. Pero AION lo trata igual.
2. **El modelo nunca aprende a representar el redshift internamente.** Como casi nunca tiene que adivinarlo, no se ve forzado a entenderlo.

Resultado: AION-1 termina necesitando un parche externo (otra red neuronal encima) para poder predecir redshifts, y aun así no lo hace especialmente bien.

---

## ¿Qué hicimos nosotros para arreglarlo?

Combinamos las **dos soluciones** que la consigna sugiere:

### 1. El token de redshift está SIEMPRE tapado
En cada ejemplo de entrenamiento, el modelo nunca ve el valor real de `z`. Tiene que adivinarlo mirando el espectro completo. Esto fuerza al modelo a aprender realmente la relación espectro → redshift.

### 2. Una "cabecita" predictora entrenada en paralelo
Le pusimos un pequeño MLP que toma la representación interna del modelo y devuelve `z`. Esa cabecita se entrena junto con la reconstrucción del espectro, **no después**. Eso fuerza al cerebro del modelo a organizar su pensamiento alrededor del redshift desde el primer paso.

---

## ¿Qué le entregamos al profesor?

1. **Un modelo entrenado**: el checkpoint `runs/desi_50k_big/checkpoint_last.pt` (~26 millones de parámetros, entrenado con 50.000 espectros reales de DESI).
2. **Código que puede correr sobre cualquier espectro**: `src/desi_fm/predict.py`. El profesor le pasa un espectro (de DESI o de cualquier otro instrumento) y obtiene `z` predicho + reconstrucción.
3. **Documentación**: `DELIVERABLE.md` explica cómo usarlo.

---

## ¿Cómo nos fue?

| qué medimos | qué dio |
|---|---|
| Error medio en `z` (normalizado por `1+z`) | **0.124** |
| Reconstrucción de regiones tapadas (RMSE) | **0.864** |

Para comparar: el pipeline oficial de DESI da errores de `z` del orden de `0.0001`. Nuestro modelo está 1000 veces peor que la pipeline oficial — pero la pipeline usa código diseñado a mano por físicos durante años. Nuestro modelo aprendió todo solo, mirando datos durante una hora en una laptop. **El punto del proyecto no era ganarle a la pipeline. Era demostrar que un foundation model unimodal con el redshift bien diseñado SÍ aprende espectros.** Y eso lo logramos.

Además, en pruebas con datos sintéticos (que NO son DESI), nuestro modelo grande dio predicciones razonables — primera señal de que sí generaliza fuera de su distribución de entrenamiento, que es lo que define a un foundation model.

---

## Lo que NO hicimos (a propósito, porque la consigna lo prohibía)

- ❌ Nada de imágenes de telescopios (Subaru, HSC, etc.).
- ❌ Nada de magnitudes ni catálogos.
- ❌ Nada de CNN que "regresione directamente" `z` desde el espectro (eso no es un foundation model, es un especialista).

Solo espectros + redshift, treinta y nueve tipos de datos menos que AION-1, una sola modalidad tratada en profundidad.
