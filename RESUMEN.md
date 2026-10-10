# Resumen histórico del proyecto v1

> Documento de la entrega inicial; no describe la versión publicada v2.1 ni
> constituye evidencia de generalización. Consultar [README.es.md](README.es.md)
> y [evidencia actual](docs/HISTORICAL_EVIDENCE.md). Las métricas históricas se
> conservan con su alcance, sin compararlas con otras ventanas.

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

La generalización a otros telescopios era un objetivo del proyecto. La interpolación de grillas no demuestra que se haya alcanzado; falta una evaluación independiente.

---

## Referencia de diseño

La consigna proponía rediseñar el tratamiento de redshift mediante un token
siempre oculto y una cabeza conjunta. AION-1 fue una referencia; no se entrenó
ni se reprodujo aquí, y este proyecto no prueba defectos generales de AION-1.

## ¿Qué hicimos nosotros para arreglarlo?

Combinamos las **dos soluciones** que la consigna sugiere:

### 1. El token de redshift está SIEMPRE tapado
En cada ejemplo de entrenamiento, el modelo nunca ve el valor real de `z`. Tiene que adivinarlo mirando el espectro completo. Esto fuerza al modelo a aprender realmente la relación espectro → redshift.

### 2. Una "cabecita" predictora entrenada en paralelo
Le pusimos un pequeño MLP que toma la representación interna del modelo y devuelve `z`. Esa cabecita se entrena junto con la reconstrucción del espectro, **no después**. Eso fuerza al cerebro del modelo a organizar su pensamiento alrededor del redshift desde el primer paso.

---

## ¿Qué le entregamos al profesor?

1. **Un modelo entrenado**: el checkpoint `runs/desi_50k_big/checkpoint_last.pt` (~26 millones de parámetros, entrenado con 50.000 espectros reales de DESI).
2. **Código de inferencia que remuestrea grillas**: `src/desi_fm/predict.py`. El profesor le pasa un espectro (de DESI o de cualquier otro instrumento) y obtiene `z` predicho + reconstrucción.
3. **Documentación**: `DELIVERABLE.md` explica cómo usarlo.

---

## ¿Cómo nos fue?

| qué medimos | qué dio |
|---|---|
| Error medio en `z` (normalizado por `1+z`) | **0.124** |
| Reconstrucción de regiones tapadas (RMSE) | **0.864** |

Esas cifras pertenecen a la evaluación antigua, con otra ventana; no miden
el checkpoint v2.1 ni generalización independiente. La reconstrucción sigue
siendo un objetivo auxiliar cuyo beneficio para redshift necesita una ablación.
Las sondas sintéticas no validan precisión en otros instrumentos.

## Lo que NO hicimos (a propósito, porque la consigna lo prohibía)

- ❌ Nada de imágenes de telescopios (Subaru, HSC, etc.).
- ❌ Nada de magnitudes ni catálogos.
- ❌ Nada de CNN que "regresione directamente" `z` desde el espectro (una comparación CNN queda para un experimento posterior).

Solo espectros + redshift, treinta y nueve tipos de datos menos que AION-1, una sola modalidad tratada en profundidad.
