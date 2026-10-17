# 5A: análisis posteriores de predicciones guardadas

No cambian la selección, el criterio predefinido ni los artefactos del 4B. Sin inferencia nueva.

| Paso de validación | A−B 17 (pp) | A−B 42 (pp) | A−B 101 (pp) |
|---:|---:|---:|---:|
| 500 | -36.10 | -23.91 | -27.03 |
| 1000 | -9.35 | +3.63 | -7.71 |
| 1500 | -2.68 | +3.69 | -0.32 |
| 2000 | +1.31 | -7.20 | -0.49 |
| 2500 | +3.31 | +2.68 | +3.61 |
| 3000 | +8.01 | +2.24 | -2.92 |
| 3500 | +4.31 | -4.21 | +3.68 |
| 4000 | +5.07 | -0.56 | +14.34 |
| 4500 | -0.72 | -1.03 | +5.80 |
| 5000 | +0.36 | +2.19 | +0.65 |
| 5500 | -4.97 | -8.59 | +1.92 |
| 6000 | -2.18 | -8.69 | -3.42 |
| 6500 | +5.48 | -9.70 | +1.27 |
| 7000 | -2.21 | -8.44 | -1.84 |
| 7500 | -1.10 | -7.62 | -3.38 |
| 8000 | +0.86 | -7.95 | -2.38 |
| 8500 | +0.24 | -7.57 | +3.24 |
| 9000 | +0.90 | -8.47 | +0.33 |
| 9500 | -0.12 | -6.15 | +1.12 |
| 10000 | +1.97 | -9.84 | +4.87 |

Las tres diferencias son negativas en 4/20 pasos; por semilla: [9, 15, 9].
ΔMAE medio en validación al paso final: +0.002896.

| Candidato | Paso seleccionado | Segundo paso | Margen (objetos/10.000) |
|---|---:|---:|---:|
| A-17 | 5500 | 7000 | 63 |
| A-42 | 10000 | 8500 | 111 |
| A-101 | 8000 | 7000 | 63 |
| B-17 | 8500 | 9500 | 27 |
| B-42 | 9500 | 8500 | 80 |
| B-101 | 2500 | 8000 | 2 |
| CNN-17 | 6000 | 8000 | 20 |
| CNN-42 | 7000 | 8500 | 14 |
| CNN-101 | 5500 | 7000 | 17 |

Promedio de los k mejores scores de cada variante (no ensemble ni nueva regla):

| k | A−B 17 (pp) | A−B 42 (pp) | A−B 101 (pp) |
|---:|---:|---:|---:|
| 1 | -0.79 | -7.88 | -2.36 |
| 2 | -0.61 | -7.73 | -2.06 |
| 3 | -0.59 | -7.87 | -1.84 |
| 5 | -0.76 | -8.01 | -0.76 |

Sensibilidad H-10 en **validación guardada**, A-101 paso 8.000 frente a B-101 pasos 2.500/8.000. No evalúa un checkpoint alternativo en test ni autoriza sustituir el seleccionado.

| Paso B | Estrato | N | Outliers A/B | Δη (pp) | ΔMAE |
|---:|---|---:|---|---:|---:|
| 2500 | all | 10000 | 2096/2332 | -2.36 | +0.032038 |
| 2500 | 2<=z<3 | 119 | 8/93 | -71.43 | -0.187971 |
| 2500 | z>=3 | 24 | 1/11 | -41.67 | -0.223513 |
| 8000 | all | 10000 | 2096/2334 | -2.38 | -0.051399 |
| 8000 | 2<=z<3 | 119 | 8/27 | -15.97 | -0.024513 |
| 8000 | z>=3 | 24 | 1/2 | -4.17 | -0.030026 |

ΔMAE de test desde predicciones guardadas; IC95 bootstrap pareado por objeto, 5.000 réplicas, semilla 20261016, condicionado a cada par seleccionado. Independencia física no establecida.

| Semilla | ΔMAE | IC95 |
|---:|---:|---|
| 17 | +0.038576 | [+0.032077, +0.045357] |
| 42 | -0.144347 | [-0.156021, -0.132731] |
| 101 | +0.026530 | [+0.018331, +0.035049] |

Identidades de entradas, tabla completa y conteos de localidad/cola: [analysis.json](batch5a-analysis.json).
