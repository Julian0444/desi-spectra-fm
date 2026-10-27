# Batch 3 measured historical results

Generated from the persisted run summary, CSV and NPZ. Source run: `historical-full-01`; full offline repeat: `historical-offline-repeat-01`. These are historical diagnostics, not a new independent test.

| Full spectrum, v2 contract | n | eta15 | MAE normalized | NMAD | Mean bias |
|---|---:|---:|---:|---:|---:|
| v1_scalar | 2000 | 22.75% | 0.118782 | 0.090253 | -0.006803 |
| v21_map | 2000 | 12.00% | 0.088689 | 0.028201 | +0.049292 |
| v21_secondary | 2000 | 14.75% | 0.084457 | 0.032707 | +0.035237 |

All 2,000 requested objects were evaluated in each task; no replacements or exclusions. Historical masked CSVs had v1 22.60% (452 outliers) and v2.1 MAP 14.95% (299). The new full-spectrum v2.1 result is 12.00%, not the earlier 7–8% projection. Weights are unchanged, while artificial masking and input support/validity differ; this does not isolate either effect. The same 150 exported cases still have 11 full-spectrum MAP outliers; their weighted projection was not a complete-population measurement.

| Original warning status | n | v1 eta15 | v2.1 MAP eta15 |
|---|---:|---:|---:|
| original_warning | 161 | 72.67% | 68.94% |
| no_original_warning | 1839 | 18.38% | 7.01% |

`ZWARN=False` is an original warning. It does not prove a wrong label. All groups remain in the primary denominator.

| Catalog z | n | v1 eta15 | v2.1 MAP eta15 |
|---|---:|---:|---:|
| [0,0.5) | 866 | 16.74% | 10.74% |
| [0.5,1) | 602 | 12.46% | 13.62% |
| [1,1.5) | 419 | 34.61% | 10.74% |
| [1.5,2) | 82 | 76.83% | 19.51% |
| [2,2.5) | 16 | 75.00% | 12.50% |
| [2.5,inf) | 15 | 100.00% | 13.33% |

Paired IDs: 2000. v2.1 corrects 295 v1 outliers and introduces 80 outliers; 1465 are successes and 160 outliers under both models. Lower global eta15 does not imply improvement in every stratum. Coverage strata, every-prefix risk/coverage and transition IDs are in the persisted summary/CSVs. Concentration is not a calibrated accuracy probability; no threshold is selected as product policy.

| Fixed-mask reconstruction method | Hidden valid pixels | RMSE in clipped arcsinh space |
|---|---:|---:|
| v1 | 7062048 | 0.821856836 |
| v21 | 7062048 | 0.818004176 |
| mean | 7062048 | 0.914105301 |
| linear | 7062048 | 1.031576897 |
| moving301 | 7062048 | 0.817984303 |

Identical saved masks across methods, seed 20261009. Normalization statistics and baselines use visible valid pixels only. Width 301 and other baseline parameters were fixed before inference; width 301 was motivated by earlier exploratory historical analysis and is not an independently selected baseline. Hidden targets only enter the score. The tiny measured difference does not establish an advantage or statistical equivalence. This normalization protocol differs from the historical reconstruction artefacts.

The main run took 49.46 s and the complete offline repeat 51.51 s on M4 Pro/MPS, batch 8. Exact repeated arrays/IDs/masks/outputs/metrics; zero attempted socket connections. Peak host RSS 2.549 GB; MPS driver allocation at closure 1.452 GB. The timing budget was estimated before the full runs, from 32 objects. These timings do not attribute any portion of the old streaming duration to a particular cause.

Individual original ID/raw evidence covers 150 exported objects; all 100,000 IDs/labels/warnings agree with recovered audit metadata. The remaining 1,850 historical identities depend on fixed snapshot/order/filter reconstruction. Original full CSVs have no contemporaneous per-row IDs. No future test was created or opened.

Reproduce from disk to a new destination (source imports; same recorded environment):

```sh
sh "<fm-repo>/scripts/reproduce_batch3.sh" "<private-state>/lote-3-20261009/my-new-run"
```

The script refuses an existing destination and blocks sockets. It produces CSV, NPZ, figures, exclusions, metrics and a complete/incomplete manifest. Compare runs with `scripts/compare_historical_runs.py`. Dataset/weights/index/results remain outside Git; light copies of manifests/summaries are versioned. No new wheel was built.

For lot 4 preparation: retain the verified snapshot/index, decide grouping limits and materialize fresh ID splits per the consensus. Random training access cost and a derived training matrix have not been profiled; the local reader is verified for ordered evaluation. New training uses fresh weights, and the new test remains closed. Credential rotation and publication remain pending.

## Adenda 5A: alcance de reproducibilidad

La repetición exacta corresponde al mismo entorno/dispositivo y código registrado.
El comparador no fija literalmente el commit 008ce03: exige igualdad de
`code_sha256` entre manifiestos. Cambios de fuente (incluso equivalentes) impiden
usar una ejecución nueva como réplica idéntica. No se elimina esta guarda.
Para reproducir el protocolo histórico usar el código registrado; ver
[BATCH_5.md](BATCH_5.md). No ejecutar el quick start histórico desde HEAD como
si certificara la misma identidad.

H-12 queda acotado: el control original de nueve objetos no certifica todos los
floats CPU/MPS. La sonda de Claude `16_cpu_tolerance_and_150.py` sobre 25 objetos
reportó máximos 3,099e-6 (v1) y 4,828e-6 (v2.1), con fallos del criterio combinado
atol=rtol=1e-6, aunque ΔRMSE fue <1e-9. En 5A se inspeccionaron código/salida;
no se repitió inferencia ni se amplió retrospectivamente la tolerancia. No se
promete paridad entre dispositivos para toda la población ni otras plataformas.
