# 10 · Embeddings + búsqueda semántica (FAISS) — ✅ COMPLETADO (2026-08-16/17)

> **Bloque:** Nivel 3 · **Tiempo real:** ~2.5 h (el indexado tomó 5.1 min, no los 30–60 estimados) · **Dependía de:** 02 (checkpoint v2.1), 07 · **Entregable:** `find_similar_spectra` funcionando sobre un índice FAISS de 15k espectros + UMAP con gradiente de z visible a simple vista en los README de ambos repos.

## Qué quedó hecho

- **`DESIFoundationModel.encode()`** (`src/desi_fm/model.py`) + **`embed_spectrum()`** (`src/desi_fm/predict.py`) en [`Julian0444/desi-spectra-fm`](https://github.com/Julian0444/desi-spectra-fm) (commit `7a2f8ab`, CI verde, tests 24→26): mean-pooling de los tokens espectrales válidos, contexto completo (sin masking), determinista, 512-d. El preprocesado se factorizó en `_prepare_inputs()` compartido con `predict_spectrum` — cero duplicación.
- **`scripts/build_index.py`** en [`Julian0444/spectra-copilot`](https://github.com/Julian0444/spectra-copilot) (commit `75997c5`, CI verde): 15.000 espectros streaming → embeddings L2-normalizados → `faiss.IndexFlatIP` (coseno). Corrió en **5.1 min** en MPS (48 espectros/s). El índice cubre los **primeros 15k del lado de entrenamiento** del split (held-out = skip 80k) → las consultas con `examples/heldout_*.npz` son out-of-index por construcción.
- **Índice publicado en el Hub**: `faiss/spectra.faiss` (30 MB) + `faiss/spectra_meta.npz` en [jirustaroure/desi-spectra-fm](https://huggingface.co/jirustaroure/desi-spectra-fm/tree/main/faiss); la model card lo documenta. Resolución en `tools.py`: `DESI_FM_INDEX_DIR` → `data/` del repo → descarga del Hub (mismo patrón que el checkpoint). `data/` quedó gitignoreado.
- **`find_similar_spectra`** integrada en las 3 capas: `tools.py` (impl con `k` clampeado, `neighbor_z_range`/`neighbor_z_median` para que el agente cite rangos), `agent.py` (`@beta_tool` + regla 4 del SYSTEM: los vecinos complementan la verificación por líneas, **nunca la reemplazan**) y `mcp_server.py` (tool + `instructions` actualizadas). Suite **17→20 tests** offline (fixture `tiny_index` en `conftest.py`: el embedding real del held-out entre 32 vectores random → rank 1 = él mismo con sim ≈ 1.0).
- **`scripts/plot_umap.py`** → **`docs/img/umap_z.png`** (UMAP coseno, n_neighbors=30, viridis sobre log(1+z) con ticks legibles en z): gradiente de z **visible a simple vista** — violeta (z≈0) → azul (z≈0.5–1) → verde (z≈2–3) — más una isla separada de z bajo. Linkeada en el README de **ambos** repos con la frase clave: nadie le enseñó al modelo a ordenarse por redshift.

## Verificación (consultas reales sobre el índice de 15k)

| query (held-out) | z_true | vecinos (k=5) | lectura |
|---|---|---|---|
| `heldout_z020` | 0.2036 | z ∈ [0.187, 0.202], sims 0.989–0.992 | cluster apretado alrededor del z verdadero — apoya el 0.204 verificado por líneas **contra** el z_pred_map 0.2267 del propio modelo |
| `heldout_z287` | 2.866 | mediana 2.898, sims 0.971–0.987 | coherente también a z alto |
| `heldout_lowconf_z157` | 1.574 | z dispersos [0.13, 1.39] con sims ~0.997 | embedding no distintivo → duda, consistente con conf 0.18 — el fallo se señala honesto |
| `trap_single_line` | — | sims máx 0.898 (vs ~0.99 de espectros reales) | lejos del manifold: similitud baja = "no confíes en estos vecinos" |

La consulta de la DoD se ejecutó además **por la capa MCP real** (`mcp.call_tool("find_similar_spectra", ...)` sobre `heldout_z020`) con idéntico resultado — el server registrado en Claude Code hereda la tool sin tocar el registro (mismo archivo).

## Adaptaciones vs el plan original

1. **faiss + torch se matan entre sí en macOS** (dos `libomp.dylib`, OMP Error #15: el proceso aborta en la primera región paralela de faiss — así aparecieron los diálogos "Python se cerró inesperadamente"). Fix centralizado en `tools._faiss()`: `KMP_DUPLICATE_LIB_OK=TRUE` (workaround documentado por Intel/LLVM) + `faiss.omp_set_num_threads(1)` (search sobre 15k flat es microsegundos igual). `build_index.py` además guarda embeddings/meta **antes** de tocar faiss, para que un abort no pierda la pasada de streaming.
2. **El índice vive en el Hub, no en git**: 30 MB binarios no van a un repo de código; `_index_paths()` los baja y cachea igual que el checkpoint. Bonus: los embeddings crudos quedan en `data/spectra_embeddings.npy` para reuso (UMAP, análisis).
3. **Sin `shuffle_buffer`** en el indexado: la membresía del índice es idéntica (`take` corre antes de `shuffle` en `HFDESISpectra`) y el orden determinista hace el rebuild reproducible.
4. **Corrida demo del agente con la tool nueva NO se hizo**: el crédito de la API está agotado ("credit balance is too low", medido 2026-08-17 ~06:40 UTC — la memoria decía ~US$ 4.50 restantes; el saldo real era cero). La integración quedó igualmente verificada offline: test `agent.find_similar_spectra.call(...) ≡ impl`, SYSTEM testeado por contrato, y la consulta real por la capa MCP. Cuando haya crédito: una corrida haiku (~$0.013) sobre `heldout_z020` debería citar "the 5 nearest neighbors have z between 0.19 and 0.20".
5. **UMAP con `random_state=42`** (reproducible) y colorbar que codifica log(1+z) pero se lee en z plano — más honesto que el label "log(1+z)" del plan.

## Definición de hecho

- [x] `embed_spectrum` con test de sanidad (mismo z → más similar) — + determinismo y shape; 26/26 en el repo principal, CI verde.
- [x] Índice de ≥ 10k espectros construido y guardado — 15.000 × 512-d, local en `data/` y publicado en el Hub.
- [x] `find_similar_spectra` integrada al agente y al MCP server; una consulta devuelve vecinos con z coherentes — tabla de arriba; consulta MCP real con z ∈ [0.187, 0.202] para z_true 0.204. (Corrida API del agente pendiente de crédito — adaptación nº 4.)
- [x] `docs/img/umap_z.png` con gradiente de z visible, linkeado en el README — en ambos repos.
- [x] Commit + tracker — `7a2f8ab` (código desi-fm) + `75997c5` (spectra-copilot) + cierre documental; CI verde en ambos.

## Si algo falla (aprendido)

- **"Python se cerró inesperadamente" / exit 139 / OMP Error #15:** es el doble libomp de torch+faiss. Importar faiss **siempre** vía `tools._faiss()`; nunca `import faiss` directo en código que también toca torch.
- **faiss-cpu no instala en Python 3.9:** confirmado el consejo del plan — todo corre en el venv 3.12 de spectra-copilot.
- **El streaming no fue el cuello de botella** (5 min, no 30–60): los shards estaban calientes en el CDN/cache; si un rebuild diera lento, `--n 8000` demuestra lo mismo.
- **El UMAP salió con estructura al primer intento** con el checkpoint v2.1; si saliera sin estructura, revisar que no se estén usando pesos random (el fallback del Hub descarga el checkpoint correcto).
