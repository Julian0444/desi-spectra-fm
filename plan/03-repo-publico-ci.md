# 03 · Repo público en GitHub + CI

> **Bloque:** Nivel 1 · **Tiempo:** 1.5–2 h · **Depende de:** — (no hace falta esperar la v2) · **Entregable:** repo público con GitHub Actions en verde

## Objetivo

Convertir la carpeta local en un repo público limpio, clonable y con CI. Es el prerequisito de todos los links del portfolio.

## Pasos

### 1. Sacar lo que no va al repo público

```bash
cd "/Users/jirustaroure/Desktop/FINAL PROJECT DEEP LEARNING"
mkdir -p ~/Documents/PHYS303-privado
mv docs.zip "docs to finish omnicursor.zip" PHYS303_Final-Project_20266.docx ~/Documents/PHYS303-privado/
mv PHYS303_Final-Project_20266.pdf ~/Documents/PHYS303-privado/   # material del profesor: fuera del repo público
rm -rf runs/desi_500 runs/desi_500_zw20 runs/desi_debug_* runs/desi_tiny* runs/desi_two_examples_exit runs/smoke*
```

(La progresión de esos runs ya está documentada en la tabla del `DELIVERABLE.md`; los directorios no aportan.)

Como el PDF de la consigna se va, verificar que el README no la referencie como archivo local — si hace falta, reemplazar por una frase que resuma la consigna.

### 2. `.gitignore`

```gitignore
__pycache__/
*.egg-info/
.pytest_cache/
.ipynb_checkpoints/
.DS_Store
external/
*.zip
*.docx
# checkpoints nunca a git; artefactos livianos del run final sí
runs/*
!runs/desi_50k_big/
!runs/desi_150k_classhead/
runs/desi_50k_big/*.pt
runs/desi_150k_classhead/*.pt
```

Con eso quedan versionados `config.json`, `metrics.jsonl`, `predictions.csv`, `reconstructions.npz` (los insumos offline del notebook, ~3 MB por run) y ningún `.pt`.

### 3. Iniciar git y crear el repo

```bash
git init -b main
git add -A
git status          # revisar: no debe aparecer ningún .pt, zip, docx ni external/
git commit -m "DESI spectra foundation model: training, evaluation notebook, inference"

# gh CLI (si falta: brew install gh && gh auth login)
gh repo create desi-spectra-fm --public --source . --push
gh repo edit --description "26M-parameter foundation model for DESI spectra — masked-token pretraining with joint redshift learning" \
  --add-topic deep-learning --add-topic transformers --add-topic astronomy \
  --add-topic foundation-models --add-topic pytorch --add-topic self-supervised-learning
```

### 4. CI — `.github/workflows/ci.yml`

```yaml
name: ci
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install torch --index-url https://download.pytorch.org/whl/cpu
      - run: pip install -e . pytest
      - run: pytest -q
```

```bash
mkdir -p .github/workflows   # crear el archivo de arriba
git add .github && git commit -m "ci: run unit tests on push" && git push
```

Cuando el workflow esté verde, badge al tope del `README.md`:

```markdown
![ci](https://github.com/TU_USUARIO/desi-spectra-fm/actions/workflows/ci.yml/badge.svg)
```

### 5. Prueba de clon limpio (lo que haría un reclutador técnico)

```bash
cd /tmp && git clone https://github.com/TU_USUARIO/desi-spectra-fm && cd desi-spectra-fm
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
pytest -q
```

## Definición de hecho

- [ ] `git log` limpio, sin `.pt`/zips/docx/external en el repo (verificar con `git ls-files | grep -E "\.pt$|\.zip$|\.docx$|external/"` → vacío).
- [ ] Actions en verde y badge visible en el README.
- [ ] El clon limpio en `/tmp` instala y pasa `pytest`.
- [ ] El notebook se ve renderizado con sus plots en la página de GitHub.
- [ ] Tracker actualizado (`plan/README.md`).

## Si algo falla

- **`gh` pide auth:** `gh auth login` → GitHub.com → HTTPS → browser.
- **Actions falla instalando torch:** fijar versión: `pip install "torch==2.4.*" --index-url https://download.pytorch.org/whl/cpu`.
- **El push rechaza archivos grandes:** algún `.pt` se coló antes del `.gitignore` — `git rm --cached <archivo>` y amend del commit (la historia es nueva, se puede reescribir tranquilo).
- **Nombre de carpeta con espacios molesta para algo:** el nombre del repo remoto ya es `desi-spectra-fm`; si querés, renombrá la carpeta local después de este plan (nada depende de la ruta vieja salvo tu sesión de terminal).
