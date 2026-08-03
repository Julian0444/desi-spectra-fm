# API REST del modelo (plan 06): imagen CPU-only, el checkpoint (~104 MB) se
# baja del Hub en el primer request y queda cacheado en el contenedor.
FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[api]"
ENV HF_HOME=/tmp/hf
EXPOSE 7860
CMD ["uvicorn", "desi_fm.api:app", "--host", "0.0.0.0", "--port", "7860"]
