# CPU / harici LLM sunucusu (Ollama, vLLM) ile çalışan hafif imaj.
# Yerel GGUF + CUDA için: --build-arg EXTRAS=llamacpp ve CUDA tabanlı imaj (README > GPU).
FROM python:3.10-slim

ARG EXTRAS=""
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MSP_DATA_DIR=/data

WORKDIR /app
COPY pyproject.toml README.md ./
COPY agent ./agent
COPY training ./training
RUN pip install --no-cache-dir -e ".${EXTRAS:+[$EXTRAS]}"

RUN useradd -r -u 1000 agent && mkdir -p /data && chown agent /data
USER agent
VOLUME /data
EXPOSE 8100
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8100/health')"
CMD ["uvicorn", "agent.service:app", "--host", "0.0.0.0", "--port", "8100", "--workers", "1"]
