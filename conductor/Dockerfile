FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY pyproject.toml ./
COPY conductor ./conductor
COPY config ./config
RUN pip install ".[otel]" && useradd -r -u 10001 conductor
USER 10001
ENV CONDUCTOR_CONFIG=/app/config/pool.yaml PORT=8080
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=3s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/healthz').status==200 else 1)"
CMD ["conductor"]
