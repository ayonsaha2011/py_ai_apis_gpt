FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
COPY services/text-worker/pyproject.toml ./pyproject.toml
RUN pip install --no-cache-dir uv && uv pip install --system -e .
COPY services/text-worker/app ./app
EXPOSE 8101
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8101"]

