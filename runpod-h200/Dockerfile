FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

LABEL org.opencontainers.image.source="https://github.com/jkpdino/parrot-h200-worker"
LABEL org.opencontainers.image.description="Runpod Serverless benchmark worker for Parrot XL training on NVIDIA H200"

WORKDIR /workspace/parrot

COPY runpod-h200/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY pyproject.toml README.md ./
COPY parrot ./parrot
COPY benchmark_cuda_training.py ./
COPY runpod-h200 ./runpod-h200
RUN pip install --no-cache-dir --no-deps .

ENV PYTHONUNBUFFERED=1
CMD ["python", "-u", "runpod-h200/handler.py"]
