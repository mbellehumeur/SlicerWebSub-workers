# Orchid /analyze adapter — TorchXRayVision on CPU.
# Weights download into ~/.torchxrayvision (mount host cache).
#
#   docker build -t websub/orchid-adapter-txrv:cpu \
#     -f orchid/adapters/Dockerfile orchid/adapters
#
#   docker run -d --name orchid-adapter-txrv -p 9201:9201 \
#     -v /var/lib/websub-workers/models/torchxrayvision:/root/.torchxrayvision \
#     websub/orchid-adapter-txrv:cpu

FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    TXRV_DEVICE=cpu \
    TXRV_WEIGHTS=densenet121-res224-all \
    PORT=9201 \
    HOME=/root

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip install --upgrade pip \
    && pip install \
        "fastapi>=0.110" \
        "uvicorn[standard]>=0.27" \
        "pydantic>=2" \
        torch \
        torchvision \
        torchxrayvision \
        pillow \
        numpy

COPY app.py /app/app.py
RUN mkdir -p /root/.torchxrayvision

EXPOSE 9201
CMD ["python", "/app/app.py"]
