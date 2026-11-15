# TorchXRayVision chest X-ray classification — Blackwell-capable (PyTorch cu128).
# Weights are NOT baked in; torchxrayvision downloads into ~/.torchxrayvision
# (mount a host cache volume at /root/.torchxrayvision).
#
#   docker build -t websub/torchxrayvision:cu128 \
#     -f torchxrayvision/docker/Dockerfile torchxrayvision/docker

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/root \
    TXRV_DEVICE=cuda \
    TXRV_WEIGHTS=densenet121-res224-all \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        ca-certificates \
        curl \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && python3 -m venv /opt/venv

# cu128 wheels for RTX 50 / sm_120
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision \
    && pip install torchxrayvision pydicom requests \
    && pip install --index-url https://download.pytorch.org/whl/cu128 --force-reinstall torchvision

WORKDIR /app
COPY entrypoint.py /app/entrypoint.py

RUN mkdir -p /root/.torchxrayvision /data/input /data/output

ENTRYPOINT ["python", "/app/entrypoint.py"]
