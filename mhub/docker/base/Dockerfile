# Local MHub base with PyTorch cu128 (Blackwell / RTX 50).
# Do NOT push this image to a public registry.
#
#   docker build -t websub/mhub-base:cu128 \
#     -f mhub/docker/base/Dockerfile mhub/docker/base

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

LABEL org.opencontainers.image.title="websub/mhub-base" \
      org.opencontainers.image.description="Local MHub base with torch cu128 — not for publication" \
      org.opencontainers.image.source="https://github.com/MHubAI/models"

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Amsterdam \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    PATH=/root/.local/bin:/root/.cargo/bin:$PATH

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && mkdir -p /app/data /app/data/input_data /app/data/output_data \
        /app/data/reference_data /app/utility/config /app/xmodules /app/xcollections

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        wget \
        curl \
        jq \
        gcc \
        g++ \
        unzip \
        sudo \
        git \
        tree \
        ca-certificates \
        libopenslide-dev \
        libvips-dev \
        plastimatch \
        ffmpeg \
        libsm6 \
        libxext6 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# uv + Python 3.11 venv (matches upstream mhubai/base)
ADD https://astral.sh/uv/0.4.4/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh \
    && uv venv -p 3.11 \
    && uv pip install --no-cache pip \
    && ln -sf "$(uv run which pip)" /usr/bin/pip3 \
    && ln -sf "$(uv run which python)" /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python

# Blackwell-capable PyTorch first, then mhubio
RUN uv pip install --index-url https://download.pytorch.org/whl/cu128 torch \
    && uv pip install "git+https://github.com/MHubAI/mhubio.git" \
    && uv tool install "git+https://github.com/LennyN95/medcmp.git" || true

# Sparse-checkout upstream base utilities (bin, buildutils, configs)
RUN git init \
    && git sparse-checkout set "base/buildutils" "base/bin" "base/configs" "base/collections.json" \
    && git fetch https://github.com/MHubAI/models.git main \
    && git merge FETCH_HEAD \
    && mv base/buildutils . \
    && chmod +x base/bin/* \
    && cp base/bin/* /usr/bin/ \
    && cp -r base/configs/* /app/utility/config \
    && cp base/collections.json /app/xcollections/collections.json \
    && rm -rf base .git

ENTRYPOINT ["mhub.run"]
CMD ["--utility"]
