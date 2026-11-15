# EUCAIM nnU-Net base — Blackwell-capable (PyTorch cu128).
# Local-only overlay; build context is upstream nnunet-base/:
#
#   docker build \
#     --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g) \
#     -t websub/eucaim-base:cu128 \
#     -f eucaim/docker/base/Dockerfile \
#     /path/to/nnunet-eucaim-docker/nnunet-base
#
# Derived from MIC-DKFZ/nnunet-eucaim-docker nnunet-base (CUDA 11.8 → 12.8).

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

LABEL maintainer="ashis.ravindran@dkfz-heidelberg.de"
LABEL version="1.0.0-cu128"
LABEL description="nnU-Net EUCAIM base with PyTorch cu128 (Blackwell)."
LABEL image.source="https://github.com/MIC-DKFZ/nnunet-eucaim-docker"
LABEL image.version="1.0.0-cu128"
ENV PYTHONUNBUFFERED=1

ARG USER_UID=2323
ARG USER_GID=2323

RUN groupadd -g $USER_GID eucaim && \
    useradd -l -r -u $USER_UID -g eucaim -d /home/eucaim -m -s /usr/sbin/nologin eucaim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    gosu \
    wget \
    python3 \
    python3-pip \
    libx11-6 \
    ca-certificates \
    linux-libc-dev \
    libtiff5-dev \
    libwrap0-dev \
    libxcomposite1 \
    libxcursor1 \
    libxi-dev \
    libxkbcommon-x11-0 \
    mesa-common-dev \
    libssl-dev \
    libglu1-mesa-dev \
    libxt-dev \
    libgomp1 \
    qtbase5-dev \
    qtscript5-dev \
    libqt5svg5-dev \
    libqt5opengl5-dev \
    libqt5xmlpatterns5-dev \
    qtwebengine5-dev \
    qttools5-dev \
    libqt5x11extras5-dev \
    qtxmlpatterns5-dev-tools \
    libqt5webengine-data \
    at-spi2-core \
    libxcb-cursor0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# PyTorch cu128 (Blackwell / sm_120+)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/cu128

RUN pip install --no-cache-dir pydicom==3.0.1 SimpleITK==2.5.2

WORKDIR /app
RUN mkdir -p nnunetv1 scripts mitk

RUN wget --no-check-certificate \
        https://github.com/MITK/MITK/releases/download/v2025.08/MITK-v2025.08-ubuntu_22_04-x86_64.tar.gz \
        -O /app/mitk.tar.gz \
    && tar -xzf /app/mitk.tar.gz --strip 1 -C /app/mitk/ MITK-v2025.08-linux-x86_64 \
    && rm -rf /app/mitk.tar.gz

RUN git clone --branch nnunetv1 https://github.com/MIC-DKFZ/nnUNet.git ./nnunetv1

COPY start_nnunet.sh ./start_nnunet.sh
COPY entrypoint.sh ./entrypoint.sh
COPY nnunetv1.patch ./nnunetv1/nnunetv1.patch
COPY scripts scripts

RUN chown -R root:root /app && chmod 755 ./start_nnunet.sh ./entrypoint.sh
RUN mkdir -p /sandbox

RUN cd nnunetv1 && git apply --ignore-space-change --ignore-whitespace nnunetv1.patch
RUN pip3 install --no-cache-dir -e ./nnunetv1

WORKDIR /home/eucaim

ENV nnUNet_input="/home/eucaim/nnUNet_input"
ENV nnUNet_output="/home/eucaim/nnUNet_output"
ENV RESULTS_FOLDER="/home/eucaim/nnUNet_results_folder"

RUN mkdir -p $nnUNet_input $nnUNet_output $RESULTS_FOLDER \
    && chown eucaim:eucaim $nnUNet_input $nnUNet_output

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["/bin/bash", "/app/start_nnunet.sh"]
