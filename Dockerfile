# UNNAT GPU image.
#
#   docker build -t unnat:gpu .
#   docker run --gpus all --rm -it -v "$PWD:/work" unnat:gpu bash scripts/harness.sh
#   docker run --gpus all --rm -it unnat:gpu python -m unnat.cli doctor
#
# rasterio's manylinux wheel bundles GDAL, so no system GDAL packages are
# installed here on purpose: fewer moving parts, and no version skew between
# the GDAL python bindings and the shared library.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3

WORKDIR /work

# Dependencies first, so a code change does not re-download torch.
COPY requirements.txt ./
RUN python -m pip install --upgrade pip wheel \
    && python -m pip install -r requirements.txt \
    && python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 \
    && python -m pip install transformers

COPY . .

# Model weights live in a cache volume so a container restart does not re-download
# 1.3 GB of ViT-L. Mount it with:  -v unnat-cache:/cache
VOLUME ["/cache"]

RUN python -c "import unnat.core.types" && python -m unnat.cli backbones

CMD ["python", "-m", "unnat.cli", "doctor"]
