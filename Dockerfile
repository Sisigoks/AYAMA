# TRAKSHA image.
#
#   docker build -t traksha .
#   docker run --rm -it -v "$PWD:/work" traksha bash scripts/harness.sh
#   docker run --rm -it traksha python -m traksha.cli doctor
#
# CPU only, and deliberately so. The one stage that could use an accelerator is
# depth inference, and measuring it showed the speed-up does not reach the
# result - an order of magnitude more backbone moved recovered relief from
# 0.05 m to 0.17 m against a true 14.4 m (README §3.2). What does recover
# structure is a scale fitted once over a dataset, which costs milliseconds.
# So there is no CUDA base image to keep current and no driver to match.
#
# rasterio's manylinux wheel bundles GDAL, so no system GDAL packages are
# installed here on purpose: fewer moving parts, and no version skew between
# the GDAL python bindings and the shared library.
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

# Dependencies first, so a code change does not re-download torch. The CPU
# wheel index keeps the image around 1 GB instead of 6.
COPY requirements.txt ./
RUN python -m pip install --upgrade pip wheel \
    && python -m pip install -r requirements.txt \
    && python -m pip install torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install transformers

COPY . .

# Model weights live in a cache volume so a container restart does not
# re-download 1.3 GB of ViT-L. Mount it with:  -v traksha-cache:/cache
VOLUME ["/cache"]

RUN python -c "import traksha.core.types" && python -m traksha.cli backbones

CMD ["python", "-m", "traksha.cli", "doctor"]
