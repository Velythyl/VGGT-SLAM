# CUDA 12.1 matches the CUDA runtime bundled with the upstream PyTorch 2.3.1
# wheels. The NVIDIA Container Toolkit exposes the host GPU when the image is
# run with Podman's `--device nvidia.com/gpu=all` option.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ARG UV_VERSION=0.9.18
ENV DEBIAN_FRONTEND=noninteractive \
    VGGT_SLAM_ROOT=/opt/vggt-slam \
    VIRTUAL_ENV=/opt/vggt-slam/.venv \
    HF_HOME=/root/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/vggt-slam/.venv/bin:/root/.local/bin:$PATH

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        libegl1 \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && curl --fail --location --silent --show-error https://astral.sh/uv/${UV_VERSION}/install.sh | sh

WORKDIR ${VGGT_SLAM_ROOT}

COPY requirements.txt setup.py ./
RUN uv python install 3.11 \
    && uv venv --seed --python 3.11 ${VIRTUAL_ENV} \
    && uv pip install --python ${VIRTUAL_ENV}/bin/python -r requirements.txt

COPY . ${VGGT_SLAM_ROOT}

# Open-set detection imports these projects lazily, but including them keeps the
# container feature-equivalent to setup.sh while leaving model checkpoints to
# Hugging Face's runtime cache.
RUN git clone --depth 1 https://github.com/Dominic101/salad.git third_party/salad \
    && git clone --depth 1 https://github.com/MIT-SPARK/VGGT_SPARK.git third_party/vggt \
    && git clone --depth 1 https://github.com/facebookresearch/perception_models.git third_party/perception_models \
    && git clone --depth 1 https://github.com/facebookresearch/sam3.git third_party/sam3 \
    && uv pip install --python ${VIRTUAL_ENV}/bin/python \
        -e third_party/salad \
        -e third_party/vggt \
        -e third_party/perception_models \
        -e third_party/sam3 \
        -e . \
    && python -c "import gtsam, torch, vggt, vggt_slam; assert torch.version.cuda == '12.1'"

WORKDIR /work
ENTRYPOINT ["python", "/opt/vggt-slam/main.py"]
CMD ["--help"]
