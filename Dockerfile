# syntax=docker/dockerfile:1
#
# LewLM portable runtime image (CPU baseline).
#
# This is the recommended cross-platform way to run LewLM, and the direct fix
# for hosts that cannot load a prebuilt `llama-cpp-python` wheel (e.g. the
# Windows `0xc000001d` / STATUS_ILLEGAL_INSTRUCTION crash on `llama.dll`):
# llama.cpp is compiled here with GGML_NATIVE=OFF so the binary does not bake in
# CPU instructions the host may lack. For NVIDIA GPU acceleration use
# Dockerfile.cuda instead.

ARG PYTHON_VERSION=3.11

###############################################################################
# Builder: compile wheels (incl. a CPU-portable llama-cpp-python) into a venv #
###############################################################################
FROM python:${PYTHON_VERSION}-slim AS builder

# Extras installed into the image. Default is a batteries-included local
# serving + conversion + documents image. Override at build time, e.g.:
#   --build-arg EXTRAS=llamacpp   # GGUF serving + conversion only
#   --build-arg EXTRAS=dev        # lightweight CI image (no torch/llama build)
ARG EXTRAS="llamacpp,documents"

# CPU-portable llama.cpp build. GGML_NATIVE=OFF avoids `-march=native`, so the
# compiled backend does NOT require CPU instructions the host might be missing.
# Override (or use Dockerfile.cuda) to add a backend, e.g. -DGGML_CUDA=ON.
ARG LLAMA_CMAKE_ARGS="-DGGML_NATIVE=OFF"

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CMAKE_ARGS="${LLAMA_CMAKE_ARGS}" \
    FORCE_CMAKE=1 \
    # Resolve torch (pulled in by the llamacpp/conversion extras) to the CPU
    # build so the default image stays lean and GPU-free.
    PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu"

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /src
COPY . .

RUN pip install --upgrade pip \
    && pip install ".[${EXTRAS}]"

###############################################################################
# Runtime: slim image carrying just the venv + required shared libraries      #
###############################################################################
FROM python:${PYTHON_VERSION}-slim AS runtime

# Runtime shared libraries:
#  - libgomp1: OpenMP, required by the compiled llama.cpp backend
#  - libpango / libcairo / libgdk-pixbuf / tesseract-ocr: used by the optional
#    `documents` extra (WeasyPrint + OCR); small enough to always include so
#    `.[documents]` works out of the box
#  - curl: used by the container HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libpango-1.0-0 libpangocairo-1.0-0 libcairo2 libgdk-pixbuf-2.0-0 \
        tesseract-ocr \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Non-root runtime user. State (registry, cache, models) lives under /data;
# mount a volume there to persist models and avoid re-downloading.
RUN useradd --create-home --uid 10001 lewlm \
    && mkdir -p /data/models \
    && chown -R lewlm:lewlm /data
USER lewlm

ENV LEWLM_HOST=0.0.0.0 \
    LEWLM_PORT=8080 \
    LEWLM_DATA_DIR=/data \
    HOME=/home/lewlm

EXPOSE 8080
VOLUME ["/data"]

# Reuses the existing `lewlm` readiness surface as a liveness probe.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=5 \
    CMD curl -fsS http://127.0.0.1:8080/v1/health || exit 1

ENTRYPOINT ["lewlm"]
CMD ["serve"]
