# syntax=docker/dockerfile:1
#
# LewLM portable runtime image (CPU baseline).
#
# This is the promoted way to run LewLM on non-Apple hosts, and the direct fix
# for hosts that cannot load a prebuilt `llama-cpp-python` wheel (e.g. the
# Windows `0xc000001d` / STATUS_ILLEGAL_INSTRUCTION crash on `llama.dll`):
# llama.cpp is compiled here with GGML_NATIVE=OFF so the binary does not bake in
# CPU instructions the host may lack. For NVIDIA GPU acceleration use
# Dockerfile.cuda instead.
#
# The image also carries the two llama.cpp tools LewLM's GGUF conversion path
# shells out to (`convert_hf_to_gguf.py` and `llama-quantize`). The Python
# wheel ships neither — it installs only the shared libraries — so without this
# stage `lewlm convert` reports `requires_install` inside the container exactly
# as it does on a bare host.

ARG PYTHON_VERSION=3.11

# llama.cpp revision used for the conversion tools. Pinned so image builds are
# reproducible and so the GGUF files written here track a known llama.cpp.
ARG LLAMA_CPP_REF=b10698

# `enabled` (default) builds the conversion tools; `disabled` skips that build
# for lean CI images. When disabled, LewLM reports the configured converter as
# not found rather than silently dropping the capability.
ARG CONVERSION_TOOLS=enabled

###############################################################################
# llama.cpp conversion tools: HF->GGUF converter + GGUF quantizer             #
###############################################################################
FROM python:${PYTHON_VERSION}-slim AS llamacpp-tools-enabled

ARG LLAMA_CPP_REF

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch "${LLAMA_CPP_REF}" \
        https://github.com/ggml-org/llama.cpp.git /src/llama.cpp

WORKDIR /src/llama.cpp

# Only `llama-quantize` is needed. Static linking keeps it a single file to
# copy, and GGML_NATIVE=OFF keeps it runnable on hosts lacking newer CPU
# instructions -- the same portability rule the runtime backend follows.
RUN cmake -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_NATIVE=OFF \
        -DBUILD_SHARED_LIBS=OFF \
        -DLLAMA_BUILD_COMMON=ON \
        -DLLAMA_BUILD_TOOLS=ON \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_SERVER=OFF \
        -DLLAMA_BUILD_APP=OFF \
        -DLLAMA_BUILD_UI=OFF \
        -DLLAMA_OPENSSL=OFF \
    && cmake --build build --target llama-quantize -j "$(nproc)"

# `convert_hf_to_gguf.py` imports `conversion` and self-inserts `gguf-py` on
# sys.path relative to its own location, so all three have to stay siblings.
# LewLM's tool resolver recognises that layout and runs the converter with the
# directory as its working root.
RUN mkdir -p /opt/llamacpp-tools \
    && cp convert_hf_to_gguf.py /opt/llamacpp-tools/ \
    && cp -r gguf-py /opt/llamacpp-tools/gguf-py \
    && cp -r conversion /opt/llamacpp-tools/conversion \
    && install -Dm755 build/bin/llama-quantize /opt/llamacpp-tools/llama-quantize

###############################################################################
# Opt-out stage: an empty tool tree so the runtime COPY stays unconditional    #
###############################################################################
FROM python:${PYTHON_VERSION}-slim AS llamacpp-tools-disabled
RUN mkdir -p /opt/llamacpp-tools

FROM llamacpp-tools-${CONVERSION_TOOLS} AS llamacpp-tools

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
COPY --from=llamacpp-tools /opt/llamacpp-tools /opt/llamacpp-tools
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
    HOME=/home/lewlm \
    # Reported by `lewlm doctor` so guidance suits the deployment shape.
    LEWLM_IN_CONTAINER=true \
    # The conversion path shells out to these; both live in the image.
    LEWLM_LLAMACPP_CONVERT_HF_TO_GGUF_PATH=/opt/llamacpp-tools/convert_hf_to_gguf.py \
    LEWLM_LLAMACPP_QUANTIZE_PATH=/opt/llamacpp-tools/llama-quantize \
    # The tool tree is read-only to this user; skip the bytecode write attempts.
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8080
VOLUME ["/data"]

# Reuses the existing `lewlm` readiness surface as a liveness probe.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=5 \
    CMD curl -fsS http://127.0.0.1:8080/v1/health || exit 1

ENTRYPOINT ["lewlm"]
CMD ["serve"]
