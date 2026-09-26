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
# Image flavors (--build-arg IMAGE_FLAVOR=...):
#   bridge   base package only: LewLM in front of Ollama / another loopback
#            server. No llama.cpp, no torch, no document libraries.
#   serving  bridge + the llama.cpp GGUF runtime (`llamacpp_runtime` extra).
#   full     serving + HF->GGUF conversion dependencies + documents/OCR
#            (`llamacpp` + `documents` extras). The default, so the existing
#            `docker build -t lewlm:cpu .` keeps producing the batteries-included
#            image.
#
# Layering is arranged so that an app-only edit never recompiles anything:
#   1. toolchain      apt build tools, ccache, an empty venv
#   2. deps           installs requirements/<flavor>.txt (or a lock named by
#                     DEPENDENCY_INPUT). llama-cpp-python is built once here
#                     under ccache, or taken from a verified wheel index.
#   3. app            copies the source and installs the LewLM wheel --no-deps
#   4. runtime        slim image with the venv and the llama.cpp tools
# Only the dependency input file and the build arguments feed layer 2, so
# editing src/ invalidates layer 3 alone. Package downloads and compiler
# output live in BuildKit cache mounts, outside the final image.
#
# The full image also carries the two llama.cpp tools LewLM's GGUF conversion
# path shells out to (`convert_hf_to_gguf.py` and `llama-quantize`). The Python
# wheel ships neither -- it installs only the shared libraries -- so without
# this stage `lewlm convert` reports `requires_install` inside the container
# exactly as it does on a bare host.

ARG PYTHON_VERSION=3.11

# llama.cpp revision used for the conversion tools. Pinned so image builds are
# reproducible and so the GGUF files written here track a known llama.cpp.
ARG LLAMA_CPP_REF=b10698

# `enabled` (default) builds the conversion tools; `disabled` skips that build
# for lean images. When disabled, LewLM reports the configured converter as not
# found rather than silently dropping the capability. Pair `disabled` with the
# `bridge` and `serving` flavors: they have no converter dependencies anyway.
ARG CONVERSION_TOOLS=enabled

ARG IMAGE_FLAVOR=full

# Parallel compile jobs for every native build in this file. 4 is deliberate:
# unbounded `nproc` exhausts memory on small CI runners and laptops building
# llama.cpp. Raise it on a large build host.
ARG BUILD_JOBS=4

###############################################################################
# llama.cpp conversion tools: HF->GGUF converter + GGUF quantizer             #
###############################################################################
FROM python:${PYTHON_VERSION}-slim AS llamacpp-tools-enabled

ARG LLAMA_CPP_REF
ARG BUILD_JOBS

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build git ccache \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch "${LLAMA_CPP_REF}" \
        https://github.com/ggml-org/llama.cpp.git /src/llama.cpp

WORKDIR /src/llama.cpp
ENV CCACHE_DIR=/root/.cache/ccache

# Only `llama-quantize` is needed. Static linking keeps it a single file to
# copy, and GGML_NATIVE=OFF keeps it runnable on hosts lacking newer CPU
# instructions -- the same portability rule the runtime backend follows.
# Quantization is CPU work, so this stage is identical for the CUDA image and
# its ccache is keyed only on the llama.cpp revision.
RUN --mount=type=cache,id=lewlm-ccache-tools-${LLAMA_CPP_REF},target=/root/.cache/ccache \
    cmake -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
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
    && cmake --build build --target llama-quantize -j "${BUILD_JOBS}" \
    && ccache --show-stats

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
# Toolchain: compilers, ccache, and an empty venv. Changes rarely.            #
###############################################################################
FROM python:${PYTHON_VERSION}-slim AS toolchain

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build git ccache \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}" \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CCACHE_DIR=/root/.cache/ccache

RUN --mount=type=cache,id=lewlm-pip,target=/root/.cache/pip \
    pip install --upgrade pip

###############################################################################
# Dependencies: keyed on requirements/<flavor>.txt, never on src/             #
###############################################################################
FROM toolchain AS deps

ARG PYTHON_VERSION
ARG IMAGE_FLAVOR
ARG BUILD_JOBS

# Dependency-layer input. The default is the unpinned export from
# pyproject.toml; point it at a lock from scripts/docker/lock_dependencies.sh
# (requirements/locks/*.txt) for a pinned, hashed build.
ARG DEPENDENCY_INPUT="requirements/${IMAGE_FLAVOR}.txt"

# CPU-portable llama.cpp build. GGML_NATIVE=OFF avoids `-march=native`, so the
# compiled backend does NOT require CPU instructions the host might be missing.
# A host-tuned local build is an explicit opt-in: -DGGML_NATIVE=ON. Other
# backends (e.g. -DGGML_VULKAN=ON) go here too; set LLAMA_BUILD_EXPECT to match.
ARG LLAMA_CMAKE_ARGS="-DGGML_NATIVE=OFF"

# Prebuilt-wheel path. When set (for example
# https://abetlen.github.io/llama-cpp-python/whl/cpu) llama-cpp-python is
# installed only as a binary wheel from that index and the source build is
# skipped entirely. Wheel availability is release-specific; an index without a
# wheel for this Python/architecture fails the build instead of silently
# compiling. The installed flavor is verified in the app stage regardless.
ARG LLAMA_CPP_PYTHON_WHEEL_INDEX=""

# Where torch resolves from when the flavor needs it (conversion). The CPU
# index keeps the default image GPU-free; the bridge/serving flavors never
# install torch at all.
ARG TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"

# Compiler-cache scope. ccache hashes compiler, flags, and sources itself, so
# this only has to separate incompatible toolchains; it is combined with the
# build flags and Python ABI below for the wheel directory.
ARG CACHE_SCOPE="cpu-py${PYTHON_VERSION}"

ENV CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS} \
    MAX_JOBS=${BUILD_JOBS} \
    PIP_EXTRA_INDEX_URL=${TORCH_INDEX_URL}

WORKDIR /build
COPY ${DEPENDENCY_INPUT} ./requirements.txt

# llama-cpp-python first, on its own. Source builds go through `pip wheel`
# with a cache directory keyed on scope + CMake flags + Python ABI, so pip can
# never hand back a wheel compiled with different flags; ccache is what makes
# a re-link after a flag change cheap. The built wheel is then installed with
# its own dependencies (numpy, jinja2, diskcache).
RUN --mount=type=cache,id=lewlm-pip,target=/root/.cache/pip \
    --mount=type=cache,id=lewlm-pip-native-${CACHE_SCOPE},target=/root/.cache/pip-native \
    --mount=type=cache,id=lewlm-ccache-${CACHE_SCOPE},target=/root/.cache/ccache \
    set -eu; \
    spec="$(tr -d '\r' < requirements.txt | grep -i '^llama[-_]cpp[-_]python' | head -n1 | sed 's/[[:space:]]*;.*$//; s/[[:space:]]*\\$//')"; \
    if [ -z "$spec" ]; then echo "flavor ${IMAGE_FLAVOR}: no llama-cpp-python in the dependency input; skipping"; exit 0; fi; \
    mkdir -p /wheels/native; \
    if [ -n "${LLAMA_CPP_PYTHON_WHEEL_INDEX}" ]; then \
        echo "llama-cpp-python: prebuilt wheel from ${LLAMA_CPP_PYTHON_WHEEL_INDEX}"; \
        pip download --only-binary llama-cpp-python --no-deps \
            --extra-index-url "${LLAMA_CPP_PYTHON_WHEEL_INDEX}" --dest /wheels/native "$spec"; \
    else \
        abi="$(python -c 'import sysconfig; print(sysconfig.get_config_var("SOABI"))')"; \
        native_key="$(printf '%s|%s|%s' "${CACHE_SCOPE}" "${LLAMA_CMAKE_ARGS}" "$abi" | sha256sum | cut -c1-16)"; \
        echo "llama-cpp-python: source build (CMAKE_ARGS='${LLAMA_CMAKE_ARGS}', jobs=${BUILD_JOBS}, cache key ${native_key})"; \
        CMAKE_ARGS="${LLAMA_CMAKE_ARGS} -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache" \
        FORCE_CMAKE=1 \
        pip --cache-dir "/root/.cache/pip-native/${native_key}" wheel --no-deps \
            --no-binary llama-cpp-python --wheel-dir /wheels/native "$spec"; \
        ccache --show-stats; \
    fi; \
    pip install /wheels/native/llama_cpp_python-*.whl

# Everything else in the flavor. llama-cpp-python is already satisfied.
RUN --mount=type=cache,id=lewlm-pip,target=/root/.cache/pip \
    pip install -r requirements.txt \
    && pip check

###############################################################################
# App: the only layer that changes when src/ changes                          #
###############################################################################
FROM deps AS app

ARG IMAGE_FLAVOR

# What the installed llama.cpp must report: `cpu` for the portable default,
# `gpu` when LLAMA_CMAKE_ARGS enabled an accelerator, `any` to only require
# that it loads. A bridge image has nothing to verify.
ARG LLAMA_BUILD_EXPECT="cpu"

# Legacy `EXTRAS` support: additional pyproject extras layered on top of the
# flavor. `--build-arg EXTRAS=dev` still yields an image with pytest; the
# flavor's dependencies are already present so only the difference resolves.
ARG EXTRAS=""

WORKDIR /src
# Exactly the files setuptools needs to build the wheel (readme + license are
# referenced from pyproject.toml), then the package itself.
COPY pyproject.toml README.md LICENSE MANIFEST.in ./
COPY src ./src
COPY scripts/verify_llamacpp_build.py ./scripts/verify_llamacpp_build.py

RUN --mount=type=cache,id=lewlm-pip,target=/root/.cache/pip \
    set -eu; \
    pip wheel --no-deps --wheel-dir /wheels/app . ; \
    wheel="$(ls /wheels/app/lewlm-*.whl)"; \
    if [ -n "${EXTRAS}" ]; then pip install "${wheel}[${EXTRAS}]"; else pip install --no-deps "${wheel}"; fi; \
    pip check

# Prove the image is what it claims: the flavor's package set is honest and
# the native backend is the intended flavor. A CPU wheel in a GPU image, or
# torch in a serving image, fails here instead of at first request.
RUN set -eu; \
    case "${IMAGE_FLAVOR}" in \
        bridge) python -c "import importlib.util as u; bad=[m for m in ('llama_cpp','torch','transformers') if u.find_spec(m)]; assert not bad, f'bridge image must not contain {bad}'" ;; \
        serving) python -c "import importlib.util as u; bad=[m for m in ('torch','transformers','weasyprint') if u.find_spec(m)]; assert not bad, f'serving image must not contain {bad}'" \
                 && python scripts/verify_llamacpp_build.py --expect "${LLAMA_BUILD_EXPECT}" ;; \
        full) python scripts/verify_llamacpp_build.py --expect "${LLAMA_BUILD_EXPECT}" ;; \
        *) echo "unknown IMAGE_FLAVOR '${IMAGE_FLAVOR}' (expected bridge, serving, or full)" >&2; exit 1 ;; \
    esac; \
    lewlm --help >/dev/null

###############################################################################
# Runtime: slim image carrying just the venv + required shared libraries      #
###############################################################################
FROM python:${PYTHON_VERSION}-slim AS runtime

ARG IMAGE_FLAVOR

# Runtime shared libraries:
#  - libgomp1: OpenMP, required by the compiled llama.cpp backend
#  - curl: used by the container HEALTHCHECK
#  - full only: libpango / libcairo / libgdk-pixbuf / tesseract-ocr for the
#    `documents` extra (WeasyPrint + OCR)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && if [ "${IMAGE_FLAVOR}" = "full" ]; then \
        apt-get install -y --no-install-recommends \
            libpango-1.0-0 libpangocairo-1.0-0 libcairo2 libgdk-pixbuf-2.0-0 \
            tesseract-ocr; \
    fi \
    && rm -rf /var/lib/apt/lists/*

COPY --from=app /opt/venv /opt/venv
COPY --from=llamacpp-tools /opt/llamacpp-tools /opt/llamacpp-tools
ENV PATH="/opt/venv/bin:${PATH}"

# Non-root runtime user. State (registry, cache, models) lives under /data;
# mount a volume there to persist models and avoid re-downloading. `lewlm
# doctor` verifies that the data directory is writable by this user.
RUN useradd --create-home --uid 10001 lewlm \
    && mkdir -p /data/models /data/cache \
    && chown -R lewlm:lewlm /data
USER lewlm

ENV LEWLM_HOST=0.0.0.0 \
    LEWLM_PORT=8080 \
    LEWLM_DATA_DIR=/data \
    HOME=/home/lewlm \
    # Reported by `lewlm doctor` so guidance suits the deployment shape.
    LEWLM_IN_CONTAINER=true \
    LEWLM_IMAGE_FLAVOR=${IMAGE_FLAVOR} \
    # The conversion path shells out to these; both live in the full image.
    LEWLM_LLAMACPP_CONVERT_HF_TO_GGUF_PATH=/opt/llamacpp-tools/convert_hf_to_gguf.py \
    LEWLM_LLAMACPP_QUANTIZE_PATH=/opt/llamacpp-tools/llama-quantize \
    # The tool tree is read-only to this user; skip the bytecode write attempts.
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8080
VOLUME ["/data"]

# Liveness through the existing `lewlm` readiness surface, plus a network check:
# the curl runs inside the container and reaches 127.0.0.1 even when the
# container was never attached to a network (e.g. its published host port was
# already taken), so on its own it would report healthy for a server nothing
# outside can reach. /proc/net/dev must list an interface other than `lo`.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=5 \
    CMD awk -F: 'NR > 2 && $1 !~ /^ *lo$/ { found = 1 } END { exit !found }' /proc/net/dev \
        && curl -fsS http://127.0.0.1:8080/v1/health || exit 1

ENTRYPOINT ["lewlm"]
CMD ["serve"]
