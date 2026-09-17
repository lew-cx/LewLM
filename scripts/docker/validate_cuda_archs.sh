#!/usr/bin/env sh
# Validate the requested CUDA architectures against the selected toolkit
# *before* a long llama.cpp CUDA build starts.
#
# CMake accepts any CUDA_ARCHITECTURES value at configure time; nvcc rejects an
# unsupported one only once compilation reaches the first .cu file, which in a
# container build is many minutes in. This check asks the toolkit which
# architectures it can actually target (`nvcc --list-gpu-arch`) and fails fast
# when one of the requested SMs is missing -- the Blackwell-on-CUDA-12.6 case.
#
# Usage:
#   validate_cuda_archs.sh "75;80;86;89"
#   validate_cuda_archs.sh "120" --list-file /path/to/nvcc-list-gpu-arch.txt
#
# `--list-file` substitutes a saved `nvcc --list-gpu-arch` output for the real
# toolkit so the script itself can be tested on hosts without CUDA. Values may
# be separated by `;` or `,`; `native` and `all*` are refused because they make
# the image depend on the build host instead of on a documented target.

set -eu

archs=${1:-}
list_file=""
if [ "${2:-}" = "--list-file" ]; then
    list_file=${3:-}
fi

if [ -z "$archs" ]; then
    echo "validate_cuda_archs: no CUDA architectures given" >&2
    exit 2
fi

if [ -n "$list_file" ]; then
    supported=$(cat "$list_file")
else
    if ! command -v nvcc >/dev/null 2>&1; then
        echo "validate_cuda_archs: nvcc is not on PATH; cannot validate ${archs}" >&2
        exit 2
    fi
    supported=$(nvcc --list-gpu-arch)
fi

status=0
for arch in $(printf '%s' "$archs" | tr ';,' '  '); do
    case "$arch" in
        native|all|all-major)
            echo "validate_cuda_archs: '${arch}' is not allowed for a distributable image; name the target SM(s) explicitly" >&2
            status=1
            continue
            ;;
    esac
    # CMake accepts suffixed forms such as 89-real / 89-virtual; the toolkit
    # list is unsuffixed.
    number=$(printf '%s' "$arch" | sed 's/-.*$//')
    case "$number" in
        ''|*[!0-9]*)
            echo "validate_cuda_archs: '${arch}' is not a CUDA architecture number" >&2
            status=1
            continue
            ;;
    esac
    if printf '%s\n' "$supported" | grep -qx "compute_${number}"; then
        echo "validate_cuda_archs: compute_${number} is supported by the selected toolkit"
    else
        echo "validate_cuda_archs: compute_${number} is NOT supported by the selected toolkit" >&2
        status=1
    fi
done

if [ "$status" -ne 0 ]; then
    echo "validate_cuda_archs: supported architectures were:" >&2
    printf '%s\n' "$supported" | sed 's/^/  /' >&2
    echo "validate_cuda_archs: pick a CUDA base image whose toolkit supports every requested SM, or drop the unsupported SM" >&2
fi
exit "$status"
