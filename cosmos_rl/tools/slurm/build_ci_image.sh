#!/usr/bin/env bash
# =============================================================================
# Build the cosmos-rl CI image and import it to an enroot .sqsh for Slurm.
# =============================================================================
# This mirrors the base image built by the GitHub Actions CI
# (.github/workflows/build-and-test.yaml) and produces a pyxis/enroot .sqsh
# that the standalone `cosmos_rl_ci_job.sh` launcher consumes via
# `srun --container-image`.
#
# Two steps:
#   1. docker build  -> a local docker image (skip with --no-build)
#   2. enroot import -> a .sqsh file on a shared filesystem
#
# NOTE: building (steps above) needs a host with BOTH `docker` and `enroot`
#       (typically a login/build node). `enroot import dockerd://...` reads the
#       image from the local docker daemon, so build and import share a daemon.
#
# Usage:
#   # build the CI image from source and import to .sqsh:
#   bash tools/slurm/build_ci_image.sh --sqsh-out /lustre/.../cosmos_rl_ci.sqsh
#
#   # reuse an already-built local docker image, only import to .sqsh:
#   bash tools/slurm/build_ci_image.sh --no-build --image-tag cosmos_rl_ci:latest \
#       --sqsh-out /lustre/.../cosmos_rl_ci.sqsh
#
#   # reuse an EXISTING published/registry image (no docker build needed; only
#   # enroot is required). Provide tests at launch via --repo-root-path:
#   bash tools/slurm/build_ci_image.sh \
#       --from-uri docker://nvcr.io#nvidia/cosmos-rl:<tag> \
#       --sqsh-out /lustre/.../cosmos_rl_ci.sqsh
#
# If you ALREADY have a .sqsh (e.g. the one your training jobs use), you don't
# need this script at all -- just launch with:
#   ./cosmos_rl_ci_job.sh --container <that.sqsh> --repo-root-path <repo> ...
# =============================================================================

set -euo pipefail

# --- Defaults ---------------------------------------------------------------
# Repo root is three levels up from this script (cosmos_rl/tools/slurm -> repo).
# Use `pwd -P` so the `tools -> cosmos_rl/tools` symlink is resolved to the real
# path; otherwise the logical path is too short and we overshoot the repo root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd -P)"

IMAGE_TAG="cosmos_rl_ci:latest"
SQSH_OUT=""
# Overwrite an existing .sqsh by default; set --no-clobber to refuse instead.
NO_CLOBBER=0
DO_BUILD=1
BAKE_TESTS=1
# When set, import this enroot source URI directly instead of building locally
# (e.g. docker://nvcr.io#nvidia/cosmos-rl:<tag> or dockerd://local-image:tag).
FROM_URI=""
# Where tests/ + configs/ get baked inside the image. The launcher falls back to
# this path when --repo-root-path is not given.
BAKED_REPO_DIR="/workspace/cosmos-rl"

# Dockerfile build args (see Dockerfile header for accepted values).
# Default to no-efa: CI runs on a single node, so the AWS-EFA stack is not
# needed and the no-efa base builds faster.
COSMOS_RL_BUILD_MODE="no-efa"
COSMOS_RL_TORCH_VARIANT="2.8"

# Parallelism cap for the from-source CUDA extension builds (flash-attn/FA3,
# apex, transformer_engine, grouped_gemm, DeepEP). Each nvcc job can use many GB,
# so building with one job per core can OOM. Empty here => auto (computed from
# RAM/cores below).
MAX_JOBS=""
NVCC_THREADS=""

usage() {
    cat <<'EOF'
Build the cosmos-rl CI image and import it to an enroot .sqsh.

Options:
  --repo-root PATH        Repo root to build from (default: inferred from script path)
  --image-tag TAG         Docker image tag to build/import (default: cosmos_rl_ci:latest)
  --sqsh-out PATH         Output .sqsh path (default: <repo-root>/cosmos_rl_ci.sqsh)
  --build-mode MODE       COSMOS_RL_BUILD_MODE build-arg: efa|no-efa (default: no-efa)
  --torch-variant VER     COSMOS_RL_TORCH_VARIANT build-arg: 2.8|2.10 (default: 2.8)
  --max-jobs N            Cap parallel compile jobs (default: auto from RAM/cores).
                          Lower this if the build OOMs; raise to build faster.
  --nvcc-threads N        Threads per nvcc invocation (default: unset)
  --no-build              Skip `docker build`; import the existing local image
  --from-uri URI          Import an existing container source URI directly (no
                          docker build / no docker needed), e.g.
                          docker://nvcr.io#nvidia/cosmos-rl:<tag>. Tests are not
                          baked in this mode -- pass --repo-root-path at launch.
  --no-bake-tests         Do not bake tests/ + configs/ into the image
                          (then --repo-root-path is required at launch time)
  --no-clobber            Refuse to overwrite an existing .sqsh
                          (default: overwrite and re-import)
  -h, --help              Show this help
EOF
}

# --- Parse args -------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo-root)      REPO_ROOT="$2"; shift 2 ;;
        --image-tag)      IMAGE_TAG="$2"; shift 2 ;;
        --sqsh-out)       SQSH_OUT="$2"; shift 2 ;;
        --build-mode)     COSMOS_RL_BUILD_MODE="$2"; shift 2 ;;
        --torch-variant)  COSMOS_RL_TORCH_VARIANT="$2"; shift 2 ;;
        --max-jobs)       MAX_JOBS="$2"; shift 2 ;;
        --nvcc-threads)   NVCC_THREADS="$2"; shift 2 ;;
        --from-uri)       FROM_URI="$2"; shift 2 ;;
        --no-build)       DO_BUILD=0; shift ;;
        --no-bake-tests)  BAKE_TESTS=0; shift ;;
        --no-clobber)     NO_CLOBBER=1; shift ;;
        --force)          shift ;;  # deprecated no-op: overwrite is now default
        -h|--help)        usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -z "${SQSH_OUT}" ]]; then
    SQSH_OUT="${REPO_ROOT}/cosmos_rl_ci.sqsh"
fi

# --from-uri reuses an existing container source: no local build, no tests baked.
if [[ -n "${FROM_URI}" ]]; then
    DO_BUILD=0
    BAKE_TESTS=0
fi

log() { echo "[build_ci_image] $*"; }

command -v enroot >/dev/null 2>&1 || { echo "ERROR: 'enroot' not found on PATH" >&2; exit 1; }
if [[ "${DO_BUILD}" -eq 1 ]]; then
    command -v docker >/dev/null 2>&1 || { echo "ERROR: 'docker' not found on PATH" >&2; exit 1; }
fi

# Auto-pick a memory-safe MAX_JOBS if not specified: the heavy nvcc jobs (esp.
# FA3) can peak well above 10 GB each, so cap by RAM as well as by core count.
# Conservative by default (better slow than OOM); raise with --max-jobs.
if [[ "${DO_BUILD}" -eq 1 && -z "${MAX_JOBS}" ]]; then
    cores=$(nproc 2>/dev/null || echo 4)
    mem_gb=$(free -g 2>/dev/null | awk '/^Mem:/{print $2}')
    [[ -z "${mem_gb}" || "${mem_gb}" -lt 1 ]] && mem_gb=8
    by_mem=$(( mem_gb / 16 ))
    [[ "${by_mem}" -lt 1 ]] && by_mem=1
    MAX_JOBS=$(( cores < by_mem ? cores : by_mem ))
    log "Auto-selected MAX_JOBS=${MAX_JOBS} (cores=${cores}, mem=${mem_gb}Gi). Override with --max-jobs."
fi

# --- 1. docker build --------------------------------------------------------
if [[ "${DO_BUILD}" -eq 1 ]]; then
    if [[ "${BAKE_TESTS}" -eq 1 ]]; then
        # Build the base image (same as GH CI) under a -base tag, then add a thin
        # layer that bakes tests/ + configs/ in (the base Dockerfile rm -rf's the
        # source tree, so tests/ is otherwise absent from the image).
        BASE_TAG="${IMAGE_TAG%:*}-base:${IMAGE_TAG##*:}"
    else
        BASE_TAG="${IMAGE_TAG}"
    fi

    log "Building base docker image '${BASE_TAG}' from ${REPO_ROOT}"
    log "  COSMOS_RL_BUILD_MODE=${COSMOS_RL_BUILD_MODE} COSMOS_RL_TORCH_VARIANT=${COSMOS_RL_TORCH_VARIANT} MAX_JOBS=${MAX_JOBS:-<all>} NVCC_THREADS=${NVCC_THREADS:-<default>}"
    # Empty MAX_JOBS/NVCC_THREADS build-args are harmless: the Dockerfile ENVs
    # treat "" as unset (torch's cpp_extension falls back to all cores).
    docker build \
        -t "${BASE_TAG}" \
        -f "${REPO_ROOT}/Dockerfile" \
        --build-arg "COSMOS_RL_BUILD_MODE=${COSMOS_RL_BUILD_MODE}" \
        --build-arg "COSMOS_RL_TORCH_VARIANT=${COSMOS_RL_TORCH_VARIANT}" \
        --build-arg "MAX_JOBS=${MAX_JOBS}" \
        --build-arg "NVCC_THREADS=${NVCC_THREADS}" \
        "${REPO_ROOT}"

    if [[ "${BAKE_TESTS}" -eq 1 ]]; then
        log "Baking tests/ + configs/ into '${IMAGE_TAG}' at ${BAKED_REPO_DIR} (layer on ${BASE_TAG})"
        # Write the thin overlay Dockerfile to a temp file (avoids inline heredoc
        # fragility). A 'scratch' default on BASE_IMAGE silences the BuildKit
        # InvalidDefaultArgInFrom warning; the real base is passed via --build-arg.
        ci_dockerfile="$(mktemp)"
        {
            echo 'ARG BASE_IMAGE=scratch'
            echo 'FROM ${BASE_IMAGE}'
            echo 'ARG BAKED_REPO_DIR=/workspace/cosmos-rl'
            echo 'COPY tests ${BAKED_REPO_DIR}/tests'
            echo 'COPY configs ${BAKED_REPO_DIR}/configs'
            echo 'WORKDIR ${BAKED_REPO_DIR}'
        } > "${ci_dockerfile}"
        docker build \
            -t "${IMAGE_TAG}" \
            --build-arg "BASE_IMAGE=${BASE_TAG}" \
            --build-arg "BAKED_REPO_DIR=${BAKED_REPO_DIR}" \
            -f "${ci_dockerfile}" \
            "${REPO_ROOT}"
        rm -f "${ci_dockerfile}"
    fi
else
    if [[ -n "${FROM_URI}" ]]; then
        log "Reusing existing container source (no docker build): ${FROM_URI}"
    else
        log "Skipping docker build (--no-build); importing existing local image '${IMAGE_TAG}'"
    fi
fi

# --- 2. enroot import -------------------------------------------------------
if [[ -n "${FROM_URI}" ]]; then
    SOURCE_URI="${FROM_URI}"
else
    SOURCE_URI="dockerd://${IMAGE_TAG}"
fi
if [[ -f "${SQSH_OUT}" ]]; then
    if [[ "${NO_CLOBBER}" -eq 1 ]]; then
        echo "ERROR: ${SQSH_OUT} already exists (--no-clobber set)." >&2
        exit 1
    fi
    log "Overwriting existing ${SQSH_OUT}"
    rm -f "${SQSH_OUT}"
fi

mkdir -p "$(dirname "${SQSH_OUT}")"
log "Importing '${SOURCE_URI}' -> ${SQSH_OUT}"
enroot import -o "${SQSH_OUT}" "${SOURCE_URI}"

log "Done. CI container image ready at: ${SQSH_OUT}"
log "Upload it to the slurm login node, then launch CI with:"
if [[ "${DO_BUILD}" -eq 1 && "${BAKE_TESTS}" -eq 1 ]]; then
    log "  ./cosmos_rl_ci_job.sh --container <uploaded.sqsh> --output-root-path <output-root>"
    log "(tests/ baked in at ${BAKED_REPO_DIR}; add --repo-root-path <repo> to override)"
else
    log "  ./cosmos_rl_ci_job.sh --container <uploaded.sqsh> --repo-root-path <repo> --output-root-path <output-root>"
    log "(no tests baked in this image; --repo-root-path is required to provide tests/)"
fi
