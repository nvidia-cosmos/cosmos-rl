#!/bin/bash
# =============================================================================
# Standalone launcher: full cosmos-rl CI suite on a single 8-GPU Slurm node.
# =============================================================================
# This ONE file is both the submitter and the Slurm batch script:
#
#   * Run it directly on a login node to submit the job:
#       ./cosmos_rl_ci_job.sh \
#           --container /lustre/.../cosmos_rl_ci.sqsh \
#           --repo-root-path /lustre/.../cosmos-rl \
#           --output-root-path /lustre/.../ci-runs
#
#   * Slurm re-invokes this same file on the compute node (SLURM_JOB_ID set),
#     where it runs `bash tests/run_test.sh` inside the container with all GPUs.
#
# It is self-contained: copy just this file (plus the .sqsh) to the login node.
# No python, no template, no repo checkout needed for the launcher itself --
# `sbatch` captures the script into its spool at submit time. A repo checkout is
# still required via --repo-root-path to provide the tests/ directory, since the
# container image does not ship it.
#
# Mirrors GitHub Actions CI (.github/workflows/build-and-test.yaml). The suite
# already contains the multi-GPU paths (torchrun --nproc_per_node=8/4/2), so one
# exclusive 8-GPU node covers it. One-shot: no autoresume/retry, so a failing CI
# run fails the Slurm job directly.
# =============================================================================

set -o pipefail

# Expose already cached model snapshots through their normal org/repo names.
# The workspace belongs to the disposable installed-wheel test container, never
# a mounted source checkout. Model resolution can then avoid repeated Hub
# metadata calls without rewriting configs, weights, tests or production code.
ci_link_cached_models() {
    local cache_root="$1" workspace="$2"
    local cached_model model_name model_owner model_repo revision snapshot target
    local linked=0
    [[ -d "$cache_root" && -d "$workspace" ]] || return 1
    for cached_model in "$cache_root"/models--*; do
        [[ -d "$cached_model" && -f "$cached_model/refs/main" ]] || continue
        model_name=${cached_model##*/}
        model_name=${model_name#models--}
        model_owner=${model_name%%--*}
        model_repo=${model_name#*--}
        [[ "$model_name" != "$model_repo" ]] || return 1
        [[ "$model_owner" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ && "$model_repo" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || return 1
        revision=$(<"$cached_model/refs/main")
        [[ "$revision" =~ ^[a-fA-F0-9]{40}$ ]] || return 1
        snapshot=$(readlink -f "$cached_model/snapshots/$revision") || return 1
        [[ -f "$snapshot/config.json" ]] || continue
        # A config-only alias shadows the Hub ID and prevents from_pretrained
        # from downloading missing weights. Validate every indexed shard before
        # exposing a local path; fail setup instead of wasting a full GPU suite.
        python -c '
import json
from pathlib import Path, PurePosixPath
import sys

root = Path(sys.argv[1])
def populated(path):
    return path.is_file() and path.stat().st_size > 0

try:
    indices = [root / name for name in (
        "model.safetensors.index.json", "pytorch_model.bin.index.json"
    ) if (root / name).is_file()]
    if indices:
        for index in indices:
            weights = json.loads(index.read_text())["weight_map"]
            assert isinstance(weights, dict) and weights, "empty weight map"
            for shard in set(weights.values()):
                assert isinstance(shard, str) and shard, "invalid shard name"
                relative = PurePosixPath(shard)
                assert not relative.is_absolute() and ".." not in relative.parts, "invalid shard path"
                assert populated(root / shard), f"missing or empty shard: {shard}"
    else:
        assert any(populated(root / name) for name in (
            "model.safetensors", "pytorch_model.bin"
        )), "no complete checkpoint or shard index"
except (AssertionError, OSError, ValueError, KeyError, TypeError) as error:
    print(f"ERROR: incomplete cached model {root}: {error}", file=sys.stderr)
    raise SystemExit(1)
' "$snapshot" || return 1
        target="$workspace/$model_owner/$model_repo"
        [[ ! -L "$workspace/$model_owner" ]] || return 1
        if [[ -e "$target" || -L "$target" ]]; then
            [[ -L "$target" && "$(readlink -f "$target")" == "$snapshot" ]] || {
                echo "ERROR: cached model would overwrite $target" >&2
                return 1
            }
        else
            mkdir -p "$workspace/$model_owner" || return 1
            ln -s "$snapshot" "$target" || return 1
        fi
        echo "Cached model path: $model_owner/$model_repo revision=$revision"
        linked=$((linked + 1))
    done
    [[ "$linked" -gt 0 ]]
}

usage() {
    cat <<'EOF'
Launch the full cosmos-rl CI suite on a single 8-GPU Slurm node.

Usage:
  ./cosmos_rl_ci_job.sh --container <.sqsh> --output-root-path <dir> [options]

Options:
  --container, --cosmos-container PATH  CI container .sqsh (or URI)   [required]
  --output-root-path DIR               Root dir for logs             [required]
  --slurm-partition NAME               SLURM partition               [required]
  --slurm-account NAME                 SLURM account                 [required]
  --scratch-path DIR                   Writable scratch for /tmp + caches inside
                                       the container. Avoids $HOME/Lustre per-user
                                       quotas (EDQUOT) on HF downloads / tmp writes.
                                       Default: node-local ${SLURM_TMPDIR:-${TMPDIR:-/tmp}}.
  --hf-cache DIR                       Pre-populated HuggingFace cache to reuse
                                       (mounted at /root/.cache/huggingface).
                                       Default: a fresh dir under --scratch-path.
  --repo-root-path DIR                 Repo to mount + test (provides tests/)
  --package-wheel PATH                Install this candidate wheel in the ephemeral
                                       container; mount only tests/ (GitHub layout).
                                       Requires --repo-root-path. No source overlay.
  --test-deps-dir DIR                  Optional offline wheelhouse for pytest>=8,<9
                                       and GitHub's ucxx-cu12>=0.40.0 extra.
  --cached-model-paths                 Expose cached main-revision model snapshots
                                       as local org/repo paths in the ephemeral
                                       test workspace; requires --package-wheel
                                       and --hf-cache. Does not skip missing assets.
  --readonly-model-cache DIR           Optional completed Hub cache (models--* root)
                                       mounted read-only for --cached-model-paths.
                                       Runtime downloads/datasets still use --hf-cache.
  --job-name NAME                      SLURM job name        (default: cosmos_ci)
  --ngpu-per-node N                    GPUs to request       (default: 8)
  --test-timeout DUR                   timeout for run_test.sh (default: 2h)
  --duration HOURS                     Override SLURM --time (hours). Default is
                                       derived from --test-timeout + 30m buffer.
  --slurm-job-time H:M:S               Override SLURM --time (overrides --duration)
  --extra-sbatch-arg ARG               Extra sbatch arg (repeatable)
  --dry-run                            Print the sbatch command, do not submit
  -h, --help                           Show this help
EOF
}

# Convert a `timeout`-style duration (e.g. 2h, 90m, 7200, 30s) to seconds.
ci_to_seconds() {
    local t="$1" num unit
    num="${t%[smhdSMHD]}"
    unit="${t#"${num}"}"
    case "${unit}" in
        s|S|"") echo "${num}" ;;
        m|M)    echo $(( num * 60 )) ;;
        h|H)    echo $(( num * 3600 )) ;;
        d|D)    echo $(( num * 86400 )) ;;
        *) echo "ERROR" ;;
    esac
}

# =============================================================================
# SUBMIT MODE (login node): no SLURM_JOB_ID -> parse args and `sbatch` self.
# =============================================================================
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    JOB_NAME="cosmos_ci"
    NGPU=8
    PARTITION=""
    ACCOUNT=""
    CONTAINER=""
    REPO_ROOT_PATH=""
    PACKAGE_WHEEL=""
    TEST_DEPS_DIR=""
    CACHED_MODEL_PATHS=0
    READONLY_MODEL_CACHE=""
    OUTPUT_ROOT=""
    SCRATCH_PATH=""
    HF_CACHE_PATH=""
    DURATION_HOURS=""
    JOB_TIME=""
    TEST_TIMEOUT="2h"
    # Extra wall-clock on top of the test timeout for container pull/startup.
    TIME_BUFFER_SEC=1800
    DRY_RUN=0
    EXTRA_SBATCH_ARGS=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --container|--cosmos-container) CONTAINER="$2"; shift 2 ;;
            --output-root-path)             OUTPUT_ROOT="$2"; shift 2 ;;
            --scratch-path)                 SCRATCH_PATH="$2"; shift 2 ;;
            --hf-cache)                     HF_CACHE_PATH="$2"; shift 2 ;;
            --repo-root-path)               REPO_ROOT_PATH="$2"; shift 2 ;;
            --package-wheel)                PACKAGE_WHEEL="$2"; shift 2 ;;
            --test-deps-dir)                TEST_DEPS_DIR="$2"; shift 2 ;;
            --cached-model-paths)           CACHED_MODEL_PATHS=1; shift ;;
            --readonly-model-cache)         READONLY_MODEL_CACHE="$2"; shift 2 ;;
            --job-name)                     JOB_NAME="$2"; shift 2 ;;
            --ngpu-per-node)                NGPU="$2"; shift 2 ;;
            --slurm-partition)              PARTITION="$2"; shift 2 ;;
            --slurm-account)                ACCOUNT="$2"; shift 2 ;;
            --duration)                     DURATION_HOURS="$2"; shift 2 ;;
            --slurm-job-time)               JOB_TIME="$2"; shift 2 ;;
            --test-timeout)                 TEST_TIMEOUT="$2"; shift 2 ;;
            --extra-sbatch-arg)             EXTRA_SBATCH_ARGS+=("$2"); shift 2 ;;
            --dry-run)                      DRY_RUN=1; shift ;;
            -h|--help)                      usage; exit 0 ;;
            *) echo "ERROR: unknown argument: $1" >&2; usage; exit 1 ;;
        esac
    done

    [[ -n "${CONTAINER}" ]] || { echo "ERROR: --container is required" >&2; exit 1; }
    [[ -n "${OUTPUT_ROOT}" ]] || { echo "ERROR: --output-root-path is required" >&2; exit 1; }
    [[ -n "${PARTITION}" ]] || { echo "ERROR: --slurm-partition is required" >&2; exit 1; }
    [[ -n "${ACCOUNT}" ]] || { echo "ERROR: --slurm-account is required" >&2; exit 1; }
    if [[ -n "${PACKAGE_WHEEL}" ]]; then
        [[ -f "${PACKAGE_WHEEL}" && -d "${REPO_ROOT_PATH}/tests" ]] || {
            echo "ERROR: --package-wheel requires an existing wheel and --repo-root-path with tests/" >&2
            exit 1
        }
        PACKAGE_WHEEL="$(readlink -f "${PACKAGE_WHEEL}")"
    fi
    if [[ -n "${TEST_DEPS_DIR}" ]]; then
        [[ -d "${TEST_DEPS_DIR}" ]] || { echo "ERROR: missing --test-deps-dir" >&2; exit 1; }
        TEST_DEPS_DIR="$(readlink -f "${TEST_DEPS_DIR}")"
    fi
    if [[ "$CACHED_MODEL_PATHS" == 1 && ( -z "$PACKAGE_WHEEL" || -z "$HF_CACHE_PATH" ) ]]; then
        echo "ERROR: --cached-model-paths requires --package-wheel and --hf-cache" >&2
        exit 1
    fi
    if [[ -n "$READONLY_MODEL_CACHE" ]]; then
        if [[ "$CACHED_MODEL_PATHS" != 1 || ! -d "$READONLY_MODEL_CACHE" ]]; then
            echo "ERROR: --readonly-model-cache requires --cached-model-paths and an existing directory" >&2
            exit 1
        fi
        READONLY_MODEL_CACHE="$(readlink -f "$READONLY_MODEL_CACHE")"
    fi

    # Resolve SLURM --time. Precedence: --slurm-job-time > --duration > derived
    # from --test-timeout + buffer (so the allocation always covers the tests).
    if [[ -n "${JOB_TIME}" ]]; then
        DURATION="${JOB_TIME}"
    elif [[ -n "${DURATION_HOURS}" ]]; then
        h=${DURATION_HOURS%.*}
        frac=0
        if [[ "${DURATION_HOURS}" == *.* ]]; then
            frac=$(awk "BEGIN{printf \"%d\", (${DURATION_HOURS} - ${h}) * 60}")
        fi
        DURATION=$(printf "%d:%02d:00" "${h}" "${frac}")
    else
        test_secs=$(ci_to_seconds "${TEST_TIMEOUT}")
        if [[ "${test_secs}" == "ERROR" || -z "${test_secs}" ]]; then
            echo "ERROR: could not parse --test-timeout '${TEST_TIMEOUT}' (use e.g. 2h, 90m, 7200)" >&2
            exit 1
        fi
        total_secs=$(( test_secs + TIME_BUFFER_SEC ))
        DURATION=$(printf "%d:%02d:%02d" $(( total_secs / 3600 )) $(( (total_secs % 3600) / 60 )) $(( total_secs % 60 )))
    fi

    # Absolute paths (the container + repo must be reachable from compute nodes).
    CONTAINER="$(readlink -f "${CONTAINER}" 2>/dev/null || echo "${CONTAINER}")"
    if [[ -n "${REPO_ROOT_PATH}" ]]; then
        REPO_ROOT_PATH="$(readlink -f "${REPO_ROOT_PATH}")"
    fi
    # Scratch/HF-cache are optional and may live on shared storage; resolve them
    # to absolute only when given (node-local default is resolved on the node).
    if [[ -n "${SCRATCH_PATH}" ]]; then
        SCRATCH_PATH="$(readlink -f "${SCRATCH_PATH}" 2>/dev/null || echo "${SCRATCH_PATH}")"
    fi
    if [[ -n "${HF_CACHE_PATH}" ]]; then
        HF_CACHE_PATH="$(readlink -f "${HF_CACHE_PATH}" 2>/dev/null || echo "${HF_CACHE_PATH}")"
    fi
    if [[ -n "$READONLY_MODEL_CACHE" ]] && {
        [[ "$READONLY_MODEL_CACHE" == "$HF_CACHE_PATH" || "$READONLY_MODEL_CACHE" == "$HF_CACHE_PATH/"* ]] ||
        [[ "$HF_CACHE_PATH" == "$READONLY_MODEL_CACHE/"* ]];
    }; then
        echo "ERROR: read-only model cache and writable HF cache must not overlap" >&2
        exit 1
    fi

    ts=$(date +%Y%m%d%H%M%S)
    OUTPUT_DIR="${OUTPUT_ROOT}/${JOB_NAME}_${ts}"
    SLURM_DIR="${OUTPUT_DIR}/slurm"

    self="$(readlink -f "$0")"

    sbatch_cmd=(
        sbatch
        --job-name="${JOB_NAME}"
        --nodes=1
        --exclusive
        --partition="${PARTITION}"
        --account="${ACCOUNT}"
        --time="${DURATION}"
        --gres=gpu:"${NGPU}"
        --output="${SLURM_DIR}/slurm_%j.log"
        --open-mode=append
        --export="ALL,COSMOS_CI_CONTAINER=${CONTAINER},COSMOS_CI_REPO_ROOT=${REPO_ROOT_PATH},COSMOS_CI_PACKAGE_WHEEL=${PACKAGE_WHEEL},COSMOS_CI_TEST_DEPS_DIR=${TEST_DEPS_DIR},COSMOS_CI_CACHED_MODEL_PATHS=${CACHED_MODEL_PATHS},COSMOS_CI_READONLY_MODEL_CACHE=${READONLY_MODEL_CACHE},COSMOS_CI_OUTPUT_DIR=${OUTPUT_DIR},COSMOS_CI_SLURM_DIR=${SLURM_DIR},COSMOS_CI_TEST_TIMEOUT=${TEST_TIMEOUT},COSMOS_CI_SCRATCH=${SCRATCH_PATH},COSMOS_CI_HF_CACHE=${HF_CACHE_PATH},COSMOS_CI_SUBMIT_USER=${USER}"
    )
    for a in "${EXTRA_SBATCH_ARGS[@]}"; do sbatch_cmd+=("${a}"); done
    sbatch_cmd+=("${self}")

    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "[cosmos-rl-ci] Output dir: ${OUTPUT_DIR}"
        echo "[cosmos-rl-ci] Would run:"
        printf '  %q' "${sbatch_cmd[@]}"; echo
        exit 0
    fi

    mkdir -p "${SLURM_DIR}"
    echo "[cosmos-rl-ci] Output dir: ${OUTPUT_DIR}"
    echo "[cosmos-rl-ci] Container : ${CONTAINER}"
    exec "${sbatch_cmd[@]}"
fi

# =============================================================================
# RUN MODE (compute node, under Slurm): execute the CI suite in the container.
# =============================================================================
export OUTPUT_DIR="${COSMOS_CI_OUTPUT_DIR}"
export SLURM_DIR="${COSMOS_CI_SLURM_DIR}"
export CONTAINER_IMAGE="${COSMOS_CI_CONTAINER}"
export TEST_TIMEOUT="${COSMOS_CI_TEST_TIMEOUT:-2h}"
export REPO_ROOT_PATH="${COSMOS_CI_REPO_ROOT}"
export SUBMIT_USER="${COSMOS_CI_SUBMIT_USER}"

# Scratch root for the container's /tmp and caches. Default to node-local scratch
# ($SLURM_TMPDIR / $TMPDIR) so HF model downloads and /tmp writes don't hit the
# submitter's $HOME or Lustre per-user quota (the EDQUOT / "Disk quota exceeded"
# failures seen otherwise). Override with --scratch-path.
SCRATCH_BASE="${COSMOS_CI_SCRATCH:-}"
if [[ -z "${SCRATCH_BASE}" ]]; then
    SCRATCH_BASE="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
fi
SCRATCH_DIR="${SCRATCH_BASE}/cosmos_ci_${SLURM_JOB_ID}"

# --- Ensure $USER is set correctly (important for containers) ---------------
if [[ -n "${SUBMIT_USER}" ]]; then
    export USER="${SUBMIT_USER}"
fi

log() {
    local message="$1"
    local timestamp
    timestamp=$(date +"%Y-%m-%d %I:%M:%S.%3N %p %Z")
    echo -e "[$timestamp][cosmos-rl-ci]: $message"
}

# --- Job info & directory setup --------------------------------------------
echo "JOBID $SLURM_JOB_ID"
log "Full-CI job started on $(hostname)"
log "User: ${USER}, Submit User: ${SUBMIT_USER}"
log "Container image: ${CONTAINER_IMAGE}"

job_dir="${SLURM_DIR}"
mkdir -p "${job_dir}"
run_dir="${job_dir}/run_${SLURM_JOB_ID}"
mkdir -p "${run_dir}"
ln -sfn "${run_dir}" "${job_dir}/latest_run"
log "Run dir: ${run_dir}"

# --- Container mounts -------------------------------------------------------
# All writable paths live under a roomy scratch dir (node-local by default) to
# avoid per-user quota (EDQUOT) on HF downloads and /tmp writes.
HOST_CACHE_DIR="${SCRATCH_DIR}/cache"        # -> /root/.cache (matches GH CI -v /tmp/cache)
HOST_TMP_DIR="${SCRATCH_DIR}/tmp"            # -> /tmp (test configs, pycache, etc.)
# HF cache: reuse a pre-populated dir if given, else a fresh one under scratch.
HOST_HF_CACHE="${COSMOS_CI_HF_CACHE:-}"
if [[ -z "${HOST_HF_CACHE}" ]]; then
    HOST_HF_CACHE="${SCRATCH_DIR}/huggingface"
fi
mkdir -p "${HOST_CACHE_DIR}" "${HOST_TMP_DIR}" "${HOST_HF_CACHE}"
log "Scratch dir: ${SCRATCH_DIR} (HF cache: ${HOST_HF_CACHE})"

# Nested binds: /root/.cache/huggingface sits inside /root/.cache.
MOUNTS="${HOST_CACHE_DIR}:/root/.cache"
MOUNTS="${MOUNTS},${HOST_TMP_DIR}:/tmp"
MOUNTS="${MOUNTS},${HOST_HF_CACHE}:/root/.cache/huggingface"
if [[ -n "${COSMOS_CI_READONLY_MODEL_CACHE:-}" ]]; then
    [[ "${COSMOS_CI_CACHED_MODEL_PATHS:-0}" == 1 && -n "${COSMOS_CI_PACKAGE_WHEEL:-}" && -d "$COSMOS_CI_READONLY_MODEL_CACHE" ]] || {
        echo "ERROR: invalid read-only model cache configuration" >&2
        exit 1
    }
    MOUNTS="${MOUNTS},${COSMOS_CI_READONLY_MODEL_CACHE}:/ci-model-cache:ro"
fi

# Mount the repo so the latest tests/ (and optionally working-tree code) run,
# rather than only the code baked into the container image.
if [[ -n "${COSMOS_CI_PACKAGE_WHEEL:-}" ]]; then
    # Mount only tests, as GitHub does. Installing into the container interpreter
    # also pins subprocess imports when a test replaces PYTHONPATH.
    MOUNTS="${MOUNTS},${REPO_ROOT_PATH}/tests:/workspace/cosmos-rl/tests:ro"
    MOUNTS="${MOUNTS},${COSMOS_CI_PACKAGE_WHEEL}:/ci-wheel/$(basename "${COSMOS_CI_PACKAGE_WHEEL}"):ro"
elif [[ -n "${REPO_ROOT_PATH}" ]]; then
    MOUNTS="${MOUNTS},${REPO_ROOT_PATH}:/opt/cosmos-rl"
fi
if [[ -n "${COSMOS_CI_TEST_DEPS_DIR:-}" ]]; then
    MOUNTS="${MOUNTS},${COSMOS_CI_TEST_DEPS_DIR}:/ci-test-deps:ro"
fi
MOUNTS="${MOUNTS},${run_dir}:/ci-results"

log "Container mounts: ${MOUNTS}"

# --- Run the CI suite -------------------------------------------------------
srun \
    --nodes=1 \
    --ntasks=1 \
    --container-image "${CONTAINER_IMAGE}" \
    --container-mounts "${MOUNTS}" \
    --no-container-mount-home \
    --export=ALL,USER=${USER} \
    -o "${run_dir}/run_test.out" \
    -e "${run_dir}/run_test.err" \
    bash -c "$(declare -f ci_link_cached_models)"'
    set -o pipefail
    export TEST_LOG_DIR=/ci-results/test-logs
    # Match GH CI environment.
    export PYTHONPYCACHEPREFIX=/tmp/pycache
    # Keep temp files + HF downloads on the mounted scratch (roomy, no quota).
    export TMPDIR=/tmp
    export HF_HOME=/root/.cache/huggingface
    # Disable NCCL NVLS to avoid potential instability on slurm clusters.
    export NCCL_NVLS_ENABLE=0

    python -c "import cosmos_rl; print(f\"cosmos_rl location: {cosmos_rl.__file__}\"); print(f\"cosmos_rl version: {cosmos_rl.__version__}\")" 2>/dev/null || true

    materialize_mounted_version_module() {
        local repo_root="$1"
        local target="${repo_root}/cosmos_rl/_version.py"
        local installed_version_file
        local temporary_target

        # setuptools-scm generates this ignored file when the package is built.
        # A raw checkout mounted over the installed package does not contain it,
        # so preserve the generated module from the image before PYTHONPATH
        # makes the checkout authoritative.
        if [[ -f "${target}" ]]; then
            return 0
        fi
        if ! installed_version_file="$(python -c "import sys; from pathlib import Path; matches = [candidate for entry in sys.path if (candidate := Path(entry) / \"cosmos_rl\" / \"_version.py\").is_file()]; print(matches[0] if matches else \"\"); raise SystemExit(not matches)")"; then
            echo "ERROR: mounted repo lacks cosmos_rl/_version.py and no installed generated version module was found" >&2
            return 1
        fi

        temporary_target="${target}.tmp.$$"
        if ! cp "${installed_version_file}" "${temporary_target}"; then
            echo "ERROR: failed to copy generated version module from ${installed_version_file}" >&2
            return 1
        fi
        if ! mv "${temporary_target}" "${target}"; then
            echo "ERROR: failed to install generated version module at ${target}" >&2
            return 1
        fi
        echo "Materialized generated version module at ${target}"
    }

    if [[ -n "${COSMOS_CI_PACKAGE_WHEEL:-}" ]]; then
        # These writes are confined to the job-local container filesystem.
        unset PYTHONPATH
        cd /tmp || exit 1
        sha256sum "/ci-wheel/$(basename "${COSMOS_CI_PACKAGE_WHEEL}")" || exit 1
        python -m pip install --no-index --no-deps --force-reinstall "/ci-wheel/$(basename "${COSMOS_CI_PACKAGE_WHEEL}")" || exit 1
        cd /workspace/cosmos-rl || exit 1
        python -c "from pathlib import Path; import cosmos_rl; p = Path(cosmos_rl.__file__).resolve(); assert Path.cwd() not in p.parents, p; print(f\"Installed candidate: {p}\")" || exit 1
        echo "Using installed wheel with tests-only mount (GitHub layout)"
    elif [[ -d /opt/cosmos-rl/tests ]]; then
        # --repo-root-path was mounted: override the baked-in code/tests and run
        # against the working-tree copy (PYTHONPATH shadows the installed pkg).
        materialize_mounted_version_module /opt/cosmos-rl || exit 1
        export PYTHONPATH="/opt/cosmos-rl:${PYTHONPATH}"
        cd /opt/cosmos-rl
        echo "Using mounted repo at /opt/cosmos-rl (overrides baked-in)"
    elif [[ -d /workspace/cosmos-rl/tests ]]; then
        # No mount: use the tests/ baked into the image (build_ci_image.sh).
        cd /workspace/cosmos-rl
        echo "Using tests baked into the image at /workspace/cosmos-rl"
    else
        echo "ERROR: no tests/ found. Build the image with tests baked in (default), or pass --repo-root-path." >&2
        exit 1
    fi

    # Match GitHub: the base TEST image includes the ucxx extra. Without it the
    # native transport suites silently skip, so fail setup rather than claiming
    # a full-CI pass against an incomplete dependency image.
    if [[ -n "${COSMOS_CI_TEST_DEPS_DIR:-}" ]]; then
        python -m pip install --no-index --find-links=/ci-test-deps "pytest>=8,<9" "ucxx-cu12>=0.40.0" || exit 1
    else
        python -m pip install "pytest>=8,<9" "ucxx-cu12>=0.40.0" || exit 1
    fi
    python -c "import torch, ucxx, pytest; assert torch.cuda.is_available() and torch.cuda.device_count() >= 8; print(f\"CI dependencies: torch={torch.__version__} ucxx={ucxx.__version__} pytest={pytest.__version__} GPUs={torch.cuda.device_count()}\")" || exit 1
    python -m pip freeze > /ci-results/environment.txt
    if [[ "${COSMOS_CI_CACHED_MODEL_PATHS:-0}" == 1 ]]; then
        [[ -n "${COSMOS_CI_PACKAGE_WHEEL:-}" ]] || exit 1
        model_cache=/root/.cache/huggingface/hub
        if [[ -n "${COSMOS_CI_READONLY_MODEL_CACHE:-}" ]]; then
            model_cache=/ci-model-cache
        fi
        ci_link_cached_models "$model_cache" "$PWD" || exit 1
    fi
    echo "Running tests from: $(pwd)"
    timeout '"${TEST_TIMEOUT}"' bash tests/run_test.sh
    ' \
    | tee "${run_dir}/run_test.log"
status=${PIPESTATUS[0]}

log "tests/run_test.sh exited with status: ${status}"
if [[ ${status} -eq 0 ]]; then
    log "================ CI PASSED ================"
else
    log "================ CI FAILED (status=${status}) ================"
fi

exit ${status}
