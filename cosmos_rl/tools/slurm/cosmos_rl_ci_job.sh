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
        --export="ALL,COSMOS_CI_CONTAINER=${CONTAINER},COSMOS_CI_REPO_ROOT=${REPO_ROOT_PATH},COSMOS_CI_OUTPUT_DIR=${OUTPUT_DIR},COSMOS_CI_SLURM_DIR=${SLURM_DIR},COSMOS_CI_TEST_TIMEOUT=${TEST_TIMEOUT},COSMOS_CI_SCRATCH=${SCRATCH_PATH},COSMOS_CI_HF_CACHE=${HF_CACHE_PATH},COSMOS_CI_SUBMIT_USER=${USER}"
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

# Mount the repo so the latest tests/ (and optionally working-tree code) run,
# rather than only the code baked into the container image.
if [[ -n "${REPO_ROOT_PATH}" ]]; then
    MOUNTS="${MOUNTS},${REPO_ROOT_PATH}:/opt/cosmos-rl"
fi

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
    bash -c '
    set -o pipefail
    # Match GH CI environment.
    export PYTHONPYCACHEPREFIX=/tmp/pycache
    # Keep temp files + HF downloads on the mounted scratch (roomy, no quota).
    export TMPDIR=/tmp
    export HF_HOME=/root/.cache/huggingface
    # Disable NCCL NVLS to avoid potential instability on slurm clusters.
    export NCCL_NVLS_ENABLE=0

    python -c "import cosmos_rl; print(f\"cosmos_rl location: {cosmos_rl.__file__}\"); print(f\"cosmos_rl version: {cosmos_rl.__version__}\")" 2>/dev/null || true

    if [[ -d /opt/cosmos-rl/tests ]]; then
        # --repo-root-path was mounted: override the baked-in code/tests and run
        # against the working-tree copy (PYTHONPATH shadows the installed pkg).
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
