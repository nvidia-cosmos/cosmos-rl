#!/bin/bash
#SBATCH --account=[[SLURM_ACCOUNT]]
#SBATCH --job-name=[[SLURM_JOB_NAME]]
#SBATCH --time=[[DURATION]]
#SBATCH --nodes=[[TOTAL_NODES]]
#SBATCH --partition=[[SLURM_PARTITION]]
#SBATCH --output=[[SLURM_DIR]]/slurm_%j.log
#SBATCH --open-mode=append
#SBATCH --signal=B:SIGUSR1@[[PRE_TIMEOUT_SIGNAL]]
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --requeue
# shellcheck disable=SC1035,SC1073,SC1020,SC1072
[[EXTRA_SBATCH_ARGS]]

# =============================================================================
# Cosmos RL SLURM Job Script
# =============================================================================
# This script handles:
# - Auto-resume on timeout (SIGUSR1 signal)
# - Auto-retry for transient failures
# - Proper $USER preservation in containers
#
# Structure:
#   1. Constants & Configuration
#   2. Function Definitions
#      2a. Logging Utilities
#      2b. Directory Management (part_N / run_N tracking)
#      2c. Signal Handling
#      2d. Auto-Resume & Auto-Retry
#   3. Main Execution
#      3a. Job Info & Directory Setup
#      3b. Register Signal Handlers
#      3c. Container Mounts
#      3d. Node List Setup
#      3e. Launch Processes (Controller, Policy, Rollout)
#      3f. Monitor Processes
#      3g. Cleanup and Exit
# =============================================================================

set -o pipefail

# =============================================================================
# 1. Constants & Configuration
# =============================================================================

# --- Template configuration ---
export OUTPUT_DIR="[[OUTPUT_DIR]]"
export SLURM_DIR="[[SLURM_DIR]]"
export WORKDIR="[[WORKDIR]]"
export CONFIG_PATH="[[CONFIG_PATH]]"
export LAUNCHER="[[LAUNCHER]]"
export LAUNCHER_ARGS="[[LAUNCHER_ARGS]]"
export MAX_RETRIES=[[RETRIES]]
export AUTORESUME_ENABLED=[[AUTORESUME]]
export SUBMIT_USER="[[SUBMIT_USER]]"
export WANDB_GIT_COMMIT="[[WANDB_GIT_COMMIT]]"
export WANDB_GIT_REMOTE_URL="[[WANDB_GIT_REMOTE_URL]]"

# --- Node configuration ---
export NUM_POLICY_NODES=[[NUM_POLICY_NODES]]
export NUM_ROLLOUT_NODES=[[NUM_ROLLOUT_NODES]]
export TOTAL_NODES=[[TOTAL_NODES]]
export NODE_LAUNCH_METADATA_POLICY='[[NODE_LAUNCH_METADATA_POLICY]]'
export NODE_LAUNCH_METADATA_ROLLOUT='[[NODE_LAUNCH_METADATA_ROLLOUT]]'

# --- Container configuration ---
export CONTAINER_IMAGE="[[CONTAINER]]"

# --- Ensure $USER is set correctly (important for containers) ---
if [[ -n "${SUBMIT_USER}" ]]; then
    export USER="${SUBMIT_USER}"
fi

# =============================================================================
# 2. Function Definitions
# =============================================================================

# -----------------------------------------------------------------------------
# 2a. Logging Utilities
# -----------------------------------------------------------------------------

# Record job start time for elapsed time calculation
JOB_START_TIME=$(date +%s)

# Log a timestamped message to stdout (captured by slurm.log).
log() {
    local message="$1"
    local timestamp=$(date +"%Y-%m-%d %I:%M:%S.%3N %p %Z")
    echo -e "[$timestamp][cosmos-rl-sbatch]: $message"
}

# Format seconds into HH:MM:SS.
format_duration() {
    local seconds=$1
    local hours=$((seconds / 3600))
    local minutes=$(((seconds % 3600) / 60))
    local secs=$((seconds % 60))
    printf "%02d:%02d:%02d" "${hours}" "${minutes}" "${secs}"
}

# -----------------------------------------------------------------------------
# 2b. Directory Management (part_N / run_N tracking)
# -----------------------------------------------------------------------------

# Create a new run directory under the current latest_part_dir.
# Prints the path of the new run directory.
add_run() {
    local latest_run_dir=$(ls -d "${latest_part_dir}"/run_* 2>/dev/null | sort -V | tail -n 1)
    local new_run_dir=""

    if [[ -z "${latest_run_dir}" ]]; then
        new_run_dir="${latest_part_dir}/run_0"
    else
        local latest_run_num=$(basename "${latest_run_dir}" | sed 's/run_//')
        local new_run_num=$((latest_run_num + 1))
        new_run_dir="${latest_part_dir}/run_${new_run_num}"
    fi

    mkdir -p "${new_run_dir}"
    echo "${new_run_dir}"
}

# Create a new part directory under job_dir (used on autoresume).
# Prints the path of the new part directory.
add_part() {
    local latest_part_num=$(basename "${latest_part_dir}" | sed 's/part_//')
    local new_part_num=$((latest_part_num + 1))
    local new_part_dir="${job_dir}/part_${new_part_num}"
    mkdir -p "${new_part_dir}"
    echo "${new_part_dir}"
}

# -----------------------------------------------------------------------------
# 2c. Signal Handling
# -----------------------------------------------------------------------------

received_signal=""

# Gracefully terminate child processes on signal, with timeout-based force kill.
sig_handler() {
    local sig=$1
    received_signal="${sig}"
    log "Signal ${sig} caught. Will terminate job and potentially requeue."

    # Send SIGTERM to all child processes
    for pid in "${pid_controller}" "${pid_policy}" "${pid_rollout}"; do
        if [[ -n "${pid}" ]]; then
            kill "${pid}" 2>/dev/null || true
        fi
    done

    # Poll for process termination (max 130s, srun default grace period is ~122s)
    local max_wait=130
    local poll_interval=5
    local start_time=$(date +%s)
    log "Waiting up to ${max_wait}s for processes to terminate gracefully..."

    while true; do
        local now=$(date +%s)
        local elapsed=$((now - start_time))

        if [[ ${elapsed} -ge ${max_wait} ]]; then
            log "Timeout reached (${max_wait}s). Force killing remaining processes..."
            for pid in "${pid_controller}" "${pid_policy}" "${pid_rollout}"; do
                if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
                    log "Force killing PID ${pid}"
                    kill -9 "${pid}" 2>/dev/null || true
                fi
            done
            break
        fi

        local all_terminated=true
        for pid in "${pid_controller}" "${pid_policy}" "${pid_rollout}"; do
            if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
                all_terminated=false
                break
            fi
        done

        if [[ "${all_terminated}" == "true" ]]; then
            log "All processes terminated gracefully after ${elapsed}s"
            break
        fi

        sleep "${poll_interval}"
    done

    log "Signal handler complete"
}

# -----------------------------------------------------------------------------
# 2d. Auto-Resume & Auto-Retry
# -----------------------------------------------------------------------------

# Requeue the job on SIGUSR1 (timeout/preemption) for automatic resume.
submit_autoresume() {
    local status=$1

    log "Checking if need to requeue for autoresume on host $(hostname)"

    if [[ "${AUTORESUME_ENABLED}" != "1" ]]; then
        log "Autoresume is disabled"
        return ${status}
    fi

    if [[ -f "${OUTPUT_DIR}/ABORT-AUTORESUME" ]]; then
        log "Abort file detected (${OUTPUT_DIR}/ABORT-AUTORESUME), will not requeue"
        return ${status}
    fi

    # Only requeue on SIGUSR1 (not SIGTERM, so scancel doesn't trigger autoresume)
    if [[ "${received_signal}" == "SIGUSR1" ]]; then
        log "Received signal ${received_signal}, will requeue for autoresume"

        new_part_dir=$(add_part)
        log "New part dir added: ${new_part_dir}"

        if [[ ${MAX_RETRIES} -gt 0 ]]; then
            log "Resetting retries for new part"
            echo "${MAX_RETRIES}" > "${new_part_dir}/remaining-retries"
        fi

        scontrol requeue ${SLURM_JOB_ID}
        local next_status=$?
        log "Requeue status: ${next_status}"
        return ${next_status}
    fi

    log "No autoresume signal received"
    return ${status}
}

# Retry the job on transient failures (decrements remaining-retries counter).
handle_auto_retry() {
    local status=$1

    # Only retry on failure
    if [[ ${status} -eq 0 ]]; then
        return ${status}
    fi

    # Don't retry on manual cancellation (SIGTERM from scancel)
    if [[ "${received_signal}" == "SIGTERM" ]]; then
        log "Job was manually cancelled (SIGTERM). Skipping auto-retry."
        return ${status}
    fi

    local remaining_retries=0
    if [[ -f "${latest_part_dir}/remaining-retries" ]]; then
        remaining_retries=$(cat "${latest_part_dir}/remaining-retries")
    fi

    if [[ ${remaining_retries} -gt 0 ]]; then
        remaining_retries=$((remaining_retries - 1))
        echo "${remaining_retries}" > "${latest_part_dir}/remaining-retries"
        log "Job failed with status ${status}. Retries remaining: ${remaining_retries}. Requeuing..."
        scontrol requeue ${SLURM_JOB_ID}
        local requeue_rc=$?
        if [[ ${requeue_rc} -ne 0 ]]; then
            log "Requeue failed with exit code ${requeue_rc}."
        fi
        return ${requeue_rc}
    else
        log "Job failed with status ${status}. No retries remaining."
        return ${status}
    fi
}

# =============================================================================
# 3. Main Execution
# =============================================================================

# --- 3a. Job Info & Directory Setup ----------------------------------------

echo "JOBID $SLURM_JOB_ID"
echo "Using ${NUM_POLICY_NODES} policy nodes and ${NUM_ROLLOUT_NODES} rollout nodes, TOTAL_NODES: ${TOTAL_NODES}"
log "Job started on $(hostname)"
log "User: ${USER}, Submit User: ${SUBMIT_USER}"

# job_dir is SLURM_DIR directly (no SLURM_JOB_ID subdirectory) so that
# part_*/run_* state persists across manual re-submissions via `sbatch`.
job_dir="${SLURM_DIR}"
mkdir -p "${job_dir}"

# Find or create the latest part directory
latest_part_dir=$(ls -d "${job_dir}"/part_* 2>/dev/null | sort -V | tail -n 1)

if [[ -z "${latest_part_dir}" ]]; then
    latest_part_dir="${job_dir}/part_0"
    log "Creating initial part directory: ${latest_part_dir}"
    mkdir -p "${latest_part_dir}"
    echo "${MAX_RETRIES}" > "${latest_part_dir}/remaining-retries"
fi

# Create a new run directory for this execution
new_run_dir=$(add_run)
log "New run dir: ${new_run_dir}"

# Symlinks for easy access
ln -sfn "${new_run_dir}" "${job_dir}/latest_run"
ln -sfn "${latest_part_dir}" "${job_dir}/latest_part"
log "Symlinks: latest_run -> ${new_run_dir}, latest_part -> ${latest_part_dir}"

# --- 3b. Register Signal Handlers ------------------------------------------

trap "sig_handler SIGUSR1" SIGUSR1
trap "sig_handler SIGTERM" SIGTERM

log "Container image: ${CONTAINER_IMAGE}"

# --- 3c. Container Mounts --------------------------------------------------

MOUNTS="/lustre:/lustre/"
MOUNTS="${MOUNTS},${HOME}/.cache/huggingface:/root/.cache/huggingface"
# wandb needs .netrc to access the wandb api
MOUNTS="${MOUNTS},${HOME}/.netrc:/root/.netrc"
MOUNTS="${MOUNTS},$(dirname "${CONFIG_PATH}"):/opt/config"
MOUNTS="${MOUNTS},${WORKDIR}:/opt/workspace"

# --- 3d. Node List Setup ---------------------------------------------------

export OUTDIR="${OUTPUT_DIR}"
mkdir -p "${OUTDIR}"

export CONTROLLER_PORT=8082
export NODELIST=$(scontrol show hostname $SLURM_JOB_NODELIST)
echo "NODELIST: $NODELIST"

# Use the first policy node for the controller
export POLICY_NODES=$(echo $NODELIST | cut -d' ' -f1-$((NUM_POLICY_NODES)))
export CONTROLLER_NODE=$(echo $POLICY_NODES | cut -d' ' -f1)
export COSMOS_CONTROLLER_HOST="${CONTROLLER_NODE}:${CONTROLLER_PORT}"

if [[ ${NUM_ROLLOUT_NODES} -gt 0 ]]; then
    export ROLLOUT_NODES=$(echo $NODELIST | cut -d' ' -f$((NUM_POLICY_NODES+1))-$((TOTAL_NODES)))
fi

log "Controller node: ${CONTROLLER_NODE}"
log "Policy nodes: ${POLICY_NODES}"
log "Rollout nodes: ${ROLLOUT_NODES:-none}"

# --- 3e. Launch Processes ---------------------------------------------------

# Controller (on first policy node)
srun \
    --overlap \
    --nodes=1 \
    --nodelist=${CONTROLLER_NODE} \
    --container-image "${CONTAINER_IMAGE}" \
    --container-mounts "${MOUNTS}" \
    --no-container-mount-home \
    --export=ALL,USER=${USER} \
    -o ${new_run_dir}/controller.out \
    -e ${new_run_dir}/controller.err \
    bash -c '
    export COSMOS_LOG_LEVEL=DEBUG
    python -c "import cosmos_rl; print(f\"cosmos_rl location: {cosmos_rl.__file__}\"); print(f\"cosmos_rl version: {cosmos_rl.__version__}\")" 2>/dev/null || true
    cosmos_dir=$(python -c "import cosmos_rl,os;print(os.path.dirname(os.path.dirname(cosmos_rl.__file__)))" 2>/dev/null | tail -1)
    if [[ -z "${cosmos_dir}" ]] || [[ ! -d "${cosmos_dir}" ]]; then
        echo "ERROR: Cannot find cosmos_rl package directory" >&2
        exit 1
    fi
    cd "${cosmos_dir}"
    ./cosmos_rl/launcher/launch_controller.sh --port ${CONTROLLER_PORT} --config /opt/config/$(basename [[CONFIG_PATH]]) --script [[LAUNCHER]] [[LAUNCHER_ARGS]]
    ' \
    &
pid_controller=$!
log "Controller started with PID: ${pid_controller}"

# Policy nodes
export LOCAL_NODE_LIST=${POLICY_NODES}
srun \
    --overlap \
    --nodes="${NUM_POLICY_NODES}" \
    --nodelist="${LOCAL_NODE_LIST}" \
    --container-image "${CONTAINER_IMAGE}" \
    --container-mounts "${MOUNTS}" \
    --no-container-mount-home \
    --export=ALL,USER=${USER} \
    -o ${new_run_dir}/policy_%t.out \
    -e ${new_run_dir}/policy_%t.err \
    bash -c '
    cd /opt/workspace
    python scripts/cosmos-rl/cosmos_rl_slurm_launch.py --type policy --config /opt/config/$(basename [[CONFIG_PATH]]) [[LAUNCHER]] [[LAUNCHER_ARGS]]
    ' \
    &
pid_policy=$!
log "Policy started with PID: ${pid_policy}"

# Rollout nodes (if any)
if [[ ${NUM_ROLLOUT_NODES} -gt 0 ]]; then
    export LOCAL_NODE_LIST=${ROLLOUT_NODES}
    srun \
        --nodes="${NUM_ROLLOUT_NODES}" \
        --nodelist="${LOCAL_NODE_LIST}" \
        --container-image "${CONTAINER_IMAGE}" \
        --container-mounts "${MOUNTS}" \
        --no-container-mount-home \
        --export=ALL,USER=${USER} \
        -o ${new_run_dir}/rollout_%t.out \
        -e ${new_run_dir}/rollout_%t.err \
        bash -c '
        cd /opt/workspace
        python scripts/cosmos-rl/cosmos_rl_slurm_launch.py --type rollout --config /opt/config/$(basename [[CONFIG_PATH]]) [[LAUNCHER]] [[LAUNCHER_ARGS]]
        ' \
        &
    pid_rollout=$!
    log "Rollout started with PID: ${pid_rollout}"
fi

log "All processes launched. Waiting for completion..."

# --- 3f. Monitor Processes --------------------------------------------------

# Track whether each process has been waited on (to avoid double-wait returning 127)
policy_waited=false
rollout_waited=false
controller_waited=false
exit_code_policy=""
exit_code_rollout=""
exit_code_controller=""

# If no rollout nodes, treat rollout as already completed successfully
if [[ ${NUM_ROLLOUT_NODES} -eq 0 ]]; then
    rollout_waited=true
    exit_code_rollout=0
fi

while true; do
    # Check if we received a signal
    if [[ -n "${received_signal}" ]]; then
        log "Signal received (${received_signal}), exiting monitoring loop"
        status=1
        break
    fi

    # Check liveness of each process (only if not already waited)
    if [[ "${policy_waited}" == "false" ]]; then
        kill -0 "$pid_policy" 2>/dev/null
        pol_alive=$?
    else
        pol_alive=1
    fi

    if [[ "${rollout_waited}" == "false" ]]; then
        kill -0 "$pid_rollout" 2>/dev/null
        roll_alive=$?
    else
        roll_alive=1
    fi

    if [[ "${controller_waited}" == "false" ]]; then
        kill -0 "$pid_controller" 2>/dev/null
        crl_alive=$?
    else
        crl_alive=1
    fi

    # Reap each dead-but-not-yet-waited process exactly once
    if [[ $pol_alive -ne 0 ]] && [[ "${policy_waited}" == "false" ]]; then
        wait "$pid_policy"
        exit_code_policy=$?
        policy_waited=true
        log "Policy process exited with code ${exit_code_policy}"
    fi

    if [[ $roll_alive -ne 0 ]] && [[ "${rollout_waited}" == "false" ]]; then
        wait "$pid_rollout"
        exit_code_rollout=$?
        rollout_waited=true
        log "Rollout process exited with code ${exit_code_rollout}"
    fi

    if [[ $crl_alive -ne 0 ]] && [[ "${controller_waited}" == "false" ]]; then
        wait "$pid_controller"
        exit_code_controller=$?
        controller_waited=true
        log "Controller process exited with code ${exit_code_controller}"
    fi

    # If any process failed, kill and reap the rest, then break
    if [[ -n "${exit_code_policy}" ]] && [[ ${exit_code_policy} -ne 0 ]]; then
        log "Policy failed with exit code ${exit_code_policy}. Terminating other processes."
        if [[ "${rollout_waited}" == "false" ]]; then
            kill "$pid_rollout" 2>/dev/null || true
            wait "$pid_rollout" 2>/dev/null || true
        fi
        if [[ "${controller_waited}" == "false" ]]; then
            kill "$pid_controller" 2>/dev/null || true
            wait "$pid_controller" 2>/dev/null || true
        fi
        status=${exit_code_policy}
        break
    fi

    if [[ -n "${exit_code_rollout}" ]] && [[ ${exit_code_rollout} -ne 0 ]]; then
        log "Rollout failed with exit code ${exit_code_rollout}. Terminating other processes."
        if [[ "${policy_waited}" == "false" ]]; then
            kill "$pid_policy" 2>/dev/null || true
            wait "$pid_policy" 2>/dev/null || true
        fi
        if [[ "${controller_waited}" == "false" ]]; then
            kill "$pid_controller" 2>/dev/null || true
            wait "$pid_controller" 2>/dev/null || true
        fi
        status=${exit_code_rollout}
        break
    fi

    if [[ -n "${exit_code_controller}" ]] && [[ ${exit_code_controller} -ne 0 ]]; then
        log "Controller failed with exit code ${exit_code_controller}. Terminating other processes."
        if [[ "${policy_waited}" == "false" ]]; then
            kill "$pid_policy" 2>/dev/null || true
            wait "$pid_policy" 2>/dev/null || true
        fi
        if [[ "${rollout_waited}" == "false" ]]; then
            kill "$pid_rollout" 2>/dev/null || true
            wait "$pid_rollout" 2>/dev/null || true
        fi
        status=${exit_code_controller}
        break
    fi

    # All processes finished successfully
    if [[ "${policy_waited}" == "true" ]] && [[ "${rollout_waited}" == "true" ]] && [[ "${controller_waited}" == "true" ]]; then
        log "All processes completed successfully"
        status=0
        break
    fi

    sleep 1
done

# --- 3g. Cleanup and Exit --------------------------------------------------

log "Main loop exited with status: ${status:-0}"

# Handle autoresume (for timeout/preemption signals)
submit_autoresume ${status:-0}
autoresume_status=$?

# Handle auto-retry (for transient failures)
handle_auto_retry ${autoresume_status}
final_status=$?

log "Final exit status: ${final_status}"
exit ${final_status}
