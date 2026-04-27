#!/usr/bin/env bash
# Run the full cosmos-rl unit-test suite.
#
# Each entry below is invoked through `run` so that:
#   * a failure in any single test does NOT stop the rest of the suite, and
#   * a failure in any single test still makes the script exit non-zero so
#     GitLab CI marks the job as failed (the previous flat command list let
#     the script exit 0 whenever the LAST test happened to pass, masking
#     real failures).
#
# Use `bash tests/run_test.sh` (not `sh ...`) — we rely on bash arrays.

set -uo pipefail

FAILED=()

run() {
    echo
    echo "================ RUN: $* ================"
    if "$@"; then
        echo "---- PASS: $* ----"
    else
        local rc=$?
        echo "---- FAIL(rc=${rc}): $* ----"
        FAILED+=("$*")
    fi
}

run python -c "from cosmos_rl._version import version; print(version)"
run python -c "import cosmos_rl, os; print('cosmos_rl imported from:', cosmos_rl.__file__)"

# run tests
run python tests/test_apex.py
run python tests/test_cosmos_hf_precision.py
run /bin/bash -c "CP_SIZE=2 TP_SIZE=1 DP_SIZE=2 torchrun --nproc_per_node=4 tests/test_context_parallel.py"
run python tests/test_cache.py
run python tests/test_comm.py
run python tests/test_fp8.py
run python tests/test_lora.py
run python tests/test_freeze_pattern.py
# run python tests/test_grad_allreduce.py
run python tests/test_high_availability_nccl.py
run python tests/test_nccl_collectives.py
run python tests/test_nccl_timeout.py
run python tests/test_parallel_map.py
run python tests/test_policy_to_policy.py
run python tests/test_policy_to_rollout.py
run python tests/test_process_flow.py
run python tests/test_custom_class.py
run python tests/test_math_verify.py
run python tests/test_policy_overfit.py
run python tests/test_data_packer.py
run python tests/test_dataset_signature.py
run python tests/test_sequence_packing.py
run python tests/test_integration.py --stream
run python tests/test_hf_models.py
run /bin/bash -c "torchrun --nproc_per_node=2 tests/test_hf_models_tp.py"
run python tests/test_activation_offload.py
run python tests/test_policy_variant.py
run python tests/test_deepep.py
run python tests/test_colocated.py
run python tests/test_teacher_model.py
run /bin/bash -c "torchrun --nproc_per_node=4 tests/test_qwen3_vl_moe.py"
run python tests/test_vllm_rollout_async.py
run python tests/test_custom_args.py
run python tests/test_colocated_separated.py
run python tests/test_load_balanced_dataset.py
run python tests/test_resume_data_index.py
run /bin/bash -c "torchrun --nproc_per_node=8 tests/test_data_loader.py"
run python tests/test_diffusion_rl_e2e.py
run /bin/bash -c "torchrun --nproc_per_node=8 tests/test_cosmos3_trajectory_equivalence.py"
run /bin/bash -c "torchrun --nproc_per_node=8 tests/test_dpo_direct.py --tp_size 8"
run python tests/test_wfm_dpo.py
run python tests/test_wfm_nft.py
run python tests/test_refactor_contracts.py

if (( ${#FAILED[@]} > 0 )); then
    echo
    echo "================ SUMMARY: ${#FAILED[@]} test(s) failed ================"
    printf '  - %s\n' "${FAILED[@]}"
    exit 1
fi

echo
echo "================ SUMMARY: all tests passed ================"
