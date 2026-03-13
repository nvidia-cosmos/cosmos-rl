python -c "from cosmos_rl._version import version; print(version)"
python -c "import cosmos_rl, os; print('cosmos_rl imported from:', cosmos_rl.__file__)"
# run tests
python tests/test_apex.py
python tests/test_cosmos_hf_precision.py
/bin/bash -c "CP_SIZE=2 TP_SIZE=1 DP_SIZE=2 torchrun --nproc_per_node=4 tests/test_context_parallel.py"
python tests/test_cache.py
python tests/test_comm.py
python tests/test_fp8.py
python tests/test_lora.py
python tests/test_freeze_pattern.py
# python tests/test_grad_allreduce.py
python tests/test_high_availability_nccl.py
python tests/test_nccl_collectives.py
python tests/test_nccl_timeout.py
python tests/test_parallel_map.py
python tests/test_policy_to_policy.py
python tests/test_policy_to_rollout.py
python tests/test_process_flow.py
python tests/test_custom_class.py
python tests/test_math_verify.py
python tests/test_policy_overfit.py
python tests/test_data_packer.py
python tests/test_dataset_signature.py
python tests/test_sequence_packing.py
python tests/test_integration.py --stream
python tests/test_hf_models.py
/bin/bash -c "torchrun --nproc_per_node=2 tests/test_hf_models_tp.py"
python tests/test_activation_offload.py
python tests/test_policy_variant.py
python tests/test_deepep.py
python tests/test_colocated.py
python tests/test_teacher_model.py
/bin/bash -c "torchrun --nproc_per_node=4 tests/test_qwen3_vl_moe.py"
python tests/test_vllm_rollout_async.py
python tests/test_custom_args.py
python tests/test_colocated_separated.py
python tests/test_load_balanced_dataset.py
python tests/test_resume_data_index.py
/bin/bash -c "torchrun --nproc_per_node=8 tests/test_data_loader.py"
