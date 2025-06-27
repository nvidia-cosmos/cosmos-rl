import torch

import triton
import triton.language as tl


@triton.jit
def _compute_logprobs_forward(
    input_ids_ptr,  # [bsz, max_len]
    full_logits_ptr,  # [bsz, max_len, vocab_size]
    seqlen_start_idx_ptr,  # [bsz]
    seqlen_end_idx_ptr,  # [bsz]
    logprobs_ptr,  # [n_logprob_tokens]
    cu_seqlens,  # [bsz + 1]
    sumexp_ptr,  # [n_logprob_tokens]
    max_len,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute the per-token log probabilities and return the flattened logprobs (total elements are)

    Args:
        input_ids_ptr (_type_): _description_
    """
    pid = tl.program_id(0)
    bsz_idx = (pid // max_len).to(tl.int64)  # avoid int32 overflow
    token_idx = (pid % max_len).to(tl.int64)
    seqlen_start_idx = tl.load(seqlen_start_idx_ptr + bsz_idx)
    seqlen_end_idx = tl.load(seqlen_end_idx_ptr + bsz_idx)
    if token_idx < seqlen_start_idx or token_idx > seqlen_end_idx:
        return

    start_addr_of_token = (
        full_logits_ptr + bsz_idx * max_len * vocab_size + token_idx * vocab_size
    )
    index_of_token = tl.load(input_ids_ptr + bsz_idx * max_len + token_idx)  # int32
    bsz_cu_seqlen = tl.load(cu_seqlens + bsz_idx)
    addr_of_logprobs = logprobs_ptr + bsz_cu_seqlen + token_idx - seqlen_start_idx
    addr_of_sumexp = sumexp_ptr + bsz_cu_seqlen + token_idx - seqlen_start_idx

    # Compute sumexp
    sum_exp = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in range(0, vocab_size, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        mask = offset < vocab_size
        x = tl.load(
            start_addr_of_token + i + tl.arange(0, BLOCK_SIZE),
            mask=mask,
            other=-float("inf"),
        ).to(tl.float32)
        sum_exp += tl.exp(x)

    sum_exp_total = tl.sum(sum_exp)
    logsum_exp = tl.log(sum_exp_total)
    tl.store(addr_of_sumexp, logsum_exp)

    logit_of_current_token = tl.load(start_addr_of_token + index_of_token).to(
        tl.float32
    )  # scalar
    # compute softmax_logsumexp
    logprob_of_current_token = logit_of_current_token - logsum_exp
    tl.store(
        addr_of_logprobs, logprob_of_current_token.to(full_logits_ptr.dtype.element_ty)
    )


@triton.jit
def _compute_logprobs_backward(
    gradient_output_ptr,  # [n_logprob_tokens]
    gradient_full_logits_ptr,  # [bsz, max_len, vocab_size]
    input_ids_ptr,  # [bsz, max_len]
    full_logits_ptr,  # [bsz, max_len, vocab_size]
    seqlen_start_idx_ptr,  # [bsz]
    seqlen_end_idx_ptr,  # [bsz]
    cu_seqlens,  # [bsz + 1]
    sumexp_ptr,  # [n_logprob_tokens]
    max_len,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    bsz_idx = (pid // max_len).to(tl.int64)  # avoid int32 overflow
    token_idx = (pid % max_len).to(tl.int64)
    seqlen_start_idx = tl.load(seqlen_start_idx_ptr + bsz_idx)
    seqlen_end_idx = tl.load(seqlen_end_idx_ptr + bsz_idx)
    if token_idx < seqlen_start_idx or token_idx > seqlen_end_idx:
        return

    bsz_cu_seqlen = tl.load(cu_seqlens + bsz_idx)

    addr_of_sumexp = sumexp_ptr + bsz_cu_seqlen + token_idx - seqlen_start_idx
    sum_exp_total = tl.load(addr_of_sumexp)

    start_addr_of_token = (
        full_logits_ptr + bsz_idx * max_len * vocab_size + token_idx * vocab_size
    )

    addr_of_gradient_output = (
        gradient_output_ptr + bsz_cu_seqlen + token_idx - seqlen_start_idx
    )
    gradient_output = tl.load(
        addr_of_gradient_output
    )  # scalar, gradient of the logit with index `index_of_token`

    # [1, 1, vocab_size]
    addr_of_gradient_full_logits = (
        gradient_full_logits_ptr
        + bsz_idx * max_len * vocab_size
        + token_idx * vocab_size
    )

    index_of_token = tl.load(input_ids_ptr + bsz_idx * max_len + token_idx)  # int32

    for i in range(0, vocab_size, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        mask = offset < vocab_size  # [BLOCK_SIZE,]

        # recompute the output of log_softmax
        x = tl.load(
            start_addr_of_token + i + tl.arange(0, BLOCK_SIZE),
            mask=mask,
            other=0,
        ).to(tl.float32)
        log_softmax_output = x - sum_exp_total
        exp_log_softmax_output = tl.exp(log_softmax_output)

        # gradient_x_i = gradient_output_i - exp(o_i) * sum_j(gradient_output_j)
        gradient_output_block = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        tl.where(offset == index_of_token, gradient_output, gradient_output_block)

        gradient_input_block = (
            gradient_output_block - exp_log_softmax_output * gradient_output
        )

        tl.store(
            addr_of_gradient_full_logits + i + tl.arange(0, BLOCK_SIZE),
            gradient_input_block,
            mask=mask,
        )


class TritonComputeLogprobs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_ids, logprob_masks, full_logits):
        bsz, max_len, vocab_size = full_logits.shape
        assert (
            input_ids.shape[:2] == logprob_masks.shape[:2]
        ), "input_ids and logprob_masks should have the same shape"
        assert (
            full_logits.shape[:2] == input_ids.shape[:2]
        ), "full_logits should have the same shape as input_ids"

        shift_input_ids = torch.empty_like(input_ids)
        shift_input_ids[:, :-1] = input_ids[:, 1:]
        shift_input_ids[:, -1] = 0

        n_logprob_tokens = logprob_masks.sum()
        logprobs = torch.empty(
            n_logprob_tokens, dtype=full_logits.dtype, device=full_logits.device
        )

        masked_seqlens = logprob_masks.sum(dim=-1)  # [bsz,]
        cu_seqlens = torch.zeros(bsz + 1, dtype=torch.int32, device=full_logits.device)
        cu_seqlens[1:] = torch.cumsum(masked_seqlens, dim=0)  # [bsz + 1,]
        int_logprob_masks = logprob_masks.int()
        seqlen_start_idx = int_logprob_masks.argmax(dim=-1)  # [bsz,]
        reversed_int_logprob_masks = int_logprob_masks.flip(dims=[-1])
        seqlen_end_idx = (
            int_logprob_masks.size(1) - 1 - reversed_int_logprob_masks.argmax(dim=-1)
        )

        sum_exp = torch.zeros(
            (n_logprob_tokens), dtype=torch.float32, device=full_logits.device
        )

        vocab_size = full_logits.shape[-1]
        MAX_FUSED_SIZE = 65536 // full_logits.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(vocab_size))
        grid = (bsz * max_len,)
        _compute_logprobs_forward[grid](
            input_ids,  # [bsz, max_len]
            full_logits,  # [bsz, max_len, vocab_size]
            seqlen_start_idx,  # [bsz,]
            seqlen_end_idx,  # [bsz,]
            logprobs,  # [n_logprob_tokens,]
            cu_seqlens,  # [bsz + 1,]
            sum_exp,  # [n_logprob_tokens,]
            max_len,
            vocab_size,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        ctx.save_for_backward(
            input_ids,
            full_logits,
            seqlen_start_idx,
            seqlen_end_idx,
            cu_seqlens,
            sum_exp,
        )
        return logprobs, cu_seqlens

    @staticmethod
    def backward(ctx, grad_output, grad_cu_seqlens):
        # https://math.stackexchange.com/questions/4258008/derivative-of-the-log-softmax-function
        # https://stackoverflow.com/questions/35304393/trying-to-understand-code-that-computes-the-gradient-wrt-to-the-input-for-logsof
        (
            input_ids,
            full_logits,
            seqlen_start_idx,
            seqlen_end_idx,
            cu_seqlens,
            sum_exp,
        ) = ctx.saved_tensors
        bsz, max_len, vocab_size = full_logits.shape

        # most of the gradient of full_logits is 0
        gradient_full_logits = torch.zeros_like(full_logits)

        MAX_FUSED_SIZE = 65536 // full_logits.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(vocab_size))
        grid = (bsz * max_len,)
        _compute_logprobs_backward[grid](
            grad_output,  # [n_logprob_tokens,]
            gradient_full_logits,  # [bsz, max_len, vocab_size]
            input_ids,  # [bsz, max_len]
            full_logits,  # [bsz, max_len, vocab_size]
            seqlen_start_idx,  # [bsz,]
            seqlen_end_idx,  # [bsz,]
            cu_seqlens,  # [bsz + 1,]
            sum_exp,  # [n_logprob_tokens,]
            max_len,
            vocab_size,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return (None, None, gradient_full_logits)
