from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def fused_moe_backward_input_kernel(
    # Pointers to matrices
    grad_output_ptr,
    weight_ptr,
    grad_input_ptr,
    grad_topk_weights_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # Strides
    stride_gom,
    stride_gon,
    stride_we,
    stride_wn,
    stride_wk,
    stride_gim,
    stride_gik,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
):
    """
    Backward kernel for computing grad_input.

    Forward: output = input @ weight.T (optionally multiplied by topk_weights)
    Backward: grad_input = grad_output @ weight (optionally multiplied by topk_weights)

    This kernel computes the weighted expert contribution for each token.
    If MUL_ROUTED_WEIGHT: grad_input[token] *= topk_weights[token]

    Parallelization: Similar to forward, parallel over M and N dimensions, loop over K.
    """
    # Map program ids to blocks (parallel over M and N, similar to forward)
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Check bounds
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)

    # Only process if this block is valid
    if pid_m * BLOCK_SIZE_M < num_tokens_post_padded:
        # Load token information
        offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
        offs_token = offs_token.to(tl.int64)
        token_mask = offs_token < num_valid_tokens

        # Get expert ID for this block
        off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

        # Only process if expert is valid
        if off_experts != -1:
            # Initialize offsets for N dimension (current block)
            offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
            offs_k = tl.arange(0, BLOCK_SIZE_K)

            # Load grad_output block: shape (BLOCK_SIZE_M, BLOCK_SIZE_N)
            grad_output_ptrs = grad_output_ptr + (
                offs_token[:, None] * stride_gom + offs_n[None, :] * stride_gon
            )
            grad_out = tl.load(
                grad_output_ptrs, mask=token_mask[:, None] & (offs_n[None, :] < N), other=0.0
            )

            # Apply topk_weights to grad_output if needed
            if MUL_ROUTED_WEIGHT:
                moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
                grad_out = grad_out * moe_weight[:, None]

            # Iterate over K dimension
            for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                # Current K offsets
                curr_offs_k = k * BLOCK_SIZE_K + offs_k

                # Load weight block: shape (BLOCK_SIZE_N, BLOCK_SIZE_K)
                # weight: shape (E, N, K)
                weight_ptrs = (
                    weight_ptr
                    + off_experts * stride_we
                    + offs_n[:, None] * stride_wn
                    + curr_offs_k[None, :] * stride_wk
                )
                w = tl.load(
                    weight_ptrs, mask=(offs_n[:, None] < N) & (curr_offs_k[None, :] < K), other=0.0
                )

                # Compute contribution: grad_out @ weight
                # grad_out: (BLOCK_SIZE_M, BLOCK_SIZE_N)
                # w: (BLOCK_SIZE_N, BLOCK_SIZE_K)
                # result: (BLOCK_SIZE_M, BLOCK_SIZE_K)
                contribution = tl.dot(grad_out, w.to(grad_out.dtype))

                # Atomic add to grad_input because different N blocks contribute to same K
                grad_input_ptrs = grad_input_ptr + (
                    (offs_token[:, None] // top_k) * stride_gim + curr_offs_k[None, :] * stride_gik
                )
                grad_input_mask = token_mask[:, None] & (curr_offs_k[None, :] < K)
                tl.atomic_add(grad_input_ptrs, contribution.to(compute_type), mask=grad_input_mask)
