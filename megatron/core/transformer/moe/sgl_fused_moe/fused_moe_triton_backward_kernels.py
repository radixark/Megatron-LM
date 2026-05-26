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


@triton.jit
def fused_moe_backward_weight_kernel(
    # Pointers to matrices
    grad_output_ptr,
    input_ptr,
    grad_weight_ptr,
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
    stride_im,
    stride_ik,
    stride_gwe,
    stride_gwn,
    stride_gwk,
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
    Backward kernel for computing grad_weight.

    Forward: output = input @ weight.T, optionally scaled by top-k weights.
    Backward accumulates each expert's weight gradient across routed tokens.

    Parallelization: Parallel over M and N dimensions with grouping, loop over K.
    """
    # Map program ids to blocks, similar to forward and backward_input.
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
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    # Get expert ID for this M block
    expert_id = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    # Only process if expert is valid
    if expert_id == -1:
        return

    # Load token information for this M block
    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_token_id = pid_m * BLOCK_SIZE_M + offs_m.to(tl.int64)
    offs_token = tl.load(
        sorted_token_ids_ptr + offs_token_id,
        mask=offs_token_id < num_tokens_post_padded,
        other=num_valid_tokens,
    )
    offs_token = offs_token.to(tl.int64)
    token_mask = (offs_token_id < num_tokens_post_padded) & (offs_token < num_valid_tokens)

    # Clamp offs_token to valid range
    offs_token_clamped = tl.where(token_mask, offs_token, 0)

    # Determine input token indices based on MUL_ROUTED_WEIGHT
    if MUL_ROUTED_WEIGHT:
        input_token_idx = offs_token_clamped
        input_mask = token_mask
    else:
        input_token_idx = offs_token_clamped // top_k
        num_input_tokens = num_valid_tokens // top_k
        input_mask = token_mask & (input_token_idx < num_input_tokens)

    # Load topk_weights if needed
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token_clamped, mask=token_mask, other=0.0)

    # Current N offset for this program
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)

    # Load grad_output for this N block: shape (M, BLOCK_SIZE_N)
    # grad_output is always indexed by sorted_token_ids (offs_token_clamped)
    # because it has shape (num_tokens * topk, N)
    grad_output_ptrs = grad_output_ptr + (
        offs_token_clamped[:, None] * stride_gom + offs_n[None, :] * stride_gon
    )
    grad_out = tl.load(
        grad_output_ptrs, mask=token_mask[:, None] & (offs_n[None, :] < N), other=0.0
    )

    # Apply topk_weights if needed
    if MUL_ROUTED_WEIGHT:
        grad_out = grad_out * moe_weight[:, None]

    # Zero out padding tokens
    token_mask_col = token_mask[:, None]
    grad_out = grad_out * token_mask_col

    # Iterate over K blocks and accumulate
    for k_block in range(tl.cdiv(K, BLOCK_SIZE_K)):
        offs_k = k_block * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)

        # Load input for this K block
        input_ptrs = input_ptr + (
            input_token_idx[:, None] * stride_im + offs_k[None, :] * stride_ik
        )
        inp = tl.load(input_ptrs, mask=input_mask[:, None] & (offs_k[None, :] < K), other=0.0)

        # Zero out padding tokens - use input_mask for input, token_mask for grad_output
        input_mask_col = input_mask[:, None]
        inp = inp * input_mask_col

        # Compute grad_weight contribution: grad_out.T @ inp
        grad_w_contribution = tl.dot(grad_out.to(compute_type).T, inp.to(compute_type))

        # Write back using atomic add
        grad_weight_ptrs = (
            grad_weight_ptr
            + expert_id * stride_gwe
            + offs_n[:, None] * stride_gwn
            + offs_k[None, :] * stride_gwk
        )
        grad_weight_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        tl.atomic_add(grad_weight_ptrs, grad_w_contribution.to(compute_type), mask=grad_weight_mask)
