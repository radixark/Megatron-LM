import pytest
import torch

from megatron.core.models.gpt.gpt_model import apply_true_on_policy_logits_contract


def test_true_on_policy_logits_contract_truncates_and_preserves_dtype():
    full_vocab_logits = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 100.0, 100.0]], dtype=torch.bfloat16
    )

    actual = apply_true_on_policy_logits_contract(full_vocab_logits, vocab_size=6)
    expected = full_vocab_logits[:, :6]

    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected)


def test_true_on_policy_logprob_uses_real_vocab_after_full_gather():
    shard_0 = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.bfloat16)
    shard_1 = torch.tensor([[5.0, 6.0, 100.0, 100.0]], dtype=torch.bfloat16)
    gathered_logits = torch.cat([shard_0, shard_1], dim=-1)

    contracted_logits = apply_true_on_policy_logits_contract(gathered_logits, vocab_size=6)
    contracted_logprob = torch.nn.functional.log_softmax(contracted_logits, dim=-1)[0, 5]

    wrong_full_vocab_logprob = torch.nn.functional.log_softmax(gathered_logits.float(), dim=-1)[
        0, 5
    ]
    expected_logprob = torch.nn.functional.log_softmax(gathered_logits[:, :6], dim=-1)[0, 5]

    torch.testing.assert_close(contracted_logprob, expected_logprob)
    assert not torch.allclose(contracted_logprob.float(), wrong_full_vocab_logprob)


def test_true_on_policy_logits_contract_rejects_vocab_larger_than_logits():
    with pytest.raises(RuntimeError, match="exceeds gathered logits width"):
        apply_true_on_policy_logits_contract(torch.randn(2, 4), vocab_size=5)
