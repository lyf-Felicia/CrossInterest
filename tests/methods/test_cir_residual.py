import copy

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

from crossinterest.methods.residual.model import FrozenCirLinear, freeze_cir_and_reset_n


def tiny_cir(dtype=torch.float32):
    torch.manual_seed(29)
    model = get_peft_model(LlamaForCausalLM(LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, tie_word_embeddings=False)).to(dtype),
        LoraConfig(r=2, lora_alpha=4, lora_dropout=0.05,
                   target_modules=["q_proj", "v_proj", "lm_head"],
                   trainable_token_indices=[30, 31], task_type="CAUSAL_LM"))
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=0.1)
    return model


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_residual_initial_function_and_gradient_isolation(dtype):
    model = tiny_cir(dtype).eval()
    inputs = torch.tensor([[1, 30, 5, 31]])
    expected = model(inputs).logits.detach()
    token_rows = {name: value.detach().clone() for name, value in model.named_parameters()
                  if "trainable_tokens_delta" in name}
    count = sum(value.numel() for value in model.parameters() if value.requires_grad)
    receipt = freeze_cir_and_reset_n(model)
    assert receipt["trainable_parameters"] == count
    torch.testing.assert_close(model(inputs).logits, expected, rtol=0, atol=0)
    frozen = {name: value.clone() for name, value in model.named_buffers() if "cir_" in name}
    optimizer = torch.optim.AdamW([value for value in model.parameters() if value.requires_grad], lr=1e-3)
    model.train()
    loss = model(inputs, labels=inputs).loss
    assert torch.isfinite(loss)
    loss.backward()
    assert any(value.grad is not None and value.grad.abs().sum() > 0
               for name, value in model.named_parameters() if "lora_B" in name)
    assert all(value.grad is None for name, value in model.named_parameters() if not value.requires_grad)
    for name, value in model.named_parameters():
        if name in token_rows:
            torch.testing.assert_close(value, token_rows[name], rtol=0, atol=0)
    optimizer.step()
    for name, value in model.named_buffers():
        if name in frozen:
            torch.testing.assert_close(value, frozen[name], rtol=0, atol=0)
    assert sum(isinstance(layer, FrozenCirLinear) for layer in model.modules()) == len(receipt["modules"])
    with pytest.raises(ValueError, match="unsupported CIR"):
        freeze_cir_and_reset_n(model)


def test_residual_adapter_reload(tmp_path):
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    model = tiny_cir()
    original = copy.deepcopy(model)
    freeze_cir_and_reset_n(model)
    inputs = torch.tensor([[1, 30, 5, 31]])
    model(inputs, labels=inputs).loss.backward()
    optimizer = torch.optim.AdamW([value for value in model.parameters() if value.requires_grad], lr=1e-3)
    optimizer.step()
    model.eval().save_pretrained(tmp_path, save_embedding_layers=False)
    state = load_file(tmp_path / "adapter_model.safetensors")
    assert not any("cir_a" in name or "cir_b" in name for name in state)
    freeze_cir_and_reset_n(original)
    result = set_peft_model_state_dict(original, state)
    assert not result.unexpected_keys
    torch.testing.assert_close(original.eval()(inputs).logits, model(inputs).logits, rtol=0, atol=0)