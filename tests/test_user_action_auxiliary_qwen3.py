import torch
from peft import LoraConfig, get_peft_model
from test_user_action_auxiliary import FakeTokenizer, make_args, make_sample
from transformers import Qwen3Config, Qwen3ForCausalLM

from llamafactory.data.action_select import ActionSelectMetadataParser
from llamafactory.train.sft.user_action_auxiliary import UserActionAuxiliaryController


def test_tiny_random_qwen3_lora_uses_one_forward_and_auxiliary_backward():
    torch.manual_seed(17)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        attention_dropout=0.0,
        use_cache=False,
    )
    model = get_peft_model(
        Qwen3ForCausalLM(config),
        LoraConfig(
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj", "o_proj", "down_proj"],
        ),
    ).to(device)
    model.eval()
    sample = make_sample()
    parser = ActionSelectMetadataParser(FakeTokenizer())
    metadata = parser.parse(sample["input_ids"], sample["labels"])
    input_ids = torch.tensor([sample["input_ids"]], device=device)
    labels = torch.tensor([sample["labels"]], device=device)
    forward_count = 0

    def count_forward(module, inputs, outputs):
        nonlocal forward_count
        forward_count += 1

    handle = model.register_forward_hook(count_forward)
    outputs = model(input_ids=input_ids, labels=labels)
    controller = UserActionAuxiliaryController(
        FakeTokenizer(), make_args(user_action_aux_vectorized_enabled=True)
    )
    result = controller.compute(outputs.logits, labels, [metadata], outputs.loss, 100)
    (outputs.loss + result.loss).backward()
    handle.remove()
    assert forward_count == 1
    assert result.loss > 0 and torch.isfinite(result.loss)
    lora_b_gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if "lora_B" in name and parameter.grad is not None
    ]
    assert lora_b_gradients and any(gradient.abs().sum() > 0 for gradient in lora_b_gradients)


if __name__ == "__main__":
    test_tiny_random_qwen3_lora_uses_one_forward_and_auxiliary_backward()
    print("PASS: tiny random Qwen3+LoRA vectorized Action auxiliary smoke")
