import torch
from transformers import Qwen3Config, Qwen3ForCausalLM


def main() -> None:
    torch.manual_seed(42)
    config = Qwen3Config.from_pretrained("Qwen/Qwen3-0.6B")
    model = Qwen3ForCausalLM(config)
    print(model)


if __name__ == "__main__":
    main()
