from pathlib import Path
import sys

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt.config import GPT, Config, Tokenizer
from transformers import AutoModelForCausalLM
import torch
from jsonargparse import CLI


def main(
    checkpoint_dir: Path = Path("checkpoints/stabilityai/stablelm-base-alpha-3b"),
) -> None:
    config = Config.from_json(checkpoint_dir / "lit_config.json")
    tokenizer = Tokenizer(checkpoint_dir)
    model = GPT(config)
    model.push_to_hub('takuma-intern/fastlabelLM')

if __name__ == "__main__":

    torch.set_float32_matmul_precision("high")
    CLI(main)
