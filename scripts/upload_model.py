# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import sys
from pathlib import Path
import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate.base import generate
from lit_gpt import Config
from lit_gpt.utils import check_valid_checkpoint_dir
from scripts.prepare_alpaca import generate_prompt
from scripts.convert_lit_checkpoint import convert_lit_checkpoint
from transformers import GPTNeoXConfig, AutoModelForCausalLM
from huggingface_hub import login

login(token='')

def convert_config(config_path):
    t_config = GPTNeoXConfig()
    config = Config.from_json(config_path)

    t_config.hidden_size = config.n_embd
    t_config.max_position_embeddings = config.block_size
    t_config.intermediate_size = config.intermediate_size

    t_config.num_attention_heads = config.n_head
    t_config.num_hidden_layers = config.n_layer
    t_config.num_key_value_heads = config.n_query_groups

    t_config.rms_norm_eps = config.norm_eps

    ## tokenizer config
    t_config.bos_token_id = 1
    t_config.eos_token_id = 2
    t_config.vocab_size = 35000 + 8

    return t_config


def main(
    checkpoint_path: Path = Path("out/fastlabelLM/iter-004096-ckpt.pth"),
    weight_path: Path = Path("out/full/alpaca/lit_model_finetuned.pth"),
    config_path: Path = Path("checkpoints/stabilityai/stablelm-base-alpha-3b"),
) -> None:
    """Generates a response based on a given instruction and an optional input.
    This script will only work with checkpoints from the instruction-tuned GPT model.
    See `finetune/full.py`.

    Args:
        prompt: The prompt/instruction (Alpaca style).
        input: Optional input (Alpaca style).
        finetuned_path: Path to the checkpoint with trained weights, which are the output of
            `finetune/full.py`.
        checkpoint_dir: The path to the checkpoint folder with pretrained GPT weights.
        quantize: Whether to quantize the model and using which method:
            - bnb.nf4, bnb.nf4-dq, bnb.fp4, bnb.fp4-dq: 4-bit quantization from bitsandbytes
            - bnb.int8: 8-bit quantization from bitsandbytes
            for more details, see https://github.com/Lightning-AI/lit-gpt/blob/main/tutorials/quantize.md
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
        precision: Indicates the Fabric precision setting to use.
    """
    convert_lit_checkpoint(
        checkpoint_path=checkpoint_path,
        output_path=weight_path,
        config_path=config_path
    )

    config = convert_config(config_path)
    model = AutoModelForCausalLM.from_config(config)

    weights = torch.load(weight_path)
    model.load_state_dict(weights)

    model.push_to_hub('Takuma-intern/fastlabelLM')




if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    CLI(main)
