import argparse
import os

import torch
from peft import AutoPeftModelForCausalLM, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def convert_checkpoint(base_model, ckpt_folder, output_folder, dtype):
    torch_dtype = DTYPE_MAP[dtype]

    if base_model:
        print(f"Loading base model from {base_model}...")
        base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch_dtype)
        print(f"Loading LoRA adapter from {ckpt_folder}...")
        model = PeftModel.from_pretrained(base, ckpt_folder, is_trainable=False)
    else:
        print(
            "Loading PEFT checkpoint (base model resolved from adapter_config.json) "
            f"from {ckpt_folder}..."
        )
        model = AutoPeftModelForCausalLM.from_pretrained(
            ckpt_folder, torch_dtype=torch_dtype
        )

    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(ckpt_folder)
    except Exception:
        if not base_model:
            raise
        tokenizer = AutoTokenizer.from_pretrained(base_model)

    print("Merging LoRA adapters into base model...")
    merged_model = model.merge_and_unload()

    print(f"Saving merged model and tokenizer to {output_folder}...")
    os.makedirs(output_folder, exist_ok=True)
    merged_model.save_pretrained(output_folder)
    tokenizer.save_pretrained(output_folder)

    print("Conversion complete.")
    print(f"Fused model saved to: {output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert PEFT LoRA checkpoint (single safetensors format) to fused format."
    )
    parser.add_argument(
        "--ckpt_folder",
        type=str,
        required=True,
        help="Path to checkpoint folder containing adapter_model.safetensors.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Path to save fused checkpoint.",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help=(
            "Optional base model name/path. "
            "If omitted, base model is read from adapter_config.json."
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
        help="Torch dtype used when loading models.",
    )

    args = parser.parse_args()
    convert_checkpoint(
        base_model=args.base_model,
        ckpt_folder=args.ckpt_folder,
        output_folder=args.output_folder,
        dtype=args.dtype,
    )
