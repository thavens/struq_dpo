import argparse
import json
import os

import torch
from peft import PeftModel, get_peft_model, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts
from gpt_moe_layer import GptOssExpertsLora
from safetensors import safe_open


DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

def load_gpt_oss(base_model, ckpt_folder, dtype):
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype, device_map=None, quantization_config=Mxfp4Config(dequantize=True))
    with open(os.path.join(ckpt_folder, "adapter_config.json")) as f:
        adapter_config = json.load(f)
    peft_config = LoraConfig(**adapter_config)
    peft_config._register_custom_module({
        GptOssExperts: GptOssExpertsLora
    })
    model = get_peft_model(model, peft_config)
    tensors = {}
    with safe_open(os.path.join(ckpt_folder, "adapter_model.safetensors"), "pt") as f:
        for k in f.keys():
            left, right = k.rsplit(".", 1)
            new_name = left + ".default." + right
            tensors[new_name] = f.get_tensor(k)
    xor = model.load_state_dict(tensors, strict=False)
    assert not xor.unexpected_keys, f"{"=" * 10} Missing Keys {"=" * 10}\n{xor.missing_keys}\n{"=" * 10} Unexpected Keys {"=" * 10}\n{xor.unexpected_keys}"
    return model
    

def convert_checkpoint(base_model, ckpt_folder, output_folder, dtype):
    torch_dtype = DTYPE_MAP[dtype]

    if "oss" in base_model:
        model = load_gpt_oss(base_model, ckpt_folder, dtype=dtype)
    else:
        print(f"Loading base model from {base_model}...")
        base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch_dtype)
        print(f"Loading LoRA adapter from {ckpt_folder}...")
        model = PeftModel.from_pretrained(base, ckpt_folder, is_trainable=False)

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
        required=True,
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
