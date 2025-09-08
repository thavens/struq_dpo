from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.distributed.checkpoint as dist_cp
from peft import LoraConfig, get_peft_model
import argparse
import os


def load_sharded_model_single_gpu(model, peft_path):
    adapter_dict = {k: v for k, v in model.state_dict().items() if "lora_" in k}
    adapter_change = {k.replace(".default", ""): k for k in adapter_dict.keys()}
    new_adapter_dict = {k.replace(".default", ""): v for k, v in adapter_dict.items()}
    state_dict = {"model": new_adapter_dict}

    dist_cp.load_state_dict(
        state_dict=state_dict,
        storage_reader=dist_cp.FileSystemReader(peft_path),
        no_dist=True,
    )

    print(f" new adapter dict keys: {list(adapter_change.items())} ")

    model.load_state_dict(
        {adapter_change[k]: v for k, v in state_dict["model"].items()}, strict=False
    )
    print(f"Sharded state checkpoint loaded from {peft_path}")
    return model


def convert_checkpoint(base_model, ckpt_folder, output_folder):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype="bfloat16",
    )

    peft_config = LoraConfig(
        r=32,
        lora_alpha=8,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_proj", "v_proj", "gate_proj", "down_proj", "up_proj"],
    )

    model = get_peft_model(model, peft_config)
    print(model.state_dict().keys())

    model = load_sharded_model_single_gpu(model, ckpt_folder)

    print("Merging LoRA adapters into the base model...")
    # This combines the adapter weights with the base model weights
    merged_model = model.merge_and_unload()

    print(f"Saving merged model and tokenizer to {output_folder}...")
    os.makedirs(output_folder, exist_ok=True)
    merged_model.save_pretrained(output_folder)
    tokenizer.save_pretrained(output_folder)

    print("\nConversion complete!")
    print(f"Fused model has been saved to: {output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert sharded checkpoint to fused format."
    )
    parser.add_argument(
        "--base_model", type=str, required=True, help="Base model name or path."
    )
    parser.add_argument(
        "--ckpt_folder",
        type=str,
        required=True,
        help="Path to sharded checkpoint folder.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Path to save fused checkpoint.",
    )

    args = parser.parse_args()
    convert_checkpoint(args.base_model, args.ckpt_folder, args.output_folder)
