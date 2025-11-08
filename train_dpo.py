#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# A script to perform DPO training on a Qwen model using the TRL library.
#
# This script is designed to be configured via a YAML file, making it easy to
# manage experiments and hyperparameters. It uses the Hugging Face
# HfArgumentParser to parse arguments defined in dataclasses.
#
# Usage:
# python train_dpo.py --config dpo_config.yaml

import json
import os
from dataclasses import dataclass, field
from typing import Optional
import yaml
import hydra
from omegaconf import DictConfig
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
)
from peft import LoraConfig
from trl import DPOTrainer, DPOConfig


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    dtype: Optional[str] = field(
        default="bfloat16",
        metadata={
            "help": "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
            "dtype will be automatically derived from the model's weights.",
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    attn_implementation: Optional[str] = field(
        default="flash_attention_2",
    )
    attention_dropout: Optional[float] = field(default=0.0)


@dataclass
class TokenizerArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    tokenizer_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained tokenizer or tokenizer identifier from huggingface.co/models"
        }
    )
    chat_template: str = field(
        default="",
        metadata={
            "help": "Path to custom chat template jinja file.",
        },
    )


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to use.
    """

    dataset_name: str = field(
        metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(cfg)
    model_args = ModelArguments(**cfg["model_args"])
    tokenizer_args = TokenizerArguments(**cfg["tokenizer_args"])
    data_args = DataArguments(**cfg["data_args"])
    trainer_args = DPOConfig(**cfg["trainer_args"])
    peft_config = LoraConfig(**cfg["lora_args"])

    # 2. Load model and tokenizer
    device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        dtype=model_args.dtype,
        device_map=device_map,
        trust_remote_code=True,
        attn_implementation=model_args.attn_implementation,
        attention_dropout=model_args.attention_dropout,
    )
    print(model)
    # The reference model is not used in the new TRL DPOTrainer,
    # but it's good practice to have it for other algorithms like PPO.
    # For DPO, the trainer creates a reference model internally.
    model_ref = None

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_args.tokenizer_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer_args.chat_template:
        with open(tokenizer_args.chat_template, "r") as f:
            chat_template = f.read()
        tokenizer.chat_template = chat_template
        print(f"Loaded custom chat template from {tokenizer_args.chat_template}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set pad_token to eos_token")

    # Add enable_thinking support to non qwen chat templates.
    if "enable_thinking" not in tokenizer.chat_template:
        tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{%- set ns.is_first = true -%}{%- else %}{{'\\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}{{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\\n<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<｜Assistant｜><think>\\n'}}{%- if enable_thinking is defined and enable_thinking is false %}\n        {{- '\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}"

    # 3. Load and process dataset
    # The thavens/Qwen3-4B-secalign dataset has the right columns: 'prompt', 'chosen', 'rejected'.
    # The data is in a conversational format, which DPOTrainer can handle directly.
    dataset = load_dataset(data_args.dataset_name)

    def format_dataset(example):
        """
        Applies the chat template to the prompt, chosen, and rejected columns.
        This is crucial for training conversational models.
        """
        if (
            isinstance(example["prompt"], list)
            and isinstance(example["chosen"], list)
            and isinstance(example["rejected"], list)
        ):
            # The DPO trainer expects string inputs, not tokenized inputs.
            # We apply the chat template to convert the list of messages to a single string.
            if "tools" in example:
                try:
                    example["prompt"] = tokenizer.apply_chat_template(
                        example["prompt"],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                        tools=json.loads(example["tools"]),
                    )
                except:
                    import pudb

                    pudb.post_mortem()
            else:
                example["prompt"] = tokenizer.apply_chat_template(
                    example["prompt"],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            example["chosen"] = example["chosen"][0]["content"] + "<|im_end|>\n"
            example["rejected"] = example["rejected"][0]["content"] + "<|im_end|>\n"

        return example

    train_dataset = dataset["train"].map(
        format_dataset,
    )
    print(train_dataset[0])

    # You can optionally format an evaluation dataset if you have one
    # eval_dataset = dataset["test"]

    # 5. Initialize DPOTrainer
    trainer = DPOTrainer(
        model,
        model_ref,
        args=trainer_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # 6. Start training
    print("Starting DPO training...")
    trainer.train()

    # 7. Save the model
    print("Saving final model...")
    trainer.save_model(trainer_args.output_dir)
    print(f"Model saved to {trainer_args.output_dir}")


if __name__ == "__main__":
    main()
