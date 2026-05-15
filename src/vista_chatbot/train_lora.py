from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from .prompts import fallback_prompt_from_messages


@dataclass
class TrainArgs:
    train_path: str
    output_dir: str
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    max_length: int = 1024
    num_train_epochs: float = 1.0
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.03
    logging_steps: int = 20
    save_steps: int = 1000
    save_total_limit: int = 2
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    load_in_4bit: bool = True
    seed: int = 42
    max_train_samples: Optional[int] = None


class ChatSFTDataset(Dataset):
    def __init__(self, path: str | Path, tokenizer, max_length: int, max_samples: Optional[int] = None):
        self.rows: List[dict] = []
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "messages" in row:
                    self.rows.append(row)
                if max_samples is not None and len(self.rows) >= max_samples:
                    break
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        messages = self.rows[idx]["messages"]
        prompt_messages = messages[:-1]
        assistant = messages[-1]

        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                prompt = self.tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
                full = prompt + str(assistant["content"]).strip() + self.tokenizer.eos_token
            except Exception:
                prompt = fallback_prompt_from_messages(prompt_messages)
                full = prompt + " " + str(assistant["content"]).strip() + self.tokenizer.eos_token
        else:
            prompt = fallback_prompt_from_messages(prompt_messages)
            full = prompt + " " + str(assistant["content"]).strip() + self.tokenizer.eos_token

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        full_ids = self.tokenizer(full, add_special_tokens=False, truncation=True, max_length=self.max_length).input_ids
        labels = full_ids.copy()
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len
        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class DataCollatorForChat:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = max(x["input_ids"].shape[0] for x in features)
        pad_id = self.tokenizer.pad_token_id
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in features:
            length = item["input_ids"].shape[0]
            pad = max_len - length
            batch["input_ids"].append(torch.cat([item["input_ids"], torch.full((pad,), pad_id, dtype=torch.long)]))
            batch["attention_mask"].append(torch.cat([item["attention_mask"], torch.zeros(pad, dtype=torch.long)]))
            batch["labels"].append(torch.cat([item["labels"], torch.full((pad,), -100, dtype=torch.long)]))
        return {k: torch.stack(v) for k, v in batch.items()}


def train_lora(args: TrainArgs) -> None:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, set_seed

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"device_map": "auto", "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32}
    if args.load_in_4bit and torch.cuda.is_available():
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    if args.load_in_4bit and torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    dataset = ChatSFTDataset(args.train_path, tokenizer, args.max_length, args.max_train_samples)
    if len(dataset) == 0:
        raise ValueError(f"No training rows found in {args.train_path}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        fp16=torch.cuda.is_available(),
        bf16=False,
        optim="paged_adamw_8bit" if args.load_in_4bit and torch.cuda.is_available() else "adamw_torch",
        lr_scheduler_type="cosine",
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForChat(tokenizer),
    )
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
