from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from vista_chatbot.train_lora import TrainArgs, train_lora


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a LoRA adapter for the Minecraft chat model.")
    parser.add_argument("--train", default=str(ROOT / "artifacts" / "corpus" / "sft_train.jsonl"))
    parser.add_argument("--output", default=str(ROOT / "artifacts" / "lora_full"))
    parser.add_argument("--base-model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    train_lora(
        TrainArgs(
            train_path=args.train,
            output_dir=args.output,
            base_model=args.base_model,
            max_length=args.max_length,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            load_in_4bit=not args.no_4bit,
            max_train_samples=args.max_train_samples,
        )
    )


if __name__ == "__main__":
    main()
