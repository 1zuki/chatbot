from __future__ import annotations

import gc
from collections import deque
from pathlib import Path
from threading import Lock
from typing import Deque, List, Mapping, Optional, Tuple

import torch

from .config import BotConfig
from .prompts import build_chat_messages, fallback_prompt_from_messages
from .retriever import NullRetriever, RetrievedDoc, load_retriever
from .text_cleaning import sanitize_chat_output

_HISTORY: Deque[dict] = deque(maxlen=12)
_HISTORY_LOCK = Lock()


def clear_history() -> None:
    with _HISTORY_LOCK:
        _HISTORY.clear()


def _append_history(role: str, content: str, max_messages: int) -> None:
    with _HISTORY_LOCK:
        _HISTORY.append({"role": role, "content": content})
        while len(_HISTORY) > max_messages:
            _HISTORY.popleft()


def get_history(max_messages: int) -> List[dict]:
    with _HISTORY_LOCK:
        return list(_HISTORY)[-max_messages:]


def _resolve_dtype(name: str):
    name = (name or "auto").lower()
    if name == "auto":
        return "auto"
    if name in {"float16", "fp16"}:
        return torch.float16
    if name in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if name in {"float32", "fp32"}:
        return torch.float32
    return "auto"


class LocalChatModel:
    def __init__(self, config: BotConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.retriever = None
        self.lock = Lock()

    def load(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_cfg = self.config.model
        kwargs = {
            "device_map": model_cfg.device_map,
            "torch_dtype": _resolve_dtype(model_cfg.torch_dtype),
        }

        if model_cfg.load_in_4bit and torch.cuda.is_available():
            try:
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            except Exception:
                # If bitsandbytes is unavailable, keep running in normal precision.
                pass

        self.tokenizer = AutoTokenizer.from_pretrained(model_cfg.base_model, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(model_cfg.base_model, **kwargs)
        adapter = self.config.adapter_dir
        if adapter.exists() and (adapter / "adapter_config.json").exists():
            from peft import PeftModel

            base = PeftModel.from_pretrained(base, str(adapter))
        self.model = base.eval()

        try:
            self.retriever = load_retriever(self.config.project_root, self.config.retrieval)
        except Exception:
            self.retriever = NullRetriever()

    def reload_retriever(self) -> str:
        try:
            self.retriever = load_retriever(self.config.project_root, self.config.retrieval)
            if isinstance(self.retriever, NullRetriever):
                return "retriever disabled or missing"
            return "retriever reloaded"
        except Exception as exc:
            self.retriever = NullRetriever()
            return f"retriever failed: {exc}"

    def _format_prompt(self, messages: List[dict]) -> str:
        assert self.tokenizer is not None
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        return fallback_prompt_from_messages(messages)

    def _retrieve(self, query: str) -> List[RetrievedDoc]:
        if self.retriever is None:
            return []
        cfg = self.config.retrieval
        try:
            return self.retriever.search(query, top_k=cfg.top_k, min_score=cfg.min_score)
        except Exception:
            return []

    @torch.inference_mode()
    def generate(self, user_message: str) -> str:
        with self.lock:
            self.load()
            assert self.model is not None and self.tokenizer is not None

            docs = self._retrieve(user_message)
            history = get_history(self.config.model.max_context_messages)
            messages = build_chat_messages(
                user_message=user_message,
                docs=docs,
                history=history,
                max_context_chars=self.config.retrieval.max_context_chars,
            )
            prompt = self._format_prompt(messages)
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3072)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            out = self.model.generate(
                **inputs,
                max_new_tokens=self.config.model.max_new_tokens,
                do_sample=self.config.model.do_sample,
                temperature=self.config.model.temperature,
                top_p=self.config.model.top_p,
                top_k=self.config.model.top_k,
                repetition_penalty=self.config.model.repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            generated = out[0][inputs["input_ids"].shape[-1] :]
            text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
            text = _postprocess_generation(text)
            text = sanitize_chat_output(text)
            if not text:
                text = "eh? say again"

            _append_history("user", user_message, self.config.model.max_context_messages)
            _append_history("assistant", text, self.config.model.max_context_messages)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            else:
                gc.collect()
            return text


def _postprocess_generation(text: str) -> str:
    # Some small instruct models continue with role labels. Cut that off.
    cut_markers = ["\nUser:", "\nAssistant:", "User:", "Assistant:", "</s>"]
    for marker in cut_markers:
        idx = text.find(marker)
        if idx > 0:
            text = text[:idx]
    return text.strip().strip('"')
