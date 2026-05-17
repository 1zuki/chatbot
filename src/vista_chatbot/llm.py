from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from .config import BotConfig
from .retriever import SearchResult, build_context, extractive_answer, extractive_candidates
from .text import safe_truncate, sanitize_chat_output


@dataclass(frozen=True)
class ChatTurn:
    role: str
    content: str


class PromptBuilder:
    def __init__(self, config: BotConfig):
        self.config = config

    def build(self, *, query: str, results: list[SearchResult], history: list[ChatTurn]) -> str:
        context = build_context(results, max_chars=self.config.retrieval.max_context_chars)
        recent = "\n".join(
            f"{turn.role}: {turn.content}" for turn in history[-self.config.model.max_context_messages :]
        )
        return (
            f"<|system|>\n{self.config.prompt.system}\n{self.config.prompt.answer_style}\n</s>\n"
            f"<|user|>\n"
            f"Recent chat context, may be empty:\n{recent or '(none)'}\n\n"
            f"Wiki context:\n{context or '(no relevant wiki context)'}\n\n"
            f"Question: {query}\n"
            f"Answer for Minecraft chat:"
            f"\n</s>\n<|assistant|>\n"
        )


class LocalGenerator:
    def __init__(self, config: BotConfig, logger: logging.Logger | None = None):
        self.config = config
        self.logger = logger or logging.getLogger("vista_chatbot.llm")
        self.tokenizer = None
        self.model = None
        self.load_error: Exception | None = None

    @property
    def loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def warmup(self) -> None:
        if not self._needs_model():
            return
        if self.loaded:
            return
        try:
            self._load()
        except Exception as exc:  # pragma: no cover - depends on local ML stack
            self.load_error = exc
            if not self.config.model.fallback_to_extractive:
                raise
            self.logger.warning("LLM failed to load; using extractive RAG fallback: %s", exc)

    def generate_or_fallback(
        self,
        *,
        prompt: str,
        query: str,
        results: list[SearchResult],
        max_chat_chars: int,
    ) -> str:
        if self._selector_enabled() and self.loaded and not self.config.model.enabled:
            return self._select_extractive_or_fallback(
                query=query,
                results=results,
                max_chat_chars=max_chat_chars,
            )
        if not self.config.model.enabled or not self.loaded:
            return extractive_answer(query, results, max_chars=max_chat_chars)
        try:
            return self.generate(prompt, max_chat_chars=max_chat_chars)
        except Exception as exc:  # pragma: no cover - depends on model runtime
            self.logger.exception("Generation failed; falling back to extractive answer: %s", exc)
            if not self.config.model.fallback_to_extractive:
                raise
            if self._selector_enabled() and self.loaded:
                return self._select_extractive_or_fallback(
                    query=query,
                    results=results,
                    max_chat_chars=max_chat_chars,
                )
            return extractive_answer(query, results, max_chars=max_chat_chars)

    def generate(self, prompt: str, *, max_chat_chars: int) -> str:
        if not self.loaded:
            raise RuntimeError("Model is not loaded")
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt")
        if hasattr(self.model, "device"):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.config.model.max_new_tokens,
                temperature=self.config.model.temperature,
                top_p=self.config.model.top_p,
                top_k=self.config.model.top_k,
                repetition_penalty=self.config.model.repetition_penalty,
                do_sample=self.config.model.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[-1] :]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        text = self._clean_generation(text)
        return safe_truncate(text, max_chat_chars)

    def _load(self) -> None:  # pragma: no cover - depends on local ML stack
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs = {
            "device_map": self.config.model.device_map,
        }
        dtype = self.config.model.torch_dtype
        if dtype != "auto":
            kwargs["torch_dtype"] = getattr(torch, dtype)
        else:
            kwargs["torch_dtype"] = "auto"
        if self.config.model.load_in_4bit:
            kwargs["load_in_4bit"] = True

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.base_model, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(self.config.model.base_model, **kwargs)
        adapter = self.config.adapter_path
        if adapter.exists() and any(adapter.iterdir()):
            try:
                from peft import PeftModel

                model = PeftModel.from_pretrained(model, str(adapter))
                self.logger.info("Loaded LoRA adapter from %s", adapter)
            except Exception as exc:
                if not self.config.model.fallback_to_extractive:
                    raise
                self.logger.warning("Failed to load LoRA adapter %s: %s", adapter, exc)
        model.eval()
        self.model = model

    @staticmethod
    def _clean_generation(text: str) -> str:
        text = text.split("</s>")[0]
        for marker in ["<|user|>", "<|assistant|>", "Question:", "Answer:"]:
            if marker in text:
                text = text.split(marker)[0]
        text = sanitize_chat_output(text)
        if not text:
            text = "I don't know from the wiki yet."
        return text

    def _needs_model(self) -> bool:
        return self.config.model.enabled or self._selector_enabled()

    def _selector_enabled(self) -> bool:
        return self.config.model.llm_select_extractive

    def _select_extractive_or_fallback(
        self,
        *,
        query: str,
        results: list[SearchResult],
        max_chat_chars: int,
    ) -> str:
        candidates = extractive_candidates(
            query,
            results,
            max_candidates=self.config.model.llm_select_max_candidates,
        )
        if not candidates:
            return extractive_answer(query, results, max_chars=max_chat_chars)

        choice = self._choose_candidate_index(query=query, candidates=candidates)
        if choice is None:
            return extractive_answer(query, results, max_chars=max_chat_chars)
        if choice == 0:
            return "I don't know from the wiki yet."

        selected = candidates[choice - 1]
        selected = sanitize_chat_output(selected)
        return safe_truncate(f"Wiki says: {selected}", max_chat_chars)

    def _choose_candidate_index(self, *, query: str, candidates: list[str]) -> int | None:
        if not self.loaded:
            return None
        prompt = self._selector_prompt(query=query, candidates=candidates)
        try:
            out = self._generate_index(prompt)
        except Exception as exc:  # pragma: no cover - depends on model runtime
            self.logger.warning("LLM candidate selection failed; using extractive fallback: %s", exc)
            return None

        m = re.search(r"\b(\d+)\b", out)
        if not m:
            return None
        idx = int(m.group(1))
        if 0 <= idx <= len(candidates):
            return idx
        return None

    def _generate_index(self, prompt: str) -> str:
        if not self.loaded:
            raise RuntimeError("Model is not loaded")
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt")
        if hasattr(self.model, "device"):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.config.model.llm_select_max_new_tokens,
                do_sample=False,
                repetition_penalty=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[-1] :]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        text = text.split("</s>")[0]
        return sanitize_chat_output(text)

    @staticmethod
    def _selector_prompt(*, query: str, candidates: list[str]) -> str:
        lines = [f"{i}. {c}" for i, c in enumerate(candidates, start=1)]
        joined = "\n".join(lines)
        return (
            "<|system|>\n"
            "You are a strict Minecraft wiki answer selector.\n"
            "Task: choose exactly one best candidate snippet for the question.\n"
            "Rules:\n"
            "- Return only one integer.\n"
            "- Return 0 if none of the snippets answer the question.\n"
            "- Prefer specific factual snippets; avoid gibberish/noise.\n"
            "- Do not explain.\n"
            "</s>\n"
            "<|user|>\n"
            f"Question: {query}\n\n"
            f"Candidates:\n{joined}\n\n"
            "Answer with one integer only:\n"
            "</s>\n<|assistant|>\n"
        )
