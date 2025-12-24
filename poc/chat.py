from __future__ import annotations

from dataclasses import dataclass

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class GenConfig:
    max_new_tokens: int = 220
    temperature: float = 0.2
    top_p: float = 0.9


def _format_chat_prompt(user_message: str) -> str:
    return (
        "### Instruction:\n"
        "You are a helpful healthcare policy assistant. If you don't know, say you don't know.\n\n"
        "### User:\n"
        f"{user_message.strip()}\n\n"
        "### Assistant:\n"
    )


def load_base(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.to(torch.device("cpu"))
    model.eval()
    return tok, model


def load_finetuned(model_id: str, adapter_dir: str):
    tok = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    base = AutoModelForCausalLM.from_pretrained(model_id)
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.to(torch.device("cpu"))
    model.eval()
    return tok, model


@torch.no_grad()
def generate(tok, model, user_message: str, cfg: GenConfig) -> str:
    prompt = _format_chat_prompt(user_message)
    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    do_sample = cfg.temperature is not None and cfg.temperature > 0
    out = model.generate(
        **inputs,
        max_new_tokens=int(cfg.max_new_tokens),
        do_sample=do_sample,
        temperature=float(cfg.temperature),
        top_p=float(cfg.top_p),
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    text = tok.decode(out[0], skip_special_tokens=True)

    # extract assistant portion for our template
    marker = "### Assistant:\n"
    if marker in text:
        text = text.split(marker, 1)[1]
    return text.strip()

