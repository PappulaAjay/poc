from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from poc.data import QAExample


@dataclass(frozen=True)
class TrainConfig:
    model_id: str
    run_dir: str
    seed: int = 42
    max_seq_len: int = 256
    train_examples: int = 160
    eval_examples: int = 40
    epochs: int = 2
    lr: float = 2e-4
    batch_size: int = 2
    grad_accum: int = 8
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    log_every: int = 1
    eval_every_steps: int = 50
    max_steps: int | None = 200
    num_threads: int = 0  # 0 = leave default


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    torch.manual_seed(seed)


def _format_example(ex: QAExample) -> tuple[str, str]:
    prompt = (
        "### Instruction:\n"
        f"{ex.instruction}\n\n"
        "### User:\n"
        f"{ex.input}\n\n"
        "### Assistant:\n"
    )
    answer = ex.output.strip()
    if not answer.endswith("\n"):
        answer += "\n"
    return prompt, answer


class SupervisedChatDataset(Dataset):
    def __init__(self, examples: list[QAExample], tokenizer: AutoTokenizer, max_seq_len: int):
        self.examples = examples
        self.tok = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.examples[idx]
        prompt, answer = _format_example(ex)

        prompt_ids = self.tok(prompt, add_special_tokens=False).input_ids
        answer_ids = self.tok(answer, add_special_tokens=False).input_ids

        # Ensure EOS at end so generation stops cleanly
        if self.tok.eos_token_id is not None and (not answer_ids or answer_ids[-1] != self.tok.eos_token_id):
            answer_ids = answer_ids + [self.tok.eos_token_id]

        input_ids = (prompt_ids + answer_ids)[: self.max_seq_len]
        labels = [-100] * min(len(prompt_ids), len(input_ids))
        labels += input_ids[len(labels) : len(input_ids)]

        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _collate(batch: list[dict[str, torch.Tensor]], pad_id: int) -> dict[str, torch.Tensor]:
    max_len = max(x["input_ids"].shape[0] for x in batch)
    input_ids = []
    attention_mask = []
    labels = []
    for x in batch:
        n = x["input_ids"].shape[0]
        pad = max_len - n
        input_ids.append(torch.cat([x["input_ids"], torch.full((pad,), pad_id, dtype=torch.long)]))
        attention_mask.append(torch.cat([x["attention_mask"], torch.zeros((pad,), dtype=torch.long)]))
        labels.append(torch.cat([x["labels"], torch.full((pad,), -100, dtype=torch.long)]))
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
    }


def infer_lora_target_modules(model) -> list[str]:
    """
    Best-effort inference of LoRA target module names across common decoder-only LMs.
    """
    leaf_names = {name.split(".")[-1] for name, _ in model.named_modules()}

    # Llama/Mistral/Gemma style projections
    if {"q_proj", "v_proj"}.issubset(leaf_names):
        return ["q_proj", "v_proj"]

    # GPT-2 style fused QKV
    if "c_attn" in leaf_names:
        return ["c_attn"]

    # Falcon / other fused variants
    for fused in ("query_key_value", "qkv_proj", "Wqkv"):
        if fused in leaf_names:
            return [fused]

    # Conservative fallback: target a few projection-ish leaves
    candidates = [n for n in leaf_names if any(k in n for k in ("proj", "attn"))]
    candidates = sorted(set(candidates))
    if candidates:
        return candidates[:8]

    raise ValueError(
        "Could not infer LoRA target modules for this model. "
        "Try a different base model or add explicit target_modules support."
    )


@torch.no_grad()
def _eval_epoch(model, loader: DataLoader) -> dict:
    model.eval()
    losses = []
    tokens = 0
    correct = 0

    for batch in loader:
        out = model(**batch)
        loss = out.loss.detach().float().item()
        losses.append(loss)

        logits = out.logits.detach()
        labels = batch["labels"]
        # token accuracy on supervised (non-masked) tokens
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        mask = shift_labels != -100
        if mask.any():
            preds = shift_logits.argmax(dim=-1)
            correct += (preds[mask] == shift_labels[mask]).sum().item()
            tokens += mask.sum().item()

    mean_loss = float(sum(losses) / max(1, len(losses)))
    ppl = float(math.exp(min(20.0, mean_loss)))  # clamp to avoid inf
    acc = float(correct / max(1, tokens))
    return {"eval_loss": mean_loss, "eval_ppl": ppl, "eval_token_acc": acc}


def train_lora(
    cfg: TrainConfig,
    examples: list[QAExample],
    *,
    ui_callback=None,
) -> dict:
    """
    CPU-only LoRA fine-tuning.

    ui_callback(metrics_dict) will be called frequently with live metrics.
    """
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "lora_adapter").mkdir(parents=True, exist_ok=True)

    set_seed(cfg.seed)
    if cfg.num_threads and cfg.num_threads > 0:
        torch.set_num_threads(int(cfg.num_threads))

    device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        # common for causal LM; use EOS as PAD
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    base_model = AutoModelForCausalLM.from_pretrained(cfg.model_id)
    base_model.to(device)

    targets = infer_lora_target_modules(base_model)
    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    model = get_peft_model(base_model, lora)
    model.to(device)
    model.train()

    # split dataset
    ex = examples[: (cfg.train_examples + cfg.eval_examples)]
    train_ex = ex[: cfg.train_examples]
    eval_ex = ex[cfg.train_examples : cfg.train_examples + cfg.eval_examples]
    if not eval_ex:
        eval_ex = train_ex[: max(1, min(16, len(train_ex)))]

    train_ds = SupervisedChatDataset(train_ex, tokenizer, cfg.max_seq_len)
    eval_ds = SupervisedChatDataset(eval_ex, tokenizer, cfg.max_seq_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=lambda b: _collate(b, pad_id=tokenizer.pad_token_id),
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=lambda b: _collate(b, pad_id=tokenizer.pad_token_id),
    )

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    metrics_path = run_dir / "metrics.jsonl"
    config_path = run_dir / "train_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    global_step = 0
    t0 = time.time()

    def log(m: dict) -> None:
        m = dict(m)
        m["step"] = global_step
        m["wall_time_sec"] = round(time.time() - t0, 3)
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(m) + "\n")
        if ui_callback is not None:
            ui_callback(m)

    # initial eval
    eval0 = _eval_epoch(model, eval_loader)
    log({"event": "eval", **eval0})

    max_steps = cfg.max_steps

    for epoch in range(cfg.epochs):
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{cfg.epochs}", leave=False)
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}

            out = model(**batch)
            loss = out.loss / cfg.grad_accum
            loss.backward()

            if (global_step + 1) % cfg.grad_accum == 0:
                optim.step()
                optim.zero_grad(set_to_none=True)

            # "train token accuracy" approximation on this batch
            with torch.no_grad():
                logits = out.logits.detach()
                labels = batch["labels"]
                shift_logits = logits[:, :-1, :]
                shift_labels = labels[:, 1:]
                mask = shift_labels != -100
                token_acc = None
                if mask.any():
                    preds = shift_logits.argmax(dim=-1)
                    token_acc = (preds[mask] == shift_labels[mask]).float().mean().item()

            if global_step % cfg.log_every == 0:
                train_loss = out.loss.detach().float().item()
                ppl = float(math.exp(min(20.0, train_loss)))
                log(
                    {
                        "event": "train",
                        "epoch": epoch + 1,
                        "train_loss": float(train_loss),
                        "train_ppl": ppl,
                        "train_token_acc": None if token_acc is None else float(token_acc),
                    }
                )

            if cfg.eval_every_steps and global_step > 0 and global_step % cfg.eval_every_steps == 0:
                ev = _eval_epoch(model, eval_loader)
                log({"event": "eval", **ev})

            global_step += 1
            if max_steps is not None and global_step >= max_steps:
                break
        if max_steps is not None and global_step >= max_steps:
            break

    # final eval + save adapter
    ev = _eval_epoch(model, eval_loader)
    log({"event": "eval", **ev})

    model.save_pretrained(run_dir / "lora_adapter")
    tokenizer.save_pretrained(run_dir / "lora_adapter")

    return {
        "run_dir": str(run_dir),
        "final_eval": ev,
        "steps": global_step,
        "seconds": round(time.time() - t0, 3),
        "train_examples": len(train_ex),
        "eval_examples": len(eval_ex),
        "model_id": cfg.model_id,
    }

