#!/usr/bin/env python3
"""QLoRA supervised fine-tuning of Gemma 4 31B on agent trajectories.

Designed for a Kaggle L4x4 notebook (4 x 24 GB, no internet): the base model
is loaded in 4-bit NF4 and sharded across the GPUs with device_map="auto";
only LoRA weights are trained. Loss is computed on assistant turns only
(thoughts + tool calls), never on system/user/tool-observation tokens.

The output directory is a standard PEFT adapter (adapter_config.json +
adapter_model.safetensors) that you copy to submission/adapters/main_lora/.

Example:
  python training/train_lora.py \
      --model /kaggle/input/gemma-4/transformers/gemma-4-31b-it/1 \
      --data training/data/sft.jsonl --out /kaggle/working/main_lora

Notes:
  * Train on the bf16 instruction-tuned checkpoint; the competition serves the
    QAT w4a16 variant of the same model, so module names match and a LoRA
    trained on the bf16 base transfers. Confirm that the evaluated harness
    accepts the target modules you pick (HARNESS_README.md).
  * Keep the dataset small and on-policy-like; with ~100 tasks, 2-3 epochs at
    a low LR is plenty. Watch for regressions on the holdout split.
  * Defaults are rank 8 on attention projections only: rank-32 all-layer
    LoRA on the compressed-tensors W4A16 model was reported unstable in vLLM
    (issue #50059), while small partial adapters load fine. The scorer serves
    with max_lora_rank=128, max_loras=8. Load-test the adapter in vLLM with the
    QAT checkpoint before submitting.
  * Mix data with repeated --data: scripted trajectories (build_sft_data.py)
    and real successful runs (traces_to_sft.py).
"""
import argparse
import json
import math
import os

import torch
from torch.utils.data import Dataset


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", action="append", required=True, help="JSONL file; repeat to mix")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=16384)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--targets", default="q_proj,k_proj,v_proj,o_proj")
    ap.add_argument("--no-4bit", action="store_true", help="load bf16 (needs much more memory)")
    return ap.parse_args()


def render(tok, messages, tools, gen_prompt=False):
    # enable_thinking=False matches serving (sampling.yaml include_thoughts: false).
    return tok.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=gen_prompt,
                                   enable_thinking=False)


def encode(tok, ex, max_len):
    """Token ids + labels where only assistant turns are supervised."""
    msgs, tools = ex["messages"], ex.get("tools")
    full = render(tok, msgs, tools)
    ids = tok(full, add_special_tokens=False)["input_ids"]
    labels = [-100] * len(ids)
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        start = len(tok(render(tok, msgs[:i], tools, gen_prompt=True), add_special_tokens=False)["input_ids"])
        end = len(tok(render(tok, msgs[: i + 1], tools), add_special_tokens=False)["input_ids"])
        for j in range(start, min(end, len(ids))):
            labels[j] = ids[j]
    if len(ids) > max_len:
        return None  # dropping beats truncating away the edit/submit turns
    return {"input_ids": ids, "labels": labels}


class Traj(Dataset):
    def __init__(self, tok, paths, max_len):
        self.items, dropped = [], 0
        for path in paths:
            for line in open(path, encoding="utf-8"):
                if not line.strip():
                    continue
                enc = encode(tok, json.loads(line), max_len)
                if enc is None:
                    dropped += 1
                else:
                    self.items.append(enc)
        n_sup = sum(sum(l != -100 for l in it["labels"]) for it in self.items)
        print(f"loaded {len(self.items)} examples ({dropped} dropped > max_len); {n_sup} supervised tokens")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), n), -100, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for k, b in enumerate(batch):
        L = len(b["input_ids"])
        ids[k, :L] = torch.tensor(b["input_ids"])
        lab[k, :L] = torch.tensor(b["labels"])
        att[k, :L] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


def main():
    a = parse()
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer,
                              TrainingArguments)

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    ds = Traj(tok, a.data, a.max_len)

    quant = None if a.no_4bit else BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16)
    load = dict(quantization_config=quant, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa")
    try:
        model = AutoModelForCausalLM.from_pretrained(a.model, **load)
    except (ValueError, KeyError):  # multimodal Gemma 4 checkpoints register under image-text-to-text
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(a.model, **load)
    model.config.use_cache = False
    if quant is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()

    lcfg = LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                      # language-model layers only; skip the vision tower if the checkpoint has one
                      target_modules=r".*language_model.*\.(" + "|".join(a.targets.split(",")) + r")$"
                      if any("language_model" in n for n, _ in model.named_modules()) else a.targets.split(","))
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    steps = math.ceil(len(ds) * a.epochs / a.grad_accum)
    args = TrainingArguments(
        output_dir=a.out, per_device_train_batch_size=1, gradient_accumulation_steps=a.grad_accum,
        num_train_epochs=a.epochs, learning_rate=a.lr, lr_scheduler_type="cosine",
        warmup_steps=max(1, steps // 10), logging_steps=1, save_strategy="epoch", bf16=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit", report_to=[], remove_unused_columns=False)
    Trainer(model=model, args=args, train_dataset=ds,
            data_collator=lambda b: collate(b, tok.pad_token_id)).train()
    model.save_pretrained(a.out, safe_serialization=True)
    print("saved adapter to", a.out, os.listdir(a.out))


if __name__ == "__main__":
    main()
