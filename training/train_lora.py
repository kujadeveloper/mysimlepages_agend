"""Dondurulmuş bir veri seti sürümüyle QLoRA ince ayarı.

Düşük kart profili: temel model 4-bit (NF4) yüklenir, yalnızca küçük LoRA ağırlıkları eğitilir, gradient checkpointing
açıktır ve batch=1 + gradient accumulation kullanılır. 1.5B model ~5 GB, 3B ~7 GB, 7B ~11 GB VRAM ile eğitilebilir.
Kayıp yalnızca asistan yanıtı üzerinden hesaplanır (prompt'u ezberlemez).

Örnek:
    python -m training.train_lora --dataset d1 --base Qwen/Qwen2.5-Coder-1.5B-Instruct --out data/adapters/run1
Sonra: GGUF'a çevirip kayıt defterine ekleyin (README > Eğitim).
"""
import argparse
import json
from pathlib import Path

from agent.config import get_settings
from agent.datasets import Datasets


def load_rows(path: Path, tasks: set[str]):
    rows = []
    for line in path.read_text("utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r["task"] in tasks:
                rows.append(r)
    return rows


def tokenize(tokenizer, row, max_len):
    msgs = [{"role": "system", "content": row["system"]}, {"role": "user", "content": row["user"]}]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(msgs + [{"role": "assistant", "content": row["assistant"]}], tokenize=False)
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    ids = tokenizer(full, add_special_tokens=False)["input_ids"][:max_len]
    labels = [-100] * min(len(p_ids), len(ids)) + ids[len(p_ids):]
    return {"input_ids": ids, "labels": labels[:len(ids)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Dondurulmuş veri seti sürümü, ör. d1")
    ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default="plan,section")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--no-4bit", action="store_true", help="Büyük VRAM varsa bf16 tam yükleme")
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForSeq2Seq,
                              Trainer, TrainingArguments)

    ds_dir = Datasets(get_settings().datasets_dir).path(args.dataset)
    tasks = set(args.tasks.split(","))
    tok = AutoTokenizer.from_pretrained(args.base)
    tok.pad_token = tok.pad_token or tok.eos_token

    def build(split):
        rows = load_rows(ds_dir / f"{split}.jsonl", tasks)
        return Dataset.from_list([tokenize(tok, r, args.max_len) for r in rows]) if rows else None

    train_ds, eval_ds = build("train"), build("eval")
    if train_ds is None:
        raise SystemExit("Eğitim verisi boş")

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    quant = None if args.no_4bit else BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(args.base, quantization_config=quant, torch_dtype=dtype,
                                                 device_map="auto", attn_implementation="sdpa")
    if quant:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
        args=TrainingArguments(
            output_dir=args.out, num_train_epochs=args.epochs, learning_rate=args.lr,
            per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=args.grad_accum,
            gradient_checkpointing=True, bf16=bf16, fp16=not bf16, optim="paged_adamw_8bit",
            lr_scheduler_type="cosine", warmup_ratio=0.03, logging_steps=10, save_strategy="epoch",
            save_total_limit=2, eval_strategy="epoch" if eval_ds else "no", report_to="none",
            group_by_length=True),  # benzer uzunlukları grupla: daha az padding, daha hızlı
    )
    trainer.train()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    meta = {"base_model": args.base, "dataset_version": args.dataset, "tasks": sorted(tasks),
            "train_examples": len(train_ds), "eval_loss": trainer.evaluate().get("eval_loss") if eval_ds else None,
            "args": vars(args)}
    Path(args.out, "training_meta.json").write_text(json.dumps(meta, indent=2), "utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
