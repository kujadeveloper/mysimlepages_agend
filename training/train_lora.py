"""Dondurulmuş bir veri seti sürümüyle QLoRA ince ayarı.

Düşük kart profili: temel model 4-bit (NF4) yüklenir, yalnızca küçük LoRA ağırlıkları eğitilir, gradient checkpointing
açıktır ve batch=1 + gradient accumulation kullanılır. 1.5B model ~5 GB, 3B ~7 GB, 7B ~11 GB VRAM ile eğitilebilir.
Kayıp yalnızca asistan yanıtı üzerinden hesaplanır (prompt'u ezberlemez).

Örnek:
    python -m training.train_lora --dataset d1 --base Qwen/Qwen2.5-Coder-1.5B-Instruct --out data/adapters/run1
Sonra: GGUF'a çevirip kayıt defterine ekleyin (README > Eğitim).
"""
import argparse
import inspect
import json
from pathlib import Path

from agent.config import get_settings
from agent.datasets import Datasets
from agent.fsutil import read_text, write_text


def load_rows(path: Path, tasks: set[str]):
    rows = []
    for line in read_text(path).splitlines():
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
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--no-4bit", action="store_true", help="Büyük VRAM varsa bf16 tam yükleme")
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments
    try:
        from transformers import BitsAndBytesConfig
    except ImportError:  # pragma: no cover
        BitsAndBytesConfig = None

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

    # Platform: CUDA (Linux/Windows) -> QLoRA 4-bit; Apple Silicon (MPS) ya da CPU -> bitsandbytes yok, 4-bit'siz LoRA
    cuda = torch.cuda.is_available()
    mps = not cuda and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    bf16 = cuda and torch.cuda.is_bf16_supported()
    fp16 = cuda and not bf16
    dtype = torch.bfloat16 if bf16 else torch.float16 if fp16 else torch.float32
    use_4bit = cuda and not args.no_4bit
    if not cuda and not args.no_4bit:
        print(f"CUDA yok ({'MPS' if mps else 'CPU'}): 4-bit kapatıldı, {dtype} ile 4-bit'siz LoRA eğitilecek (daha çok bellek)")
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype) if use_4bit else None
    model = AutoModelForCausalLM.from_pretrained(args.base, quantization_config=quant, dtype=dtype,
                                                 device_map="auto" if cuda else None, attn_implementation="sdpa")
    if mps:
        model.to("mps")
    # prepare_model_for_kbit_training kullanılmaz: 4-bit dışı katmanları (embedding, lm_head) fp32'ye çevirir; 7B'de
    # bu ~4 GB ek bellek demek ve 16 GB kartı taşırır. LoRA ağırlıkları zaten ayrı eğitilir, bf16 yeterlidir.
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()  # gradient checkpointing + LoRA için gerekli
    model =get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    # transformers 4.x ve 5.x arasında değişen adlar: warmup_ratio -> warmup_steps (0-1 arası oran),
    # group_by_length -> train_sampling_strategy="group_by_length"
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    compat = {"warmup_ratio": 0.03} if "warmup_ratio" in ta_params else {"warmup_steps": 0.03}
    if "group_by_length" in ta_params:
        compat["group_by_length"] = True  # benzer uzunlukları grupla: daha az padding, daha hızlı
    elif "train_sampling_strategy" in ta_params:
        compat["train_sampling_strategy"] = "group_by_length"
    if eval_ds:  # son tur değil, ölçüm kaybı en düşük tur kaydedilir (küçük veride son tur genelde ezberler)
        compat.update(load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False)

    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
        args=TrainingArguments(
            output_dir=args.out, num_train_epochs=args.epochs, learning_rate=args.lr,
            per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=args.grad_accum,
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}, bf16=bf16, fp16=fp16,
            optim="paged_adamw_8bit" if use_4bit else "adamw_torch",  # 8-bit optimizer yalnızca CUDA
            dataloader_num_workers=0,  # Windows/macOS: alt süreç (spawn) sorunlarını önler
            lr_scheduler_type="cosine", logging_steps=10, save_strategy="epoch",
            save_total_limit=2, eval_strategy="epoch" if eval_ds else "no", report_to="none", **compat),
    )
    trainer.train()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    meta = {"base_model": args.base, "dataset_version": args.dataset, "tasks": sorted(tasks),
            "train_examples": len(train_ds), "eval_loss": trainer.evaluate().get("eval_loss") if eval_ds else None,
            "args": vars(args)}
    write_text(Path(args.out, "training_meta.json"), json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
