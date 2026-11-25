"""vLLM başlatıcı: /adapters altındaki her LoRA adaptörünü "msp-<klasör>" adıyla --lora-modules'a ekler.

Bir klasör adaptör sayılır, içinde adapter_config.json ve adapter_model.safetensors varsa (train_lora çıktısı).
Ara checkpoint klasörleri (checkpoint-*) atlanır.
"""
import os
import shlex
from pathlib import Path

args = ["vllm", "serve", os.environ["VLLM_BASE_MODEL"],
        "--served-model-name", os.environ.get("VLLM_SERVED_NAME", "base"),
        "--host", "0.0.0.0", "--port", "8000"]
args += shlex.split(os.environ.get("VLLM_EXTRA_ARGS", ""))

root = Path("/adapters")
loras = []
if root.is_dir():
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if (d / "adapter_config.json").exists() and (d / "adapter_model.safetensors").exists():
            loras.append(f"msp-{d.name}={d}")
if loras:
    args += ["--lora-modules", *loras]
print("LoRA adaptörleri:", ", ".join(loras) or "yok", flush=True)
print("Komut:", shlex.join(args), flush=True)
os.execvp(args[0], args)
