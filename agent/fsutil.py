"""Windows / Linux / macOS'ta aynı davranan dosya yardımcıları.

- Metin her zaman UTF-8 ve LF ("\\n") satır sonuyla yazılır. Windows'ta Python metin modunda "\\n"i "\\r\\n"e çevirir;
  bu hem veri setlerinin sha256 özetini platforma bağlı yapar hem de JSONL dosyalarını karıştırır.
- Windows'ta os.replace, hedef dosya o an başka bir iş parçacığı/süreç tarafından okunuyorsa PermissionError verebilir;
  kısa aralıklarla yeniden denenir.
"""
import os
import time
from pathlib import Path


def read_text(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def write_text(path: Path, text: str):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def append_text(path: Path, text: str):
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(text)


def atomic_write(path: Path, text: str, retries: int = 20):
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    write_text(tmp, text)
    for i in range(retries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:  # yalnızca Windows'ta: dosya açık
            if i == retries - 1:
                tmp.unlink(missing_ok=True)
                raise
            time.sleep(0.05)
