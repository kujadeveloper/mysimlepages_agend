# MySimplePage AI Agent

"AI Destekli Tasarım (BETA) — Hayal Edin, Biz Kodlayalım" özelliğinin servisi. Kullanıcının anlattığı siteyi
**tek sayfa HTML + Tailwind** olarak üretir. Kendi modelinizle çalışır, kullanıcı geri bildirimiyle **eğitilir**,
model / veri / prompt **sürümlüdür** ve **düşük VRAM'li kartlarda** (4–8 GB) verimli çalışacak şekilde tasarlanmıştır.

## Mimari

```
Next.js ──> Django backend ──(X-API-Key, iç ağ)──> agent servisi (FastAPI) ──> LLM arka ucu
                                                    │  SQLite iş kuyruğu         ├─ llamacpp: yerel GGUF (+LoRA)
                                                    │  1 işçi = 1 GPU işi         ├─ openai: Ollama / vLLM / llama.cpp server
                                                    │                            └─ mock: test
                                                    └─ data/  registry/ (model sürümleri)  datasets/ (veri sürümleri)
```

**Üretim hattı** (`agent/pipeline.py`):
1. **Plan**: açıklama → JSON site planı (başlık, dil, palet, yazı tipi, bölümler). JSON modu/grameri ile kısıtlanır,
   geçersizse bir kez daha denenir, yine olmazsa makul bir varsayılan plan kullanılır.
2. **Bölümler**: her bölüm ayrı ve kısa bir çağrıyla üretilir. Küçük modeller uzun çıktıda dağılır; bölüm bölüm
   üretim hem kaliteyi hem de bellek kullanımını (4K bağlam yeter) iyileştirir. Bozuk bölüm tek başına yeniden denenir.
3. **Temizlik** (`agent/sanitize.py`): `<script>`, `on*` öznitelikleri, `javascript:` bağlantıları, iframe vb. silinir,
   kapanmamış etiketler kapatılır. Bu sayılar kalite metriği olarak da kullanılır.
4. **Birleştirme**: tek HTML, Tailwind CDN, palet CSS değişkenleri ve Google Font ile.

Her LLM çağrısı `trace` olarak saklanır. Kullanıcı puan verdiğinde bu çağrılar doğrudan eğitim örneğine dönüşür.
Ekibin düzelttiği bölümler en değerli veridir.

## Hızlı başlangıç

Servis Linux, Windows ve macOS'ta çalışır (CI üç platformda da test eder: `.github/workflows/tests.yml`).

### Docker (tüm platformlar)

LLM sunucusu **vLLM**'dir (NVIDIA GPU gerekir). Temel model HF biçiminde `models/` altında durur; eğitilen LoRA
adaptörleri (`data/adapters/<ad>`) vLLM açılışında `msp-<ad>` adıyla otomatik yüklenir (GGUF dönüşümü gerekmez).

```bash
cp .env.example .env            # Windows PowerShell: Copy-Item .env.example .env   — MSP_API_KEY'i değiştirin
docker compose -f docker-compose.train.yaml run --rm train hf download Qwen/Qwen2.5-Coder-7B-Instruct --local-dir models/Qwen2.5-Coder-7B-Instruct
docker compose up -d            # vLLM (FP8, ~8 GB VRAM) + ajan
```

| Platform | Not |
|----------|-----|
| Linux + NVIDIA | nvidia-container-toolkit gerekir |
| Windows + NVIDIA | Docker Desktop (WSL2 arka ucu) |
| macOS / GPU'suz | vLLM çalışmaz. Başka bir OpenAI uyumlu sunucu (ör. Ollama) kurup yalnızca ajanı başlatın: `LLM_URL=http://host.docker.internal:11434/v1 docker compose up -d --no-deps agent` ve `.env` içinde `MSP_OPENAI_MODEL` |

Deneme isteği:

```bash
curl -H "X-API-Key: ..." -H "Content-Type: application/json" \
     -d '{"brief":"Kafe için tek sayfalık site, sıcak kahve tonları"}' http://127.0.0.1:8100/v1/designs
```

Windows PowerShell'de `curl` yerine:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8100/v1/designs -Headers @{"X-API-Key"="..."} `
  -ContentType "application/json; charset=utf-8" -Body '{"brief":"Kafe için tek sayfalık site, sıcak kahve tonları"}'
```

### Docker'sız geliştirme (Python 3.10+)

macOS / Linux:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
MSP_BACKEND=mock msp-agent generate "Kafe sitesi" -o site.html
msp-agent serve                # http://127.0.0.1:8100
pytest
```

Windows (PowerShell):

```powershell
py -3 -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
$env:MSP_BACKEND="mock"; msp-agent generate "Kafe sitesi" -o site.html
msp-agent serve
pytest
```

Windows (cmd): `.venv\Scripts\activate.bat` ve `set MSP_BACKEND=mock`. Ayarlar `.env` dosyasından da okunur,
bu yüzden ortam değişkeni yerine `.env` içine `MSP_BACKEND=mock` yazmak her platformda aynı şekilde çalışır.

Platform notları:
- Dosyalar (JSONL veri setleri, manifest'ler, üretilen HTML) her platformda UTF-8 ve LF ile yazılır; veri seti
  sha256 özetleri platformdan bağımsızdır. `.gitattributes` depo içeriğini Windows'ta da LF tutar.
- Windows konsolunda Türkçe karakterler için CLI çıktısı UTF-8'e ayarlanır. Yine de bozuk görünürse: `chcp 65001`.

## Düşük kart ayarları

| VRAM | Önerilen model (GGUF Q4_K_M) | Not |
|------|------------------------------|-----|
| CPU / 4 GB | Qwen2.5-Coder-1.5B-Instruct | `MSP_N_GPU_LAYERS=0` ya da kısmi (ör. 16) |
| 6–8 GB | Qwen2.5-Coder-3B-Instruct | varsayılan |
| 12–16 GB | Qwen2.5-Coder-7B-Instruct | daha iyi kalite |

- `MSP_N_CTX=4096`: bölüm bölüm üretim sayesinde küçük bağlam yeterli; KV önbelleği küçük kalır.
- `MSP_WORKERS=1`: aynı anda tek üretim; istekler kuyrukta bekler, bellek taşmaz.
- vLLM: `--quantization fp8` (7B ~8 GB), `--max-model-len 8192`, `--gpu-memory-utilization 0.85` (compose'ta).
  Eğitim sırasında GPU'yu boşaltmak için: `docker compose stop vllm`.
- Yerel GGUF: `pip install -e ".[llamacpp]"`, `MSP_BACKEND=llamacpp`,
  `MSP_DEFAULT_MODEL_PATH=models/...gguf`. Eğitilen LoRA, temel modelle birleştirilmeden yüklenebilir.

  | Platform | llama-cpp-python GPU kurulumu |
  |----------|-------------------------------|
  | Linux (CUDA) | `CMAKE_ARGS="-DGGML_CUDA=on" pip install -e ".[llamacpp]"` |
  | Windows (CUDA) | PowerShell: `$env:CMAKE_ARGS="-DGGML_CUDA=on"; pip install -e ".[llamacpp]"` (Visual Studio Build Tools + CUDA Toolkit gerekir) |
  | macOS (Apple Silicon) | `pip install -e ".[llamacpp]"` — Metal varsayılan olarak açık |
  | CPU (hepsi) | `pip install -e ".[llamacpp]"` ve `MSP_N_GPU_LAYERS=0` |

## API

| Yöntem | Yol | Açıklama |
|--------|-----|----------|
| GET | `/health` | durum + etkin model |
| POST | `/v1/designs` | `{brief, user_ref?}` → `202 {id, position}` |
| GET | `/v1/designs/{id}` | durum, ilerleme (`stage`, `progress`), plan, metrikler, bitince `html` |
| GET | `/v1/designs/{id}/preview` | HTML önizleme (CSP'li; sandbox iframe'de, ayrı kökenden gösterin) |
| POST | `/v1/designs/{id}/revise` | `{instruction}` → `202 {id, parent_id}`: yeni sürüm. Yalnızca değişmesi gereken bölümler yeniden yazılır; renk/font plandan uygulanır; eklenen bölüm üretilir, menü ve footer güncellenir. Sonuçta `parent_id`, `instruction`, `revision {summary, changed, added, removed}` |
| POST | `/v1/designs/{id}/feedback` | `{rating 1-5, comment?, corrections?: {section_id: html}}` → eğitim verisi |
| GET/POST | `/v1/models`, `/v1/models/{v}/activate` | model sürümleri |
| GET/POST | `/v1/datasets`, `/v1/datasets/freeze` | veri seti sürümleri |

Tüm `/v1` uçları `X-API-Key` ister. Servis internete açılmamalı; kullanıcı istekleri, kota ve ödeme kontrolü
Django backend'de yapılır (`user_ref` yalnızca kimlik numarasıdır, kişisel veri göndermeyin).

## Eğitim ve sürümleme döngüsü

```
geri bildirim / distill ──> data/datasets/raw/*.jsonl
        │ msp-agent freeze --min-rating 4            → d1 (değişmez, sha256'lı, train/eval)
        │ python -m training.train_lora --dataset d1 --base models/Qwen2.5-Coder-7B-Instruct --out data/adapters/run1
        │ docker compose restart vllm                 → adaptör "msp-run1" adıyla sunulur
        │ msp-agent register --backend openai --model-ref msp-run1 \
        │        --base-model Qwen/Qwen2.5-Coder-7B-Instruct --dataset d1   → v2
        │ (llamacpp arka ucu için: convert_lora_to_gguf.py ile GGUF'a çevirip --adapter-path verin)
        │ msp-agent eval v2                           → bench metrikleri manifest'e yazılır
        └ msp-agent activate v2                       → servis bir sonraki işte yeni modeli yükler
geri almak: msp-agent activate v1
```

- **Elle örnek ekleme**: `training/examples/*.json` içine yalnızca müşteri açıklaması (`brief`) ve beklenen çıktıyı
  (`output`) yazın; `python -m training.build_examples` system/user metinlerini servisin şablonlarıyla birebir üretip
  doğrular ve `data/datasets/raw/manual.jsonl` dosyasına yazar. Biçim için mevcut üç örneğe bakın.
- **Windows'ta eğitim (Docker)**: Akıllı Uygulama Denetimi açıksa yerel PyTorch DLL'leri engellenir (`WinError 4551`).
  Eğitimi GPU'lu konteynerde çalıştırın: `docker-compose.train.yaml` başındaki komutlar (model indirme, örnek
  derleme, freeze, train_lora). Proje klasörü konteynere bağlanır; `data/` ve `models/` host'ta kalır.
- **Başlangıç verisi (distill)**: Henüz geri bildirim yokken, büyük bir öğretmen modeli (ör. 14B/32B; geçici olarak
  büyük bir GPU'da) `msp-agent distill briefs.jsonl --model <öğretmen>` ile çalıştırın. Çıktıları gözden geçirip
  küçük modeli bunlarla eğitin. Böylece küçük model, büyük modelin bu işteki becerisini öğrenir.
- **QLoRA** (Linux/Windows + NVIDIA): 4-bit temel model + LoRA, gradient checkpointing, batch 1 + accumulation, 8-bit optimizer.
  1.5B ~5 GB, 3B ~7 GB, 7B ~11 GB VRAM ile eğitilebilir. Kayıp yalnızca asistan yanıtı üzerinden hesaplanır.
- **macOS / CPU eğitimi**: bitsandbytes yalnızca CUDA'da çalışır. `train_lora.py` CUDA yoksa bunu algılar,
  4-bit'i kapatır ve Apple Silicon'da MPS ile (yoksa CPU'da) LoRA eğitir. Daha çok bellek ister; 1.5B model önerilir.
- **Sürümlenenler**: model (`data/registry/vN/manifest.json`: temel model, adaptör, veri seti, prompt sürümü,
  üst sürüm, metrikler), veri seti (`data/datasets/dN`), prompt şablonları (`agent/prompts/vN`). Model eğitildiği
  prompt sürümüyle çalıştırılır; prompt değişikliği yeni klasör (`v2`) olarak eklenir, eskisi silinmez.
- **Değerlendirme** (`training/evaluate.py`): sabit açıklama setinde `plan_ok`, `section_ok`, `clean`, `balanced`,
  `seconds`. Yeni sürümü yalnızca öncekinden kötü değilse etkinleştirin. İleride görsel/insan puanı eklenebilir.

## Dizin yapısı

```
agent/          servis, hat, temizleyici, kuyruk, kayıt defteri, veri setleri, LLM arka uçları, prompts/v1
training/       train_lora.py (QLoRA), evaluate.py, bench_briefs.jsonl
tests/          mock arka uçla uçtan uca testler
data/           (git dışında) sqlite, registry/, datasets/, adapters/
```
