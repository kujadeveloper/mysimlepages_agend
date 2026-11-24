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

```bash
cp .env.example .env            # MSP_API_KEY'i değiştirin
docker compose up -d
docker compose exec ollama ollama pull qwen2.5-coder:3b
curl -H "X-API-Key: ..." -H "Content-Type: application/json" \
     -d '{"brief":"Kafe için tek sayfalık site, sıcak kahve tonları"}' http://127.0.0.1:8100/v1/designs
```

Geliştirme (Python 3.10+):

```bash
pip install -e ".[dev]"
MSP_BACKEND=mock msp-agent generate "Kafe sitesi" -o site.html
pytest
```

## Düşük kart ayarları

| VRAM | Önerilen model (GGUF Q4_K_M) | Not |
|------|------------------------------|-----|
| CPU / 4 GB | Qwen2.5-Coder-1.5B-Instruct | `MSP_N_GPU_LAYERS=0` ya da kısmi (ör. 16) |
| 6–8 GB | Qwen2.5-Coder-3B-Instruct | varsayılan |
| 12–16 GB | Qwen2.5-Coder-7B-Instruct | daha iyi kalite |

- `MSP_N_CTX=4096`: bölüm bölüm üretim sayesinde küçük bağlam yeterli; KV önbelleği küçük kalır.
- `MSP_WORKERS=1`: aynı anda tek üretim; istekler kuyrukta bekler, bellek taşmaz.
- Ollama'da `OLLAMA_KV_CACHE_TYPE=q8_0`, `OLLAMA_FLASH_ATTENTION=1` (compose'ta açık).
- Yerel GGUF: `pip install -e ".[llamacpp]"` (CUDA: `CMAKE_ARGS="-DGGML_CUDA=on"`), `MSP_BACKEND=llamacpp`,
  `MSP_DEFAULT_MODEL_PATH=models/...gguf`. Eğitilen LoRA, temel modelle birleştirilmeden yüklenebilir.

## API

| Yöntem | Yol | Açıklama |
|--------|-----|----------|
| GET | `/health` | durum + etkin model |
| POST | `/v1/designs` | `{brief, user_ref?}` → `202 {id, position}` |
| GET | `/v1/designs/{id}` | durum, ilerleme (`stage`, `progress`), plan, metrikler, bitince `html` |
| GET | `/v1/designs/{id}/preview` | HTML önizleme (CSP'li; sandbox iframe'de, ayrı kökenden gösterin) |
| POST | `/v1/designs/{id}/feedback` | `{rating 1-5, comment?, corrections?: {section_id: html}}` → eğitim verisi |
| GET/POST | `/v1/models`, `/v1/models/{v}/activate` | model sürümleri |
| GET/POST | `/v1/datasets`, `/v1/datasets/freeze` | veri seti sürümleri |

Tüm `/v1` uçları `X-API-Key` ister. Servis internete açılmamalı; kullanıcı istekleri, kota ve ödeme kontrolü
Django backend'de yapılır (`user_ref` yalnızca kimlik numarasıdır, kişisel veri göndermeyin).

## Eğitim ve sürümleme döngüsü

```
geri bildirim / distill ──> data/datasets/raw/*.jsonl
        │ msp-agent freeze --min-rating 4            → d1 (değişmez, sha256'lı, train/eval)
        │ python -m training.train_lora --dataset d1 --base Qwen/Qwen2.5-Coder-1.5B-Instruct --out data/adapters/run1
        │ llama.cpp: convert_lora_to_gguf.py data/adapters/run1 --outfile data/adapters/run1/lora.gguf
        │ msp-agent register --backend llamacpp --model-ref models/base.gguf \
        │        --adapter-path data/adapters/run1/lora.gguf --base-model Qwen/... --dataset d1   → v2
        │ msp-agent eval v2                           → bench metrikleri manifest'e yazılır
        └ msp-agent activate v2                       → servis bir sonraki işte yeni modeli yükler
geri almak: msp-agent activate v1
```

- **Başlangıç verisi (distill)**: Henüz geri bildirim yokken, büyük bir öğretmen modeli (ör. 14B/32B; geçici olarak
  büyük bir GPU'da) `msp-agent distill briefs.jsonl --model <öğretmen>` ile çalıştırın. Çıktıları gözden geçirip
  küçük modeli bunlarla eğitin. Böylece küçük model, büyük modelin bu işteki becerisini öğrenir.
- **QLoRA**: 4-bit temel model + LoRA, gradient checkpointing, batch 1 + accumulation, 8-bit optimizer.
  1.5B ~5 GB, 3B ~7 GB, 7B ~11 GB VRAM ile eğitilebilir. Kayıp yalnızca asistan yanıtı üzerinden hesaplanır.
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
