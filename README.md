# Turkish Legal RAG — Proje Kurulum Rehberi

Merhaba! Bu dosya projeyi sıfırdan kurma adımlarını içeriyor.

---

## ✅ Adım 1: Python Ortamı Kur

```powershell
# Conda kullanıyorsan (önerilen):
conda create -n legal-rag python=3.10 -y
conda activate legal-rag

# veya sadece venv:
python -m venv .venv
.venv\Scripts\activate
```

---

## ✅ Adım 2: Bağımlılıkları Yükle

```powershell
pip install -r requirements.txt
```

> ⚠️ Bu adım 5-10 dakika sürebilir.

---

## ✅ Adım 3: API Token'larını Ayarla

`.env.example` dosyasını `.env` olarak kopyala:

```powershell
Copy-Item .env.example .env
```

Sonra `.env` dosyasını aç ve şu bilgileri doldur:
- `HF_TOKEN` → https://huggingface.co/settings/tokens
- `KAGGLE_API_TOKEN` → https://www.kaggle.com/settings > API > Create New Token (KGAT_ ile başlayan token)

---

## ✅ Adım 4: Veriyi İndir

```powershell
python src/data/download.py
```

---

## ✅ Adım 5: Veriyi İşle

```powershell
python src/data/preprocess.py
```

---

## ✅ Adım 6: Baseline RAG'ı Test Et

```powershell
python src/pipeline/baseline_rag.py
```

---

## 📁 Proje Yapısı

```
turkish-legal-rag/
├── data/
│   ├── raw/            ← Ham veriler
│   ├── processed/      ← İşlenmiş veriler + FAISS index
│   └── benchmark/      ← Test seti (150-300 soru)
├── src/
│   ├── data/           ← download.py, preprocess.py
│   ├── pipeline/       ← baseline_rag.py, ...
│   ├── retrieval/      ← embedder, BM25, FAISS
│   ├── reranker/       ← cross-encoder
│   ├── llm/            ← prompt templates, LoRA
│   └── evaluation/     ← metrikler
├── notebooks/          ← Jupyter analiz notebook'ları
├── configs/            ← Hyperparameter dosyaları
├── demo/               ← Gradio arayüzü
├── requirements.txt
├── .env.example        ← Bunu .env olarak kopyala!
└── README.md
```
