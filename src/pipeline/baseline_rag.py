"""
Ablasyon 1: Baseline RAG Pipeline
- Embedding: paraphrase-multilingual-mpnet-base-v2
- Retrieval: FAISS (dense only)
- LLM: Qwen2.5-3B-Instruct (local) 
"""

import os
import json
import torch
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

load_dotenv()

PROCESSED_DIR = Path("data/processed")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🖥️  Kullanılan cihaz: {DEVICE}")
if DEVICE == "cuda":
    print(f"🎮  GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ─── 1. Corpus Yükle ──────────────────────────────────────────────────────────
def load_corpus() -> List[Dict]:
    corpus_path = PROCESSED_DIR / "retrieval_corpus.jsonl"
    if not corpus_path.exists():
        raise FileNotFoundError("❌ Corpus bulunamadı! Önce preprocess.py çalıştırın.")

    corpus = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            corpus.append(json.loads(line))
    print(f"📚 {len(corpus)} doküman yüklendi.")
    return corpus


# ─── 2. Embedding Modeli ──────────────────────────────────────────────────────
class EmbeddingModel:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        print(f"⚙️  Embedding modeli yükleniyor: {model_name}")
        self.model = SentenceTransformer(model_name, device=DEVICE)
        print("✅ Embedding modeli hazır.")

    def encode(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # Cosine similarity için
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)


# ─── 3. FAISS Index ───────────────────────────────────────────────────────────
class FAISSRetriever:
    def __init__(self, embedding_dim: int = 768):
        self.index = faiss.IndexFlatIP(embedding_dim)   # Inner Product (cosine için)
        self.doc_ids: List[str] = []
        self.texts: List[str]   = []

    def add_documents(self, corpus: List[Dict], embeddings: np.ndarray):
        self.doc_ids = [d["doc_id"] for d in corpus]
        self.texts   = [d["text"]   for d in corpus]
        self.index.add(embeddings)
        print(f"✅ FAISS index'e {self.index.ntotal} doküman eklendi.")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                "doc_id": self.doc_ids[idx],
                "text":   self.texts[idx],
                "score":  float(score),
            })
        return results

    def save(self, path: str = "data/processed/faiss_index"):
        Path(path).mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, f"{path}/index.faiss")
        with open(f"{path}/metadata.json", "w", encoding="utf-8") as f:
            json.dump({"doc_ids": self.doc_ids, "texts": self.texts}, f, ensure_ascii=False)
        print(f"✅ FAISS index kaydedildi → {path}")

    def load(self, path: str = "data/processed/faiss_index"):
        self.index = faiss.read_index(f"{path}/index.faiss")
        with open(f"{path}/metadata.json", encoding="utf-8") as f:
            meta = json.load(f)
        self.doc_ids = meta["doc_ids"]
        self.texts   = meta["texts"]
        print(f"✅ FAISS index yüklendi: {self.index.ntotal} doküman.")


# ─── 4. LLM ───────────────────────────────────────────────────────────────────
class LLMGenerator:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct"):
        print(f"⚙️  LLM yükleniyor: {model_name} (bu biraz zaman alabilir...)")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        print("✅ LLM hazır.")

    def generate(self, question: str, context_docs: List[Dict], max_new_tokens: int = 512) -> str:
        context = "\n\n---\n\n".join([
            f"[Kaynak {i+1}]: {doc['text'][:500]}"
            for i, doc in enumerate(context_docs)
        ])

        prompt = f"""Sen bir Türk hukuku uzmanısın. Aşağıdaki hukuki belgeleri kullanarak soruyu yanıtla.
Yanıtında kaynak göster (örn: "Kaynak 1'e göre..."). Eğer yanıtı bilmiyorsan "Bu konuda yeterli bilgi bulunamadı." de.

Hukuki Belgeler:
{context}

Soru: {question}

Yanıt:"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        return response.strip()


# ─── 5. Baseline RAG Pipeline ─────────────────────────────────────────────────
class BaselineRAG:
    def __init__(self, top_k: int = 5):
        self.top_k    = top_k
        self.embedder = EmbeddingModel()
        self.retriever = FAISSRetriever()
        self.llm      = None   # İlk çalıştırmada None, build() ile oluşur

    def build_index(self, corpus: List[Dict] = None):
        """Corpus'u encode edip FAISS index'e ekle."""
        index_path = "data/processed/faiss_index"

        # Kayıtlı index varsa yükle
        if Path(f"{index_path}/index.faiss").exists():
            print("ℹ️  Kayıtlı FAISS index bulundu, yükleniyor...")
            self.retriever.load(index_path)
            return

        if corpus is None:
            corpus = load_corpus()

        texts = [d["text"] for d in corpus]
        print(f"⚙️  {len(texts)} doküman encode ediliyor...")
        embeddings = self.embedder.encode(texts)
        self.retriever.add_documents(corpus, embeddings)
        self.retriever.save(index_path)

    def load_llm(self):
        self.llm = LLMGenerator()

    def query(self, question: str, use_llm: bool = True) -> Dict:
        # 1. Soruyu encode et
        q_embedding = self.embedder.encode([question], show_progress=False)[0]

        # 2. Retrieval
        docs = self.retriever.search(q_embedding, top_k=self.top_k)

        # 3. LLM ile cevap üret
        answer = ""
        if use_llm:
            if self.llm is None:
                self.load_llm()
            answer = self.llm.generate(question, docs)

        return {
            "question": question,
            "retrieved_docs": docs,
            "answer": answer,
        }


# ─── Ana Program ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  TÜRK HUKUK RAG - BASELINE PIPELINE")
    print("=" * 60)

    rag = BaselineRAG(top_k=5)

    # Index oluştur
    corpus = load_corpus()
    rag.build_index(corpus)

    # Test sorusu (LLM olmadan, sadece retrieval test)
    test_question = "İş akdinin feshinde ihbar süresi ne kadar olmalıdır?"
    print(f"\n🔍 Test Sorusu: {test_question}")

    result = rag.query(test_question, use_llm=False)   # use_llm=True yaparsan LLM de çalışır

    print("\n📄 Bulunan Dokümanlar:")
    for i, doc in enumerate(result["retrieved_docs"]):
        print(f"\n[{i+1}] Score: {doc['score']:.4f}")
        print(f"     {doc['text'][:200]}...")

    print("\n✅ Baseline RAG hazır!")
