# -*- coding: utf-8 -*-
"""
BM25 Sparse Retriever
- rank_bm25 kütüphanesi ile Okapi BM25
- Türkçe tokenization (boşluk bazlı, stop-words opsiyonel)
- FAISS ile Hybrid Retrieval için ReciprocAl Rank Fusion (RRF)
"""

import json
import pickle
import math
import sys
from pathlib import Path
from typing import List, Dict, Optional

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# ─── Türkçe Stop Words (minimal) ─────────────────────────────────────────────
TR_STOP_WORDS = {
    "bir", "bu", "ve", "ile", "için", "de", "da", "ki",
    "mi", "mu", "mü", "mı", "bir", "bu", "şu", "o",
    "ben", "sen", "biz", "siz", "onlar", "ise", "ama",
    "fakat", "ancak", "veya", "ya", "hem", "ne", "nasıl",
    "olan", "olarak", "olan", "gibi", "kadar", "daha",
    "en", "çok", "az", "her", "hiç", "herhangi",
}


def tokenize_tr(text: str, remove_stopwords: bool = True) -> List[str]:
    """Basit Türkçe tokenizer: küçük harfe çevir, noktalama sil, tokenize et."""
    import re
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)   # noktalama → boşluk
    tokens = text.split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in TR_STOP_WORDS and len(t) > 1]
    return tokens


# ─── BM25 Retriever ──────────────────────────────────────────────────────────
class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75, remove_stopwords: bool = True):
        """
        Args:
            k1: Term frequency saturation parametresi (tipik: 1.2–2.0)
            b:  Belge uzunluğu normalizasyonu (0=devre dışı, 1=tam normalizasyon)
            remove_stopwords: Türkçe stop-word filtresi
        """
        from rank_bm25 import BM25Okapi
        self.BM25Okapi = BM25Okapi
        self.k1 = k1
        self.b = b
        self.remove_stopwords = remove_stopwords

        self.bm25: Optional[object] = None
        self.doc_ids: List[str] = []
        self.texts:   List[str] = []
        self._tokenized_corpus: List[List[str]] = []

    def build(self, corpus: List[Dict]) -> None:
        """Corpus'tan BM25 index oluştur."""
        print(f"[BM25] {len(corpus)} doküman tokenize ediliyor...")
        self.doc_ids = [d["doc_id"] for d in corpus]
        self.texts   = [d["text"]   for d in corpus]

        self._tokenized_corpus = [
            tokenize_tr(text, self.remove_stopwords)
            for text in self.texts
        ]
        self.bm25 = self.BM25Okapi(
            self._tokenized_corpus,
            k1=self.k1,
            b=self.b,
        )
        avg_len = sum(len(t) for t in self._tokenized_corpus) / len(self._tokenized_corpus)
        print(f"[BM25] Index hazır. Ort. token/belge: {avg_len:.1f}")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Query'yi tokenize edip BM25 skorlarını hesapla."""
        if self.bm25 is None:
            raise RuntimeError("BM25 index henüz oluşturulmadı. build() çağırın.")

        query_tokens = tokenize_tr(query, self.remove_stopwords)
        scores = self.bm25.get_scores(query_tokens)

        # Top-K index'leri bul
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "doc_id": self.doc_ids[idx],
                "text":   self.texts[idx],
                "score":  float(scores[idx]),
            })
        return results

    def save(self, path: str = "data/processed/bm25_index") -> None:
        """BM25 index ve metadata'yı pickle ile kaydet."""
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)

        with open(out / "bm25.pkl", "wb") as f:
            pickle.dump({
                "bm25":               self.bm25,
                "doc_ids":            self.doc_ids,
                "texts":              self.texts,
                "tokenized_corpus":   self._tokenized_corpus,
                "k1":                 self.k1,
                "b":                  self.b,
            }, f)
        print(f"[BM25] Index kaydedildi → {out}")

    def load(self, path: str = "data/processed/bm25_index") -> None:
        """Kayıtlı BM25 index'i yükle."""
        pkl_path = Path(path) / "bm25.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(f"BM25 index bulunamadı: {pkl_path}")

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        self.bm25               = data["bm25"]
        self.doc_ids            = data["doc_ids"]
        self.texts              = data["texts"]
        self._tokenized_corpus  = data["tokenized_corpus"]
        self.k1                 = data["k1"]
        self.b                  = data["b"]
        print(f"[BM25] Index yüklendi: {len(self.doc_ids)} doküman.")


# ─── Reciprocal Rank Fusion ───────────────────────────────────────────────────
def reciprocal_rank_fusion(
    dense_results: List[Dict],
    sparse_results: List[Dict],
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
    k: int = 60,
) -> List[Dict]:
    """
    Dense (FAISS) + Sparse (BM25) sonuçlarını RRF ile birleştir.

    RRF skoru: w_dense * 1/(k + rank_dense) + w_sparse * 1/(k + rank_sparse)

    Args:
        k: RRF sabitesi (rank etkisini yumuşatır, standart: 60)
        dense_weight / sparse_weight: Ağırlık dengeleri (toplamı önemli değil)
    """
    # doc_id → rrf_score
    rrf_scores: Dict[str, float] = {}
    doc_texts: Dict[str, str] = {}

    # Dense katkısı
    for rank, doc in enumerate(dense_results, start=1):
        did = doc["doc_id"]
        rrf_scores[did] = rrf_scores.get(did, 0.0) + dense_weight * (1.0 / (k + rank))
        doc_texts[did]  = doc["text"]

    # Sparse katkısı
    for rank, doc in enumerate(sparse_results, start=1):
        did = doc["doc_id"]
        rrf_scores[did] = rrf_scores.get(did, 0.0) + sparse_weight * (1.0 / (k + rank))
        doc_texts[did]  = doc.get("text", doc_texts.get(did, ""))

    # Sırala
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    return [
        {"doc_id": did, "text": doc_texts[did], "score": score}
        for did, score in sorted_docs
    ]


# ─── Test ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from pipeline.baseline_rag import load_corpus

    print("=" * 55)
    print("  BM25 RETRIEVER — TEST")
    print("=" * 55)

    corpus = load_corpus()
    retriever = BM25Retriever()

    # Kayıtlı index varsa yükle, yoksa oluştur
    bm25_path = "data/processed/bm25_index"
    if (Path(bm25_path) / "bm25.pkl").exists():
        print("[INFO] Kayıtlı BM25 index bulundu, yükleniyor...")
        retriever.load(bm25_path)
    else:
        retriever.build(corpus)
        retriever.save(bm25_path)

    # Test sorgusu
    query = "İş akdinin feshinde ihbar süresi ne kadar olmalıdır?"
    print(f"\n🔍 Sorgu: {query}\n")
    results = retriever.search(query, top_k=5)

    for i, r in enumerate(results, 1):
        print(f"[{i}] score={r['score']:.4f}  {r['text'][:120]}...")
