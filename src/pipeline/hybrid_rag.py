# -*- coding: utf-8 -*-
"""
Ablasyon 2: Hybrid RAG Pipeline
- Dense Retrieval: FAISS + paraphrase-multilingual-mpnet-base-v2
- Sparse Retrieval: BM25Okapi
- Fusion: Reciprocal Rank Fusion (RRF)
- LLM: Qwen2.5-3B-Instruct (aynı)

Baseline ile aynı arayüzü paylaşır — Evaluator direkt kullanabilir.
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Proje src klasörü path'e ekle
sys.path.insert(0, str(Path(__file__).parents[1]))

from pipeline.baseline_rag import (
    load_corpus,
    EmbeddingModel,
    FAISSRetriever,
    LLMGenerator,
    PROCESSED_DIR,
    DEVICE,
)
from retrieval.bm25_retriever import BM25Retriever, reciprocal_rank_fusion


class HybridRAG:
    """
    Baseline RAG ile aynı .query() arayüzüne sahip Hybrid pipeline.
    Evaluator `rag_pipeline.embedder`, `rag_pipeline.retriever`,
    `rag_pipeline.top_k` ve `rag_pipeline.llm` özelliklerine erişir,
    bu yüzden hepsini burada da tanımlıyoruz.
    """

    def __init__(
        self,
        top_k: int = 5,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        rrf_k: int = 60,
        bm25_candidate_k: int = 20,   # BM25 ve dense her biri kaç aday getirsin
    ):
        self.top_k = top_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k
        self.bm25_candidate_k = bm25_candidate_k

        # Evaluator bunlara doğrudan erişir
        self.embedder  = EmbeddingModel()
        self.retriever = FAISSRetriever()      # Dense retriever
        self.bm25      = BM25Retriever()        # Sparse retriever
        self.llm: LLMGenerator = None

    # ── Index Oluşturma ──────────────────────────────────────────────────────
    def build_index(self, corpus: List[Dict] = None) -> None:
        """Dense (FAISS) ve Sparse (BM25) index'leri oluştur veya yükle."""
        faiss_path = "data/processed/faiss_index"
        bm25_path  = "data/processed/bm25_index"

        if corpus is None:
            corpus = load_corpus()

        # ── FAISS ──
        if Path(f"{faiss_path}/index.faiss").exists():
            print("[HYBRID] Kayıtlı FAISS index bulundu, yükleniyor...")
            self.retriever.load(faiss_path)
        else:
            texts = [d["text"] for d in corpus]
            print(f"[HYBRID] {len(texts)} doküman encode ediliyor (FAISS)...")
            embeddings = self.embedder.encode(texts)
            self.retriever.add_documents(corpus, embeddings)
            self.retriever.save(faiss_path)

        # ── BM25 ──
        if (Path(bm25_path) / "bm25.pkl").exists():
            print("[HYBRID] Kayıtlı BM25 index bulundu, yükleniyor...")
            self.bm25.load(bm25_path)
        else:
            self.bm25.build(corpus)
            self.bm25.save(bm25_path)

    def load_llm(self) -> None:
        self.llm = LLMGenerator()

    # ── Sorgu ────────────────────────────────────────────────────────────────
    def query(self, question: str, use_llm: bool = True) -> Dict:
        # 1. Dense retrieval
        q_emb = self.embedder.encode([question], show_progress=False)[0]
        dense_docs = self.retriever.search(q_emb, top_k=self.bm25_candidate_k)

        # 2. Sparse retrieval
        sparse_docs = self.bm25.search(question, top_k=self.bm25_candidate_k)

        # 3. RRF fusion → top_k al
        fused = reciprocal_rank_fusion(
            dense_results  = dense_docs,
            sparse_results = sparse_docs,
            dense_weight   = self.dense_weight,
            sparse_weight  = self.sparse_weight,
            k              = self.rrf_k,
        )[: self.top_k]

        # 4. LLM
        answer = ""
        if use_llm:
            if self.llm is None:
                self.load_llm()
            answer = self.llm.generate(question, fused)

        return {
            "question":       question,
            "retrieved_docs": fused,
            "answer":         answer,
            "debug": {
                "dense_top1":  dense_docs[0]["score"] if dense_docs else 0.0,
                "sparse_top1": sparse_docs[0]["score"] if sparse_docs else 0.0,
            },
        }


# ─── Ana Program ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  TÜRK HUKUK RAG — HYBRİD PIPELINE (Dense + BM25 + RRF)")
    print("=" * 60)

    rag = HybridRAG(top_k=5)
    corpus = load_corpus()
    rag.build_index(corpus)

    # Retrieval-only test
    questions = [
        "İş akdinin feshinde ihbar süresi ne kadar olmalıdır?",
        "Anayasaya göre temel haklar nelerdir?",
        "Kira sözleşmesinin sona ermesi için hangi koşullar gereklidir?",
    ]

    for q in questions:
        print(f"\n🔍 {q}")
        result = rag.query(q, use_llm=False)
        docs = result["retrieved_docs"]
        print(f"   dense_top1={result['debug']['dense_top1']:.4f}  "
              f"sparse_top1={result['debug']['sparse_top1']:.4f}")
        for i, doc in enumerate(docs[:3], 1):
            print(f"   [{i}] rrf={doc['score']:.5f}  {doc['text'][:100]}...")

    print("\n✅ Hybrid RAG hazır!")
