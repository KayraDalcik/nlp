# -*- coding: utf-8 -*-
"""
Ablasyon Değerlendirme Runner
- Baseline RAG (FAISS only) ve Hybrid RAG (FAISS + BM25 + RRF) karşılaştırır
- Retrieval-only modda çalışır (LLM'siz) — hızlı benchmark için
- use_llm=True yapılırsa tam RAG değerlendirmesi yapılır

Kullanım:
    python run_eval.py                          # retrieval-only (hızlı)
    python run_eval.py --llm                    # LLM ile (yavaş)
    python run_eval.py --samples 20             # sadece 20 soru
"""

import sys
import json
import time
import argparse
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent / "src"))

from pipeline.baseline_rag import BaselineRAG, load_corpus
from pipeline.hybrid_rag   import HybridRAG
from evaluation.evaluator   import RAGEvaluator


BENCHMARK_PATH = "data/benchmark/test_questions.jsonl"
RESULTS_DIR    = "results"


def run_experiment(name: str, pipeline, evaluator: RAGEvaluator,
                   benchmark_path: str, max_samples: int, use_llm: bool):
    print(f"\n{'#'*60}")
    print(f"  DENEY: {name}")
    print(f"{'#'*60}")
    t0 = time.perf_counter()
    result = evaluator.evaluate(
        rag_pipeline   = pipeline,
        benchmark_path = benchmark_path,
        max_samples    = max_samples,
        use_llm        = use_llm,
    )
    elapsed = time.perf_counter() - t0
    print(f"\n⏱️  Toplam süre: {elapsed/60:.1f} dakika")
    evaluator.save_result(result, results_dir=RESULTS_DIR)
    return result


def main():
    parser = argparse.ArgumentParser(description="Türk Hukuk RAG — Ablasyon Değerlendirmesi")
    parser.add_argument("--llm",     action="store_true", help="LLM ile tam RAG değerlendirmesi")
    parser.add_argument("--samples", type=int, default=None, help="Kaç soru kullanılsın (None=hepsi)")
    parser.add_argument("--experiment", choices=["baseline", "hybrid", "both"], default="both")
    args = parser.parse_args()

    use_llm     = args.llm
    max_samples = args.samples

    print("=" * 60)
    print("  TÜRK HUKUK RAG — ABLASYON DEĞERLENDİRMESİ")
    print(f"  Mod: {'LLM + Retrieval' if use_llm else 'Sadece Retrieval (Hızlı)'}")
    print(f"  Benchmark: {BENCHMARK_PATH}")
    print(f"  Max örnek: {max_samples or 'Hepsi'}")
    print("=" * 60)

    # Corpus yükle (her iki pipeline da bunu paylaşır)
    corpus = load_corpus()

    results = {}

    # ── 1. Baseline RAG ──────────────────────────────────────────────────────
    if args.experiment in ("baseline", "both"):
        print("\n[1/2] Baseline RAG hazırlanıyor...")
        baseline = BaselineRAG(top_k=5)
        baseline.build_index(corpus)
        if use_llm:
            baseline.load_llm()

        evaluator_b = RAGEvaluator("baseline_rag")
        results["baseline"] = run_experiment(
            name           = "Baseline RAG (FAISS Dense)",
            pipeline       = baseline,
            evaluator      = evaluator_b,
            benchmark_path = BENCHMARK_PATH,
            max_samples    = max_samples,
            use_llm        = use_llm,
        )

    # ── 2. Hybrid RAG ────────────────────────────────────────────────────────
    if args.experiment in ("hybrid", "both"):
        print("\n[2/2] Hybrid RAG hazırlanıyor...")
        hybrid = HybridRAG(top_k=5, dense_weight=0.6, sparse_weight=0.4)
        hybrid.build_index(corpus)
        if use_llm:
            hybrid.load_llm()

        evaluator_h = RAGEvaluator("hybrid_rag")
        results["hybrid"] = run_experiment(
            name           = "Hybrid RAG (FAISS + BM25 + RRF)",
            pipeline       = hybrid,
            evaluator      = evaluator_h,
            benchmark_path = BENCHMARK_PATH,
            max_samples    = max_samples,
            use_llm        = use_llm,
        )

    # ── Karşılaştırma Tablosu ─────────────────────────────────────────────
    if len(results) == 2:
        b = results["baseline"]
        h = results["hybrid"]

        print("\n" + "=" * 60)
        print("  KARŞILAŞTIRMA TABLOSU")
        print("=" * 60)
        print(f"{'Metrik':<30} {'Baseline':>10} {'Hybrid':>10} {'Δ':>10}")
        print("-" * 60)
        metrics = [
            ("ROUGE-L F1",        b.rouge_l_f1,        h.rouge_l_f1),
            ("BERTScore F1",       b.bert_score_f1,     h.bert_score_f1),
            ("Recall@1",           b.recall_at_1,       h.recall_at_1),
            ("Recall@3",           b.recall_at_3,       h.recall_at_3),
            ("Recall@5",           b.recall_at_5,       h.recall_at_5),
            ("MRR",                b.mrr,               h.mrr),
            ("nDCG@5",             b.ndcg_at_5,         h.ndcg_at_5),
            ("Avg Retrieval Skor", b.avg_retrieval_score, h.avg_retrieval_score),
            ("Avg Ret. Süresi(ms)",b.avg_retrieval_time_ms, h.avg_retrieval_time_ms),
        ]
        for name, bv, hv in metrics:
            diff = hv - bv
            sign = "+" if diff >= 0 else ""
            print(f"  {name:<28} {bv:>10.4f} {hv:>10.4f} {sign}{diff:>+10.4f}")
        print("=" * 60)

        # Karşılaştırmayı kaydet
        Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
        comparison = {
            "baseline": {
                "rouge_l_f1":           b.rouge_l_f1,
                "bert_score_f1":        b.bert_score_f1,
                "recall_at_1":          b.recall_at_1,
                "recall_at_3":          b.recall_at_3,
                "recall_at_5":          b.recall_at_5,
                "mrr":                  b.mrr,
                "ndcg_at_5":            b.ndcg_at_5,
                "avg_retrieval_score":  b.avg_retrieval_score,
                "avg_retrieval_time_ms": b.avg_retrieval_time_ms,
            },
            "hybrid": {
                "rouge_l_f1":           h.rouge_l_f1,
                "bert_score_f1":        h.bert_score_f1,
                "recall_at_1":          h.recall_at_1,
                "recall_at_3":          h.recall_at_3,
                "recall_at_5":          h.recall_at_5,
                "mrr":                  h.mrr,
                "ndcg_at_5":            h.ndcg_at_5,
                "avg_retrieval_score":  h.avg_retrieval_score,
                "avg_retrieval_time_ms": h.avg_retrieval_time_ms,
            },
        }
        out_path = Path(RESULTS_DIR) / "comparison.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] Karşılaştırma → {out_path}")

    print("\n✅ Değerlendirme tamamlandı!")


if __name__ == "__main__":
    main()
