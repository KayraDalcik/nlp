# -*- coding: utf-8 -*-
"""
Evaluation Metrics — ROUGE-L, BERTScore, (opsiyonel RAGAS)
Tüm ablasyon deneyleri bu modül üzerinden karşılaştırılır.
"""

import sys
import json
import time
import warnings
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

import numpy as np

# Suppress verbose warnings
warnings.filterwarnings("ignore", category=FutureWarning)

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


# ─── Sonuç Veri Yapısı ────────────────────────────────────────────────────────
@dataclass
class EvalResult:
    experiment_name: str
    num_samples:     int
    rouge_l_f1:      float
    rouge_l_precision: float
    rouge_l_recall:  float
    bert_score_f1:   float
    bert_score_precision: float
    bert_score_recall:    float
    avg_retrieval_score:  float     # FAISS top-1 similarity score
    avg_retrieval_time_ms: float
    avg_generation_time_ms: float
    # RAGAS (opsiyonel)
    ragas_faithfulness:      Optional[float] = None
    ragas_answer_relevance:  Optional[float] = None
    ragas_context_recall:    Optional[float] = None

    def summary(self) -> str:
        lines = [
            f"\n{'='*55}",
            f"  Deney: {self.experiment_name}",
            f"{'='*55}",
            f"  Örnek sayısı       : {self.num_samples}",
            f"  ROUGE-L F1         : {self.rouge_l_f1:.4f}",
            f"  BERTScore F1 (tr)  : {self.bert_score_f1:.4f}",
            f"  Avg Retrieval Skoru: {self.avg_retrieval_score:.4f}",
            f"  Retrieval süresi   : {self.avg_retrieval_time_ms:.1f} ms/soru",
            f"  Generation süresi  : {self.avg_generation_time_ms:.1f} ms/soru",
        ]
        if self.ragas_faithfulness is not None:
            lines += [
                f"  RAGAS Faithfulness : {self.ragas_faithfulness:.4f}",
                f"  RAGAS Ans.Relev.   : {self.ragas_answer_relevance:.4f}",
                f"  RAGAS Ctx.Recall   : {self.ragas_context_recall:.4f}",
            ]
        lines.append("=" * 55)
        return "\n".join(lines)


# ─── Metrik Hesaplayıcılar ────────────────────────────────────────────────────
class RougeEvaluator:
    def __init__(self):
        from rouge_score import rouge_scorer
        self.scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    def score(self, prediction: str, reference: str) -> Dict[str, float]:
        result = self.scorer.score(reference, prediction)
        rl = result["rougeL"]
        return {
            "f1":        round(rl.fmeasure,  4),
            "precision": round(rl.precision, 4),
            "recall":    round(rl.recall,    4),
        }

    def batch_score(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        scores = [self.score(p, r) for p, r in zip(predictions, references)]
        return {
            "f1":        float(np.mean([s["f1"]        for s in scores])),
            "precision": float(np.mean([s["precision"] for s in scores])),
            "recall":    float(np.mean([s["recall"]    for s in scores])),
        }


class BertScoreEvaluator:
    def __init__(self, lang: str = "tr"):
        self.lang = lang
        self._model_loaded = False

    def _lazy_load(self):
        if not self._model_loaded:
            import bert_score  # noqa — lazy import
            self._bert_score = bert_score
            self._model_loaded = True

    def batch_score(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        self._lazy_load()
        P, R, F = self._bert_score.score(
            predictions, references,
            lang=self.lang,
            verbose=False,
            batch_size=8,
        )
        return {
            "f1":        float(F.mean().item()),
            "precision": float(P.mean().item()),
            "recall":    float(R.mean().item()),
        }


# ─── Ana Evaluatör ────────────────────────────────────────────────────────────
class RAGEvaluator:
    def __init__(self, experiment_name: str, bert_lang: str = "tr"):
        self.experiment_name = experiment_name
        self.rouge = RougeEvaluator()
        self.bert  = BertScoreEvaluator(lang=bert_lang)
        print(f"[EVAL] Evaluatör hazır: '{experiment_name}'")

    def evaluate(
        self,
        rag_pipeline,           # BaselineRAG veya türevleri
        benchmark_path: str,
        max_samples: Optional[int] = None,
        use_llm: bool = True,
    ) -> EvalResult:
        """
        Benchmark dosyasındaki sorular üzerinde RAG pipeline'ını çalıştır,
        metrikleri hesapla ve EvalResult döndür.
        """
        # ── Benchmark yükle ──
        bench_path = Path(benchmark_path)
        if not bench_path.exists():
            raise FileNotFoundError(f"Benchmark bulunamadı: {bench_path}")

        samples = []
        with open(bench_path, encoding="utf-8") as f:
            for line in f:
                samples.append(json.loads(line))

        if max_samples:
            samples = samples[:max_samples]

        print(f"\n[EVAL] {len(samples)} soru değerlendiriliyor...")

        predictions  = []
        references   = []
        ret_scores   = []
        ret_times    = []
        gen_times    = []

        for i, sample in enumerate(samples, 1):
            question  = sample["question"]
            reference = sample["reference_answer"]

            # ── Retrieval ──
            t0 = time.perf_counter()
            q_emb = rag_pipeline.embedder.encode([question], show_progress=False)[0]
            docs  = rag_pipeline.retriever.search(q_emb, top_k=rag_pipeline.top_k)
            ret_time_ms = (time.perf_counter() - t0) * 1000
            ret_times.append(ret_time_ms)

            top_score = docs[0]["score"] if docs else 0.0
            ret_scores.append(top_score)

            # ── Generation ──
            if use_llm:
                t1 = time.perf_counter()
                answer = rag_pipeline.llm.generate(question, docs)
                gen_times.append((time.perf_counter() - t1) * 1000)
            else:
                # Retrieval-only: top belgeyi cevap olarak kullan
                answer = docs[0]["text"][:500] if docs else ""
                gen_times.append(0.0)

            predictions.append(answer)
            references.append(reference)

            if i % 10 == 0 or i == len(samples):
                print(f"  [{i}/{len(samples)}] ✓  ret={ret_time_ms:.0f}ms  score={top_score:.3f}")

        # ── ROUGE ──
        print("\n[EVAL] ROUGE-L hesaplanıyor...")
        rouge_scores = self.rouge.batch_score(predictions, references)

        # ── BERTScore ──
        print("[EVAL] BERTScore hesaplanıyor (ilk seferinde model indirilir)...")
        bert_scores = self.bert.batch_score(predictions, references)

        result = EvalResult(
            experiment_name        = self.experiment_name,
            num_samples            = len(samples),
            rouge_l_f1             = rouge_scores["f1"],
            rouge_l_precision      = rouge_scores["precision"],
            rouge_l_recall         = rouge_scores["recall"],
            bert_score_f1          = bert_scores["f1"],
            bert_score_precision   = bert_scores["precision"],
            bert_score_recall      = bert_scores["recall"],
            avg_retrieval_score    = float(np.mean(ret_scores)),
            avg_retrieval_time_ms  = float(np.mean(ret_times)),
            avg_generation_time_ms = float(np.mean(gen_times)),
        )

        print(result.summary())
        return result

    def save_result(self, result: EvalResult, results_dir: str = "results"):
        """Sonuçları JSON olarak kaydet."""
        out_dir = Path(results_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f"{result.experiment_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, ensure_ascii=False, indent=2)
        print(f"[OK] Sonuçlar kaydedildi → {out_path}")
        return out_path
