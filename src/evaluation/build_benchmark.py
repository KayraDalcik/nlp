# -*- coding: utf-8 -*-
"""
Adım 3: Benchmark Test Seti Hazırlama
- hf_processed.jsonl içindeki test split'inden sorular çıkar
- Yeterli test verisi yoksa train'den random sample al
- data/benchmark/test_questions.jsonl olarak kaydet
"""

import sys
import json
import random
import pandas as pd
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

PROCESSED_DIR = Path("data/processed")
BENCHMARK_DIR = Path("data/benchmark")
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

BENCHMARK_FILE = BENCHMARK_DIR / "test_questions.jsonl"
MAX_SAMPLES    = 100
RANDOM_SEED    = 42


def build_benchmark():
    hf_path = PROCESSED_DIR / "hf_processed.jsonl"
    if not hf_path.exists():
        print("[ERR] hf_processed.jsonl bulunamadı. Önce preprocess.py çalıştırın.")
        return

    df = pd.read_json(hf_path, lines=True)
    print(f"[INFO] Toplam kayıt: {len(df)}")
    print(f"[INFO] Split dağılımı:\n{df['split'].value_counts().to_string()}")

    # Önce test split'ini dene
    df_test = df[df["split"] == "test"].copy()

    if len(df_test) >= 20:
        print(f"\n[OK] Test split bulundu: {len(df_test)} örnek")
        source = df_test
    else:
        print(f"\n[WARN] Test split yetersiz ({len(df_test)} örnek). Train'den örnekleniyor...")
        df_train = df[df["split"] != "test"].copy()
        source = df_train.sample(min(MAX_SAMPLES, len(df_train)), random_state=RANDOM_SEED)

    # İstenen miktarı al
    df_bench = source.sample(min(MAX_SAMPLES, len(source)), random_state=RANDOM_SEED)

    # Benchmark formatına dönüştür
    records = []
    for i, row in df_bench.reset_index(drop=True).iterrows():
        records.append({
            "id":            f"bench_{i:04d}",
            "question":      row["question"],
            "reference_answer": row["answer"],
            "source_split":  row.get("split", "unknown"),
            "original_id":   row.get("id", ""),
        })

    # Kaydet
    with open(BENCHMARK_FILE, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n[OK] {len(records)} benchmark sorusu → {BENCHMARK_FILE}")

    # Önizleme
    print("\n[ÖRNEK] İlk 3 soru:")
    for r in records[:3]:
        print(f"  [{r['id']}] {r['question'][:80]}...")


if __name__ == "__main__":
    print("=" * 60)
    print("  TÜRK HUKUK RAG - BENCHMARK HAZIRLAMA")
    print("=" * 60)
    build_benchmark()
