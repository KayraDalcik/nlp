# -*- coding: utf-8 -*-
"""
Adim 2: Veri On Isleme
- Ham veriyi temizle ve standart formata cevir
- Retrieval corpus olustur
- Fine-tuning icin veri hazirla
"""

import sys
import json
import re
import pandas as pd
from pathlib import Path
from datasets import load_from_disk

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# --- Metin Temizleme ---
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


# --- HuggingFace Verisini Isle ---
def process_hf_data():
    hf_path = RAW_DIR / "hf_lawchatbot"
    if not hf_path.exists():
        print("[ERR] HuggingFace verisi bulunamadi. Once download.py calistirin.")
        return None

    print("[HF] HuggingFace verisi isleniyor...")
    dataset = load_from_disk(str(hf_path))

    records = []
    for split in dataset.keys():
        for i, row in enumerate(dataset[split]):
            # Kolon adlari: 'Soru' ve 'Cevap' (HuggingFace dataseti)
            question = clean_text(row.get("Soru", row.get("question", row.get("input", ""))))
            answer   = clean_text(row.get("Cevap", row.get("answer",   row.get("output", ""))))
            context  = clean_text(row.get("context", row.get("Baglam", "")))

            if question and answer:
                records.append({
                    "id":       f"hf_{split}_{i}",
                    "source":   "turkish-law-chatbot",
                    "split":    split,
                    "question": question,
                    "answer":   answer,
                    "context":  context,
                })

    df = pd.DataFrame(records)
    out_path = PROCESSED_DIR / "hf_processed.jsonl"
    df.to_json(out_path, orient="records", lines=True, force_ascii=False)
    print(f"[OK] {len(df)} satir -> {out_path}")

    # Split dagilimi
    for split in df["split"].unique():
        n = len(df[df["split"] == split])
        print(f"  {split}: {n} ornek")

    return df


# --- Kaggle Verisini Isle (varsa) ---
def process_kaggle_data():
    kaggle_files = (list(RAW_DIR.glob("*.json"))
                  + list(RAW_DIR.glob("*.csv"))
                  + list(RAW_DIR.glob("*.jsonl")))

    # hf_ ile baslayan dosyalari filtrele (bunlar HF metadata)
    kaggle_files = [f for f in kaggle_files
                    if not f.name.startswith("dataset")]

    if not kaggle_files:
        print("[SKIP] Kaggle verisi bulunamadi, atlaniyor.")
        return None

    print(f"\n[KG] Kaggle verisi isleniyor... ({len(kaggle_files)} dosya)")
    all_records = []

    for f in kaggle_files:
        try:
            if f.suffix == ".csv":
                df_raw = pd.read_csv(f)
            elif f.suffix in [".json", ".jsonl"]:
                df_raw = pd.read_json(f, lines=(f.suffix == ".jsonl"))
            else:
                continue

            print(f"  - {f.name}: {len(df_raw)} satir | kolonlar: {list(df_raw.columns)}")

            for i, row in df_raw.iterrows():
                # Farkli kolon isimlerini dene
                question = clean_text(str(row.get("Soru",     row.get("soru",
                           row.get("question", row.get("input",  ""))))))
                answer   = clean_text(str(row.get("Cevap",    row.get("cevap",
                           row.get("answer",   row.get("output", ""))))))
                context  = clean_text(str(row.get("Baglam",   row.get("context", ""))))

                if question and answer and question != "nan":
                    all_records.append({
                        "id":       f"kaggle_{len(all_records)}",
                        "source":   f.name,
                        "split":    "train",
                        "question": question,
                        "answer":   answer,
                        "context":  context,
                    })
        except Exception as e:
            print(f"  [WARN] {f.name} islenemedi: {e}")

    if not all_records:
        return None

    df = pd.DataFrame(all_records)
    out_path = PROCESSED_DIR / "kaggle_processed.jsonl"
    df.to_json(out_path, orient="records", lines=True, force_ascii=False)
    print(f"[OK] {len(df)} satir -> {out_path}")
    return df


# --- Retrieval Corpus Olustur ---
def build_retrieval_corpus(dfs: list):
    print("\n[CORPUS] Retrieval corpus olusturuluyor...")
    docs = []

    for df in dfs:
        if df is None:
            continue
        for _, row in df.iterrows():
            # Contexti varsa kullan, yoksa answer'i kullan
            text = row.get("context", "")
            if not text or len(text) < 30:
                text = row.get("answer", "")

            if text and len(text) >= 30:
                docs.append({
                    "doc_id": f"{row['source']}_{row['id']}",
                    "text":   text,
                    "source": row["source"],
                    "qa_id":  row["id"],
                })

    df_docs = pd.DataFrame(docs).drop_duplicates(subset=["text"])
    out_path = PROCESSED_DIR / "retrieval_corpus.jsonl"
    df_docs.to_json(out_path, orient="records", lines=True, force_ascii=False)
    print(f"[OK] {len(df_docs)} dokuman -> {out_path}")
    return df_docs


# --- Fine-tuning Formati Olustur (Alpaca/ChatML) ---
def build_finetune_data(dfs: list):
    print("\n[FT] Fine-tuning verisi hazirlaniyor...")
    records = []

    SYSTEM_MSG = (
        "Sen bir Turk hukuku uzmanissin. "
        "Hukuki sorulari net, kaynak gosterir sekilde yanitla."
    )

    for df in dfs:
        if df is None:
            continue
        for _, row in df.iterrows():
            if row.get("split") == "test":
                continue  # Test verisini fine-tuning'e katma

            records.append({
                "messages": [
                    {"role": "system",    "content": SYSTEM_MSG},
                    {"role": "user",      "content": row["question"]},
                    {"role": "assistant", "content": row["answer"]},
                ]
            })

    df_ft = pd.DataFrame(records)
    out_path = PROCESSED_DIR / "finetune_data.jsonl"
    df_ft.to_json(out_path, orient="records", lines=True, force_ascii=False)
    print(f"[OK] {len(df_ft)} ornek -> {out_path}")
    return df_ft


# --- Ozet ---
def print_stats(df_hf, df_kg, df_docs, df_ft):
    print("\n" + "=" * 55)
    print("  VERI OZETI")
    print("=" * 55)
    if df_hf  is not None: print(f"  HuggingFace Q&A:      {len(df_hf):>6} satir")
    if df_kg  is not None: print(f"  Kaggle Q&A:           {len(df_kg):>6} satir")
    if df_docs is not None: print(f"  Retrieval corpus:     {len(df_docs):>6} dokuman")
    if df_ft  is not None: print(f"  Fine-tuning ornekleri:{len(df_ft):>6} ornek")
    print("=" * 55)
    print(f"\n[DIR] {PROCESSED_DIR.resolve()}")
    for f in PROCESSED_DIR.glob("*.jsonl"):
        size = f.stat().st_size / 1e6
        print(f"  {f.name}  ({size:.2f} MB)")


if __name__ == "__main__":
    print("=" * 60)
    print("  TURK HUKUK RAG - VERI ON ISLEME")
    print("=" * 60)

    df_hf  = process_hf_data()
    df_kg  = process_kaggle_data()
    df_docs = build_retrieval_corpus([df_hf, df_kg])
    df_ft  = build_finetune_data([df_hf, df_kg])
    print_stats(df_hf, df_kg, df_docs, df_ft)
