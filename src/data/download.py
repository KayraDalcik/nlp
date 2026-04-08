# -*- coding: utf-8 -*-
"""
Adim 1: Veri Indirme
- HuggingFace: turkish-law-chatbot dataseti
- Kaggle: turkishlaw-dataset-for-llm-finetuning
"""

import os
import sys
import json
from pathlib import Path
from datasets import load_dataset
from dotenv import load_dotenv

# Windows encoding fix
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


# --- 1. HuggingFace Verisi ---
def download_huggingface_data():
    print("[HF] HuggingFace verisi indiriliyor: Renicames/turkish-law-chatbot ...")
    dataset = load_dataset("Renicames/turkish-law-chatbot", token=os.getenv("HF_TOKEN"))

    save_path = RAW_DIR / "hf_lawchatbot"
    dataset.save_to_disk(str(save_path))
    print(f"[OK] Kaydedildi -> {save_path}")

    print("\n[INFO] Veri Onizlemesi:")
    for split in dataset.keys():
        print(f"  {split}: {len(dataset[split])} ornek")
        print(f"  Sutunlar: {dataset[split].column_names}")
        print(f"  Ilk ornek: {dataset[split][0]}\n")

    return dataset


# --- 2. Kaggle Verisi ---
def download_kaggle_data():
    print("\n[KG] Kaggle verisi indiriliyor: turkishlaw-dataset-for-llm-finetuning ...")

    # Yeni Kaggle token sistemi: KGAT_... formatı KAGGLE_API_TOKEN env variable olarak set edilir
    kaggle_token = os.getenv("KAGGLE_API_TOKEN")
    if not kaggle_token:
        print("[ERR] KAGGLE_API_TOKEN .env'de bulunamadi!")
        return

    # Kaggle CLI icin environment'a ekle
    env = {**os.environ, "KAGGLE_API_TOKEN": kaggle_token}

    try:
        import subprocess

        # Once versiyonu kontrol et
        ver = subprocess.run(["kaggle", "--version"], capture_output=True, text=True, env=env)
        print(f"[INFO] Kaggle CLI: {ver.stdout.strip()}")

        result = subprocess.run(
            ["kaggle", "datasets", "download",
             "-d", "batuhankalem/turkishlaw-dataset-for-llm-finetuning",
             "-p", str(RAW_DIR), "--unzip"],
            capture_output=True, text=True,
            env=env
        )

        if result.returncode == 0:
            print(f"[OK] Kaggle verisi indirildi -> {RAW_DIR}")
            if result.stdout:
                print(result.stdout)
        else:
            print(f"[ERR] Kaggle hatasi:\n{result.stderr}")
            print("\n[INFO] Alternatif: Manuel indirme adimlarini deneyin.")
            print("  1. https://www.kaggle.com/datasets/batuhankalem/turkishlaw-dataset-for-llm-finetuning")
            print("  2. 'Download' butonuna basin")
            print(f"  3. ZIP dosyasini surayi acin: {RAW_DIR.resolve()}")

    except FileNotFoundError:
        print("[ERR] kaggle komutu bulunamadi. 'pip install kaggle' calistirin.")


# --- 3. Veri Ozeti ---
def summarize_data():
    print("\n[DIR] Raw Veri Klasoru:")
    found = False
    for f in RAW_DIR.rglob("*"):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.relative_to(RAW_DIR)}  ({size_mb:.2f} MB)")
            found = True
    if not found:
        print("  (bos - hicbir dosya indirilemedi)")


if __name__ == "__main__":
    print("=" * 60)
    print("  TURK HUKUK RAG - VERI INDIRME")
    print("=" * 60)

    # HuggingFace
    try:
        hf_data = download_huggingface_data()
    except Exception as e:
        print(f"[ERR] HuggingFace hatasi: {e}")

    # Kaggle
    download_kaggle_data()

    # Ozet
    summarize_data()
