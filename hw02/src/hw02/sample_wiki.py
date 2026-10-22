from datasets import load_dataset
from datasets import load_dataset_builder
import numpy as np
import json
from pathlib import Path
from huggingface_hub import snapshot_download

# LOAD DATASET

local_dir = snapshot_download("wikimedia/wikipedia", repo_type="dataset", allow_patterns="20231101.vi/*")
files = f"{local_dir}/20231101.vi/*.parquet"
ds = load_dataset("parquet", data_files=files, split="train", streaming=True)

builder = load_dataset_builder("wikimedia/wikipedia", "20231101.vi")
total = builder.info.splits["train"].num_examples
print("Total articles:", total)

# KEEP CANDS

CANDIDATES = 85000
k = total // CANDIDATES
print("k = ", k)

MIN_CHARS = 500
LIMIT_RUNS = total

kept = []
for i, article in enumerate(ds):
    if i >= LIMIT_RUNS:
        break
    if i % k != 0:
        continue
    text = article["text"].strip()
    if len(text) >= MIN_CHARS:
        kept.append({"id": article["id"], "title": article["title"], "text": text})

print("Scanned: ", LIMIT_RUNS, "articles")
print("num_kept_articles:", len(kept))
print("first_kept_article:", kept[0]["title"])

# SELECT TARGET FROM KEPT

TARGET = 10000
if len(kept) < TARGET:
    raise SystemExit(f"There's only {len(kept)} articles, less than TARGET = {TARGET}")

idx = np.linspace(0, len(kept)-1, TARGET).round().astype(int)
print("Num of diff idx: ", len(set(idx)))

selected = [kept[j] for j in idx]
print("Selected: ", len(selected), "articles")

# TO JSONL

ROOT = Path(__file__).resolve().parents[2]
print(ROOT)
OUT = ROOT / "data" / "raw" / "raw_viwiki.jsonl"
OUT.parent.mkdir(parents=True, exist_ok=True)

with open(OUT, "w", encoding= "utf-8") as f:
    for a in selected:
        f.write(json.dumps(a, ensure_ascii=False) + "\n")

print("Saved: ", len(selected), "articles ->", OUT)