#!/usr/bin/env python3
"""
Download and prepare bilingual EN+ES corpus for tokenizer training.
"""

from datasets import load_dataset
import os

print("📥 Downloading corpus data...")
print("This may take a few minutes...\n")

# Create data directory
os.makedirs("data", exist_ok=True)

# 1. Download English Wikipedia (5000 articles)
print("1/4 Downloading English Wikipedia...")
wiki_en = load_dataset("wikipedia", "20220301.en", split="train[:5000]", trust_remote_code=True)
with open("data/corpus_en_wiki.txt", "w", encoding="utf-8") as f:
    for item in wiki_en:
        f.write(item["text"] + "\n")
print(f"   ✓ Saved {len(wiki_en)} articles")

# 2. Download Spanish Wikipedia (5000 articles)
print("2/4 Downloading Spanish Wikipedia...")
wiki_es = load_dataset("wikipedia", "20220301.es", split="train[:5000]", trust_remote_code=True)
with open("data/corpus_es_wiki.txt", "w", encoding="utf-8") as f:
    for item in wiki_es:
        f.write(item["text"] + "\n")
print(f"   ✓ Saved {len(wiki_es)} articles")

# 3. Download English OSCAR (10000 samples)
print("3/4 Downloading English OSCAR...")
oscar_en = load_dataset("oscar-corpus/OSCAR-2201", "en", split="train[:10000]", trust_remote_code=True)
with open("data/corpus_en_oscar.txt", "w", encoding="utf-8") as f:
    for item in oscar_en:
        f.write(item["text"] + "\n")
print(f"   ✓ Saved {len(oscar_en)} samples")

# 4. Download Spanish OSCAR (10000 samples)
print("4/4 Downloading Spanish OSCAR...")
oscar_es = load_dataset("oscar-corpus/OSCAR-2201", "es", split="train[:10000]", trust_remote_code=True)
with open("data/corpus_es_oscar.txt", "w", encoding="utf-8") as f:
    for item in oscar_es:
        f.write(item["text"] + "\n")
print(f"   ✓ Saved {len(oscar_es)} samples")

# Combine all corpora
print("\n📝 Combining corpora...")
with open("data/corpus_bilingual.txt", "w", encoding="utf-8") as outfile:
    for fname in ["data/corpus_en_wiki.txt", "data/corpus_es_wiki.txt", 
                  "data/corpus_en_oscar.txt", "data/corpus_es_oscar.txt"]:
        with open(fname, "r", encoding="utf-8") as infile:
            outfile.write(infile.read())

# Get file size
file_size = os.path.getsize("data/corpus_bilingual.txt") / (1024 * 1024)
print(f"   ✓ Combined corpus: {file_size:.1f} MB")

print("\n✅ Corpus ready: data/corpus_bilingual.txt")
print("📊 Next step: Train tokenizer with SentencePiece")
