#!/usr/bin/env python3
"""
Create a small sample corpus from simple text for quick tokenizer training.
"""

import os
import urllib.request

print("📥 Downloading sample text corpus...")
os.makedirs("data", exist_ok=True)

# Download English text samples
print("1/2 Downloading English samples...")
en_urls = [
    "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",  # Pride and Prejudice
    "https://www.gutenberg.org/cache/epub/11/pg11.txt",      # Alice in Wonderland
]

en_text = []
for url in en_urls:
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            text = response.read().decode('utf-8')
            en_text.append(text)
        print(f"   ✓ Downloaded {url.split('/')[-1]}")
    except Exception as e:
        print(f"   ⚠ Skipped {url}: {e}")

# Download Spanish text samples  
print("2/2 Downloading Spanish samples...")
es_urls = [
    "https://www.gutenberg.org/cache/epub/2000/pg2000.txt",  # Don Quijote
]

es_text = []
for url in es_urls:
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            text = response.read().decode('utf-8', errors='ignore')
            es_text.append(text)
        print(f"   ✓ Downloaded {url.split('/')[-1]}")
    except Exception as e:
        print(f"   ⚠ Skipped {url}: {e}")

# Combine and save
print("\n📝 Creating bilingual corpus...")
with open("data/corpus_bilingual.txt", "w", encoding="utf-8") as f:
    for text in en_text + es_text:
        # Clean up Project Gutenberg headers/footers
        lines = text.split('\n')
        cleaned = [line for line in lines if line.strip()]
        f.write('\n'.join(cleaned) + '\n')

file_size = os.path.getsize("data/corpus_bilingual.txt") / (1024 * 1024)
print(f"   ✓ Created corpus: {file_size:.1f} MB")
print("\n✅ Corpus ready: data/corpus_bilingual.txt")
