#!/usr/bin/env python3
"""
Train SentencePiece tokenizer on bilingual corpus.
"""

import sentencepiece as spm
import os
import argparse
from pathlib import Path

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train SentencePiece tokenizer on bilingual corpus')
    parser.add_argument('--vocab-size', type=int, default=8000, help='Vocabulary size')
    parser.add_argument('--output-dir', type=str, default='data', help='Output directory for tokenizer model')
    parser.add_argument('--input-file', type=str, default='data/corpus_bilingual.txt', help='Input corpus file')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set model prefix path
    model_prefix = output_dir / f'bilingual_{args.vocab_size}'
    
    print("🔧 Training SentencePiece tokenizer...")
    print("Configuration:")
    print(f"  - Vocabulary size: {args.vocab_size}")
    print(f"  - Model type: unigram")
    print(f"  - Character coverage: 0.9995")
    print(f"  - Input: {args.input_file}")
    print(f"  - Output directory: {output_dir}\n")
    
    # Train tokenizer
    spm.SentencePieceTrainer.train(
        input=args.input_file,
        model_prefix=str(model_prefix),
        vocab_size=args.vocab_size,
        character_coverage=0.9995,
        model_type='unigram',
        pad_id=2,
        bos_id=0,
        eos_id=1,
        unk_id=3,
        input_sentence_size=1000000,
        shuffle_input_sentence=True,
    )
    
    model_file = f'{model_prefix}.model'
    vocab_file = f'{model_prefix}.vocab'
    
    print("✅ Tokenizer training complete!")
    print("\nGenerated files:")
    print(f"  - {model_file} ({os.path.getsize(model_file) / 1024:.1f} KB)")
    print(f"  - {vocab_file} ({os.path.getsize(vocab_file) / 1024:.1f} KB)")
    
    # Test the tokenizer
    sp = spm.SentencePieceProcessor(model_file=model_file)
    
    print("\n🧪 Testing tokenizer:")
    test_sentences = [
        "Hello world! How are you?",
        "Hola mundo! ¿Cómo estás?",
    ]
    
    for sentence in test_sentences:
        tokens = sp.encode(sentence, out_type=str)
        ids = sp.encode(sentence, out_type=int)
        print(f"\n  Text: {sentence}")
        print(f"  Tokens: {tokens[:10]}..." if len(tokens) > 10 else f"  Tokens: {tokens}")
        print(f"  IDs: {ids[:10]}..." if len(ids) > 10 else f"  IDs: {ids}")


if __name__ == '__main__':
    main()
    print(f"  Count: {len(tokens)} tokens")

print("\n📊 Vocabulary stats:")
print(f"  Total vocab size: {sp.vocab_size()}")
print(f"  Special tokens:")
print(f"    <BOS>: {sp.bos_id()}")
print(f"    <EOS>: {sp.eos_id()}")
print(f"    <PAD>: {sp.pad_id()}")
print(f"    <UNK>: {sp.unk_id()}")
