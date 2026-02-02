import argparse
import json
from pathlib import Path
from src.services.generate import generate_with_safety
from src.lib.runtime import load_bundle_metadata, verify_bundle_integrity


def show_bundle_info(metadata_path: Path):
    """Display model bundle information."""
    bundle = load_bundle_metadata(metadata_path)
    if bundle is None:
        print("Error: Bundle metadata not found")
        return
    
    integrity_ok = verify_bundle_integrity(bundle)
    
    info = {
        "id": bundle.id,
        "version": bundle.version,
        "size_mb": round(bundle.size_bytes / (1024 * 1024), 2),
        "hash": bundle.hash_sha256[:16] + "...",
        "quantization": bundle.quantization_type,
        "tokenizer_version": bundle.tokenizer_version,
        "integrity_verified": integrity_ok,
    }
    print(json.dumps(info, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Minimal on-device LLM")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate text")
    gen_parser.add_argument("--prompt", type=str, required=True, help="Text prompt for generation")
    gen_parser.add_argument("--max_tokens", type=int, default=64, help="Max tokens to generate")
    gen_parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    gen_parser.add_argument("--json", action="store_true", help="Output JSON format with metrics")
    gen_parser.add_argument("--safety-mode", choices=["strict", "normal"], default="normal", help="Safety filtering mode")
    
    # Bundle info command
    info_parser = subparsers.add_parser("bundle-info", help="Show bundle information")
    info_parser.add_argument("--metadata", type=str, default="bundle_metadata.json", help="Path to metadata file")
    
    args = parser.parse_args()
    
    if args.command == "generate":
        result = generate_with_safety(args.prompt, max_tokens=args.max_tokens, temperature=args.temperature)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(result["text"])
            if result["metrics"]:
                print(f"\nMetrics: {result['metrics']}")
            print(f"Safety: {result['safety']['status']} - {result['safety']['rationale']}")
    elif args.command == "bundle-info":
        show_bundle_info(Path(args.metadata))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
