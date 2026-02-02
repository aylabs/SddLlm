from dataclasses import dataclass
import psutil
import os
import hashlib
import json
from pathlib import Path
from typing import Optional


@dataclass
class GenerationMetrics:
    latency_p95_ms: int
    tokens_per_sec: float
    rss_mb: int
    peak_mb: int


@dataclass
class ModelBundle:
    id: str
    version: str
    size_bytes: int
    hash_sha256: str
    path: str
    quantization_type: str = "int8"
    tokenizer_version: str = "0.1.0"
    integrity_verified: bool = False


def current_rss_mb() -> int:
    """Get current RSS memory usage in MB."""
    process = psutil.Process(os.getpid())
    return int(process.memory_info().rss / (1024 * 1024))


def check_resource_budgets(metrics: GenerationMetrics) -> dict:
    """Verify resource budgets per constitution: RSS ≤ 400MB, Peak ≤ 512MB, p95 ≤ 250ms."""
    violations = []
    if metrics.rss_mb > 400:
        violations.append(f"RSS {metrics.rss_mb}MB exceeds 400MB budget")
    if metrics.peak_mb > 512:
        violations.append(f"Peak {metrics.peak_mb}MB exceeds 512MB budget")
    if metrics.latency_p95_ms > 250:
        violations.append(f"p95 latency {metrics.latency_p95_ms}ms exceeds 250ms budget")
    
    return {
        "passed": len(violations) == 0,
        "violations": violations,
        "budgets": {"rss_mb_max": 400, "peak_mb_max": 512, "p95_ms_max": 250},
    }


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def verify_bundle_integrity(bundle: ModelBundle) -> bool:
    """Verify SHA-256 integrity of model bundle."""
    if not Path(bundle.path).exists():
        return False
    computed_hash = compute_file_hash(Path(bundle.path))
    return computed_hash == bundle.hash_sha256


def check_semantic_version_compatible(current: str, required: str) -> tuple[bool, str]:
    """Check semantic version compatibility (MAJOR.MINOR.PATCH)."""
    def parse_version(v: str) -> tuple[int, int, int]:
        parts = v.split(".")
        return (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0, int(parts[2]) if len(parts) > 2 else 0)
    
    curr_major, curr_minor, _ = parse_version(current)
    req_major, req_minor, _ = parse_version(required)
    
    if curr_major != req_major:
        return False, f"Major version mismatch: {current} vs {required} (breaking change)"
    if curr_minor < req_minor:
        return False, f"Minor version too old: {current} < {required}"
    return True, "Compatible"


def load_bundle_metadata(metadata_path: Path) -> Optional[ModelBundle]:
    """Load model bundle metadata from JSON file."""
    if not metadata_path.exists():
        return None
    with open(metadata_path, "r") as f:
        data = json.load(f)
    return ModelBundle(**data)


def cold_start_initialize(bundle_path: Path, metadata_path: Path) -> dict:
    """Initialize model bundle with integrity and version checks."""
    rss_start = current_rss_mb()
    
    # Load metadata
    bundle = load_bundle_metadata(metadata_path)
    if bundle is None:
        return {"status": "error", "message": "Metadata not found"}
    
    # Verify integrity
    if not verify_bundle_integrity(bundle):
        return {"status": "error", "message": "Integrity check failed"}
    
    bundle.integrity_verified = True
    
    # Check version compatibility
    compatible, msg = check_semantic_version_compatible(bundle.version, "0.1.0")
    if not compatible:
        return {"status": "error", "message": msg}
    
    rss_end = current_rss_mb()
    return {
        "status": "success",
        "bundle": bundle,
        "init_rss_mb": rss_end - rss_start,
        "message": "Bundle initialized successfully",
    }
