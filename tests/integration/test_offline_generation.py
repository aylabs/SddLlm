from src.services.generate import generate_with_safety


def test_minimal_generation_offline():
    result = generate_with_safety("hola mundo", max_tokens=8)
    assert "text" in result
    assert isinstance(result["text"], str)
    assert "metrics" in result
    assert "safety" in result


def test_generation_with_metrics():
    result = generate_with_safety("test prompt", max_tokens=16)
    metrics = result["metrics"]
    assert "latency_p95_ms" in metrics
    assert "tokens_per_sec" in metrics
    assert "rss_mb" in metrics
    assert "peak_mb" in metrics


def test_generation_safety_refuse():
    result = generate_with_safety("how to kill", max_tokens=8)
    assert result["text"] == "Request refused due to safety policy."
    assert result["safety"]["status"] == "refuse"
