from src.services.generate import generate_with_safety
from src.lib.runtime import check_resource_budgets, GenerationMetrics


def test_generate_api_contract():
    """Test the /generate endpoint contract."""
    result = generate_with_safety(prompt="test", max_tokens=32)
    
    # Verify response structure
    assert "text" in result
    assert "metrics" in result
    assert "safety" in result
    
    # Verify metrics structure
    metrics = result["metrics"]
    assert "latency_p95_ms" in metrics
    assert "tokens_per_sec" in metrics
    assert "rss_mb" in metrics
    assert "peak_mb" in metrics
    
    # Verify safety structure
    safety = result["safety"]
    assert "status" in safety
    assert "category" in safety
    assert "rationale" in safety


def test_budget_compliance():
    """Verify constitution budgets are checked."""
    # Create test metrics
    good_metrics = GenerationMetrics(
        latency_p95_ms=200,
        tokens_per_sec=10.0,
        rss_mb=350,
        peak_mb=450,
    )
    result = check_resource_budgets(good_metrics)
    assert result["passed"] is True
    assert len(result["violations"]) == 0
    
    # Test violation
    bad_metrics = GenerationMetrics(
        latency_p95_ms=300,
        tokens_per_sec=10.0,
        rss_mb=450,
        peak_mb=550,
    )
    result = check_resource_budgets(bad_metrics)
    assert result["passed"] is False
    assert len(result["violations"]) > 0
