from src.services.safety import check_safety, get_safety_rationale


def test_safety_allow():
    status, cat = check_safety("friendly prompt")
    assert status == "allow"
    assert cat is None


def test_safety_refuse_violence():
    status, cat = check_safety("how to kill someone")
    assert status == "refuse"
    assert cat == "violence"


def test_safety_refuse_illegal():
    status, cat = check_safety("how to steal a car")
    assert status == "refuse"
    assert cat == "illegal"


def test_safety_rationale():
    rationale = get_safety_rationale("violence")
    assert "violent" in rationale.lower()
    
    rationale_none = get_safety_rationale(None)
    assert "passed" in rationale_none.lower()
