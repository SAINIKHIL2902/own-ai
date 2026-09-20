from app.behavior.preferences import calculate_confidence, extract_preferences


def test_extract_code_preference():
    prefs = extract_preferences("Can you write a python script to parse CSV files?")
    assert prefs.get("prefers_code") == "true"


def test_extract_step_by_step_preference():
    prefs = extract_preferences("Give me a step-by-step walkthrough of database indexing.")
    assert prefs.get("prefers_step_by_step") == "true"


def test_extract_concise_preference():
    prefs = extract_preferences("Explain quantum computing in one sentence briefly.")
    assert prefs.get("prefers_concise_answers") == "true"


def test_extract_correction_preference():
    prefs = extract_preferences("No, explain this using an Amazon production case study instead.")
    assert prefs.get("prefers_real_world_examples") == "true"


def test_confidence_scaling():
    assert calculate_confidence(1) == 0.2
    assert calculate_confidence(3) == 0.6
    assert calculate_confidence(5) == 1.0
    assert calculate_confidence(10) == 1.0  # Maxes out at 1.0
