import pytest
from app.behavior.style import response_style_analyzer


def test_classify_step_by_step_code_response():
    prompt = "Show me how to build a FastAPI app with tests"
    response = (
        "Here is how to set up FastAPI step by step:\n\n"
        "### Step 1: Install FastAPI\n"
        "Run the following pip command:\n"
        "```bash\npip install fastapi uvicorn\n```\n\n"
        "### Step 2: Create main.py\n"
        "Write your app code:\n"
        "```python\nfrom fastapi import FastAPI\napp = FastAPI()\n```\n\n"
        "### Step 3: Run the server\n"
        "Start uvicorn.\n"
    )
    result = response_style_analyzer.analyze(prompt, response)
    assert result["primary_style"] == "step_by_step_code"
    assert result["has_code"] is True
    assert result["is_step_by_step"] is True
    assert "executable_code" in result["style_tags"]
    assert len(result["confusion_triggers"]) == 0


def test_classify_concise_bullets_response():
    prompt = "Summarize the benefits of SQLite"
    response = (
        "- Zero configuration and serverless\n"
        "- Single disk file database\n"
        "- High performance read operations with WAL mode\n"
        "- ACID compliant transactions\n"
    )
    result = response_style_analyzer.analyze(prompt, response)
    assert result["primary_style"] == "concise_bullets"
    assert result["is_concise"] is True
    assert "bulleted_list" in result["style_tags"]


def test_detect_ambiguity_and_confusion_triggers():
    prompt = "How does deep learning work?"
    response = (
        "Well, maybe it works through some things. Perhaps there are some layers, "
        "and sort of connections, and probably some mathematics, etc. "
        "It might be neural networks, and could be training weights."
    )
    result = response_style_analyzer.analyze(prompt, response)
    assert "ambiguous_or_vague" in result["confusion_triggers"]
    assert result["hedge_density"] > 1.5


def test_detect_missing_code_when_code_requested():
    prompt = "Write a python script to parse a json file"
    response = (
        "To parse a JSON file, one must understand that JSON represents javascript object notation. "
        "Data structures are mapped into keys and values."
    )
    result = response_style_analyzer.analyze(prompt, response)
    assert "missing_concrete_code" in result["confusion_triggers"]
