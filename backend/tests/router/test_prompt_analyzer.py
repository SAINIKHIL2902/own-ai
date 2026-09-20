from app.router.prompt_analyzer import PromptAnalyzer


def test_analyze_simple_question():
    analyzer = PromptAnalyzer()
    res = analyzer.analyze("What is Python?")
    assert res.task_type in ["explanation", "question_answering"]
    assert res.difficulty == "easy"
    assert res.complexity_score < 0.35


def test_analyze_coding_task():
    analyzer = PromptAnalyzer()
    res = analyzer.analyze("Write a Python function to reverse a linked list.")
    assert res.task_type == "coding"
    assert "coding" in res.required_capabilities
    assert res.required_capabilities["coding"] >= 0.85
    assert res.difficulty in ["easy", "medium"]


def test_analyze_complex_reasoning_and_hard_difficulty():
    analyzer = PromptAnalyzer()
    res = analyzer.analyze(
        "Analyze the consistency trade-offs between Multi-Raft active-active replication "
        "and Paxos in a distributed system with high availability and network partitions."
    )
    assert res.task_type in ["reasoning", "analysis"]
    assert res.difficulty == "hard"
    assert res.reasoning_requirement >= 0.50
    assert "complex_reasoning" in res.required_capabilities


def test_analyze_structured_output():
    analyzer = PromptAnalyzer()
    res = analyzer.analyze("Return a JSON format list of users with id and name.")
    assert res.structured_output_required is True
    assert "structured_output" in res.required_capabilities


def test_analyze_long_context():
    analyzer = PromptAnalyzer()
    long_prompt = "Document snippet: " + ("lorem ipsum text " * 400) + "\nSummarize key points."
    res = analyzer.analyze(long_prompt)
    assert res.context_requirement >= 0.80
    assert res.difficulty == "hard"
