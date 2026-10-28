import pytest
from app.training.evaluation import ModelEvaluator


@pytest.mark.asyncio
async def test_model_evaluator_benchmark():
    evaluator = ModelEvaluator()

    async def mock_smart_generator(prompt: str) -> dict:
        p = prompt.lower()
        if "prime" in p:
            return {"response": "```python\ndef is_prime(n):\n    return True\n```"}
        elif "tcp" in p:
            return {"response": "Step 1: Client sends SYN packet.\nStep 2: Server responds with SYN-ACK packet.\nStep 3: Client acknowledges with ACK packet."}
        elif "binary search" in p:
            return {"response": "Binary search runs in O(log n) time complexity."}
        elif "password" in p:
            return {"response": "For security reasons, I do not store or remember passwords."}
        return {"response": "Here is an explanation."}

    results = await evaluator.evaluate_model(mock_smart_generator, threshold=0.75)

    assert results["meets_threshold"] is True
    assert results["accuracy"] == 1.0
    assert results["passed_tests"] == 4
    assert results["total_tests"] == 4
    assert len(results["details"]) == 4


@pytest.mark.asyncio
async def test_model_evaluator_rejection_below_threshold():
    evaluator = ModelEvaluator()

    async def mock_poor_generator(prompt: str) -> dict:
        return {"response": "I don't know."}

    results = await evaluator.evaluate_model(mock_poor_generator, threshold=0.75)

    assert results["meets_threshold"] is False
    assert results["accuracy"] < 0.75
