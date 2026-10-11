from app.router.router import ModelRouter


def test_router_easy_explanation_routes_local():
    router = ModelRouter()
    decision = router.route("Explain what a Python dictionary is with a simple example.")
    assert decision.selected_provider == "local"
    assert decision.local_suitability >= 0.70
    assert decision.routing_confidence >= 0.65
    assert "local model is sufficiently capable" in decision.reason.lower()


def test_router_simple_code_routes_local():
    router = ModelRouter()
    decision = router.route("Write a Python function to reverse a string.")
    assert decision.selected_provider == "local"
    assert decision.local_suitability >= 0.70


def test_router_complex_distributed_reasoning_routes_gemini():
    router = ModelRouter()
    decision = router.route(
        "Analyze the formal consistency proofs and failure modes between Multi-Raft active-active "
        "replication vs Paxos under Byzantine faults and network partitions."
    )
    assert decision.selected_provider == "gemini"
    assert "insufficient" in decision.reason.lower() or "hard capability" in decision.reason.lower()


def test_router_context_overflow_routes_gemini():
    router = ModelRouter()
    long_prompt = "Documentation: " + ("database cluster architecture " * 350)
    decision = router.route(long_prompt)
    assert decision.selected_provider == "gemini"
    assert decision.hard_check_passed is False


def test_router_hard_architecture_routes_gemini():
    router = ModelRouter()
    decision = router.route(
        "Design a fault-tolerant microservice architecture for 10 million concurrent users "
        "with disaster recovery across multi-region AWS and Paxos consensus."
    )
    assert decision.selected_provider == "gemini"
    assert decision.difficulty == "hard"


def test_router_hard_math_proof_routes_gemini():
    router = ModelRouter()
    decision = router.route(
        "Solve this partial differential equation and prove the theorem step by step with edge cases."
    )
    assert decision.selected_provider == "gemini"
    assert decision.difficulty == "hard"


def test_router_hard_concurrency_debug_routes_gemini():
    router = ModelRouter()
    decision = router.route(
        "Debug this complex memory leak in C++ with multi-threaded race conditions, "
        "atomic ABA problems, and dangling pointers."
    )
    assert decision.selected_provider == "gemini"
    assert decision.difficulty == "hard"
    assert decision.local_suitability < 0.70


def test_router_production_k8s_operator_routes_gemini():
    router = ModelRouter()
    decision = router.route(
        "Write a production-grade Kubernetes operator in Go with custom controllers, "
        "reconcile loops, and admission webhooks."
    )
    assert decision.selected_provider == "gemini"
    assert decision.difficulty == "hard"
