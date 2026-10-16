from typing import List, Optional, Tuple

from app.config.settings import settings
from app.memory.lifecycle import LifecycleManager, lifecycle_manager
from app.memory.models import MemoryRecord
from app.memory.repository import MemoryRepository, memory_repo
from app.memory.scorer import MemoryScorer, memory_scorer


class MemoryRetriever:
    """
    Retrieves active, unexpired memories relevant to a given user prompt,
    ranked by relevance score.
    """

    def __init__(
        self,
        repository: Optional[MemoryRepository] = None,
        scorer: Optional[MemoryScorer] = None,
        lifecycle: Optional[LifecycleManager] = None,
    ):
        self.repo = repository or memory_repo
        self.scorer = scorer or memory_scorer
        self.lifecycle = lifecycle or lifecycle_manager

    def retrieve_relevant(
        self,
        prompt: str,
        user_id: str = "local_user",
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List[MemoryRecord]:
        """
        Fetch top relevant memories for a prompt above the relevance threshold.
        """
        limit = top_k if top_k is not None else settings.MEMORY_TOP_K
        min_score = threshold if threshold is not None else settings.MEMORY_RELEVANCE_THRESHOLD

        active_memories = self.repo.get_active_memories(user_id=user_id)
        scored: List[Tuple[MemoryRecord, float]] = []

        for mem in active_memories:
            if not self.lifecycle.is_retrievable(mem):
                continue

            score = self.scorer.score(prompt, mem)
            if score >= min_score:
                scored.append((mem, score))

        # Sort by relevance score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        return [mem for mem, _ in scored[:limit]]


memory_retriever = MemoryRetriever()
