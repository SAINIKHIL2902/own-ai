from typing import Dict, List

from app.memory.models import MemoryRecord


class ContextBuilder:
    """
    Builds personalized model context by injecting relevant memories.
    Enforces the mandatory rule: Current user instructions ALWAYS override
    remembered background preferences if there is any conflict.
    """

    SYSTEM_DIRECTIVE = (
        "IMPORTANT: The user's explicit instructions in the current conversation "
        "always supersede these background memories if any conflict arises."
    )

    def format_memory_block(self, memories: List[MemoryRecord]) -> str:
        """Format a list of memories into structured context text."""
        if not memories:
            return ""

        lines = ["[User Personal Context & Background Memory]:"]
        for mem in memories:
            type_label = mem.memory_type.capitalize()
            lines.append(f"- [{type_label}]: {mem.content}")

        lines.append(f"\n{self.SYSTEM_DIRECTIVE}")
        return "\n".join(lines)

    def personalize_messages(
        self,
        messages: List[Dict[str, str]],
        memories: List[MemoryRecord],
    ) -> List[Dict[str, str]]:
        """
        Inject formatted memory context into message list.
        If a system message already exists, appends to it.
        Otherwise prepends a new system message with the memory context.
        """
        if not memories:
            return list(messages)

        memory_text = self.format_memory_block(memories)
        new_messages = []
        system_found = False

        for msg in messages:
            if msg.get("role") == "system" and not system_found:
                existing_content = msg.get("content", "")
                merged_content = f"{existing_content}\n\n{memory_text}".strip()
                new_messages.append({"role": "system", "content": merged_content})
                system_found = True
            else:
                new_messages.append(dict(msg))

        if not system_found:
            new_messages.insert(0, {"role": "system", "content": memory_text})

        return new_messages


context_builder = ContextBuilder()
