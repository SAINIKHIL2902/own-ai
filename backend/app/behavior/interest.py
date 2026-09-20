import re
from typing import List, Set

# Curated deterministic topic ontology for local AI modeling
TOPIC_KEYWORDS = {
    # Programming Languages
    "python": "Python",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "rust": "Rust",
    "golang": "Go",
    "java": "Java",
    "c++": "C++",
    "sql": "SQL",
    "bash": "Bash",
    # Frameworks & Platforms
    "fastapi": "FastAPI",
    "flask": "Flask",
    "django": "Django",
    "react": "React",
    "next.js": "Next.js",
    "kafka": "Kafka",
    "docker": "Docker",
    "kubernetes": "Kubernetes",
    "git": "Git",
    "github": "GitHub",
    "linux": "Linux",
    "ollama": "Ollama",
    "pytorch": "PyTorch",
    "sqlite": "SQLite",
    # Domains & Concepts
    "ai": "Artificial Intelligence",
    "llm": "LLMs",
    "llms": "LLMs",
    "nlp": "NLP",
    "mlops": "MLOps",
    "machine learning": "Machine Learning",
    "deep learning": "Deep Learning",
    "data science": "Data Science",
    "data engineering": "Data Engineering",
    "etl": "Data Engineering",
    "concurrency": "Concurrency",
    "microservices": "Microservices",
    "api design": "API Design",
    "api": "API Design",
    "rest api": "API Design",
    "algorithms": "Algorithms",
    "database": "Databases",
    "databases": "Databases",
    "compiler": "Compilers",
    "testing": "Testing",
    "pytest": "Testing",
    "devops": "DevOps",
    "ci/cd": "CI/CD",
    "cloud": "Cloud Computing",
    "security": "Cybersecurity",
    "frontend": "Frontend Development",
    "backend": "Backend Development",
    "html": "HTML/CSS",
    "css": "HTML/CSS",
    "performance": "Performance Optimization",
    "architecture": "System Architecture",
}

INTENT_PATTERNS = [
    re.compile(r"(?:learn|explain|understand|how does|what is|tell me about)\s+([a-zA-Z0-9_\-\.\+]+)", re.IGNORECASE),
    re.compile(r"(?:code|script|program|build|write a)\s+(?:in\s+)?([a-zA-Z0-9_\-\.\+]+)", re.IGNORECASE),
]


def extract_topics(prompt: str, response: str = "") -> List[str]:
    """Deterministically extracts matched topic tags from user prompt and assistant response."""
    found: Set[str] = set()
    combined_text = f"{prompt} {response}".lower()

    # 1. Exact ontology matches
    for keyword, canonical_name in TOPIC_KEYWORDS.items():
        pattern = r"\b" + re.escape(keyword) + r"\b"
        if re.search(pattern, combined_text):
            found.add(canonical_name)

    # 2. Heuristic regex matches for common target terms
    for pattern in INTENT_PATTERNS:
        match = pattern.search(prompt)
        if match:
            raw_target = match.group(1).strip().lower()
            if raw_target in TOPIC_KEYWORDS:
                found.add(TOPIC_KEYWORDS[raw_target])

    # Default fallback if no specific domain matched
    if not found:
        # Check for general question
        if prompt.strip().endswith("?"):
            found.add("General Q&A")
        else:
            found.add("General Assistant")

    return sorted(list(found))
