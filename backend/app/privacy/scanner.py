from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Tuple

from app.privacy.patterns import PATTERNS

logger = logging.getLogger(__name__)


@dataclass
class PrivacyScanResult:
    """Result of privacy scan and sanitization."""
    contains_sensitive_data: bool
    detected_types: List[str] = field(default_factory=list)
    sanitized_prompt: str = ""


class PrivacyScanner:
    """
    Scans and sanitizes user messages before transmission to external APIs (Gemini).

    Guarantees:
    - Any sensitive information matching supported detectors (API keys, passwords,
      tokens, private keys, connection strings, emails, phone numbers) is redacted.
    - Raw sensitive values are never logged, stored in routing metadata, or transmitted.
    - Ordinary technical discussions (e.g. "Explain OAuth authentication") remain intact.

    Disclaimer:
    Pattern-based detection covers known sensitive credential formats and PII,
    but cannot mathematically guarantee identification of every unstructured private fact.
    """

    def __init__(self):
        self._patterns = PATTERNS

    def scan(self, text: str) -> PrivacyScanResult:
        """Scan text and return sanitized text along with detected sensitivity categories."""
        if not text:
            return PrivacyScanResult(contains_sensitive_data=False, sanitized_prompt="")

        sanitized = text
        detected: List[str] = []

        for category, pattern, replacement in self._patterns:
            if pattern.search(sanitized):
                if category not in detected:
                    detected.append(category)
                sanitized = pattern.sub(replacement, sanitized)

        contains_sensitive = len(detected) > 0
        if contains_sensitive:
            logger.info(
                f"PrivacyScanner detected and redacted sensitive categories: {', '.join(detected)}"
            )

        return PrivacyScanResult(
            contains_sensitive_data=contains_sensitive,
            detected_types=detected,
            sanitized_prompt=sanitized,
        )

    def sanitize_messages(
        self, messages: List[Dict[str, str]]
    ) -> Tuple[List[Dict[str, str]], PrivacyScanResult]:
        """
        Sanitize an entire list of conversation messages.
        Returns a tuple of (sanitized_messages, aggregate_scan_result).
        """
        sanitized_messages: List[Dict[str, str]] = []
        all_detected: List[str] = []
        last_sanitized_prompt = ""

        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "user")
            res = self.scan(content)

            for d in res.detected_types:
                if d not in all_detected:
                    all_detected.append(d)

            sanitized_messages.append({"role": role, "content": res.sanitized_prompt})
            if role == "user":
                last_sanitized_prompt = res.sanitized_prompt

        aggregate_result = PrivacyScanResult(
            contains_sensitive_data=len(all_detected) > 0,
            detected_types=all_detected,
            sanitized_prompt=last_sanitized_prompt,
        )

        return sanitized_messages, aggregate_result


privacy_scanner = PrivacyScanner()
