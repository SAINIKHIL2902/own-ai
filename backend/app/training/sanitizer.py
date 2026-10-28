import logging
from typing import List, Tuple

from app.privacy.patterns import PATTERNS

logger = logging.getLogger(__name__)

# Categories that strictly disqualify an example from entering the training dataset
HARD_EXCLUDE_CATEGORIES = {
    "private_key",
    "database_url",
    "api_key",
    "bearer_token",
    "password_or_token",
}


class TrainingSanitizer:
    """
    Enforces privacy, secret-redaction, and data-minimization rules on training data.
    Ensures zero credentials, private keys, API tokens, or PII leak into model weights.
    """

    def __init__(self):
        self._patterns = PATTERNS

    def check_sensitivity(self, text: str) -> Tuple[bool, List[str]]:
        """
        Check if text contains sensitive data categories.
        Returns (is_sensitive, detected_categories).
        """
        if not text:
            return False, []

        detected = []
        for category, pattern, _ in self._patterns:
            if pattern.search(text):
                if category not in detected:
                    detected.append(category)

        return len(detected) > 0, detected

    def is_eligible_for_dataset(self, user_content: str, assistant_content: str) -> Tuple[bool, str]:
        """
        Evaluate if a user/assistant pair is safe to include in the training set.
        If credentials or secrets are present, the example is strictly excluded.
        """
        # 1. Check user content
        has_user_sensitive, user_cats = self.check_sensitivity(user_content)
        for cat in user_cats:
            if cat in HARD_EXCLUDE_CATEGORIES:
                return False, f"EXCLUDED_CREDENTIAL_IN_PROMPT ({cat})"

        # 2. Check assistant content
        has_asst_sensitive, asst_cats = self.check_sensitivity(assistant_content)
        for cat in asst_cats:
            if cat in HARD_EXCLUDE_CATEGORIES:
                return False, f"EXCLUDED_CREDENTIAL_IN_RESPONSE ({cat})"

        # 3. Minimum quality checks
        if len(user_content.strip()) < 5:
            return False, "EXCLUDED_PROMPT_TOO_SHORT"
        if len(assistant_content.strip()) < 10:
            return False, "EXCLUDED_RESPONSE_TOO_SHORT"

        return True, "ELIGIBLE"

    def redact_pii(self, text: str) -> str:
        """Sanitizes PII such as emails and phone numbers with placeholders."""
        sanitized = text
        for category, pattern, replacement in self._patterns:
            if category in ["email", "phone"]:
                sanitized = pattern.sub(replacement, sanitized)
        return sanitized

    def sanitize_text(self, text: str) -> str:
        """Redacts all credentials, secrets, API keys, tokens, and PII from text."""
        if not text:
            return ""
        sanitized = text
        for _, pattern, replacement in self._patterns:
            sanitized = pattern.sub(replacement, sanitized)
        return sanitized


training_sanitizer = TrainingSanitizer()
