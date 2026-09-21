import re
from typing import Dict, List, Pattern, Tuple

# Regular expressions for sensitive data patterns
# Format: (category_name, regex_pattern, replacement_token)

PATTERNS: List[Tuple[str, Pattern, str]] = [
    # 1. Private Keys
    (
        "private_key",
        re.compile(
            r"-----BEGIN[ A-Z0-9_-]*PRIVATE KEY-----[\s\S]*?-----END[ A-Z0-9_-]*PRIVATE KEY-----",
            re.IGNORECASE,
        ),
        "[CREDENTIAL_REDACTED]",
    ),
    # 2. Database Connection Strings
    (
        "database_url",
        re.compile(
            r"\b(postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|mssql):\/\/[^\s\'\"]+:[^\s\'\"]+@[^\s\'\"]+",
            re.IGNORECASE,
        ),
        "[CREDENTIAL_REDACTED]",
    ),
    # 3. Known Provider API Keys and Tokens
    (
        "api_key",
        re.compile(
            r"\b(sk-[a-zA-Z0-9_-]{20,}|ghp_[a-zA-Z0-9]{36}|gho_[a-zA-Z0-9]{36}|AKIA[0-9A-Z]{16}|xox[baprs]-[0-9a-zA-Z]{10,}|AIza[0-9A-Za-z-_]{35})\b"
        ),
        "********",
    ),
    # 4. Bearer / Authorization Tokens
    (
        "bearer_token",
        re.compile(
            r"(?i)\b(bearer\s+)([A-Za-z0-9\-._~+/]+=*)\b"
        ),
        r"\1********",
    ),
    # 5. Explicit Key/Value Credential Pairs (e.g. api_key = "abc12345", password: "xyz")
    (
        "password_or_token",
        re.compile(
            r"(?i)\b(password|passwd|pwd|api[_-]?key|secret[_-]?key|auth[_-]?token|access[_-]?token)\s*[:=]\s*['\"]?([^\s'\",;]{6,})['\"]?"
        ),
        r"\1: [CREDENTIAL_REDACTED]",
    ),
    # 6. Email Addresses
    (
        "email",
        re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,7}\b"
        ),
        "[EMAIL_REDACTED]",
    ),
    # 7. Phone Numbers (7-15 digits with optional country code, delimiters, and parentheses)
    (
        "phone",
        re.compile(
            r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b"
        ),
        "[PHONE_REDACTED]",
    ),
]
