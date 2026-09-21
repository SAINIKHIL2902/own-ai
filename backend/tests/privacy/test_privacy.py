import pytest
from app.privacy.patterns import PATTERNS
from app.privacy.scanner import PrivacyScanner, privacy_scanner


def test_redact_api_keys():
    scanner = PrivacyScanner()
    text = "Here is my OpenAI key sk-1234567890abcdef1234567890abcdef and github ghp_123456789012345678901234567890123456."
    res = scanner.scan(text)
    assert res.contains_sensitive_data is True
    assert "api_key" in res.detected_types
    assert "sk-1234567890abcdef" not in res.sanitized_prompt
    assert "ghp_1234567890" not in res.sanitized_prompt
    assert "********" in res.sanitized_prompt


def test_redact_bearer_token():
    scanner = PrivacyScanner()
    text = "Use Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.e30.t-ID for auth."
    res = scanner.scan(text)
    assert res.contains_sensitive_data is True
    assert "bearer_token" in res.detected_types
    assert "eyJhbGci" not in res.sanitized_prompt
    assert "Bearer ********" in res.sanitized_prompt


def test_redact_database_url():
    scanner = PrivacyScanner()
    text = "Connect to postgresql://admin:supersecret123@db.internal:5432/production"
    res = scanner.scan(text)
    assert res.contains_sensitive_data is True
    assert "database_url" in res.detected_types
    assert "supersecret123" not in res.sanitized_prompt
    assert "[CREDENTIAL_REDACTED]" in res.sanitized_prompt


def test_redact_passwords_and_credentials():
    scanner = PrivacyScanner()
    text = "My credentials are password: 'MySuperPassword99!' and api_key: 'my_secret_token_12345'"
    res = scanner.scan(text)
    assert res.contains_sensitive_data is True
    assert "password_or_token" in res.detected_types
    assert "MySuperPassword99!" not in res.sanitized_prompt
    assert "my_secret_token_12345" not in res.sanitized_prompt


def test_redact_emails_and_phones():
    scanner = PrivacyScanner()
    text = "Contact me at alice.smith@example.com or call +1-555-839-2049."
    res = scanner.scan(text)
    assert res.contains_sensitive_data is True
    assert "email" in res.detected_types
    assert "phone" in res.detected_types
    assert "alice.smith@example.com" not in res.sanitized_prompt
    assert "+1-555-839-2049" not in res.sanitized_prompt
    assert "[EMAIL_REDACTED]" in res.sanitized_prompt
    assert "[PHONE_REDACTED]" in res.sanitized_prompt


def test_technical_prompts_not_falsely_redacted():
    scanner = PrivacyScanner()
    text = "Explain how Kafka authentication works with OAuth and JWT tokens."
    res = scanner.scan(text)
    assert res.contains_sensitive_data is False
    assert res.sanitized_prompt == text
    assert "OAuth" in res.sanitized_prompt
    assert "authentication" in res.sanitized_prompt


def test_sanitize_messages_preserves_roles_and_sanitizes_contents():
    scanner = PrivacyScanner()
    raw_msgs = [
        {"role": "user", "content": "My secret key is sk-abcdef123456789012345678. How do I use it?"},
        {"role": "assistant", "content": "You should never share your API keys."},
        {"role": "user", "content": "My email is bob@corp.com. Send it there."},
    ]
    sanitized, summary = scanner.sanitize_messages(raw_msgs)
    assert summary.contains_sensitive_data is True
    assert "api_key" in summary.detected_types
    assert "email" in summary.detected_types

    # Ensure raw secret is NOT anywhere in the sanitized list
    assert "sk-abcdef" not in sanitized[0]["content"]
    assert "bob@corp.com" not in sanitized[2]["content"]
    assert "********" in sanitized[0]["content"]
    assert "[EMAIL_REDACTED]" in sanitized[2]["content"]
