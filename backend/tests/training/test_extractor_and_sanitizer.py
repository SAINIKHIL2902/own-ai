import json
import pytest
from app.storage.repositories import raw_repo
from app.training.extractor import ConsentAwareExtractor
from app.training.models import TrainingExample
from app.training.repository import TrainingRepository
from app.training.sanitizer import TrainingSanitizer


def test_sanitizer_masks_pii():
    sanitizer = TrainingSanitizer()
    raw_text = "My email is user@domain.com and my phone number is 415-555-2671."
    redacted = sanitizer.redact_pii(raw_text)

    assert "user@domain.com" not in redacted
    assert "[EMAIL_REDACTED]" in redacted
    assert "415-555-2671" not in redacted
    assert "[PHONE_REDACTED]" in redacted


def test_sanitizer_disqualifies_credentials():
    sanitizer = TrainingSanitizer()
    good_prompt = "Can you help me sort this list in Python?"
    good_resp = "Here is a clean implementation using sorted(my_list)."

    is_safe, reason = sanitizer.is_eligible_for_dataset(good_prompt, good_resp)
    assert is_safe is True
    assert reason == "ELIGIBLE"

    # API key in prompt
    bad_prompt = "My OpenAI key is sk-abcdef1234567890abcdef1234567890 how do I use it?"
    is_safe, reason = sanitizer.is_eligible_for_dataset(bad_prompt, good_resp)
    assert is_safe is False
    assert "EXCLUDED_CREDENTIAL" in reason

    # Private key in response
    bad_resp = "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA0...\n-----END RSA PRIVATE KEY-----"
    is_safe, reason = sanitizer.is_eligible_for_dataset(good_prompt, bad_resp)
    assert is_safe is False
    assert "EXCLUDED_CREDENTIAL" in reason


def test_extractor_filters_negative_feedback_and_sanitizes(tmp_path):
    repo = TrainingRepository()
    user_id = "test_extract_user"
    repo.set_consent(user_id, opted_in=True)

    # 1. Create a conversation with negative feedback
    conv_bad = "conv_bad_feedback_1"
    raw_repo.save_conversation(conv_bad, "Bad Feedback Conv")
    raw_repo.save_message("msg_user_bad", conv_bad, "user", "What is an apple?", "llama3.2:1b", 20.0, user_id=user_id)
    raw_repo.save_message("msg_asst_bad", conv_bad, "assistant", "An apple is an aircraft.", "llama3.2:1b", 50.0, user_id=user_id)
    raw_repo.save_feedback("fb_1", "msg_asst_bad", conv_bad, feedback_type="thumbs_down", rating=1, user_id=user_id)

    # 2. Create a conversation with positive feedback and PII
    conv_good = "conv_good_pii_1"
    raw_repo.save_conversation(conv_good, "Good Conv with PII")
    raw_repo.save_message("msg_user_good", conv_good, "user", "Please send the file to test.engineer@domain.com right away.", "llama3.2:1b", 20.0, user_id=user_id)
    raw_repo.save_message("msg_asst_good", conv_good, "assistant", "I will prepare the file for test.engineer@domain.com now.", "llama3.2:1b", 50.0, user_id=user_id)
    raw_repo.save_feedback("fb_2", "msg_asst_good", conv_good, feedback_type="thumbs_up", rating=5, user_id=user_id)

    extractor = ConsentAwareExtractor(raw_repository=raw_repo, training_repository=repo)
    report = extractor.extract_user_examples(user_id=user_id)

    assert report.consent_opted_in is True
    # The negative example must be excluded
    asst_texts = [ex.messages[1]["content"] for ex in report.examples]
    assert "An apple is an aircraft." not in asst_texts

    # The good example must be included with redacted PII
    found_good = [ex for ex in report.examples if "[EMAIL_REDACTED]" in ex.messages[0]["content"]]
    assert len(found_good) >= 1
    assert "test.engineer@domain.com" not in found_good[0].messages[0]["content"]
    assert "test.engineer@domain.com" not in found_good[0].messages[1]["content"]
