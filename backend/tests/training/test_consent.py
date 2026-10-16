import pytest
from app.training.models import TrainingConsentRecord
from app.training.repository import TrainingRepository
from app.training.extractor import ConsentAwareExtractor
from app.training.dataset import DatasetBuilder
from app.training.models import TrainingConfig


def test_consent_defaults_to_none_or_false():
    repo = TrainingRepository()
    consent = repo.get_consent("unregistered_user_123")
    assert consent is None


def test_set_and_get_consent():
    repo = TrainingRepository()
    user_id = "consent_user_xyz"
    
    # Opt-in
    saved = repo.set_consent(user_id, opted_in=True)
    assert saved.opted_in is True
    assert saved.user_id == user_id
    
    fetched = repo.get_consent(user_id)
    assert fetched is not None
    assert fetched.opted_in is True

    # Opt-out
    revoked = repo.set_consent(user_id, opted_in=False)
    assert revoked.opted_in is False

    fetched_revoked = repo.get_consent(user_id)
    assert fetched_revoked.opted_in is False


def test_extractor_respects_unconsented_user():
    repo = TrainingRepository()
    user_id = "unconsented_user"
    repo.set_consent(user_id, opted_in=False)

    extractor = ConsentAwareExtractor(training_repository=repo)
    report = extractor.extract_user_examples(user_id=user_id)

    assert report.consent_opted_in is False
    assert report.accepted_count == 0
    assert "NO_CONSENT_GRANTED" in report.rejected_counts


def test_dataset_builder_raises_on_unconsented_user():
    repo = TrainingRepository()
    user_id = "unconsented_builder_user"
    repo.set_consent(user_id, opted_in=False)

    extractor = ConsentAwareExtractor(training_repository=repo)
    builder = DatasetBuilder(extractor=extractor)
    with pytest.raises(PermissionError) as exc_info:
        builder.build_dataset(TrainingConfig(), user_id=user_id)

    assert "User 'unconsented_builder_user' has not granted consent" in str(exc_info.value)
