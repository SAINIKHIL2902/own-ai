from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional, Tuple

from app.storage.repositories import RawRepository, raw_repo
from app.training.models import TrainingExample
from app.training.repository import TrainingRepository, training_repo
from app.training.sanitizer import TrainingSanitizer, training_sanitizer

logger = logging.getLogger(__name__)


@dataclass
class ExtractionReport:
    user_id: str
    consent_opted_in: bool
    total_evaluated: int = 0
    accepted_count: int = 0
    rejected_counts: Dict[str, int] = field(default_factory=dict)
    examples: List[TrainingExample] = field(default_factory=list)


class ConsentAwareExtractor:
    """
    Read-only extraction layer that retrieves eligible behavioral examples from
    conversations and feedback while strictly respecting user consent, privacy rules,
    and quality filters.
    """

    def __init__(
        self,
        raw_repository: Optional[RawRepository] = None,
        training_repository: Optional[TrainingRepository] = None,
        sanitizer: Optional[TrainingSanitizer] = None,
    ):
        self.raw_repo = raw_repository or raw_repo
        self.training_repo = training_repository or training_repo
        self.sanitizer = sanitizer or training_sanitizer

    def extract_user_examples(self, user_id: str = "local_user") -> ExtractionReport:
        """
        Extract approved, high-quality, sanitized training examples for a user.
        If the user has not explicitly opted into training, returns zero examples.
        """
        report = ExtractionReport(user_id=user_id, consent_opted_in=False)

        # 1. Mandatory Consent Check
        consent = self.training_repo.get_consent(user_id)
        if not consent or not consent.opted_in:
            logger.info(f"Training extraction aborted for user '{user_id}': User has not opted into training.")
            report.rejected_counts["NO_CONSENT_GRANTED"] = 1
            return report

        report.consent_opted_in = True

        # 2. Fetch conversations
        with self.raw_repo.manager.get_connection() as conn:
            conv_rows = conn.execute(
                "SELECT id, title FROM conversations ORDER BY updated_at DESC"
            ).fetchall()

        for crow in conv_rows:
            conv_id = crow["id"]
            messages = self.raw_repo.get_messages(conv_id)

            # Pair consecutive (user -> assistant) messages
            for i in range(len(messages) - 1):
                msg = messages[i]
                next_msg = messages[i + 1]

                if msg.get("role") == "user" and next_msg.get("role") == "assistant":
                    report.total_evaluated += 1
                    user_text = msg.get("content", "").strip()
                    asst_text = next_msg.get("content", "").strip()
                    msg_id = next_msg.get("message_id") or next_msg.get("id")

                    # 3. Quality & Feedback Check
                    fb = self.raw_repo.get_feedback_by_message_id(msg_id) if msg_id else None
                    if fb:
                        fb_type = fb.get("feedback_type", "").lower()
                        rating = fb.get("rating")
                        if fb_type in ["negative", "thumbs_down"] or (rating is not None and rating <= 2):
                            report.rejected_counts["EXCLUDED_NEGATIVE_FEEDBACK"] = (
                                report.rejected_counts.get("EXCLUDED_NEGATIVE_FEEDBACK", 0) + 1
                            )
                            continue

                    # 4. Privacy & Secret Filter
                    is_eligible, reason = self.sanitizer.is_eligible_for_dataset(user_text, asst_text)
                    if not is_eligible:
                        report.rejected_counts[reason] = report.rejected_counts.get(reason, 0) + 1
                        continue

                    # 5. Sanitize PII
                    clean_user = self.sanitizer.redact_pii(user_text)
                    clean_asst = self.sanitizer.redact_pii(asst_text)

                    example = TrainingExample(
                        messages=[
                            {"role": "user", "content": clean_user},
                            {"role": "assistant", "content": clean_asst},
                        ],
                        metadata={
                            "conversation_id": conv_id,
                            "message_id": msg_id,
                            "user_id": user_id,
                            "feedback_type": fb.get("feedback_type") if fb else "implicit_positive",
                        },
                    )
                    report.examples.append(example)

        report.accepted_count = len(report.examples)
        logger.info(
            f"Extraction completed for user '{user_id}': {report.accepted_count} accepted, "
            f"{sum(report.rejected_counts.values())} rejected."
        )
        return report


consent_aware_extractor = ConsentAwareExtractor()
