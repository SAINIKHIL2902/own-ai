import asyncio
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from aiokafka import AIOKafkaConsumer

from app.behavior.context import extract_contexts
from app.behavior.interest import extract_topics
from app.behavior.preferences import calculate_confidence, extract_preferences
from app.behavior.scoring import update_interest_score
from app.behavior.style import response_style_analyzer
from app.config.settings import settings
from app.storage.models import InterestRecord, PreferenceRecord, ResponseStyleRecord, utc_now
from app.storage.repositories import profile_repo, raw_repo

logger = logging.getLogger(__name__)

SPOOL_DIR = Path(settings.resolved_db_path).parent / "events"


class BehaviorAnalyzer:
    def __init__(self):
        self.raw_repo = raw_repo
        self.profile_repo = profile_repo

    def process_event_dict(self, event_data: Dict[str, Any]) -> bool:
        """Processes an interaction or feedback event dictionary with strict idempotency."""
        event_id = event_data.get("event_id")
        if not event_id:
            logger.warning("Received event without event_id, skipping.")
            return False

        # Idempotency Gate: Do not re-process already analyzed events
        if self.raw_repo.is_event_processed(event_id):
            logger.info(f"Event {event_id} already processed. Skipping to maintain idempotency.")
            return True

        event_type = event_data.get("event_type", "chat_interaction")

        if event_type == "chat_interaction":
            self._analyze_interaction(event_data)
        elif event_type == "user_feedback":
            self._analyze_feedback(event_data)
        else:
            logger.info(f"Unhandled event type: {event_type}")

        # Mark processed
        self.raw_repo.mark_event_processed(event_id)
        logger.info(f"Successfully analyzed and updated profile for event {event_id}")
        return True

    def _analyze_interaction(self, event: Dict[str, Any]) -> None:
        prompt = event.get("prompt", "")
        response = event.get("response", "")
        now_iso = event.get("timestamp") or utc_now()
        prompt_len = len(prompt)

        # 1. Topic & Interest Modeling
        topics = extract_topics(prompt, response)
        for topic in topics:
            existing = self.profile_repo.get_interest(topic)
            if existing:
                new_score = update_interest_score(
                    current_score=existing.score,
                    last_seen_iso=existing.last_seen,
                    prompt_len=prompt_len,
                )
                updated_record = InterestRecord(
                    topic=topic,
                    score=new_score,
                    interaction_count=existing.interaction_count + 1,
                    recent_interactions=existing.recent_interactions + 1,
                    first_seen=existing.first_seen,
                    last_seen=now_iso,
                    updated_at=utc_now(),
                )
            else:
                initial_score = update_interest_score(
                    current_score=0.2,
                    last_seen_iso=now_iso,
                    prompt_len=prompt_len,
                )
                updated_record = InterestRecord(
                    topic=topic,
                    score=initial_score,
                    interaction_count=1,
                    recent_interactions=1,
                    first_seen=now_iso,
                    last_seen=now_iso,
                    updated_at=utc_now(),
                )
            self.profile_repo.upsert_interest(updated_record)

        # 2. Preference Modeling
        detected_prefs = extract_preferences(prompt)
        for pref_key, val in detected_prefs.items():
            existing_pref = self.profile_repo.get_preference(pref_key)
            if existing_pref:
                new_count = existing_pref.evidence_count + 1
                new_conf = calculate_confidence(new_count)
                pref_record = PreferenceRecord(
                    preference=pref_key,
                    value=val,
                    confidence=new_conf,
                    evidence_count=new_count,
                    updated_at=utc_now(),
                )
            else:
                pref_record = PreferenceRecord(
                    preference=pref_key,
                    value=val,
                    confidence=calculate_confidence(1),
                    evidence_count=1,
                    updated_at=utc_now(),
                )
            self.profile_repo.upsert_preference(pref_record)

        # 3. Context Tracking
        contexts = extract_contexts(prompt)
        for ctx in contexts:
            self.profile_repo.upsert_context(ctx)

        # 4. Behavioral Statistics
        self._update_behavior_stats(prompt, event)

    def _analyze_feedback(self, event: Dict[str, Any]) -> None:
        feedback_type = event.get("feedback_type", "thumbs_up")
        rating = event.get("rating")
        message_id = event.get("message_id")
        conversation_id = event.get("conversation_id")

        # 1. Normalize sentiment
        is_positive = feedback_type in ("positive", "thumbs_up") or (rating is not None and rating >= 4)
        is_negative = feedback_type in ("negative", "thumbs_down") or (rating is not None and rating <= 2)
        category = "positive" if is_positive else ("negative" if is_negative else "correction")

        stats = self.profile_repo.get_all_stats()
        current_count = int(stats.get(f"feedback_{category}_count", 0)) + 1
        self.profile_repo.set_stat(f"feedback_{category}_count", str(current_count))

        # 2. Correlate feedback with target assistant response
        if not message_id:
            return

        msg_row = self.raw_repo.get_message_by_id(message_id)
        if not msg_row:
            return

        asst_response = msg_row.get("content", "")
        # Find user prompt from conversation history
        user_prompt = ""
        conv_id = conversation_id or msg_row.get("conversation_id")
        if conv_id:
            conv_msgs = self.raw_repo.get_messages(conv_id)
            user_msgs = [m["content"] for m in conv_msgs if m.get("role") == "user"]
            if user_msgs:
                user_prompt = user_msgs[-1]

        # 3. Analyze response style and ambiguity triggers
        style_info = response_style_analyzer.analyze(user_prompt, asst_response)
        primary_key = style_info["primary_style"]
        style_name = style_info["style_name"]
        description = style_info["description"]
        confusion_triggers = style_info.get("confusion_triggers", [])

        # 4. Update response style profile affinity
        existing = self.profile_repo.get_response_style(primary_key)
        liked_cnt = existing.liked_count if existing else 0
        disliked_cnt = existing.disliked_count if existing else 0

        if is_positive:
            liked_cnt += 1
        elif is_negative:
            disliked_cnt += 1

        total = liked_cnt + disliked_cnt
        affinity = round(liked_cnt / float(total), 2) if total > 0 else 0.5

        self.profile_repo.upsert_response_style(
            ResponseStyleRecord(
                style_key=primary_key,
                style_name=style_name,
                description=description,
                liked_count=liked_cnt,
                disliked_count=disliked_cnt,
                affinity_score=affinity,
                updated_at=utc_now(),
            )
        )

        # 5. Track top liked style or ambiguity trigger in behavior stats
        if is_positive:
            self.profile_repo.set_stat("top_liked_style", style_name)
        elif is_negative and confusion_triggers:
            self.profile_repo.set_stat("top_confusion_trigger", confusion_triggers[0].replace("_", " ").title())

        # 6. Save/update feedback record in raw SQLite with style metadata
        metadata = {
            "primary_style": primary_key,
            "style_name": style_name,
            "style_tags": style_info.get("style_tags", []),
            "confusion_triggers": confusion_triggers,
            "prompt_snippet": user_prompt[:120],
            "response_snippet": asst_response[:120],
        }
        self.raw_repo.save_feedback(
            fb_id=event.get("event_id") or f"fb_{message_id}",
            message_id=message_id,
            conversation_id=conv_id,
            feedback_type=feedback_type,
            feedback_value="positive" if is_positive else "negative",
            rating=rating,
            comment=event.get("comment"),
            metadata_json=json.dumps(metadata),
        )

    def _update_behavior_stats(self, prompt: str, event: Dict[str, Any]) -> None:
        stats = self.profile_repo.get_all_stats()
        total_msgs = int(stats.get("total_interactions", 0)) + 1
        self.profile_repo.set_stat("total_interactions", str(total_msgs))

        # Average prompt length tracking
        total_len = int(stats.get("total_prompt_length", 0)) + len(prompt)
        self.profile_repo.set_stat("total_prompt_length", str(total_len))
        self.profile_repo.set_stat("avg_prompt_length", str(round(total_len / total_msgs, 1)))

        # Active hour tracking
        current_hour = datetime.now(timezone.utc).hour
        self.profile_repo.set_stat("last_active_utc_hour", str(current_hour))

        # Model preference
        if event.get("model"):
            self.profile_repo.set_stat("last_model_used", event["model"])

        # Latency tracking
        latency = float(event.get("latency_ms", 0.0) or 0.0)
        if latency > 0:
            total_latency = float(stats.get("total_latency_ms", 0.0) or 0.0) + latency
            self.profile_repo.set_stat("total_latency_ms", str(round(total_latency, 2)))
            self.profile_repo.set_stat("avg_latency_ms", str(round(total_latency / total_msgs, 1)))

    def drain_unprocessed(self) -> int:
        """Processes all pending events from local spool files and unprocessed SQLite raw_events."""
        count = 0
        # 1. Drain from local file spool if any files exist
        if SPOOL_DIR.exists():
            for spool_file in sorted(SPOOL_DIR.glob("*.json")):
                try:
                    with open(spool_file, "r", encoding="utf-8") as f:
                        event_data = json.load(f)
                    if self.process_event_dict(event_data):
                        count += 1
                    spool_file.unlink(missing_ok=True)
                except Exception as e:
                    logger.error(f"Error processing spooled file {spool_file}: {e}")

        # 2. Drain any unprocessed records stored in raw_events table
        unprocessed_raw = self.raw_repo.get_unprocessed_raw_events()
        for event_data in unprocessed_raw:
            try:
                if self.process_event_dict(event_data):
                    count += 1
            except Exception as e:
                logger.error(f"Error processing unprocessed raw event: {e}")

        if count > 0:
            logger.info(f"Drained and processed {count} pending events.")
        return count

    def drain_local_spool(self) -> int:
        """Alias for backward compatibility."""
        return self.drain_unprocessed()


behavior_analyzer = BehaviorAnalyzer()


async def run_behavior_consumer():
    """Continuous background consumer and local drain worker with auto-reconnect."""
    while True:
        try:
            # Drain all pending offline events on every iteration
            behavior_analyzer.drain_unprocessed()

            if not settings.KAFKA_ENABLED:
                await asyncio.sleep(3)
                continue

            consumer = AIOKafkaConsumer(
                settings.KAFKA_TOPIC_INTERACTIONS,
                bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
                group_id=settings.KAFKA_CONSUMER_GROUP,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                auto_offset_reset="earliest",
                enable_auto_commit=True,
                request_timeout_ms=3000,
            )
            await consumer.start()
            logger.info(f"Behavior analyzer consumer started on topic '{settings.KAFKA_TOPIC_INTERACTIONS}'")

            try:
                async for msg in consumer:
                    try:
                        event_data = msg.value
                        behavior_analyzer.process_event_dict(event_data)
                    except Exception as exc:
                        logger.error(f"Error processing Kafka message: {exc}", exc_info=True)
            finally:
                await consumer.stop()

        except asyncio.CancelledError:
            logger.info("Behavior consumer task cancelled.")
            break
        except Exception as exc:
            logger.debug(
                f"Kafka consumer connection attempt: {exc}. "
                "Continuing offline mode with local event processing. Retrying Kafka in 5s."
            )
            await asyncio.sleep(5)
