# Local Kafka Setup for Phase 2

Phase 2 uses **Apache Kafka** as an event streaming backbone to decouple user interactions from behavioral analytics.

---

## 1. Quick Start (Using Docker Compose)

To start the local KRaft-mode Kafka broker (zero ZooKeeper required):

```bash
cd "own-ai/kafka"
docker compose up -d
```

To view broker logs:
```bash
docker compose logs -f kafka
```

To stop Kafka:
```bash
docker compose down
```

---

## 2. Running Kafka via Homebrew (Alternative)

If you prefer running Kafka directly on macOS without Docker:

```bash
brew install kafka
brew services start kafka
```

---

## 3. Creating & Inspecting Topics

### Create the interactions topic:
```bash
# Via Docker
docker exec -it own-ai-kafka /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 \
  --create --topic interaction.events \
  --partitions 3 --replication-factor 1

# Or via Homebrew
kafka-topics --bootstrap-server localhost:9092 \
  --create --topic interaction.events \
  --partitions 3 --replication-factor 1
```

### Listen to live interaction events in terminal:
```bash
# Via Docker
docker exec -it own-ai-kafka /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic interaction.events \
  --from-beginning
```

---

## 4. Offline Resilience Guarantee

If Kafka is **NOT running**, the Personal Local AI application **does not crash**:
1. All user chat responses continue to work at full speed.
2. The producer automatically detects that Kafka is down.
3. Interaction events are safely spooled to `data/events/*.json`.
4. As soon as the system starts or `/behavior/drain` is called, all spooled events are processed and incorporated into your user profile.
