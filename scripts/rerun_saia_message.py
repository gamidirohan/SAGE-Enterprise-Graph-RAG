#!/usr/bin/env python3
import json
from app import utils
from app import saia

DOC_ID = "chat-msg-67d5d586-7702-41e8-8d15-3bbafc183914"


def fetch_message_record(doc_id: str):
    driver = utils.create_neo4j_driver()
    try:
        with utils.open_neo4j_session(driver, utils.NEO4J_DATABASE) as session:
            rows = session.run(
                """
                MATCH (d:Document {doc_id: $doc_id})
                RETURN d.sender AS sender, d.receivers AS receivers, d.conversation_id AS conversation_id,
                       d.conversation_type AS conversation_type, d.group_id AS group_id, d.timestamp AS sent_at,
                       d.content AS content
                LIMIT 1
                """,
                doc_id=doc_id,
            ).data()
            if not rows:
                raise SystemExit(f"Document {doc_id} not found in graph")
            row = rows[0]
            return {
                "sender_id": row.get("sender") or "",
                "receiver_ids": row.get("receivers") or [],
                "conversation_id": row.get("conversation_id"),
                "conversation_type": row.get("conversation_type"),
                "group_id": row.get("group_id"),
                "sent_at": row.get("sent_at") or "",
                "content": row.get("content") or "",
            }
    finally:
        driver.close()


def main():
    record = fetch_message_record(DOC_ID)
    # derive message id (strip chat-msg- prefix)
    message_id = DOC_ID.replace("chat-msg-", "")
    print("Re-running SAIA on:", DOC_ID)
    result = saia.process_chat_message(
        message_id=message_id,
        sender_id=record["sender_id"],
        receiver_ids=record["receiver_ids"],
        conversation_id=record["conversation_id"],
        conversation_type=record["conversation_type"],
        group_id=record["group_id"],
        sent_at=record["sent_at"] or "",
        content=record["content"],
        source="chat_message",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
