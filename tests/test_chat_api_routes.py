from fastapi.testclient import TestClient

from app import backend


client = TestClient(backend.app)


def test_auth_register_returns_safe_user(monkeypatch):
    monkeypatch.setattr(
        backend.chat_store,
        "register_user",
        lambda payload: {
            "id": "u1",
            "name": payload["name"],
            "email": payload["email"],
            "avatar": None,
            "team": [],
            "isBot": False,
            "isPinned": False,
        },
    )

    response = client.post(
        "/api/auth/register",
        json={"name": "Alice", "email": "alice@example.com", "password": "secret123"},
    )

    assert response.status_code == 200
    assert response.json()["email"] == "alice@example.com"
    assert "password" not in response.json()


def test_auth_login_rejects_invalid_credentials(monkeypatch):
    monkeypatch.setattr(backend.chat_store, "authenticate_user", lambda email, password: None)

    response = client.post(
        "/api/auth/login",
        json={"email": "alice@example.com", "password": "wrong"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid email or password"


def test_auth_session_returns_user(monkeypatch):
    monkeypatch.setattr(
        backend.chat_store,
        "get_user_by_id",
        lambda user_id: {
            "id": user_id,
            "name": "Alice",
            "email": "alice@example.com",
            "avatar": None,
            "team": [],
            "isBot": False,
            "isPinned": False,
        },
    )

    response = client.post("/api/auth/session", json={"user_id": "u1"})

    assert response.status_code == 200
    assert response.json()["id"] == "u1"


def test_profile_update_requires_authenticated_header():
    response = client.put(
        "/api/profile",
        json={"name": "Alice", "email": "alice@example.com", "avatar": "A"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Missing authenticated user id"


def test_list_groups_requires_authenticated_header():
    response = client.get("/api/groups")

    assert response.status_code == 401
    assert response.json()["detail"] == "Missing authenticated user id"


def test_list_conversations_returns_only_authorized_summaries(monkeypatch):
    monkeypatch.setattr(
        backend.chat_store,
        "list_conversation_summaries",
        lambda user_id: [
            {
                "id": "direct:u1:u2",
                "type": "direct",
                "title": "Bob",
                "avatar": "B",
                "unreadCount": 0,
                "groupId": None,
                "participantIds": ["u2"],
                "otherUser": {"id": "u2", "name": "Bob", "email": "bob@example.com"},
                "lastMessage": None,
            }
        ],
    )

    response = client.get("/api/conversations", headers={"x-user-id": "u1"})

    assert response.status_code == 200
    assert response.json()["conversations"][0]["id"] == "direct:u1:u2"


def test_list_conversation_messages_denies_non_member(monkeypatch):
    def fake_get_conversation_messages(user_id, conversation_id):
        raise ValueError("Conversation not found or access denied")

    monkeypatch.setattr(backend.chat_store, "get_conversation_messages", fake_get_conversation_messages)

    response = client.get("/api/conversations/direct:u1:u2/messages", headers={"x-user-id": "u9"})

    assert response.status_code == 403
    assert response.json()["detail"] == "Conversation not found or access denied"


def test_create_conversation_message_returns_notify_user_ids(monkeypatch):
    monkeypatch.setattr(
        backend.chat_store,
        "create_message_for_conversation",
        lambda authenticated_user_id, conversation_id, payload: {
            "message": {
                "id": "m1",
                "conversationId": conversation_id,
                "conversationType": "direct",
                "senderId": authenticated_user_id,
                "receiverId": "u2",
                "content": payload["content"],
                "sentAt": "2026-04-01T10:00:00Z",
            },
            "notifyUserIds": ["u2"],
        },
    )

    response = client.post(
        "/api/conversations/direct:u1:u2/messages",
        headers={"x-user-id": "u1"},
        json={"content": "hello", "receiverId": "u2"},
    )

    assert response.status_code == 200
    assert response.json()["message"]["content"] == "hello"
    assert response.json()["notifyUserIds"] == ["u2"]


def test_mark_message_read_rejects_unrelated_user(monkeypatch):
    monkeypatch.setattr(
        backend.chat_store,
        "mark_message_read",
        lambda user_id, message_id: (_ for _ in ()).throw(ValueError("Message not found or access denied")),
    )

    response = client.post("/api/messages/m1/read", headers={"x-user-id": "u9"})

    assert response.status_code == 403
    assert response.json()["detail"] == "Message not found or access denied"
