from app import chat_store


def test_hash_password_falls_back_to_pbkdf2_when_bcrypt_missing(monkeypatch):
    monkeypatch.setattr(chat_store, "bcrypt", None)

    password_hash = chat_store.hash_password("secret123")

    assert password_hash.startswith("pbkdf2_sha256$")
    assert chat_store.verify_password("secret123", password_hash) is True
    assert chat_store.verify_password("wrong", password_hash) is False


def test_verify_password_rejects_legacy_bcrypt_hash_without_bcrypt(monkeypatch):
    monkeypatch.setattr(chat_store, "bcrypt", None)

    assert chat_store.verify_password("secret123", "$2b$12$not-a-real-bcrypt-hash") is False
