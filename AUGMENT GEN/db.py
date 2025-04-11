import json
import os
import hashlib
from datetime import datetime
import uuid

# Ensure data directory exists
DATA_DIR = "data"
USERS_FILE = os.path.join(DATA_DIR, "users.json")
CHATS_FILE = os.path.join(DATA_DIR, "chats.json")
MESSAGES_FILE = os.path.join(DATA_DIR, "messages.json")
DOCUMENTS_FILE = os.path.join(DATA_DIR, "documents.json")

def ensure_data_dir():
    """Ensure data directory and files exist"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    # Initialize users file if it doesn't exist
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'w') as f:
            json.dump([], f)
    
    # Initialize chats file if it doesn't exist
    if not os.path.exists(CHATS_FILE):
        with open(CHATS_FILE, 'w') as f:
            json.dump([], f)
    
    # Initialize messages file if it doesn't exist
    if not os.path.exists(MESSAGES_FILE):
        with open(MESSAGES_FILE, 'w') as f:
            json.dump([], f)
    
    # Initialize documents file if it doesn't exist
    if not os.path.exists(DOCUMENTS_FILE):
        with open(DOCUMENTS_FILE, 'w') as f:
            json.dump([], f)

# User functions
def hash_password(password):
    """Hash a password for storing"""
    return hashlib.sha256(password.encode()).hexdigest()

def add_user(username, password):
    """Add a new user to the database"""
    ensure_data_dir()
    
    with open(USERS_FILE, 'r') as f:
        users = json.load(f)
    
    # Check if user already exists
    for user in users:
        if user['username'] == username:
            return False
    
    # Add new user
    user_id = str(uuid.uuid4())
    users.append({
        'id': user_id,
        'username': username,
        'password_hash': hash_password(password),
        'created_at': datetime.now().isoformat()
    })
    
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)
    
    # Create AI chat for the new user
    create_ai_chat(user_id)
    
    return user_id

def verify_user(username, password):
    """Verify user credentials"""
    ensure_data_dir()
    
    with open(USERS_FILE, 'r') as f:
        users = json.load(f)
    
    for user in users:
        if user['username'] == username and user['password_hash'] == hash_password(password):
            return user['id']
    
    return None

def get_user_by_id(user_id):
    """Get user by ID"""
    ensure_data_dir()
    
    with open(USERS_FILE, 'r') as f:
        users = json.load(f)
    
    for user in users:
        if user['id'] == user_id:
            return user
    
    return None

def get_all_users():
    """Get all users"""
    ensure_data_dir()
    
    with open(USERS_FILE, 'r') as f:
        users = json.load(f)
    
    return users

# Chat functions
def create_ai_chat(user_id):
    """Create an AI chat for a user"""
    ensure_data_dir()
    
    with open(CHATS_FILE, 'r') as f:
        chats = json.load(f)
    
    chat_id = str(uuid.uuid4())
    chats.append({
        'id': chat_id,
        'name': 'AI Assistant',
        'type': 'ai',
        'participants': [user_id],
        'created_at': datetime.now().isoformat(),
        'updated_at': datetime.now().isoformat(),
        'is_pinned': True
    })
    
    with open(CHATS_FILE, 'w') as f:
        json.dump(chats, f, indent=2)
    
    return chat_id

def create_chat(user_id, other_user_id, chat_name=None):
    """Create a new chat between users"""
    ensure_data_dir()
    
    with open(CHATS_FILE, 'r') as f:
        chats = json.load(f)
    
    # Check if chat already exists
    for chat in chats:
        if chat['type'] == 'direct' and set(chat['participants']) == set([user_id, other_user_id]):
            return chat['id']
    
    # Get other user's name if chat_name not provided
    if not chat_name:
        other_user = get_user_by_id(other_user_id)
        chat_name = other_user['username'] if other_user else "New Chat"
    
    chat_id = str(uuid.uuid4())
    chats.append({
        'id': chat_id,
        'name': chat_name,
        'type': 'direct',
        'participants': [user_id, other_user_id],
        'created_at': datetime.now().isoformat(),
        'updated_at': datetime.now().isoformat(),
        'is_pinned': False
    })
    
    with open(CHATS_FILE, 'w') as f:
        json.dump(chats, f, indent=2)
    
    return chat_id

def get_user_chats(user_id):
    """Get all chats for a user"""
    ensure_data_dir()
    
    with open(CHATS_FILE, 'r') as f:
        chats = json.load(f)
    
    user_chats = []
    for chat in chats:
        if user_id in chat['participants']:
            user_chats.append(chat)
    
    # Sort chats: pinned first, then by updated_at
    user_chats.sort(key=lambda x: (not x['is_pinned'], x['updated_at']), reverse=True)
    
    return user_chats

def get_chat_by_id(chat_id):
    """Get chat by ID"""
    ensure_data_dir()
    
    with open(CHATS_FILE, 'r') as f:
        chats = json.load(f)
    
    for chat in chats:
        if chat['id'] == chat_id:
            return chat
    
    return None

def update_chat_timestamp(chat_id):
    """Update the timestamp of a chat"""
    ensure_data_dir()
    
    with open(CHATS_FILE, 'r') as f:
        chats = json.load(f)
    
    for chat in chats:
        if chat['id'] == chat_id:
            chat['updated_at'] = datetime.now().isoformat()
            break
    
    with open(CHATS_FILE, 'w') as f:
        json.dump(chats, f, indent=2)

# Message functions
def add_message(chat_id, sender_id, content, message_type='text', document_id=None):
    """Add a message to a chat"""
    ensure_data_dir()
    
    with open(MESSAGES_FILE, 'r') as f:
        messages = json.load(f)
    
    message_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    
    messages.append({
        'id': message_id,
        'chat_id': chat_id,
        'sender_id': sender_id,
        'content': content,
        'type': message_type,
        'document_id': document_id,
        'timestamp': timestamp,
        'is_read': False
    })
    
    with open(MESSAGES_FILE, 'w') as f:
        json.dump(messages, f, indent=2)
    
    # Update chat timestamp
    update_chat_timestamp(chat_id)
    
    return message_id

def get_chat_messages(chat_id):
    """Get all messages for a chat"""
    ensure_data_dir()
    
    with open(MESSAGES_FILE, 'r') as f:
        messages = json.load(f)
    
    chat_messages = []
    for message in messages:
        if message['chat_id'] == chat_id:
            chat_messages.append(message)
    
    # Sort messages by timestamp
    chat_messages.sort(key=lambda x: x['timestamp'])
    
    return chat_messages

def mark_messages_as_read(chat_id, user_id):
    """Mark all messages in a chat as read for a user"""
    ensure_data_dir()
    
    with open(MESSAGES_FILE, 'r') as f:
        messages = json.load(f)
    
    updated = False
    for message in messages:
        if message['chat_id'] == chat_id and message['sender_id'] != user_id and not message['is_read']:
            message['is_read'] = True
            updated = True
    
    if updated:
        with open(MESSAGES_FILE, 'w') as f:
            json.dump(messages, f, indent=2)

# Document functions
def save_document(filename, file_path, uploader_id, file_type, file_size):
    """Save document metadata to the database"""
    ensure_data_dir()
    
    with open(DOCUMENTS_FILE, 'r') as f:
        documents = json.load(f)
    
    document_id = str(uuid.uuid4())
    documents.append({
        'id': document_id,
        'filename': filename,
        'file_path': file_path,
        'uploader_id': uploader_id,
        'file_type': file_type,
        'file_size': file_size,
        'uploaded_at': datetime.now().isoformat()
    })
    
    with open(DOCUMENTS_FILE, 'w') as f:
        json.dump(documents, f, indent=2)
    
    return document_id

def get_document_by_id(document_id):
    """Get document by ID"""
    ensure_data_dir()
    
    with open(DOCUMENTS_FILE, 'r') as f:
        documents = json.load(f)
    
    for document in documents:
        if document['id'] == document_id:
            return document
    
    return None

# Initialize data directory and files
ensure_data_dir()
