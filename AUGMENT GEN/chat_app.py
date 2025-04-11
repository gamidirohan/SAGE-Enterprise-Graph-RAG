import streamlit as st
import os
import time
from datetime import datetime
import auth
import db
import graph_integration

# Set page configuration
st.set_page_config(
    page_title="SAGE Chat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide the top padding
st.markdown("""
<style>
    .block-container {
        padding-top: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
auth.init_session_state()

# Create data directories
if not os.path.exists("data"):
    os.makedirs("data")
if not os.path.exists("uploads"):
    os.makedirs("uploads")

# Custom CSS
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #f5f5f5;
        padding: 0 !important;
        max-width: 100% !important;
    }

    /* Chat button styling */
    div[data-testid="stButton"] {
        margin-bottom: 1px;
        padding: 0;
    }

    div[data-testid="stButton"] button {
        text-align: left;
        padding: 8px 10px;
        border-radius: 5px;
        display: flex;
        align-items: center;
        position: relative;
        height: 60px;
        transition: background-color 0.2s;
    }

    div[data-testid="stButton"] button:hover {
        background-color: #f5f5f5;
    }

    /* Remove extra padding */
    .block-container {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 100% !important;
    }

    /* App header */
    .app-header {
        background-color: white;
        padding: 8px 15px;
        border-bottom: 1px solid #eee;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 5px;
    }

    .app-header h3 {
        margin: 0;
        font-size: 1.2rem;
        font-weight: 500;
    }

    /* Chat container styling */
    .chat-container {
        height: calc(100vh - 180px); /* Adjusted to leave space for input area */
        overflow-y: auto;
        padding: 10px;
        background-color: #e5ddd5; /* WhatsApp-like background */
        background-image: url('https://web.whatsapp.com/img/bg-chat-tile-light_a4be512e7195b6b733d9110b408f075d.png');
        background-repeat: repeat;
        margin-bottom: 100px; /* Space for fixed input area */
        display: flex;
        flex-direction: column;
        border-radius: 5px;
        padding-bottom: 120px; /* Extra padding at bottom to ensure messages aren't hidden behind input */
    }

    /* Message styling */
    .message {
        margin-bottom: 10px;
        padding: 10px 15px;
        max-width: 70%;
        word-wrap: break-word;
        position: relative;
        clear: both;
        line-height: 1.4;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }

    .user-message {
        color: #303030;
        float: right;
        margin-left: 50px;
    }

    .other-message, .ai-message {
        color: #303030;
        float: left;
        margin-right: 50px;
    }

    /* Message container with avatar */
    .message-container {
        display: flex;
        margin-bottom: 10px;
        clear: both;
        align-items: flex-end;
        width: 100%;
    }

    /* Message alignment classes */
    .message-right {
        flex-direction: row-reverse;
        justify-content: flex-start;
    }

    .message-left {
        flex-direction: row;
        justify-content: flex-start;
    }

    /* Avatar styling */
    .avatar {
        width: 35px;
        height: 35px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
        margin: 0 8px;
    }

    .ai-message {
        background-color: #f0f2f5;
        color: #333;
        float: left;
        border-bottom-left-radius: 5px;
        margin-right: 50px;
    }

    /* Avatar styling */
    .avatar {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background-color: #ccc;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
        margin-bottom: 5px;
    }

    .avatar.user {
        float: right;
        margin-left: 10px;
    }

    .avatar.other {
        float: left;
        margin-right: 10px;
    }

    /* Chat list styling */
    .chat-list {
        padding: 0;
        margin: 0;
        list-style: none;
    }

    .chat-list-item {
        padding: 12px 15px;
        border-bottom: 1px solid #eee;
        display: flex;
        align-items: center;
        cursor: pointer;
        transition: background-color 0.2s;
        margin-bottom: 2px;
        border-radius: 5px;
    }

    .chat-list-item:hover {
        background-color: #f0f2f5;
    }

    .chat-list-item.active {
        background-color: #e6f7ff;
    }

    .chat-list-item.pinned {
        border-left: 3px solid #1e88e5;
        background-color: #f8f9fa;
    }

    .chat-avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background-color: #1e88e5;
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin-right: 15px;
        flex-shrink: 0;
    }

    .chat-info {
        flex-grow: 1;
        overflow: hidden;
    }

    .chat-name {
        font-weight: 500;
        margin-bottom: 3px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .chat-preview {
        color: #666;
        font-size: 0.85em;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .chat-time {
        color: #999;
        font-size: 0.75em;
        white-space: nowrap;
        margin-left: 10px;
    }

    /* Document styling */
    .document-message {
        background-color: #F0F0F0;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 5px;
    }

    /* Timestamp styling */
    .timestamp {
        font-size: 0.7em;
        color: #888;
        margin-top: 5px;
        clear: both;
    }

    /* Chat header */
    .chat-header {
        padding: 15px 20px;
        border-bottom: 1px solid #eee;
        display: flex;
        align-items: center;
        background-color: white;
    }

    .chat-header-avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background-color: #1e88e5;
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin-right: 15px;
    }

    .chat-header-info {
        flex-grow: 1;
    }

    .chat-header-name {
        font-weight: 500;
        margin-bottom: 3px;
    }

    .chat-header-status {
        color: #4CAF50;
        font-size: 0.8em;
    }

    /* Chat input styling */
    .chat-input-area {
        padding: 10px 15px;
        background-color: white;
        border-top: 1px solid #eee;
        position: sticky;
        bottom: 0;
        display: flex;
        align-items: center;
    }

    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    header {visibility: hidden;}

    /* Make the app full width with small left gap */
    .block-container {
        max-width: 100% !important;
        padding-left: 10px !important;
        padding-right: 0 !important;
    }

    /* Unread indicator */
    .unread-indicator {
        width: 10px;
        height: 10px;
        background-color: #4CAF50;
        border-radius: 50%;
        display: inline-block;
        margin-left: 5px;
    }

    /* Message container to clear floats */
    .message-container {
        overflow: hidden;
        margin-bottom: 20px;
        clear: both;
    }

    /* Button styling */
    .stButton>button {
        background-color: white;
        border: 1px solid #eee;
        padding: 10px 15px;
        border-radius: 5px;
        transition: all 0.3s;
        width: 100%;
        text-align: left;
    }

    .stButton>button:hover {
        background-color: #f8f9fa;
        border-color: #ddd;
    }

    .stButton>button:active {
        background-color: #e6f7ff;
        border-color: #1e88e5;
    }

    /* Form styling */
    .stForm {
        background-color: white;
        padding: 0;
        border-radius: 0;
    }

    /* Text area styling */
    .stTextArea>div>div>textarea {
        border-radius: 20px;
        padding: 10px 15px;
        resize: none;
        min-height: 68px !important;
        max-height: 100px;
        overflow-y: auto;
    }

    /* Reduce text area label spacing */
    .stTextArea label {
        display: none;
    }

    /* Adjust text area container padding */
    .stTextArea>div {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }

    /* Contact list styling */
    .contact-list {
        margin-top: 5px;
        max-height: calc(100vh - 80px);
        overflow-y: auto;
    }

    /* AI chat styling - make it stand out */
    .ai-contact {
        border-left: 3px solid #1e88e5 !important;
        background-color: #f0f7ff !important;
        margin-bottom: 10px !important;
    }

    /* Divider styling */
    .divider {
        height: 1px;
        background-color: #ddd;
        margin: 10px 0;
    }

    /* Search box styling */
    .search-box {
        padding: 8px;
        border-radius: 20px;
        border: 1px solid #ddd;
        width: 100%;
        margin-bottom: 10px;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Main application
def main():
    # Check if user is authenticated
    if not auth.is_authenticated():
        auth.render_login_page()
    else:
        render_chat_interface()

def render_chat_interface():
    # Auto-refresh mechanism
    if 'last_refresh_time' not in st.session_state:
        st.session_state.last_refresh_time = time.time()
        st.session_state.last_message_count = 0

    # Set up auto-refresh every second
    refresh_interval = 1.0  # seconds
    current_time = time.time()

    # Check if it's time to refresh
    if current_time - st.session_state.last_refresh_time > refresh_interval:
        # Update the last refresh time
        st.session_state.last_refresh_time = current_time

        # Check if there are new messages in the current chat
        if st.session_state.current_chat_id:
            current_messages = db.get_chat_messages(st.session_state.current_chat_id)
            current_message_count = len(current_messages)

            # If there are new messages, rerun to update the UI
            if current_message_count > st.session_state.last_message_count:
                st.session_state.last_message_count = current_message_count
                st.rerun()

    # Get current user
    current_user = auth.get_current_user()
    current_user_id = current_user['id']

    # Initialize current chat ID if not present
    if "current_chat_id" not in st.session_state:
        # Try to find an AI chat
        user_chats = db.get_user_chats(current_user_id)
        ai_chat = next((chat for chat in user_chats if chat['type'] == 'ai'), None)

        if ai_chat:
            st.session_state.current_chat_id = ai_chat['id']
        elif user_chats:
            # If no AI chat, use the first available chat
            st.session_state.current_chat_id = user_chats[0]['id']
        else:
            # Create an AI chat if no chats exist
            ai_chat_id = db.create_ai_chat(current_user_id, "AI Assistant")
            st.session_state.current_chat_id = ai_chat_id

    # Debug section (collapsed by default)
    with st.expander("Debug Info", expanded=False):
        st.write(f"Current user: {current_user['username']} (ID: {current_user_id})")
        st.write(f"Current chat ID: {st.session_state.current_chat_id}")

        # Show all session state variables
        st.write("Session State:")
        for key, value in st.session_state.items():
            st.write(f"{key}: {value}")

        # Show all chats for this user
        st.write("User Chats:")
        chats = db.get_user_chats(current_user_id)
        for chat in chats:
            st.write(f"Chat ID: {chat['id']}, Type: {chat['type']}, Participants: {chat['participants']}")

    # Create a two-column layout
    col1, col2 = st.columns([1, 3])

    with col1:
        # App header with logout button
        col_header, col_logout = st.columns([3, 1])
        with col_header:
            st.markdown('<div class="app-header"><h3>CHATS</h3></div>', unsafe_allow_html=True)
        with col_logout:
            if st.button("🚪", key="logout_btn", help="Logout"):
                auth.logout_user()
                st.rerun()

        # Optional search box (commented out for now as it's not implemented)
        # search_query = st.text_input("", placeholder="Search...", key="search_box")

        # Get user's chats
        chats = db.get_user_chats(current_user_id)

        # Get all users for display
        all_users = db.get_all_users()
        other_users = [user for user in all_users if user['id'] != current_user_id]

        # Contact list container
        st.markdown('<div class="contact-list">', unsafe_allow_html=True)

        # First, display the AI Assistant as a special contact
        ai_chat = next((chat for chat in chats if chat['type'] == 'ai'), None)
        if not ai_chat:
            # Create AI chat if it doesn't exist
            ai_chat_id = db.create_ai_chat(current_user_id, "AI Assistant")
            ai_chat = {
                'id': ai_chat_id,
                'name': 'AI Assistant',
                'type': 'ai',
                'participants': [current_user_id]
            }
            # Force reload to get the updated chat
            st.rerun()

        # Display AI chat button
        ai_active = ai_chat['id'] == st.session_state.current_chat_id
        ai_button_key = f"chat_button_ai_{ai_chat['id']}"

        if st.button(
            "AI Assistant",
            key=ai_button_key,
            use_container_width=True,
            help="Chat with AI Assistant"
        ):
            st.session_state.current_chat_id = ai_chat['id']
            st.rerun()

        # Style the AI button
        st.markdown(f"""
        <style>
            div[data-testid="stButton"]:has(button[kind="secondary"]:contains("AI Assistant")) {{
                margin-bottom: 2px;
            }}

            div[data-testid="stButton"]:has(button[kind="secondary"]:contains("AI Assistant")) button {{
                background-color: {"#e6f7ff" if ai_active else "#f0f7ff"};
                border-left: 3px solid #1e88e5;
                text-align: left;
                padding: 10px;
                border-radius: 5px;
                display: flex;
                align-items: center;
                position: relative;
            }}

            div[data-testid="stButton"]:has(button[kind="secondary"]:contains("AI Assistant")) button::before {{
                content: "A";
                background-color: #1e88e5;
                color: white;
                width: 40px;
                height: 40px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                margin-right: 10px;
                font-weight: bold;
            }}
        </style>
        """, unsafe_allow_html=True)

        # Add divider
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Display all users as contacts
        st.markdown('<p style="font-weight: 500; margin-bottom: 5px;">Contacts</p>', unsafe_allow_html=True)

        for user in other_users:
            # Find existing chat with this user or create chat data
            existing_chat = None
            for chat in chats:
                if chat['type'] == 'direct' and user['id'] in chat['participants']:
                    existing_chat = chat
                    break

            # Get the first letter of the username for the avatar
            avatar_letter = user['username'][0].upper() if user['username'] else "?"

            # Create a unique key for this user button
            user_button_key = f"user_button_{user['id']}"

            # Check if this user chat is active
            is_active = existing_chat and existing_chat['id'] == st.session_state.current_chat_id

            # Create a clickable button for each user
            if st.button(
                f"{user['username']}",
                key=user_button_key,
                use_container_width=True
            ):
                if existing_chat:
                    # Use existing chat
                    st.session_state.current_chat_id = existing_chat['id']
                else:
                    # Create a new chat with this user
                    new_chat_id = db.create_chat(current_user_id, user['id'])
                    st.session_state.current_chat_id = new_chat_id

                # Force a rerun
                st.rerun()

            # Add custom styling to make it look like a user item
            st.markdown(f"""
            <style>
                div[data-testid="stButton"]:has(button[kind="secondary"]:contains("{user['username']}")) {{
                    margin-bottom: 2px;
                }}

                div[data-testid="stButton"]:has(button[kind="secondary"]:contains("{user['username']}")) button {{
                    background-color: {"#e6f7ff" if is_active else "white"};
                    text-align: left;
                    padding: 10px;
                    border-radius: 5px;
                    display: flex;
                    align-items: center;
                }}

                div[data-testid="stButton"]:has(button[kind="secondary"]:contains("{user['username']}")) button::before {{
                    content: "{avatar_letter}";
                    background-color: #9e9e9e;
                    color: white;
                    width: 40px;
                    height: 40px;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin-right: 10px;
                    font-weight: bold;
                }}
            </style>
            """, unsafe_allow_html=True)

        # Close contact list container
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        # Check if a chat is selected
        if st.session_state.current_chat_id:
            current_chat = db.get_chat_by_id(st.session_state.current_chat_id)

            if current_chat:
                # Determine chat details
                if current_chat['type'] == 'direct':
                    other_user_id = next((uid for uid in current_chat['participants'] if uid != current_user_id), None)
                    other_user = db.get_user_by_id(other_user_id) if other_user_id else None
                    chat_title = other_user['username'] if other_user else current_chat['name']
                    avatar_letter = chat_title[0].upper() if chat_title else "?"
                    avatar_color = "#9e9e9e"
                    is_ai_chat = False
                else:
                    chat_title = current_chat['name']
                    avatar_letter = "A"
                    avatar_color = "#1e88e5"
                    is_ai_chat = True

                # Display chat header with refresh indicator
                last_refresh = time.time() - st.session_state.last_refresh_time
                refresh_indicator = '⟳' if last_refresh < 0.5 else ''

                st.markdown(f"""
                <div class="chat-header">
                    <div class="chat-header-avatar" style="background-color: {avatar_color};">{avatar_letter}</div>
                    <div class="chat-header-info">
                        <div class="chat-header-name">{chat_title} {refresh_indicator}</div>
                        <div class="chat-header-status">{'AI Assistant' if is_ai_chat else 'Online'}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Mark messages as read
                db.mark_messages_as_read(current_chat['id'], current_user_id)

                # Get chat messages
                messages = db.get_chat_messages(current_chat['id'])

                # Update the message count in session state for auto-refresh comparison
                st.session_state.last_message_count = len(messages)

                # Create a container for chat messages
                chat_container = st.container()

                # Create HTML for all messages with avatars and better styling
                messages_html = ''
                for message in messages:
                    # Determine message class and sender info
                    if message['sender_id'] == current_user_id:
                        message_class = "message user-message"
                        sender_name = "You"
                        avatar_class = "user"
                        avatar_letter = sender_name[0].upper()
                        avatar_color = "#3a3f47"
                        align_class = "message-right"
                    elif message['sender_id'] == "ai":
                        message_class = "message ai-message"
                        sender_name = "AI Assistant"
                        avatar_class = "other"
                        avatar_letter = "A"
                        avatar_color = "#1e88e5"
                        align_class = "message-left"
                    else:
                        message_class = "message other-message"
                        sender = db.get_user_by_id(message['sender_id'])
                        sender_name = sender['username'] if sender else "Unknown"
                        avatar_class = "other"
                        avatar_letter = sender_name[0].upper()
                        avatar_color = "#9e9e9e"
                        align_class = "message-left"

                    # Format timestamp
                    timestamp = datetime.fromisoformat(message['timestamp'])
                    formatted_time = timestamp.strftime("%I:%M %p")

                    # Add message container with avatar
                    messages_html += f'<div class="message-container {align_class}">'

                    # Add avatar
                    messages_html += f'<div class="avatar {avatar_class}" style="background-color: {avatar_color};">{avatar_letter}</div>'

                    # Add message based on type
                    if message['type'] == 'text':
                        # Different styling for user vs AI/other messages
                        if message['sender_id'] == current_user_id:
                            # User message styling (right-aligned, green)
                            messages_html += f"""
                            <div class="{message_class}" style="background-color: #dcf8c6; border-radius: 15px 0px 15px 15px; max-width: 65%;">
                                <div style="padding: 6px 12px;">{message['content']}</div>
                                <div class="timestamp" style="text-align: right; font-size: 11px; color: #999; padding: 0 7px 3px 0;">{formatted_time} ✓✓</div>
                            </div>
                            """
                        else:
                            # AI/other message styling (left-aligned, white)
                            messages_html += f"""
                            <div class="{message_class}" style="background-color: white; border-radius: 0px 15px 15px 15px; max-width: 65%;">
                                <div style="padding: 6px 12px;">{message['content']}</div>
                                <div class="timestamp" style="text-align: right; font-size: 11px; color: #999; padding: 0 7px 3px 0;">{formatted_time}</div>
                            </div>
                            """
                    elif message['type'] == 'document':
                        document = db.get_document_by_id(message['document_id'])
                        if document:
                            messages_html += f"""
                            <div class="{message_class}">
                                <div class="document-message">
                                    <i class="fas fa-file"></i> {document['filename']}
                                </div>
                                <p>{message['content']}</p>
                                <div class="timestamp">{formatted_time}</div>
                            </div>
                            """

                    # Close message container
                    messages_html += '</div>'

                # Display all messages in the container
                with chat_container:
                    st.markdown(f'<div class="chat-container" id="chat-container">{messages_html}</div>', unsafe_allow_html=True)

                    # Add JavaScript to auto-scroll to the bottom of the chat
                    st.markdown("""
                    <script>
                        // Function to scroll to the bottom of the chat container
                        function scrollToBottom() {
                            const chatContainer = document.getElementById('chat-container');
                            if (chatContainer) {
                                chatContainer.scrollTop = chatContainer.scrollHeight;
                            }
                        }

                        // Scroll to bottom when page loads
                        window.addEventListener('load', scrollToBottom);

                        // Also scroll after a short delay to ensure all content is loaded
                        setTimeout(scrollToBottom, 500);
                    </script>
                    """, unsafe_allow_html=True)

                # Chat input area with better styling
                # Add custom styling for the input area
                st.markdown("""
                <style>
                    /* Chat input area styling */
                    .chat-input-container {
                        display: flex;
                        align-items: center;
                        background-color: #f0f0f0;
                        border-radius: 24px;
                        padding: 8px 16px;
                        margin-top: 10px;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                        position: fixed;
                        bottom: 20px;
                        left: 50%;
                        transform: translateX(-50%);
                        width: 65%;
                        max-width: 800px;
                        z-index: 1000; /* Ensure it's above other elements */
                    }

                    /* Hide Streamlit elements' default styling */
                    div[data-testid="stTextArea"] > div {
                        border: none !important;
                        box-shadow: none !important;
                        padding: 0 !important;
                    }

                    div[data-testid="stTextArea"] textarea {
                        padding: 8px !important;
                        font-size: 15px !important;
                        min-height: 40px !important;
                        background-color: transparent !important;
                    }

                    /* Style the send button */
                    div[data-testid="stButton"]:has(button:contains("➤")) button {
                        border-radius: 50%;
                        width: 40px;
                        height: 40px;
                        padding: 0;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        background-color: #00a884; /* WhatsApp green */
                        color: white;
                        box-shadow: none;
                        border: none;
                        font-size: 18px;
                    }

                    /* Hide file uploader styling */
                    div[data-testid="stFileUploader"] {
                        width: 40px;
                        height: 40px;
                        overflow: hidden;
                    }

                    div[data-testid="stFileUploader"] button {
                        border-radius: 50%;
                        width: 40px;
                        height: 40px;
                        padding: 0;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        background-color: transparent;
                        color: #666;
                        box-shadow: none;
                        border: none;
                    }

                    /* Hide the file uploader text */
                    div[data-testid="stFileUploader"] p {
                        display: none;
                    }

                    /* Hide the file uploader drag area */
                    div[data-testid="stFileUploader"] div[data-testid="stFileUploadDropzone"] {
                        height: 40px;
                        min-height: 40px;
                        padding: 0;
                        border: none;
                        background: none;
                    }
                </style>
                <div class="chat-input-container">
                """, unsafe_allow_html=True)

                # Use columns for input and buttons
                input_col, upload_col, send_col = st.columns([6, 1, 1])

                with input_col:
                    message_input = st.text_area("Type a message...", key=f"message_input_{current_chat['id']}", label_visibility="collapsed", height=68, placeholder="Type a message...")

                    # Add JavaScript for Ctrl+Enter to send message
                    st.markdown(f"""
                    <script>
                        // Get the textarea element
                        const textareas = document.querySelectorAll('textarea');
                        textareas.forEach(textarea => {{
                            textarea.addEventListener('keydown', function(e) {{
                                // Check if Ctrl+Enter or Cmd+Enter (for Mac) was pressed
                                if ((e.ctrlKey || e.metaKey) && e.keyCode === 13) {{
                                    // Find the send button
                                    const sendButton = document.querySelector('button[kind="secondary"]:has-text("➤")') ||
                                                      document.querySelector('button:contains("➤")') ||
                                                      document.querySelector('button[data-testid="stButton"]');
                                    if (sendButton) {{
                                        // Click the send button
                                        sendButton.click();
                                        e.preventDefault();
                                    }}
                                }}
                            }});
                        }});
                    </script>
                    """, unsafe_allow_html=True)

                with upload_col:
                    # Add a paperclip icon for the file uploader
                    st.markdown('<div style="display: flex; justify-content: center; align-items: center; height: 40px;">📎</div>', unsafe_allow_html=True)
                    uploaded_file = st.file_uploader("", type=["pdf", "txt"], key=f"file_uploader_{current_chat['id']}", label_visibility="collapsed")

                with send_col:
                    # Use an icon for the send button
                    send_button = st.button("➤", key=f"send_button_{current_chat['id']}")

                st.markdown('</div>', unsafe_allow_html=True)

                # Process message when send button is clicked
                if send_button and (message_input or uploaded_file):
                    # Process uploaded file if any
                    document_id = None
                    if uploaded_file:
                        # Save the file
                        file_path = os.path.join("uploads", uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                        # Save document metadata
                        document_id = db.save_document(
                            filename=uploaded_file.name,
                            file_path=file_path,
                            uploader_id=current_user_id,
                            file_type=uploaded_file.type,
                            file_size=uploaded_file.size
                        )

                        # Process document for graph database
                        _, message = graph_integration.process_document(file_path, uploaded_file.name, current_user_id)

                        # Add a message about the document
                        if message_input:
                            document_message = message_input
                        else:
                            document_message = f"Shared a document: {uploaded_file.name}"

                        # Add message to chat
                        db.add_message(
                            chat_id=current_chat['id'],
                            sender_id=current_user_id,
                            content=document_message,
                            message_type='document',
                            document_id=document_id
                        )

                    # Process text message if any
                    elif message_input:
                        # Add message to chat
                        db.add_message(
                            chat_id=current_chat['id'],
                            sender_id=current_user_id,
                            content=message_input
                        )

                        # Store message in Neo4j
                        graph_integration.store_message_in_neo4j(
                            message_content=message_input,
                            sender_id=current_user_id,
                            chat_id=current_chat['id']
                        )

                        # Only generate AI response for AI chat
                        if is_ai_chat:
                            # Show a loading message
                            with st.spinner("AI is thinking..."):
                                # Query the graph database
                                graph_results = graph_integration.query_graph(message_input)

                                # Generate a response
                                answer, _ = graph_integration.generate_groq_response(message_input, graph_results)

                                # Add AI response to chat
                                db.add_message(
                                    chat_id=current_chat['id'],
                                    sender_id="ai",
                                    content=answer
                                )

                    # Rerun to update the UI
                    time.sleep(0.1)  # Small delay to ensure database writes complete
                    st.rerun()

                # JavaScript to scroll to bottom of chat
                st.markdown("""
                <script>
                    function scrollToBottom() {
                        const chatContainer = document.getElementById('chat-container');
                        if (chatContainer) {
                            chatContainer.scrollTop = chatContainer.scrollHeight;
                        }
                    }

                    // Scroll when page loads
                    window.addEventListener('load', scrollToBottom);

                    // Also try scrolling after a short delay (for dynamic content)
                    setTimeout(scrollToBottom, 200);

                    // Add event listener for Enter key in text area
                    document.addEventListener('DOMContentLoaded', function() {
                        const textArea = document.querySelector('textarea');
                        if (textArea) {
                            textArea.addEventListener('keydown', function(e) {
                                if (e.key === 'Enter' && !e.shiftKey) {
                                    e.preventDefault();
                                    const sendButton = document.querySelector('button[data-testid="stButton"]');
                                    if (sendButton) {
                                        sendButton.click();
                                    }
                                }
                            });
                        }
                    });
                </script>
                """, unsafe_allow_html=True)
            else:
                st.warning("Selected chat not found.")
        else:
            # Display welcome message when no chat is selected
            st.markdown("""
            <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 80vh; text-align: center;">
                <h2>Welcome to SAGE Chat</h2>
                <p>Select a chat from the list or create a new one to start messaging.</p>
                <p>The AI Assistant chat is always available to answer your questions about the knowledge base.</p>
            </div>
            """, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()
