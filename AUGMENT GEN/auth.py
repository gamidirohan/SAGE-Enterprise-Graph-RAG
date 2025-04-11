import streamlit as st
import db

def init_session_state():
    """Initialize session state variables for authentication"""
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'is_authenticated' not in st.session_state:
        st.session_state.is_authenticated = False
    if 'current_chat_id' not in st.session_state:
        st.session_state.current_chat_id = None

def login_user(username, password):
    """Log in a user"""
    user_id = db.verify_user(username, password)
    if user_id:
        st.session_state.user_id = user_id
        st.session_state.username = username
        st.session_state.is_authenticated = True
        
        # Get user's chats
        chats = db.get_user_chats(user_id)
        
        # Set current chat to AI chat if it exists
        ai_chat = next((chat for chat in chats if chat['type'] == 'ai'), None)
        if ai_chat:
            st.session_state.current_chat_id = ai_chat['id']
        elif chats:
            st.session_state.current_chat_id = chats[0]['id']
        
        return True
    return False

def register_user(username, password, confirm_password):
    """Register a new user"""
    if password != confirm_password:
        return False, "Passwords do not match"
    
    if len(password) < 6:
        return False, "Password must be at least 6 characters"
    
    user_id = db.add_user(username, password)
    if user_id:
        # Automatically log in the user
        st.session_state.user_id = user_id
        st.session_state.username = username
        st.session_state.is_authenticated = True
        
        # Get user's chats (should only be the AI chat at this point)
        chats = db.get_user_chats(user_id)
        if chats:
            st.session_state.current_chat_id = chats[0]['id']
        
        return True, "Registration successful"
    
    return False, "Username already exists"

def logout_user():
    """Log out the current user"""
    st.session_state.user_id = None
    st.session_state.username = None
    st.session_state.is_authenticated = False
    st.session_state.current_chat_id = None

def is_authenticated():
    """Check if a user is authenticated"""
    return st.session_state.is_authenticated

def get_current_user():
    """Get the current user"""
    if is_authenticated():
        return {
            'id': st.session_state.user_id,
            'username': st.session_state.username
        }
    return None

def auth_required(func):
    """Decorator to require authentication for a function"""
    def wrapper(*args, **kwargs):
        if is_authenticated():
            return func(*args, **kwargs)
        else:
            st.warning("Please log in to access this feature")
            return None
    return wrapper

def render_login_page():
    """Render the login page"""
    st.title("Login")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Login")
            
            if submit_button:
                if username and password:
                    if login_user(username, password):
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
                else:
                    st.error("Please enter both username and password")
    
    with tab2:
        with st.form("register_form"):
            new_username = st.text_input("Username")
            new_password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            submit_button = st.form_submit_button("Register")
            
            if submit_button:
                if new_username and new_password and confirm_password:
                    success, message = register_user(new_username, new_password, confirm_password)
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.error("Please fill in all fields")
