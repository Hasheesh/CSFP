"""session_manager.py

This file handles session management functionality for the AI tutor.
It manages chat sessions, including creation, deletion, renaming, and history loading.

"""
from chat_database import ChatDatabase


class SessionManager:
    """Enhanced session manager that handles all session operations for the UI."""
    
    def __init__(self, db_path="chat_sessions.sqlite3"):
        """Initializes the SessionManager and connects to the database."""
        self.database = ChatDatabase(db_path)
        self._current_session_id = None
        
        # initialize with the latest session
        self._initialize_current_session()

    def _initialize_current_session(self):
        """Initializes the current session with the latest session from the database."""
        latest_session_id = self.database.latest_session_id()
        if latest_session_id:
            self._current_session_id = latest_session_id

    def _load_chat_history(self, session_id):
        """Loads chat history from the database."""
        return self.database.get_messages(session_id)

    def generate_title(self, session_id, user_content):
        """Generates a title from the first user message if the session is still named 'New Chat'."""
        try:
            # get current session name
            sessions = self.database.list_sessions()
            current_session = next((s for s in sessions if s['id'] == session_id), None)
            
            if current_session and current_session['name'] == "New Chat":
                # generate title from user content
                title = self._extract_title(user_content)
                if title and title != "New Chat":
                    self.database.rename_session(session_id, title)
        except Exception as e:
            print(f"Error generating title: {e}")

    def _extract_title(self, content):
        """Extracts a meaningful title from the user's content."""
        if not content or content.strip() == "[No input]":
            return "New Chat"
        
        # clean the content
        cleaned = content.strip()
        
        # remove common prefixes
        prefixes_to_remove = [
            "[Image OCR]\n", "[Audio]\n", "User input:\n"
        ]
        for prefix in prefixes_to_remove:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        
        # extract first 50 characters
        title = cleaned[:50] + "..." if len(cleaned) > 50 else cleaned
        
        # clean up title
        title = title.replace("\n", " ").replace("\t", " ")
        title = " ".join(title.split())  # remove extra whitespace
        
        # ensure minimum length
        if len(title) < 3:
            return "New Chat"
        
        return title

    def add_message_to_database(self, session_id, role, content, translated_content=None):
        """Adds a message to the database with optional translated content."""
        self.database.add_message(session_id, role, content, translated_content)
        
        # auto-generate title from first user message
        if role == "user":
            self.generate_title(session_id, content)

    def get_current_session_id(self):
        """Gets the current session ID."""
        return self._current_session_id

    def set_current_session_id(self, session_id):
        """Sets the current session ID."""
        self._current_session_id = session_id

    def list_session_choices(self):
        """Returns a list of session choices formatted for the Gradio radio component."""
        sessions = self.database.list_sessions()
        return [[f"{s['id']} - {s['name']}", s['id']] for s in sessions]

    def new_session(self):
        """Creates a new, empty chat session in the database."""
        session_id = self.database.create_session("New Chat")
        # set as current session
        self._current_session_id = session_id
        return session_id

    def delete_session(self, session_id):
        """Deletes the selected chat session and returns the new active session ID."""
        self.database.delete_session(session_id)
        # this function also ensures that if it was the last, a new empty chat is created
        new_session_id = self.database.latest_session_id()
        # update current session
        self._current_session_id = new_session_id
        return new_session_id

    def rename_session(self, session_id, new_name):
        """Manually renames a session."""
        try:
            self.database.rename_session(session_id, new_name)
            return True
        except Exception as e:
            print(f"Error renaming session: {e}")
            return False

    def get_history(self, session_id, current_lang="en"):
        """Returns the chat history formatted for the Gradio chatbot display."""
        messages = self._load_chat_history(session_id)
        display_messages = []
        
        for msg in messages:
            role = msg["role"]
            # default to the 'content' column (which is always English)
            content_to_display = msg["content"]
            
            if current_lang == 'ur':
                # for Urdu UI, if a translated version exists, use it.
                # this applies to both 'user' (original Urdu) and 'assistant' (translated Urdu) roles.
                if msg.get("translated_content"):
                    content_to_display = msg["translated_content"]
            
            display_messages.append({"role": role, "content": content_to_display})
            
        return display_messages

    def get_history_for_llm(self, session_id):
        """Returns the chat history for the LLM's context, always in the original English format."""
        messages = self._load_chat_history(session_id)
        # ensure we only pass the original english content to the LLM
        return [{"role": msg["role"], "content": msg["content"]} for msg in messages]