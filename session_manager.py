"""session_manager.py

This file handles session management functionality for the AI tutor.
It manages chat sessions, including creation, deletion, renaming, and history loading.

"""

import gradio as gr
from chat_database import ChatDatabase


class SessionManager:
    # enhanced session manager that handles all session operations for the ui
    
    def __init__(self, db_path="chat_sessions.sqlite3"):
        self.database = ChatDatabase(db_path)
        self._current_session_id = None
        
        # initialize with the latest session
        self._initialize_current_session()

    def _initialize_current_session(self):
        # initialize the current session with the latest session from the database
        latest_session_id = self.database.latest_session_id()
        if latest_session_id:
            self._current_session_id = latest_session_id

    def _load_chat_history(self, session_id):
        # load chat history from database
        return self.database.get_messages(session_id)

    def generate_title(self, session_id, user_content):
        # generate a title from the first user message if session is still 'new chat'
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
        # extract a meaningful title from user content
        if not content or content.strip() == "[No input]":
            return "New Chat"
        
        # clean the content
        cleaned = content.strip()
        
        # remove common prefixes
        prefixes_to_remove = [
            "[Image OCR]\n", "[Audio]\n", "Student's question:\n"
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

    def add_message_to_database(self, session_id, role, content):
        # add a message to database
        self.database.add_message(session_id, role, content)
        
        # auto-generate title from first user message
        if role == "user":
            self.generate_title(session_id, content)

    def get_current_session_id(self):
        # get the current session id
        return self._current_session_id

    def set_current_session_id(self, session_id):
        # set the current session id
        self._current_session_id = session_id

    # session operations used by ui events
    def list_session_choices(self):
        # returns session choices for gradio radio
        # returns list of [label, value] lists for maximum raspberry pi performance
        sessions = self.database.list_sessions()
        return [[f"{s['id']} - {s['name']}", s['id']] for s in sessions]

    def new_session(self):
        # creates a new empty chat in database
        session_id = self.database.create_session("New Chat")
        if session_id is None:
            # fallback to latest session if creation failed
            session_id = self.database.latest_session_id()
        # set as current session
        self._current_session_id = session_id
        return session_id

    def delete_session(self, session_id):
        # deletes selected chat and returns new active session
        self.database.delete_session(session_id)
        # this function also ensures that if it was the last, a new empty chat is created
        new_session_id = self.database.latest_session_id()
        # update current session
        self._current_session_id = new_session_id
        return new_session_id

    def rename_session(self, session_id, new_name):
        # manually rename a session
        try:
            self.database.rename_session(session_id, new_name)
            return True
        except Exception as e:
            print(f"Error renaming session: {e}")
            return False

    def get_history(self, session_id, current_lang="en", translator=None):
        # returns history for gradio chatbot display in messages format
        messages = self._load_chat_history(session_id)
        
        # translate assistant messages if current language is urdu
        if current_lang == 'ur' and translator:
            translated_messages = []
            for msg in messages:
                if msg["role"] == "assistant":
                    # translate assistant messages from english to urdu
                    translated_content = translator.process_input(msg["content"], "en")
                    # convert to urdu numerals for display (not words, just numerals ۱۲۳)
                    # wrap with rtl html attributes
                    rtl_content = f'<div dir="rtl" lang="ur">{translated_content}</div>'
                    translated_messages.append({"role": msg["role"], "content": rtl_content})
                else:
                    # keep user messages as-is (they're already in original language)
                    # convert any numbers in user messages to urdu numerals for consistency
                    user_content = msg["content"]
                    translated_messages.append({"role": msg["role"], "content": user_content})
            return translated_messages
        else:
            return [{"role": msg["role"], "content": msg["content"]} for msg in messages]