"""
chat_database.py
This file is the database for the ai tutor.
It handles the database management for the chat sessions and messages.

the parts of code not written by me are referenced from the following sources:
- code to handle all the database operations from https://www.sqlitetutorial.net/sqlite-python/
"""


import sqlite3
from datetime import datetime, timezone

class ChatDatabase:
    """sqlite database for chat sessions and messages
       each session has many messages
       if all sessions are deleted then autocreate an empty default session
    """

    def __init__(self, path="chat_sessions.sqlite3"):
        self.path = path
        # allow use from multiple threads to work with gradio
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self._init_schema()

    # core database schema setup
    def _init_schema(self):
        try:
            connection = self.conn
            cursor = connection.cursor()
            # create sessions table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """)
            # create messages table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                translated_content TEXT,  
                created_at TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );
            """)
           
            # create indexes to speed up data retrieval
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at DESC);")

            cursor.close()
        except Exception as e:
            print(e)

        # ensure at least one empty chat exists
        self.default_session()

    # helper functions to manage chat sessions
    def time_now(self):
        # get current utc time as iso string till seconds
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def default_session(self):
        # if there are no sessions then create a new chat
        try:
            connection = self.conn
            cursor = connection.cursor()
            cursor.execute("SELECT COUNT(*) FROM sessions")
            for result in cursor.fetchall():
                count = result[0]
                if count == 0:
                    now = self.time_now()
                    cursor.execute(
                        "INSERT INTO sessions(name, created_at, updated_at) VALUES (?,?,?)",
                        ("New Chat", now, now),
                    )
                    connection.commit()
            cursor.close()
        except Exception as e:
            print(e)

    def create_session(self, name):
        # create a new chat session
        now = self.time_now()
        try:
            connection = self.conn
            cursor = connection.cursor()
            cursor.execute(
                "INSERT INTO sessions(name, created_at, updated_at) VALUES (?,?,?)",
                (name, now, now),
            )
            connection.commit()
            session_id = cursor.lastrowid
            cursor.close()
            return session_id
        except Exception as e:
            print(f"Error creating session: {e}")
            return None

    def list_sessions(self):
        # return all sessions with newest updated first
        # order by updated_at and id both to ensure the newest session is first
        try:
            connection = self.conn
            cursor = connection.cursor()
            cursor.execute("""
            SELECT id, name, created_at, updated_at
            FROM sessions
            ORDER BY updated_at DESC, id DESC
            """)
            results = []
            for result in cursor.fetchall():
                results.append({
                    "id": result[0],
                    "name": result[1], 
                    "created_at": result[2],
                    "updated_at": result[3]
                })
            cursor.close()
            return results
        except Exception as e:
            print(e)
            return []

    def latest_session_id(self):
        # get the most recently updated session id
        try:
            connection = self.conn
            cursor = connection.cursor()
            cursor.execute(
                "SELECT id FROM sessions ORDER BY updated_at DESC, id DESC LIMIT 1"
            )
            for result in cursor.fetchall():
                cursor.close()
                return int(result[0])
            cursor.close()
        except Exception as e:
            print(e)
        

    def rename_session(self, session_id, name):
        # rename a session
        try:
            connection = self.conn
            cursor = connection.cursor()
            cursor.execute(
                "UPDATE sessions SET name=?, updated_at=? WHERE id=?",
                (name, self.time_now(), session_id),
            )
            connection.commit()
            cursor.close()
        except Exception as e:
            print(e)

    def delete_session(self, session_id):
        # delete a session and all its messages
        try:
            connection = self.conn
            cursor = connection.cursor()
            cursor.execute("DELETE FROM sessions WHERE id=?", (session_id,))
            connection.commit()
            cursor.close()
            # recreate an empty chat if nothing remains
            self.default_session()
            # return the latest session id after deletion
            return self.latest_session_id()
            
        except Exception as e:
            print(e)
            return None

    def add_message(self, session_id, role, content, translated_content=None):
        # add a message to a session with optional translated content
        now = self.time_now()
        try:
            connection = self.conn
            cursor = connection.cursor()
            cursor.execute(
                "INSERT INTO messages(session_id, role, content, translated_content, created_at) VALUES (?,?,?,?,?)",
                (session_id, role, content, translated_content, now),
            )
            # update session's updated_at so it sorts to the top
            cursor.execute(
                "UPDATE sessions SET updated_at=? WHERE id=?", (now, session_id)
            )
            connection.commit()
            cursor.close() 
        except Exception as e:
            print(e)

    def get_messages(self, session_id):
        # get all messages with oldest first
        # returns a list of message dictionaries
        try:
            connection = self.conn
            cursor = connection.cursor()
            cursor.execute(
                "SELECT role, content, created_at, translated_content FROM messages WHERE session_id=? ORDER BY id ASC",
                (session_id,)
            )
            
            results = [] 
            for result in cursor.fetchall():
                results.append({
                    "role": result[0],
                    "content": result[1],
                    "created_at": result[2],
                    "translated_content": result[3],
                })
            cursor.close()
            return results

        except Exception as e:
            print(e)
            return []
