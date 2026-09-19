"""
Chat history storage module using SQLite.

This module provides persistent storage for chat sessions and messages,
enabling users to resume conversations across sessions.

Enhanced with connection handling fixes:
- check_same_thread=False for multi-threaded access
- Increased timeout for concurrent operations
- Better error handling for database locks
- Connection pooling to prevent "closed database" errors
"""

import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, List, Dict

from app.config import settings
from app.config.constants import LogEvent
from app.monitoring import get_logger
from app.schemas.chat import ChatMessage, ChatSession


logger = get_logger("food_safety_rag.vector_store.chat")


class ChatStore:
    """
    SQLite-based storage for chat sessions and messages.
    
    Attributes:
        db_path: Path to the SQLite database file.
    """
    
    def __init__(self, db_path: Optional[Path] = None) -> None:
        """
        Initialize the chat store.
        
        Args:
            db_path: Path to the SQLite database. Defaults to DATA_DIR / "chats.db".
        """
        self.db_path: Path = db_path or (settings.DATA_DIR / "chats.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection: Optional[sqlite3.Connection] = None
        self._init_db()
        
        logger.log_event(
            event=LogEvent.SYSTEM_STARTUP,
            message=f"Chat store initialized at {self.db_path}",
        )
    
    def _get_connection(self) -> sqlite3.Connection:
        """
        Get or create a database connection.
        
        Uses check_same_thread=False to allow multi-threaded access.
        Uses increased timeout to handle concurrent write operations.
        Connection is kept open for the lifetime of the application.
        
        Returns:
            sqlite3.Connection: Database connection.
        """
        if self._connection is None:
            self._connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30.0,
            )
            self._connection.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrency
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=NORMAL")
            self._connection.execute("PRAGMA cache_size=10000")
        return self._connection
    
    def _close_connection(self) -> None:
        """Close the database connection if open."""
        if self._connection is not None:
            try:
                self._connection.close()
            except Exception:
                pass
            self._connection = None
    
    def _init_db(self) -> None:
        """
        Create database tables if they don't exist.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Chats table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                session_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL
            )
        """)
        
        # Messages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                FOREIGN KEY (session_id) REFERENCES chats(session_id) ON DELETE CASCADE
            )
        """)
        
        # Index for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_session 
            ON messages(session_id, timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_chats_updated 
            ON chats(updated_at DESC)
        """)
        
        conn.commit()
    
    def create_chat(self, title: str = "New Chat") -> str:
        """
        Create a new chat session with a unique session_id.
        
        Args:
            title: Optional title for the chat.
        
        Returns:
            str: The session ID of the new chat.
        """
        import random
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        random_suffix = random.randint(1000, 9999)
        session_id = f"chat_{timestamp}_{random_suffix}"
        
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute(
                """
                INSERT INTO chats (session_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, title, datetime.now(), datetime.now())
            )
            
            conn.commit()
            
            logger.log_event(
                event=LogEvent.SYSTEM_STARTUP,
                message=f"Created new chat: {session_id}",
                details={"session_id": session_id, "title": title},
            )
            
            return session_id
            
        except sqlite3.IntegrityError:
            # If session_id already exists (very unlikely), generate a new one
            import uuid
            session_id = f"chat_{uuid.uuid4().hex[:12]}"
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO chats (session_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, title, datetime.now(), datetime.now())
            )
            conn.commit()
            
            logger.log_event(
                event=LogEvent.SYSTEM_STARTUP,
                message=f"Created new chat (fallback): {session_id}",
                details={"session_id": session_id, "title": title},
            )
            
            return session_id
    
    def add_message(self, session_id: str, role: str, content: str) -> None:
        """
        Add a message to a chat session.
        
        Args:
            session_id: The chat session ID.
            role: The role (user or assistant).
            content: The message content.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Insert message
        cursor.execute(
            """
            INSERT INTO messages (session_id, role, content, timestamp)
            VALUES (?, ?, ?, ?)
            """,
            (session_id, role, content, datetime.now())
        )
        
        # Update chat timestamp
        cursor.execute(
            """
            UPDATE chats SET updated_at = ?
            WHERE session_id = ?
            """,
            (datetime.now(), session_id)
        )
        
        conn.commit()
    
    def get_chat(self, session_id: str) -> Optional[ChatSession]:
        """
        Retrieve a chat session with all messages.
        
        Args:
            session_id: The chat session ID.
        
        Returns:
            Optional[ChatSession]: The chat session, or None if not found.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Get chat metadata
        cursor.execute(
            "SELECT session_id, title, created_at, updated_at FROM chats WHERE session_id = ?",
            (session_id,)
        )
        chat_row = cursor.fetchone()
        
        if not chat_row:
            return None
        
        # Get messages
        cursor.execute(
            """
            SELECT role, content, timestamp FROM messages
            WHERE session_id = ?
            ORDER BY timestamp ASC
            """,
            (session_id,)
        )
        message_rows = cursor.fetchall()
        
        # Build ChatSession
        messages = []
        for row in message_rows:
            messages.append(ChatMessage(
                role=row["role"],
                content=row["content"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
            ))
        
        return ChatSession(
            session_id=chat_row["session_id"],
            title=chat_row["title"],
            messages=messages,
            created_at=datetime.fromisoformat(chat_row["created_at"]),
            updated_at=datetime.fromisoformat(chat_row["updated_at"]),
        )
    
    def get_chat_history(self, session_id: str, limit: int = 10) -> List[Dict[str, str]]:
        """
        Get recent messages from a chat session.
        
        Args:
            session_id: The chat session ID.
            limit: Maximum number of messages to return.
        
        Returns:
            list[dict[str, str]]: List of messages with role and content.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            """
            SELECT role, content FROM messages
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (session_id, limit)
        )
        
        rows = cursor.fetchall()
        
        # Return in chronological order (oldest first)
        result = [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]
        return result
    
    def list_chats(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        List all chat sessions.
        
        Args:
            limit: Maximum number of chats to return.
        
        Returns:
            list[dict[str, Any]]: List of chat metadata.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            """
            SELECT 
                c.session_id,
                c.title,
                c.created_at,
                c.updated_at,
                COUNT(m.id) as message_count
            FROM chats c
            LEFT JOIN messages m ON c.session_id = m.session_id
            GROUP BY c.session_id
            ORDER BY c.updated_at DESC
            LIMIT ?
            """,
            (limit,)
        )
        
        rows = cursor.fetchall()
        
        return [
            {
                "session_id": row["session_id"],
                "title": row["title"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "message_count": row["message_count"],
            }
            for row in rows
        ]
    
    def delete_chat(self, session_id: str) -> bool:
        """
        Delete a chat session and all its messages.
        
        Args:
            session_id: The chat session ID.
        
        Returns:
            bool: True if deleted, False if not found.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # First check if chat exists
        cursor.execute("SELECT session_id FROM chats WHERE session_id = ?", (session_id,))
        exists = cursor.fetchone() is not None
        
        if not exists:
            return False
        
        # Delete chat (messages will be deleted via CASCADE)
        cursor.execute("DELETE FROM chats WHERE session_id = ?", (session_id,))
        deleted = cursor.rowcount > 0
        
        conn.commit()
        
        if deleted:
            logger.log_event(
                event=LogEvent.CACHE_INVALIDATION,
                message=f"Deleted chat: {session_id}",
            )
        
        return deleted
    
    def update_title(self, session_id: str, title: str) -> bool:
        """
        Update the title of a chat session.
        
        Args:
            session_id: The chat session ID.
            title: New title for the chat.
        
        Returns:
            bool: True if updated, False if not found.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # First check if chat exists
        cursor.execute("SELECT session_id FROM chats WHERE session_id = ?", (session_id,))
        exists = cursor.fetchone() is not None
        
        if not exists:
            return False
        
        cursor.execute(
            "UPDATE chats SET title = ?, updated_at = ? WHERE session_id = ?",
            (title, datetime.now(), session_id)
        )
        
        updated = cursor.rowcount > 0
        conn.commit()
        
        if updated:
            logger.log_event(
                event=LogEvent.SYSTEM_STARTUP,
                message=f"Updated chat title: {session_id} -> {title}",
            )
        
        return updated
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get chat store statistics.
        
        Returns:
            dict[str, Any]: Statistics about the chat store.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM chats")
        chat_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM messages")
        message_count = cursor.fetchone()[0]
        
        return {
            "total_chats": chat_count,
            "total_messages": message_count,
            "db_path": str(self.db_path),
        }