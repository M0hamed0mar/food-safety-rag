"""
Chat schema definitions for multi-turn conversations.

This module defines Pydantic models for chat sessions, messages,
and conversation history management.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """
    A single message in a chat conversation.
    
    Attributes:
        role: The role of the message sender (user or assistant).
        content: The message content.
        timestamp: When the message was sent.
    """
    
    role: str = Field(
        ..., 
        description="Role of the message sender (user or assistant)",
        pattern="^(user|assistant)$"
    )
    content: str = Field(
        ..., 
        description="Message content",
        min_length=1
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the message was sent"
    )
    
    def to_dict(self) -> dict[str, str]:
        """Convert to a dictionary for LLM context."""
        return {
            "role": self.role,
            "content": self.content
        }


class ChatSession(BaseModel):
    """
    A complete chat session with message history.
    
    Attributes:
        session_id: Unique identifier for the chat session.
        title: User-friendly title for the chat.
        messages: List of messages in the conversation.
        created_at: When the chat was created.
        updated_at: When the chat was last updated.
    """
    
    session_id: str = Field(
        ..., 
        description="Unique identifier for the chat session"
    )
    title: str = Field(
        default="New Chat",
        description="User-friendly title for the chat"
    )
    messages: list[ChatMessage] = Field(
        default_factory=list,
        description="List of messages in the conversation"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the chat was created"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the chat was last updated"
    )
    
    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the chat session.
        
        Args:
            role: The role (user or assistant).
            content: The message content.
        """
        self.messages.append(ChatMessage(role=role, content=content))
        self.updated_at = datetime.utcnow()
    
    def get_last_n_messages(self, n: int = 3) -> list[ChatMessage]:
        """
        Get the last N messages (user + assistant pairs).
        
        Args:
            n: Number of recent messages to return.
        
        Returns:
            list[ChatMessage]: The most recent messages.
        """
        return self.messages[-n:] if self.messages else []
    
    def get_context_messages(self, n: int = 3) -> list[dict[str, str]]:
        """
        Get formatted messages for LLM context.
        
        Args:
            n: Number of recent messages to include.
        
        Returns:
            list[dict[str, str]]: List of message dicts for context.
        """
        messages = self.get_last_n_messages(n)
        return [m.to_dict() for m in messages]
    
    def get_conversation_history_text(self, n: int = 3) -> str:
        """
        Get formatted conversation history as text.
        
        Args:
            n: Number of recent messages to include.
        
        Returns:
            str: Formatted conversation history.
        """
        messages = self.get_last_n_messages(n)
        if not messages:
            return ""
        
        lines = ["Previous conversation:"]
        for msg in messages:
            role = "User" if msg.role == "user" else "Assistant"
            lines.append(f"{role}: {msg.content}")
        
        return "\n".join(lines)


class ChatCreateRequest(BaseModel):
    """
    Request model for creating a new chat.
    
    Attributes:
        title: Optional title for the chat.
    """
    
    title: Optional[str] = Field(
        default="New Chat",
        description="Title for the chat"
    )


class ChatListResponse(BaseModel):
    """
    Response model for listing chats.
    
    Attributes:
        session_id: Unique identifier for the chat session.
        title: Chat title.
        message_count: Number of messages in the chat.
        created_at: When the chat was created.
        updated_at: When the chat was last updated.
    """
    
    session_id: str
    title: str
    message_count: int
    created_at: datetime
    updated_at: datetime