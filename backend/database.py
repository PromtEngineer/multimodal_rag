import sqlite3
import uuid
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple

class ChatDatabase:
    def __init__(self, db_path: str = "chat_history.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        
        # Enable foreign keys
        conn.execute("PRAGMA foreign_keys = ON")
        
        # Sessions table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                model_used TEXT NOT NULL,
                message_count INTEGER DEFAULT 0
            )
        ''')
        
        # Messages table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                content TEXT NOT NULL,
                sender TEXT NOT NULL CHECK (sender IN ('user', 'assistant')),
                timestamp TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
            )
        ''')
        
        # Create indexes for better performance
        conn.execute('CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at)')
        
        conn.commit()
        conn.close()
        print("✅ Database initialized successfully")
    
    def create_session(self, title: str, model: str) -> str:
        """Create a new chat session"""
        session_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            INSERT INTO sessions (id, title, created_at, updated_at, model_used)
            VALUES (?, ?, ?, ?, ?)
        ''', (session_id, title, now, now, model))
        conn.commit()
        conn.close()
        
        print(f"📝 Created new session: {session_id[:8]}... - {title}")
        return session_id
    
    def get_sessions(self, limit: int = 50) -> List[Dict]:
        """Get all chat sessions, ordered by most recent"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        cursor = conn.execute('''
            SELECT id, title, created_at, updated_at, model_used, message_count
            FROM sessions
            ORDER BY updated_at DESC
            LIMIT ?
        ''', (limit,))
        
        sessions = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return sessions
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get a specific session"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        cursor = conn.execute('''
            SELECT id, title, created_at, updated_at, model_used, message_count
            FROM sessions
            WHERE id = ?
        ''', (session_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        return dict(row) if row else None
    
    def add_message(self, session_id: str, content: str, sender: str, metadata: Dict = None) -> str:
        """Add a message to a session"""
        message_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        metadata_json = json.dumps(metadata or {})
        
        conn = sqlite3.connect(self.db_path)
        
        # Add the message
        conn.execute('''
            INSERT INTO messages (id, session_id, content, sender, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (message_id, session_id, content, sender, now, metadata_json))
        
        # Update session timestamp and message count
        conn.execute('''
            UPDATE sessions 
            SET updated_at = ?, 
                message_count = message_count + 1
            WHERE id = ?
        ''', (now, session_id))
        
        conn.commit()
        conn.close()
        
        return message_id
    
    def get_messages(self, session_id: str, limit: int = 100) -> List[Dict]:
        """Get all messages for a session"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        cursor = conn.execute('''
            SELECT id, content, sender, timestamp, metadata
            FROM messages
            WHERE session_id = ?
            ORDER BY timestamp ASC
            LIMIT ?
        ''', (session_id, limit))
        
        messages = []
        for row in cursor.fetchall():
            message = dict(row)
            message['metadata'] = json.loads(message['metadata'])
            messages.append(message)
        
        conn.close()
        return messages
    
    def get_conversation_history(self, session_id: str) -> List[Dict]:
        """Get conversation history in the format expected by Ollama"""
        messages = self.get_messages(session_id)
        
        history = []
        for msg in messages:
            history.append({
                "role": msg["sender"],
                "content": msg["content"]
            })
        
        return history
    
    def update_session_title(self, session_id: str, title: str):
        """Update session title"""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            UPDATE sessions 
            SET title = ?, updated_at = ?
            WHERE id = ?
        ''', (title, datetime.now().isoformat(), session_id))
        conn.commit()
        conn.close()
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its messages"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute('DELETE FROM sessions WHERE id = ?', (session_id,))
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        if deleted:
            print(f"🗑️ Deleted session: {session_id[:8]}...")
        
        return deleted
    
    def cleanup_empty_sessions(self) -> int:
        """Remove sessions with no messages"""
        conn = sqlite3.connect(self.db_path)
        
        # Find sessions with no messages
        cursor = conn.execute('''
            SELECT s.id FROM sessions s
            LEFT JOIN messages m ON s.id = m.session_id
            WHERE m.id IS NULL
        ''')
        
        empty_sessions = [row[0] for row in cursor.fetchall()]
        
        # Delete empty sessions
        deleted_count = 0
        for session_id in empty_sessions:
            cursor = conn.execute('DELETE FROM sessions WHERE id = ?', (session_id,))
            if cursor.rowcount > 0:
                deleted_count += 1
                print(f"🗑️ Cleaned up empty session: {session_id[:8]}...")
        
        conn.commit()
        conn.close()
        
        if deleted_count > 0:
            print(f"✨ Cleaned up {deleted_count} empty sessions")
        
        return deleted_count
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        conn = sqlite3.connect(self.db_path)
        
        # Get session count
        cursor = conn.execute('SELECT COUNT(*) FROM sessions')
        session_count = cursor.fetchone()[0]
        
        # Get message count
        cursor = conn.execute('SELECT COUNT(*) FROM messages')
        message_count = cursor.fetchone()[0]
        
        # Get most used model
        cursor = conn.execute('''
            SELECT model_used, COUNT(*) as count
            FROM sessions
            GROUP BY model_used
            ORDER BY count DESC
            LIMIT 1
        ''')
        most_used_model = cursor.fetchone()
        
        conn.close()
        
        return {
            "total_sessions": session_count,
            "total_messages": message_count,
            "most_used_model": most_used_model[0] if most_used_model else None
        }

def generate_session_title(first_message: str, max_length: int = 50) -> str:
    """Generate a session title from the first message"""
    # Clean up the message
    title = first_message.strip()
    
    # Remove common prefixes
    prefixes = ["hey", "hi", "hello", "can you", "please", "i want", "i need"]
    title_lower = title.lower()
    for prefix in prefixes:
        if title_lower.startswith(prefix):
            title = title[len(prefix):].strip()
            break
    
    # Capitalize first letter
    if title:
        title = title[0].upper() + title[1:]
    
    # Truncate if too long
    if len(title) > max_length:
        title = title[:max_length].strip() + "..."
    
    # Fallback
    if not title or len(title) < 3:
        title = "New Chat"
    
    return title

# Global database instance
db = ChatDatabase()

if __name__ == "__main__":
    # Test the database
    print("🧪 Testing database...")
    
    # Create a test session
    session_id = db.create_session("Test Chat", "llama3.2:latest")
    
    # Add some messages
    db.add_message(session_id, "Hello!", "user")
    db.add_message(session_id, "Hi there! How can I help you?", "assistant")
    
    # Get messages
    messages = db.get_messages(session_id)
    print(f"📨 Messages: {len(messages)}")
    
    # Get sessions
    sessions = db.get_sessions()
    print(f"📋 Sessions: {len(sessions)}")
    
    # Get stats
    stats = db.get_stats()
    print(f"📊 Stats: {stats}")
    
    print("✅ Database test completed!") 