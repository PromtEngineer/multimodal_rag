import uuid
import json
from datetime import datetime
from typing import List, Dict, Optional
try:
    from .database_manager import DatabaseManager
except ImportError:
    from database_manager import DatabaseManager
import logging

logger = logging.getLogger(__name__)

class EnhancedChatDatabase:
    """Enhanced database class using connection pooling and proper transaction management"""
    
    def __init__(self, db_path: str = "chat_history.db"):
        self.db_manager = DatabaseManager(db_path)
    
    # =============================================
    # Session Management
    # =============================================
    
    def create_session(self, title: str, model: str) -> str:
        """Create a new chat session"""
        session_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        self.db_manager.execute_update('''
            INSERT INTO sessions (id, title, created_at, updated_at, model_used)
            VALUES (?, ?, ?, ?, ?)
        ''', (session_id, title, now, now, model))
        
        logger.info(f"📝 Created new session: {session_id[:8]}... - {title}")
        return session_id
    
    def get_sessions(self, limit: int = 50) -> List[Dict]:
        """Get all chat sessions, ordered by most recent"""
        rows = self.db_manager.execute_query('''
            SELECT id, title, created_at, updated_at, model_used, message_count
            FROM sessions
            ORDER BY updated_at DESC
            LIMIT ?
        ''', (limit,), fetch_all=True)
        
        return [dict(row) for row in rows] if rows else []
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get a specific session"""
        row = self.db_manager.execute_query('''
            SELECT id, title, created_at, updated_at, model_used, message_count
            FROM sessions
            WHERE id = ?
        ''', (session_id,), fetch_one=True)
        
        return dict(row) if row else None
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its messages (cascade)"""
        affected = self.db_manager.execute_update('''
            DELETE FROM sessions WHERE id = ?
        ''', (session_id,))
        
        if affected > 0:
            logger.info(f"🗑️ Deleted session: {session_id[:8]}...")
            return True
        return False
    
    def update_session_title(self, session_id: str, title: str):
        """Update session title"""
        now = datetime.now().isoformat()
        self.db_manager.execute_update('''
            UPDATE sessions 
            SET title = ?, updated_at = ?
            WHERE id = ?
        ''', (title, now, session_id))
    
    def cleanup_empty_sessions(self) -> int:
        """Clean up sessions with no messages"""
        affected = self.db_manager.execute_update('''
            DELETE FROM sessions 
            WHERE id NOT IN (SELECT DISTINCT session_id FROM messages)
        ''')
        
        if affected > 0:
            logger.info(f"🧹 Cleaned up {affected} empty sessions")
        return affected
    
    # =============================================
    # Message Management
    # =============================================
    
    def add_message(self, session_id: str, content: str, sender: str, metadata: Dict = None) -> str:
        """Add a message to a session with transaction support"""
        message_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        metadata_json = json.dumps(metadata or {})
        
        # Use transaction to ensure both operations succeed or both fail
        with self.db_manager.transaction() as conn:
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
        
        return message_id
    
    def get_messages(self, session_id: str, limit: int = 100) -> List[Dict]:
        """Get all messages for a session"""
        rows = self.db_manager.execute_query('''
            SELECT id, content, sender, timestamp, metadata
            FROM messages
            WHERE session_id = ?
            ORDER BY timestamp ASC
            LIMIT ?
        ''', (session_id, limit), fetch_all=True)
        
        messages = []
        if rows:
            for row in rows:
                message = dict(row)
                try:
                    message['metadata'] = json.loads(message['metadata'])
                except json.JSONDecodeError:
                    message['metadata'] = {}
                messages.append(message)
        
        return messages
    
    def get_conversation_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        """Get conversation history for Ollama"""
        rows = self.db_manager.execute_query('''
            SELECT content, sender
            FROM messages
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (session_id, limit * 2), fetch_all=True)  # Get more to account for pairs
        
        if not rows:
            return []
        
        # Convert to Ollama format and reverse to chronological order
        history = []
        for row in reversed(rows):
            role = "user" if row['sender'] == "user" else "assistant"
            history.append({"role": role, "content": row['content']})
        
        return history[-limit:] if len(history) > limit else history
    
    # =============================================
    # Document Management
    # =============================================
    
    def add_document_to_session(self, session_id: str, file_path: str) -> int:
        """Add a document to a session (compatible with original database.py schema)"""
        result = self.db_manager.execute_update('''
            INSERT INTO session_documents (session_id, file_path, indexed)
            VALUES (?, ?, 0)
        ''', (session_id, file_path))
        
        logger.info(f"📄 Added document '{file_path}' to session {session_id[:8]}...")
        return result
    
    def get_documents_for_session(self, session_id: str) -> List[str]:
        """Get file paths for all documents in a session (compatible with original database.py schema)"""
        rows = self.db_manager.execute_query('''
            SELECT file_path FROM session_documents WHERE session_id = ?
        ''', (session_id,), fetch_all=True)
        
        return [row['file_path'] for row in rows] if rows else []
    
    # =============================================
    # Index Management
    # =============================================
    
    def create_index(self, name: str, description: str = None, metadata: Dict = None) -> str:
        """Create a new index (compatible with original database schema)"""
        index_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        metadata_json = json.dumps(metadata or {})
        vector_table_name = f"text_pages_{index_id}"  # Same format as original
        
        self.db_manager.execute_update('''
            INSERT INTO indexes (id, name, description, created_at, updated_at, vector_table_name, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (index_id, name, description, now, now, vector_table_name, metadata_json))
        
        logger.info(f"📊 Created new index: {index_id[:8]}... - {name}")
        return index_id
    
    def list_indexes(self) -> List[Dict]:
        """List all indexes with their documents"""
        rows = self.db_manager.execute_query('''
            SELECT id, name, description, created_at, updated_at, vector_table_name, metadata
            FROM indexes
            ORDER BY created_at DESC
        ''', fetch_all=True)
        
        indexes = []
        if rows:
            for row in rows:
                index = dict(row)
                try:
                    index['metadata'] = json.loads(index['metadata'])
                except json.JSONDecodeError:
                    index['metadata'] = {}
                
                # Fetch associated documents for this index (use old column names)
                doc_rows = self.db_manager.execute_query('''
                    SELECT original_filename, stored_path
                    FROM index_documents
                    WHERE index_id = ?
                ''', (index['id'],), fetch_all=True)
                
                # Add documents list (matching old database format)
                index['documents'] = [
                    {'filename': doc_row['original_filename'], 'stored_path': doc_row['stored_path']}
                    for doc_row in (doc_rows or [])
                ]
                
                indexes.append(index)
        
        return indexes
    
    def get_index(self, index_id: str) -> Optional[Dict]:
        """Get a specific index with its documents"""
        row = self.db_manager.execute_query('''
            SELECT id, name, description, created_at, updated_at, vector_table_name, metadata
            FROM indexes
            WHERE id = ?
        ''', (index_id,), fetch_one=True)
        
        if row:
            index = dict(row)
            try:
                index['metadata'] = json.loads(index['metadata'])
            except json.JSONDecodeError:
                index['metadata'] = {}
            
            # Fetch associated documents for this index (use old column names)
            doc_rows = self.db_manager.execute_query('''
                SELECT original_filename, stored_path
                FROM index_documents
                WHERE index_id = ?
            ''', (index_id,), fetch_all=True)
            
            # Add documents list (matching old database format)
            index['documents'] = [
                {'filename': doc_row['original_filename'], 'stored_path': doc_row['stored_path']}
                for doc_row in (doc_rows or [])
            ]
            
            return index
        return None
    
    def delete_index(self, index_id: str) -> bool:
        """Delete an index and all its documents (cascade)"""
        affected = self.db_manager.execute_update('''
            DELETE FROM indexes WHERE id = ?
        ''', (index_id,))
        
        if affected > 0:
            logger.info(f"🗑️ Deleted index: {index_id[:8]}...")
            return True
        return False
    
    def add_document_to_index(self, index_id: str, filename: str, file_path: str):
        """Add a document to an index (compatible with original database.py schema)"""
        self.db_manager.execute_update('''
            INSERT INTO index_documents (index_id, original_filename, stored_path)
            VALUES (?, ?, ?)
        ''', (index_id, filename, file_path))
    
    def link_index_to_session(self, session_id: str, index_id: str):
        """Link an index to a session (compatible with original schema)"""
        now = datetime.now().isoformat()
        
        self.db_manager.execute_update('''
            INSERT INTO session_indexes (session_id, index_id, linked_at)
            VALUES (?, ?, ?)
        ''', (session_id, index_id, now))
    
    def get_indexes_for_session(self, session_id: str) -> List[str]:
        """Get all index IDs linked to a session (compatible with original schema)"""
        rows = self.db_manager.execute_query('''
            SELECT index_id FROM session_indexes WHERE session_id = ? ORDER BY linked_at
        ''', (session_id,), fetch_all=True)
        
        return [row['index_id'] for row in rows] if rows else []
    
    # =============================================
    # Statistics and Utilities
    # =============================================
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        stats = {}
        
        # Total sessions
        row = self.db_manager.execute_query('''
            SELECT COUNT(*) as count FROM sessions
        ''', fetch_one=True)
        stats['total_sessions'] = row['count'] if row else 0
        
        # Total messages
        row = self.db_manager.execute_query('''
            SELECT COUNT(*) as count FROM messages
        ''', fetch_one=True)
        stats['total_messages'] = row['count'] if row else 0
        
        # Most used model
        row = self.db_manager.execute_query('''
            SELECT model_used, COUNT(*) as count 
            FROM sessions 
            GROUP BY model_used 
            ORDER BY count DESC 
            LIMIT 1
        ''', fetch_one=True)
        stats['most_used_model'] = row['model_used'] if row else 'unknown'
        
        # Total indexes
        row = self.db_manager.execute_query('''
            SELECT COUNT(*) as count FROM indexes
        ''', fetch_one=True)
        stats['total_indexes'] = row['count'] if row else 0
        
        return stats
    
    def close(self):
        """Close the database manager"""
        self.db_manager.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# For backward compatibility, provide a function to generate session titles
def generate_session_title(first_message: str, max_length: int = 50) -> str:
    """Generate a session title from the first message"""
    if not first_message:
        return "New Chat"
    
    # Clean up the message
    title = first_message.strip()
    
    # Remove common prefixes
    prefixes_to_remove = ["please", "can you", "could you", "would you", "help me", "i need"]
    title_lower = title.lower()
    for prefix in prefixes_to_remove:
        if title_lower.startswith(prefix):
            title = title[len(prefix):].strip()
            break
    
    # Capitalize first letter
    if title:
        title = title[0].upper() + title[1:]
    
    # Truncate if too long
    if len(title) > max_length:
        title = title[:max_length-3] + "..."
    
    return title or "New Chat"


# Create a global instance for backward compatibility
enhanced_db = EnhancedChatDatabase() 