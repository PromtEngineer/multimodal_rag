import sqlite3
import threading
import contextlib
from typing import Optional, Any, Dict, List, Tuple
import logging
from queue import Queue, Empty
import time

logger = logging.getLogger(__name__)

class DatabaseConnectionPool:
    """Thread-safe SQLite connection pool"""
    
    def __init__(self, db_path: str, max_connections: int = 10, timeout: float = 30.0):
        self.db_path = db_path
        self.max_connections = max_connections
        self.timeout = timeout
        self._pool = Queue(maxsize=max_connections)
        self._lock = threading.Lock()
        self._created_connections = 0
        
        # Initialize with one connection to ensure DB exists
        self._create_connection()
    
    def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection with proper settings"""
        conn = sqlite3.connect(
            self.db_path,
            timeout=self.timeout,
            check_same_thread=False  # Allow sharing across threads
        )
        
        # Enable foreign keys and WAL mode for better performance
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA temp_store = MEMORY")
        
        return conn
    
    def get_connection(self) -> sqlite3.Connection:
        """Get a connection from the pool"""
        try:
            # Try to get existing connection from pool
            conn = self._pool.get_nowait()
            # Test connection is still valid
            conn.execute("SELECT 1").fetchone()
            return conn
        except (Empty, sqlite3.Error):
            # Pool is empty or connection is bad, create new one
            with self._lock:
                if self._created_connections < self.max_connections:
                    self._created_connections += 1
                    return self._create_connection()
                else:
                    # Wait for a connection to be returned
                    try:
                        conn = self._pool.get(timeout=self.timeout)
                        conn.execute("SELECT 1").fetchone()  # Test connection
                        return conn
                    except (Empty, sqlite3.Error):
                        raise sqlite3.OperationalError("Unable to get database connection")
    
    def return_connection(self, conn: sqlite3.Connection):
        """Return a connection to the pool"""
        try:
            # Test connection is still valid
            conn.execute("SELECT 1").fetchone()
            self._pool.put_nowait(conn)
        except:
            # Connection is bad, don't return it to pool
            with self._lock:
                self._created_connections -= 1
    
    def close_all(self):
        """Close all connections in the pool"""
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except Empty:
                break


class DatabaseManager:
    """Enhanced database manager with connection pooling and transaction support"""
    
    def __init__(self, db_path: str = "chat_history.db", max_connections: int = 10):
        self.db_path = db_path
        self.pool = DatabaseConnectionPool(db_path, max_connections)
        self.init_database()
    
    @contextlib.contextmanager
    def get_connection(self, read_only: bool = False):
        """Context manager for database connections"""
        conn = None
        try:
            conn = self.pool.get_connection()
            # Don't start transaction for read-only operations
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                self.pool.return_connection(conn)
    
    @contextlib.contextmanager
    def transaction(self):
        """Context manager for database transactions"""
        conn = None
        try:
            conn = self.pool.get_connection()
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Transaction error: {e}")
            raise
        finally:
            if conn:
                self.pool.return_connection(conn)
    
    def execute_query(self, query: str, params: Tuple = (), fetch_one: bool = False, fetch_all: bool = False) -> Any:
        """Execute a query and return results"""
        with self.get_connection(read_only=True) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            
            if fetch_one:
                return cursor.fetchone()
            elif fetch_all:
                return cursor.fetchall()
            else:
                return cursor
    
    def execute_update(self, query: str, params: Tuple = ()) -> int:
        """Execute an update/insert/delete query and return affected rows"""
        with self.transaction() as conn:
            cursor = conn.execute(query, params)
            return cursor.rowcount
    
    def execute_many(self, query: str, params_list: List[Tuple]) -> int:
        """Execute multiple queries in a single transaction"""
        with self.transaction() as conn:
            cursor = conn.executemany(query, params_list)
            return cursor.rowcount
    
    def init_database(self):
        """Initialize the SQLite database with required tables"""
        with self.get_connection() as conn:
            # Manually handle transaction for initialization
            conn.execute("BEGIN IMMEDIATE")
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
            
            # Documents table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    original_filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    uploaded_at TEXT NOT NULL,
                    content TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
                )
            ''')
            
            # Document chunks table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    chunk_text TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
                )
            ''')
            
            # Indexes table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS indexes (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    created_at TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}'
                )
            ''')
            
            # Index documents table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS index_documents (
                    id TEXT PRIMARY KEY,
                    index_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    uploaded_at TEXT NOT NULL,
                    FOREIGN KEY (index_id) REFERENCES indexes (id) ON DELETE CASCADE
                )
            ''')
            
            # Session index links table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS session_index_links (
                    session_id TEXT NOT NULL,
                    index_id TEXT NOT NULL,
                    linked_at TEXT NOT NULL,
                    PRIMARY KEY (session_id, index_id),
                    FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE,
                    FOREIGN KEY (index_id) REFERENCES indexes (id) ON DELETE CASCADE
                )
            ''')
            
            # Create indexes for better performance
            indexes_to_create = [
                'CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id)',
                'CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)',
                'CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at)',
                'CREATE INDEX IF NOT EXISTS idx_documents_session_id ON documents(session_id)',
                'CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id ON document_chunks(document_id)',
                'CREATE INDEX IF NOT EXISTS idx_index_documents_index_id ON index_documents(index_id)',
                'CREATE INDEX IF NOT EXISTS idx_session_index_links_session_id ON session_index_links(session_id)',
                'CREATE INDEX IF NOT EXISTS idx_session_index_links_index_id ON session_index_links(index_id)',
            ]
            
            for index_sql in indexes_to_create:
                conn.execute(index_sql)
            
            # Commit the transaction
            conn.commit()
    
    def close(self):
        """Close all database connections"""
        self.pool.close_all()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close() 