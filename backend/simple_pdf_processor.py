"""
Simple PDF Processing Service
Handles PDF upload, text extraction, and simple text search for RAG functionality
"""

import os
import uuid
from typing import List, Dict, Any
import PyPDF2
from io import BytesIO
import sqlite3
import json
from datetime import datetime

class SimplePDFProcessor:
    def __init__(self, db_path: str = "chat_data.db"):
        """Initialize simple PDF processor with SQLite storage"""
        self.db_path = db_path
        self.init_database()
        print("✅ Simple PDF processor initialized")
    
    def init_database(self):
        """Initialize SQLite database for storing PDF content"""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS pdf_documents (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                content TEXT NOT NULL,
                chunks INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS pdf_chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                FOREIGN KEY (document_id) REFERENCES pdf_documents (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def extract_text_from_pdf(self, pdf_bytes: bytes) -> str:
        """Extract text from PDF bytes"""
        try:
            print(f"📄 Starting PDF text extraction ({len(pdf_bytes)} bytes)")
            pdf_file = BytesIO(pdf_bytes)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            print(f"📖 PDF has {len(pdf_reader.pages)} pages")
            
            text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                print(f"📄 Processing page {page_num + 1}")
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text += f"\n--- Page {page_num + 1} ---\n"
                        text += page_text + "\n"
                    print(f"✅ Page {page_num + 1}: extracted {len(page_text)} characters")
                except Exception as page_error:
                    print(f"❌ Error on page {page_num + 1}: {str(page_error)}")
                    continue
            
            print(f"📄 Total extracted text: {len(text)} characters")
            return text.strip()
            
        except Exception as e:
            print(f"❌ Error extracting text from PDF: {str(e)}")
            print(f"❌ Error type: {type(e).__name__}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        if not text.strip():
            return []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            
            # If this isn't the last chunk, try to break at a sentence or paragraph
            if end < text_length:
                # Look for sentence endings near the chunk boundary
                for boundary in ['. ', '.\n', '!\n', '?\n', '\n\n']:
                    boundary_pos = text.rfind(boundary, start, end)
                    if boundary_pos > start + chunk_size // 2:  # Only if we find a good break point
                        end = boundary_pos + len(boundary)
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
        
        return chunks
    
    def process_pdf(self, pdf_bytes: bytes, filename: str, session_id: str) -> Dict[str, Any]:
        """Process a PDF file and store in database"""
        print(f"📄 Processing PDF: {filename}")
        
        # Extract text
        text = self.extract_text_from_pdf(pdf_bytes)
        if not text:
            return {
                "success": False,
                "error": "Could not extract text from PDF",
                "filename": filename,
                "chunks": 0
            }
        
        print(f"📝 Extracted {len(text)} characters from {filename}")
        
        # Chunk text
        chunks = self.chunk_text(text)
        if not chunks:
            return {
                "success": False,
                "error": "Could not create text chunks",
                "filename": filename,
                "chunks": 0
            }
        
        print(f"✂️ Created {len(chunks)} chunks from {filename}")
        
        # Store in database
        document_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Store document
            conn.execute('''
                INSERT INTO pdf_documents (id, session_id, filename, content, chunks, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (document_id, session_id, filename, text, len(chunks), now))
            
            # Store chunks
            for i, chunk in enumerate(chunks):
                chunk_id = f"{document_id}_{i}"
                conn.execute('''
                    INSERT INTO pdf_chunks (id, document_id, chunk_index, content)
                    VALUES (?, ?, ?, ?)
                ''', (chunk_id, document_id, i, chunk))
            
            conn.commit()
            conn.close()
            
            print(f"💾 Stored {len(chunks)} chunks for {filename} in database")
            
            return {
                "success": True,
                "filename": filename,
                "file_id": document_id,
                "chunks": len(chunks),
                "text_length": len(text)
            }
            
        except Exception as e:
            print(f"❌ Error storing in database: {str(e)}")
            return {
                "success": False,
                "error": f"Database storage failed: {str(e)}",
                "filename": filename,
                "chunks": len(chunks)
            }
    
    def search_relevant_chunks(self, query: str, session_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant chunks using simple text matching"""
        try:
            query_lower = query.lower()
            query_words = set(query_lower.split())
            
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            
            # Get all chunks for this session
            cursor = conn.execute('''
                SELECT c.*, d.filename 
                FROM pdf_chunks c
                JOIN pdf_documents d ON c.document_id = d.id
                WHERE d.session_id = ?
                ORDER BY c.chunk_index
            ''', (session_id,))
            
            chunks = cursor.fetchall()
            conn.close()
            
            # Score chunks based on word overlap
            scored_chunks = []
            for chunk in chunks:
                content_lower = chunk['content'].lower()
                content_words = set(content_lower.split())
                
                # Calculate simple overlap score
                overlap = len(query_words.intersection(content_words))
                if overlap > 0 or query_lower in content_lower:
                    # Boost score if query appears as a phrase
                    phrase_boost = 2 if query_lower in content_lower else 1
                    score = overlap * phrase_boost
                    
                    scored_chunks.append({
                        "text": chunk['content'],
                        "metadata": {
                            "filename": chunk['filename'],
                            "chunk_index": chunk['chunk_index']
                        },
                        "score": score
                    })
            
            # Sort by score and return top_k
            scored_chunks.sort(key=lambda x: x['score'], reverse=True)
            relevant_chunks = scored_chunks[:top_k]
            
            print(f"🔍 Found {len(relevant_chunks)} relevant chunks for query")
            return relevant_chunks
            
        except Exception as e:
            print(f"❌ Error searching for relevant chunks: {str(e)}")
            return []
    
    def get_session_documents(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all documents for a session"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            
            cursor = conn.execute('''
                SELECT id, filename, chunks, created_at
                FROM pdf_documents
                WHERE session_id = ?
                ORDER BY created_at DESC
            ''', (session_id,))
            
            documents = [dict(row) for row in cursor.fetchall()]
            conn.close()
            
            return documents
            
        except Exception as e:
            print(f"❌ Error getting session documents: {str(e)}")
            return []
    
    def get_document_content(self, document_id: str) -> str:
        """Get the full content of a document"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            cursor = conn.execute('''
                SELECT content FROM pdf_documents WHERE id = ?
            ''', (document_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            return result[0] if result else ""
            
        except Exception as e:
            print(f"❌ Error getting document content: {str(e)}")
            return ""
    
    def delete_session_documents(self, session_id: str) -> bool:
        """Delete all documents for a session"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Delete chunks first (due to foreign key)
            conn.execute('''
                DELETE FROM pdf_chunks 
                WHERE document_id IN (
                    SELECT id FROM pdf_documents WHERE session_id = ?
                )
            ''', (session_id,))
            
            # Delete documents
            cursor = conn.execute('''
                DELETE FROM pdf_documents WHERE session_id = ?
            ''', (session_id,))
            
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            if deleted_count > 0:
                print(f"🗑️ Deleted {deleted_count} documents for session {session_id[:8]}...")
            
            return True
            
        except Exception as e:
            print(f"❌ Error deleting session documents: {str(e)}")
            return False

# Global PDF processor instance
simple_pdf_processor = None

def initialize_simple_pdf_processor():
    """Initialize simple PDF processor with error handling"""
    global simple_pdf_processor
    try:
        simple_pdf_processor = SimplePDFProcessor()
        return True
    except Exception as e:
        print(f"❌ Failed to initialize simple PDF processor: {str(e)}")
        print("⚠️ PDF functionality will be disabled")
        return False

if __name__ == "__main__":
    # Test the simple PDF processor
    print("🧪 Testing simple PDF processor...")
    
    processor = SimplePDFProcessor()
    print("✅ Simple PDF processor test completed!") 