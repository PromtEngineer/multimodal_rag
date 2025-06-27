#!/usr/bin/env python3
"""
Enhanced Flask-based server with connection pooling, middleware, and cleaner architecture
Replaces the old HTTP server with better error handling and code organization
"""

import os
import sys
import logging
from flask import Flask, request
from typing import Dict, Callable

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

from enhanced_database import EnhancedChatDatabase, generate_session_title
from middleware import (
    setup_middleware, create_error_response, create_success_response, 
    require_json, validate_fields, stream_response
)
from ollama_client import OllamaClient
import simple_pdf_processor as pdf_module

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.name = "MultimodalRAG-Backend"

# Initialize components
DB_PATH = os.environ.get("DB_PATH", os.path.join(os.path.dirname(__file__), '..', 'chat_data.db'))
logger.info(f"Connecting to database at: {os.path.abspath(DB_PATH)}")
db = EnhancedChatDatabase(db_path=DB_PATH)
ollama_client = OllamaClient()

# =============================================
# Health Check Functions for Middleware
# =============================================

def check_ollama_health() -> bool:
    """Check if Ollama service is running"""
    try:
        return ollama_client.is_ollama_running()
    except:
        return False

def check_database_health() -> bool:
    """Check if database is accessible"""
    try:
        stats = db.get_stats()
        return isinstance(stats, dict)
    except:
        return False

def check_pdf_processor_health() -> bool:
    """Check if PDF processor is working"""
    try:
        return hasattr(pdf_module, 'initialize_simple_pdf_processor')
    except:
        return False

# Setup middleware with health checks
setup_middleware(app, {
    "ollama": check_ollama_health,
    "database": check_database_health,
    "pdf_processor": check_pdf_processor_health
})

# =============================================
# Session Management Endpoints
# =============================================

@app.route('/sessions', methods=['GET'])
def get_sessions():
    """Get all chat sessions"""
    try:
        sessions = db.get_sessions()
        return create_success_response({
            "sessions": sessions,
            "total": len(sessions)
        })
    except Exception as e:
        logger.error(f"Failed to get sessions: {e}")
        return create_error_response("Failed to retrieve sessions", 500)

@app.route('/sessions', methods=['POST'])
@require_json
@validate_fields(['title'], ['model'])
def create_session():
    """Create a new chat session"""
    try:
        data = request.get_json()
        title = data['title']
        model = data.get('model', 'llama3.2:latest')
        
        session_id = db.create_session(title, model)
        session = db.get_session(session_id)
        
        return create_success_response(
            {"session": session}, 
            "Session created successfully"
        ), 201
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        return create_error_response("Failed to create session", 500)

@app.route('/sessions/<session_id>', methods=['GET'])
def get_session(session_id: str):
    """Get a specific session with its messages"""
    try:
        session = db.get_session(session_id)
        if not session:
            return create_error_response("Session not found", 404)
        
        messages = db.get_messages(session_id)
        
        return create_success_response({
            "session": session,
            "messages": messages
        })
    except Exception as e:
        logger.error(f"Failed to get session {session_id}: {e}")
        return create_error_response("Failed to retrieve session", 500)

@app.route('/sessions/<session_id>', methods=['DELETE'])
def delete_session(session_id: str):
    """Delete a session and all its messages"""
    try:
        success = db.delete_session(session_id)
        if not success:
            return create_error_response("Session not found", 404)
        
        return create_success_response(message="Session deleted successfully")
    except Exception as e:
        logger.error(f"Failed to delete session {session_id}: {e}")
        return create_error_response("Failed to delete session", 500)

@app.route('/sessions/cleanup', methods=['POST'])
def cleanup_sessions():
    """Clean up empty sessions"""
    try:
        cleanup_count = db.cleanup_empty_sessions()
        return create_success_response({
            "cleanup_count": cleanup_count
        }, f"Cleaned up {cleanup_count} empty sessions")
    except Exception as e:
        logger.error(f"Failed to cleanup sessions: {e}")
        return create_error_response("Failed to cleanup sessions", 500)

# =============================================
# Message/Chat Endpoints
# =============================================

@app.route('/chat', methods=['POST'])
@require_json
@validate_fields(['message'], ['model', 'conversation_history'])
def legacy_chat():
    """Handle legacy chat requests (without sessions)"""
    try:
        data = request.get_json()
        message = data['message']
        model = data.get('model', 'llama3.2:latest')
        conversation_history = data.get('conversation_history', [])
        
        # Check if Ollama is running
        if not ollama_client.is_ollama_running():
            return create_error_response("Ollama is not running. Please start Ollama first.", 503)
        
        # Get response from Ollama
        response = ollama_client.chat(message, model, conversation_history)
        
        return create_success_response({
            "response": response,
            "model": model,
            "message_count": len(conversation_history) + 1
        })
    except Exception as e:
        logger.error(f"Failed to handle chat: {e}")
        return create_error_response("Failed to process chat request", 500)

@app.route('/sessions/<session_id>/messages', methods=['POST'])
@require_json
@validate_fields(['message'], ['model'])
def session_chat(session_id: str):
    """Send a message in a session context"""
    try:
        data = request.get_json()
        message = data['message']
        model = data.get('model', 'llama3.2:latest')
        
        # Verify session exists
        session = db.get_session(session_id)
        if not session:
            return create_error_response("Session not found", 404)
        
        # Check if Ollama is running
        if not ollama_client.is_ollama_running():
            return create_error_response("Ollama is not running. Please start Ollama first.", 503)
        
        # Get conversation history
        conversation_history = db.get_conversation_history(session_id)
        
        # Add user message to database
        user_message_id = db.add_message(session_id, message, "user")
        
        # Get response from Ollama
        response = ollama_client.chat(message, model, conversation_history)
        
        # Add assistant response to database
        assistant_message_id = db.add_message(session_id, response, "assistant")
        
        return create_success_response({
            "response": response,
            "user_message_id": user_message_id,
            "assistant_message_id": assistant_message_id,
            "model": model
        })
    except Exception as e:
        logger.error(f"Failed to handle session chat: {e}")
        return create_error_response("Failed to process session chat", 500)

# =============================================
# Index Management Endpoints
# =============================================

@app.route('/indexes', methods=['GET'])
def get_indexes():
    """Get all indexes"""
    try:
        indexes = db.list_indexes()
        return create_success_response({"indexes": indexes})
    except Exception as e:
        logger.error(f"Failed to get indexes: {e}")
        return create_error_response("Failed to retrieve indexes", 500)

@app.route('/indexes', methods=['POST'])
@require_json
@validate_fields(['name'], ['description', 'metadata'])
def create_index():
    """Create a new index"""
    try:
        data = request.get_json()
        name = data['name']
        description = data.get('description')
        metadata = data.get('metadata', {})
        
        index_id = db.create_index(name, description, metadata)
        index = db.get_index(index_id)
        
        return create_success_response(
            {"index": index}, 
            "Index created successfully"
        ), 201
    except Exception as e:
        logger.error(f"Failed to create index: {e}")
        return create_error_response("Failed to create index", 500)

@app.route('/indexes/<index_id>', methods=['GET'])
def get_index(index_id: str):
    """Get a specific index"""
    try:
        index = db.get_index(index_id)
        if not index:
            return create_error_response("Index not found", 404)
        
        return create_success_response({"index": index})
    except Exception as e:
        logger.error(f"Failed to get index {index_id}: {e}")
        return create_error_response("Failed to retrieve index", 500)

@app.route('/indexes/<index_id>', methods=['DELETE'])
def delete_index(index_id: str):
    """Delete an index"""
    try:
        success = db.delete_index(index_id)
        if not success:
            return create_error_response("Index not found", 404)
        
        return create_success_response(message="Index deleted successfully")
    except Exception as e:
        logger.error(f"Failed to delete index {index_id}: {e}")
        return create_error_response("Failed to delete index", 500)

@app.route('/sessions/<session_id>/indexes/<index_id>', methods=['POST'])
def link_index_to_session(session_id: str, index_id: str):
    """Link an index to a session"""
    try:
        # Verify both session and index exist
        session = db.get_session(session_id)
        if not session:
            return create_error_response("Session not found", 404)
        
        index = db.get_index(index_id)
        if not index:
            return create_error_response("Index not found", 404)
        
        db.link_index_to_session(session_id, index_id)
        
        return create_success_response(message="Index linked to session successfully")
    except Exception as e:
        logger.error(f"Failed to link index {index_id} to session {session_id}: {e}")
        return create_error_response("Failed to link index to session", 500)

@app.route('/sessions/<session_id>/indexes', methods=['GET'])
def get_session_indexes(session_id: str):
    """Get all indexes linked to a session"""
    try:
        session = db.get_session(session_id)
        if not session:
            return create_error_response("Session not found", 404)
        
        index_ids = db.get_indexes_for_session(session_id)
        
        # Get full index details
        indexes = []
        for index_id in index_ids:
            index = db.get_index(index_id)
            if index:
                indexes.append(index)
        
        return create_success_response({"indexes": indexes})
    except Exception as e:
        logger.error(f"Failed to get session indexes for {session_id}: {e}")
        return create_error_response("Failed to retrieve session indexes", 500)

@app.route('/indexes/<index_id>/build', methods=['POST'])
def build_index(index_id: str):
    """Build/process an index by running the RAG indexing pipeline"""
    try:
        # Verify index exists
        index = db.get_index(index_id)
        if not index:
            return create_error_response("Index not found", 404)
        
        # Get file paths from index documents
        file_paths = [doc['stored_path'] for doc in index.get('documents', [])]
        if not file_paths:
            return create_error_response("No documents to index", 400)
        
        # Parse optional flags and configuration from request body
        data = request.get_json() or {}
        latechunk = bool(data.get('latechunk', False))
        docling_chunk = bool(data.get('doclingChunk', False))
        
        # Build configuration overrides from frontend parameters
        config_overrides = {}
        
        # Chunking configuration
        if data.get('chunkSize') is not None or data.get('chunkOverlap') is not None:
            config_overrides['chunking'] = {}
            if data.get('chunkSize') is not None:
                config_overrides['chunking']['max_chunk_size'] = int(data.get('chunkSize'))
            if data.get('chunkOverlap') is not None:
                config_overrides['chunking']['chunk_overlap'] = int(data.get('chunkOverlap'))
        
        # Contextual enricher configuration
        contextual_config = {}
        if data.get('enableContextualEnrich') is not None:
            contextual_config['enabled'] = bool(data.get('enableContextualEnrich'))
        if data.get('contextWindow') is not None:
            contextual_config['window_size'] = int(data.get('contextWindow'))
        if contextual_config:
            config_overrides['contextual_enricher'] = contextual_config
        
        # Indexing batch sizes
        indexing_config = {}
        if data.get('embeddingBatchSize') is not None:
            indexing_config['embedding_batch_size'] = int(data.get('embeddingBatchSize'))
        if data.get('enrichmentBatchSize') is not None:
            indexing_config['enrichment_batch_size'] = int(data.get('enrichmentBatchSize'))
        if indexing_config:
            config_overrides['indexing'] = indexing_config
        
        # Embedding model
        if data.get('embeddingModel'):
            config_overrides['embedding_model_name'] = data.get('embeddingModel')
        
        # Retrieval mode
        if data.get('retrievalMode'):
            retrieval_mode = data.get('retrievalMode')
            config_overrides['retrievers'] = {
                'dense': {'enabled': retrieval_mode in ['hybrid', 'vector']},
                'bm25': {'enabled': retrieval_mode in ['hybrid', 'bm25']}
            }
        
        # Delegate to advanced RAG API
        import requests
        rag_api_url = "http://localhost:8001/index"
        payload = {
            "file_paths": file_paths, 
            "session_id": index_id,  # Use index_id as session_id for table naming
            "config_overrides": config_overrides
        }
        if latechunk:
            payload["enable_latechunk"] = True
        if docling_chunk:
            payload["enable_docling_chunk"] = True
            
        rag_response = requests.post(rag_api_url, json=payload, timeout=300)  # 5 min timeout
        
        if rag_response.status_code == 200:
            response_data = rag_response.json()
            return create_success_response({
                "message": f"Index built successfully with {len(file_paths)} documents",
                "details": response_data,
                "latechunk": latechunk,
                "docling_chunk": docling_chunk,
                "table_name": f"text_pages_{index_id}"
            })
        else:
            logger.error(f"RAG indexing failed: {rag_response.text}")
            return create_error_response(f"RAG indexing failed: {rag_response.text}", 500)
            
    except requests.exceptions.Timeout:
        return create_error_response("Indexing timeout - process may still be running", 408)
    except Exception as e:
        logger.error(f"Failed to build index {index_id}: {e}")
        return create_error_response("Failed to build index", 500)

@app.route('/indexes/<index_id>/upload', methods=['POST'])
def upload_files_to_index(index_id: str):
    """Upload files to an index"""
    try:
        # Verify index exists
        index = db.get_index(index_id)
        if not index:
            return create_error_response("Index not found", 404)
        
        if 'files' not in request.files:
            return create_error_response("No files provided", 400)
        
        files = request.files.getlist('files')
        if not files or all(f.filename == '' for f in files):
            return create_error_response("No valid files provided", 400)
        
        import os
        import uuid
        
        upload_dir = 'shared_uploads'
        os.makedirs(upload_dir, exist_ok=True)
        
        uploaded_files = []
        for file in files:
            if file.filename and file.filename != '':
                # Generate unique filename
                unique_filename = f"{uuid.uuid4()}_{file.filename}"
                file_path = os.path.join(upload_dir, unique_filename)
                
                # Save file
                file.save(file_path)
                absolute_path = os.path.abspath(file_path)
                
                # Add to database
                db.add_document_to_index(index_id, file.filename, absolute_path)
                
                uploaded_files.append({
                    'filename': file.filename,
                    'stored_path': absolute_path
                })
        
        if not uploaded_files:
            return create_error_response("No files were successfully uploaded", 400)
        
        return create_success_response({
            "uploaded_files": uploaded_files,
            "count": len(uploaded_files)
        }, f"Successfully uploaded {len(uploaded_files)} files")
        
    except Exception as e:
        logger.error(f"Failed to upload files to index {index_id}: {e}")
        return create_error_response("Failed to upload files", 500)

# =============================================
# File Upload and Document Management
# =============================================

@app.route('/sessions/<session_id>/upload', methods=['POST'])
def upload_file_to_session(session_id: str):
    """Upload a file to a session"""
    try:
        session = db.get_session(session_id)
        if not session:
            return create_error_response("Session not found", 404)
        
        if 'file' not in request.files:
            return create_error_response("No file provided", 400)
        
        file = request.files['file']
        if file.filename == '':
            return create_error_response("No file selected", 400)
        
        # Save file to shared uploads directory
        upload_dir = "shared_uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, file.filename)
        file.save(file_path)
        
        # Add document to session
        document_id = db.add_document_to_session(session_id, file_path)
        
        return create_success_response({
            "document_id": document_id,
            "filename": file.filename,
            "file_path": file_path
        }, "File uploaded successfully")
    except Exception as e:
        logger.error(f"Failed to upload file to session {session_id}: {e}")
        return create_error_response("Failed to upload file", 500)

@app.route('/sessions/<session_id>/documents', methods=['GET'])
def get_session_documents(session_id: str):
    """Get all documents for a session"""
    try:
        session = db.get_session(session_id)
        if not session:
            return create_error_response("Session not found", 404)
        
        documents = db.get_documents_for_session(session_id)
        
        return create_success_response({"documents": documents})
    except Exception as e:
        logger.error(f"Failed to get session documents for {session_id}: {e}")
        return create_error_response("Failed to retrieve session documents", 500)

# =============================================
# Statistics and Information Endpoints
# =============================================

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get database and system statistics"""
    try:
        db_stats = db.get_stats()
        ollama_stats = {
            "is_running": ollama_client.is_ollama_running(),
            "available_models": ollama_client.list_models() if ollama_client.is_ollama_running() else []
        }
        
        return create_success_response({
            "database": db_stats,
            "ollama": ollama_stats
        })
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        return create_error_response("Failed to retrieve statistics", 500)

# =============================================
# Application Lifecycle
# =============================================

@app.teardown_appcontext
def close_db(error):
    """Clean up database connections on app teardown"""
    pass  # EnhancedDatabase handles its own cleanup

def shutdown_server():
    """Gracefully shutdown the server"""
    logger.info("🛑 Shutting down server...")
    db.close()

if __name__ == '__main__':
    try:
        logger.info("🚀 Starting Enhanced Backend Server...")
        logger.info(f"📊 Database stats: {db.get_stats()}")
        logger.info(f"🤖 Ollama running: {ollama_client.is_ollama_running()}")
        
        # Run the Flask app
        app.run(
            host='0.0.0.0',
            port=8000,
            debug=False,  # Set to True for development
            threaded=True
        )
    except KeyboardInterrupt:
        shutdown_server()
    except Exception as e:
        logger.error(f"❌ Server startup failed: {e}")
        shutdown_server() 