#!/usr/bin/env python3
"""
Consolidated RAG Server - Production Ready
Combines frontend API, session management, and RAG processing into a single server.
"""
import json
import http.server
import socketserver
import cgi
import os
import uuid
import sys
import re
from urllib.parse import urlparse, parse_qs
from typing import List, Dict, Any, Optional, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add backend directory to path for database imports
backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

# Import database and RAG components
try:
    from backend.database import ChatDatabase, generate_session_title
    from backend.ollama_client import OllamaClient
    from rag_system.main import get_agent
    from rag_system.factory import get_indexing_pipeline
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    sys.exit(1)

# Initialize global components
db = ChatDatabase()
ollama_client = OllamaClient()

# Initialize RAG components
AGENT_MODE = os.getenv("RAG_CONFIG_MODE", "default")
logger.info(f"Initializing RAG Agent in '{AGENT_MODE}' mode...")

try:
    RAG_AGENT = get_agent(AGENT_MODE)
    INDEXING_PIPELINE = get_indexing_pipeline(AGENT_MODE)
    if RAG_AGENT is None:
        raise RuntimeError("RAG Agent initialization failed")
    logger.info("✅ RAG Agent initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize RAG components: {e}")
    sys.exit(1)

class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True

class ConsolidatedHandler(http.server.BaseHTTPRequestHandler):
    """Unified handler for all frontend and RAG functionality"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urlparse(self.path)
        
        # Health check
        if parsed_path.path == '/health':
            self.handle_health()
        # Session management
        elif parsed_path.path == '/sessions':
            self.handle_get_sessions()
        elif parsed_path.path == '/sessions/cleanup':
            self.handle_cleanup_sessions()
        elif parsed_path.path.startswith('/sessions/') and parsed_path.path.endswith('/documents'):
            session_id = parsed_path.path.split('/')[-2]
            self.handle_get_session_documents(session_id)
        elif parsed_path.path.startswith('/sessions/') and parsed_path.path.endswith('/indexes'):
            session_id = parsed_path.path.split('/')[-2]
            self.handle_get_session_indexes(session_id)
        elif parsed_path.path.startswith('/sessions/') and parsed_path.path.count('/') == 2:
            session_id = parsed_path.path.split('/')[-1]
            self.handle_get_session(session_id)
        # Index management
        elif parsed_path.path == '/indexes':
            self.handle_get_indexes()
        elif parsed_path.path.startswith('/indexes/') and parsed_path.path.count('/') == 2:
            index_id = parsed_path.path.split('/')[-1]
            self.handle_get_index(index_id)
        # Models
        elif parsed_path.path == '/models':
            self.handle_get_models()
        else:
            self.send_json_response({"error": "Not Found"}, status_code=404)
    
    def do_POST(self):
        """Handle POST requests"""
        parsed_path = urlparse(self.path)
        
        # Legacy chat endpoint
        if parsed_path.path == '/chat':
            self.handle_legacy_chat()
        # Session management
        elif parsed_path.path == '/sessions':
            self.handle_create_session()
        elif parsed_path.path.startswith('/sessions/') and parsed_path.path.endswith('/messages'):
            session_id = parsed_path.path.split('/')[-2]
            self.handle_session_chat(session_id)
        elif parsed_path.path.startswith('/sessions/') and parsed_path.path.endswith('/upload'):
            session_id = parsed_path.path.split('/')[-2]
            self.handle_file_upload(session_id)
        elif parsed_path.path.startswith('/sessions/') and parsed_path.path.endswith('/index'):
            session_id = parsed_path.path.split('/')[-2]
            self.handle_index_documents(session_id)
        elif parsed_path.path.startswith('/sessions/') and parsed_path.path.endswith('/rename'):
            session_id = parsed_path.path.split('/')[-2]
            self.handle_rename_session(session_id)
        # Index management
        elif parsed_path.path == '/indexes':
            self.handle_create_index()
        elif parsed_path.path.startswith('/indexes/') and parsed_path.path.endswith('/upload'):
            index_id = parsed_path.path.split('/')[-2]
            self.handle_index_file_upload(index_id)
        elif parsed_path.path.startswith('/indexes/') and parsed_path.path.endswith('/build'):
            index_id = parsed_path.path.split('/')[-2]
            self.handle_build_index(index_id)
        elif parsed_path.path.startswith('/sessions/') and '/indexes/' in parsed_path.path:
            parts = parsed_path.path.split('/')
            session_id = parts[2]
            index_id = parts[4]
            self.handle_link_index_to_session(session_id, index_id)
        # RAG endpoints
        elif parsed_path.path == '/rag/chat':
            self.handle_rag_chat()
        elif parsed_path.path == '/rag/chat/stream':
            self.handle_rag_chat_stream()
        elif parsed_path.path == '/rag/index':
            self.handle_rag_index()
        else:
            self.send_json_response({"error": "Not Found"}, status_code=404)

    def do_DELETE(self):
        """Handle DELETE requests"""
        parsed_path = urlparse(self.path)
        
        if parsed_path.path.startswith('/sessions/') and parsed_path.path.count('/') == 2:
            session_id = parsed_path.path.split('/')[-1]
            self.handle_delete_session(session_id)
        elif parsed_path.path.startswith('/indexes/') and parsed_path.path.count('/') == 2:
            index_id = parsed_path.path.split('/')[-1]
            self.handle_delete_index(index_id)
        else:
            self.send_json_response({"error": "Not Found"}, status_code=404)

    # Health and system endpoints
    def handle_health(self):
        """System health check"""
        try:
            self.send_json_response({
                "status": "ok",
                "ollama_running": ollama_client.is_ollama_running(),
                "available_models": ollama_client.list_models(),
                "database_stats": db.get_stats(),
                "rag_agent_status": "initialized" if RAG_AGENT else "error"
            })
        except Exception as e:
            self.send_json_response({
                "status": "error",
                "error": str(e)
            }, status_code=500)

    # Session management methods
    def handle_get_sessions(self):
        """Get all chat sessions"""
        try:
            sessions = db.get_sessions()
            self.send_json_response({
                "sessions": sessions,
                "total": len(sessions)
            })
        except Exception as e:
            self.send_json_response({
                "error": f"Failed to get sessions: {str(e)}"
            }, status_code=500)

    def handle_create_session(self):
        """Create a new chat session"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            title = data.get('title', 'New Chat')
            model = data.get('model', 'qwen3:8b')
            
            session_id = db.create_session(title, model)
            session = db.get_session(session_id)
            
            self.send_json_response({
                "session": session,
                "session_id": session_id
            }, status_code=201)
            
        except json.JSONDecodeError:
            self.send_json_response({
                "error": "Invalid JSON"
            }, status_code=400)
        except Exception as e:
            self.send_json_response({
                "error": f"Failed to create session: {str(e)}"
            }, status_code=500)

    def handle_session_chat(self, session_id: str):
        """Handle chat within a specific session with smart routing"""
        try:
            session = db.get_session(session_id)
            if not session:
                self.send_json_response({"error": "Session not found"}, status_code=404)
                return
            
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            message = data.get('message', '')

            if not message:
                self.send_json_response({"error": "Message is required"}, status_code=400)
                return

            # Add user message to database first
            user_message_id = db.add_message(session_id, message, "user")
            
            # Update session title if first message
            if session['message_count'] == 0:
                title = generate_session_title(message)
                db.update_session_title(session_id, title)
            
            # Smart routing: decide between direct LLM vs RAG
            idx_ids = db.get_indexes_for_session(session_id)
            force_rag = bool(data.get("force_rag", False))
            use_rag = True if force_rag else self._should_use_rag(message, idx_ids)
            
            if use_rag:
                # Use RAG pipeline
                logger.info(f"Using RAG pipeline for: '{message[:50]}...'")
                response_text, source_docs = self._handle_rag_query_internal(session_id, message, data, idx_ids)
            else:
                # Use direct LLM
                logger.info(f"Using direct LLM for: '{message[:50]}...'")
                response_text, source_docs = self._handle_direct_llm_query(session_id, message, session)

            # Add AI response to database
            ai_message_id = db.add_message(session_id, response_text, "assistant")
            
            updated_session = db.get_session(session_id)
            
            self.send_json_response({
                "response": response_text,
                "session": updated_session,
                "source_documents": source_docs,
                "used_rag": use_rag
            })
            
        except Exception as e:
            logger.error(f"Session chat error: {e}")
            self.send_json_response({
                "error": f"Server error: {str(e)}"
            }, status_code=500)

    def _should_use_rag(self, message: str, idx_ids: List[str]) -> bool:
        """Determine if query should use RAG based on content and available indexes"""
        if not idx_ids:
            return False

        # Load document overviews for intelligent routing
        try:
            doc_overviews = self._load_document_overviews(idx_ids)
            if doc_overviews:
                return self._route_using_overviews(message, doc_overviews)
        except Exception as e:
            logger.warning(f"Overview-based routing failed: {e}")
        
        # Fallback to simple pattern matching
        return self._simple_pattern_routing(message)

    def _simple_pattern_routing(self, message: str) -> bool:
        """Simple pattern-based routing fallback"""
        message_lower = message.lower()
        
        # Greeting patterns - use direct LLM
        greeting_patterns = [
            'hello', 'hi', 'hey', 'thanks', 'thank you', 'bye', 'goodbye'
        ]
        
        for pattern in greeting_patterns:
            if pattern in message_lower:
                return False
        
        # Document-related patterns - use RAG
        rag_indicators = [
            'document', 'doc', 'file', 'pdf', 'summarize', 'analyze', 
            'what does', 'according to', 'mentioned', 'extract', 'find'
        ]
        
        for indicator in rag_indicators:
            if indicator in message_lower:
                return True
        
        # Default based on length and question words
        question_words = ['what', 'how', 'when', 'where', 'why', 'who']
        starts_with_question = any(message_lower.startswith(word) for word in question_words)
        
        return starts_with_question and len(message) > 40

    def _handle_direct_llm_query(self, session_id: str, message: str, session: dict) -> Tuple[str, List]:
        """Handle query using direct Ollama client"""
        try:
            conversation_history = db.get_conversation_history(session_id)
            model = session.get('model', 'qwen3:8b')
            
            response_text = ollama_client.chat(
                message=message,
                model=model,
                conversation_history=conversation_history,
                enable_thinking=False
            )
            
            return response_text, []
            
        except Exception as e:
            logger.error(f"Direct LLM error: {e}")
            return f"Error processing query: {str(e)}", []

    def _handle_rag_query_internal(self, session_id: str, message: str, data: dict, idx_ids: List[str]) -> Tuple[str, List]:
        """Handle RAG query using internal RAG agent (no HTTP calls)"""
        try:
            # Apply index embedding model if specified
            self._apply_index_embedding_model(idx_ids)
            
            # Load overviews for session
            RAG_AGENT.load_overviews_for_indexes(idx_ids)
            
            # Configure table name
            table_name = f"text_pages_{idx_ids[-1]}" if idx_ids else f"text_pages_{session_id}"
            
            # Extract parameters
            retrieval_k = data.get('retrieval_k', 20)
            context_window_size = data.get('context_window_size', 1)
            reranker_top_k = data.get('reranker_top_k', 10)
            search_type = data.get('search_type', 'hybrid')
            dense_weight = data.get('dense_weight', 0.7)
            force_rag = bool(data.get('force_rag', False))
            
            # Configure RAG pipeline
            rp_cfg = RAG_AGENT.retrieval_pipeline.config
            rp_cfg["retrieval_k"] = retrieval_k
            rp_cfg.setdefault("reranker", {})["top_k"] = reranker_top_k
            rp_cfg.setdefault("retrieval", {})["search_type"] = search_type
            rp_cfg.setdefault("retrieval", {}).setdefault("dense", {})["weight"] = dense_weight
            rp_cfg["overview_path"] = f"index_store/overviews/{session_id}.jsonl"
            rp_cfg.setdefault("retrievers", {}).setdefault("latechunk", {})["enabled"] = True
            
            # Run RAG agent
            if force_rag:
                result = RAG_AGENT.retrieval_pipeline.run(
                    message,
                    table_name=table_name,
                    window_size_override=context_window_size
                )
            else:
                result = RAG_AGENT.run(
                    message,
                    table_name=table_name,
                    session_id=session_id,
                    retrieval_k=retrieval_k,
                    context_window_size=context_window_size,
                    reranker_top_k=reranker_top_k,
                    search_type=search_type,
                    dense_weight=dense_weight
                )
            
            response_text = result.get("answer", "No answer found.")
            source_docs = result.get("source_documents", [])
            
            # Clean any thinking tags
            response_text = re.sub(r'<(think|thinking)>.*?</\1>', '', response_text, flags=re.DOTALL | re.IGNORECASE).strip()
            
            return response_text, source_docs
            
        except Exception as e:
            logger.error(f"RAG processing error: {e}")
            return f"Error processing RAG query: {str(e)}", []

    def _apply_index_embedding_model(self, idx_ids: List[str]):
        """Apply embedding model from index metadata"""
        if not idx_ids:
            return
        try:
            idx = db.get_index(idx_ids[0])
            model = (idx.get("metadata") or {}).get("embedding_model")
            if model:
                RAG_AGENT.retrieval_pipeline.update_embedding_model(model)
        except Exception as e:
            logger.warning(f"Could not apply index embedding model: {e}")

    # RAG-specific endpoints
    def handle_rag_chat(self):
        """Direct RAG chat endpoint"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            query = data.get('query')
            session_id = data.get('session_id')
            
            if not query:
                self.send_json_response({"error": "Query is required"}, status_code=400)
                return

            # Update session title and store message if session provided
            if session_id:
                try:
                    session = db.get_session(session_id)
                    if session and session.get('message_count', 0) == 0:
                        title = generate_session_title(query)
                        db.update_session_title(session_id, title)
                    
                    db.add_message(session_id, query, "user")
                except Exception as e:
                    logger.warning(f"Failed to update session: {e}")

            # Configure table name
            table_name = data.get('table_name')
            if not table_name and session_id:
                table_name = f"text_pages_{session_id}"

            # Apply index embedding model
            if session_id:
                idx_ids = db.get_indexes_for_session(session_id)
                self._apply_index_embedding_model(idx_ids)
                RAG_AGENT.load_overviews_for_indexes(idx_ids)

            # Extract parameters
            retrieval_k = data.get('retrieval_k', 20)
            context_window_size = data.get('context_window_size', 1)
            reranker_top_k = data.get('reranker_top_k', 10)
            search_type = data.get('search_type', 'hybrid')
            dense_weight = data.get('dense_weight', 0.7)
            force_rag = bool(data.get('force_rag', False))

            # Configure pipeline
            rp_cfg = RAG_AGENT.retrieval_pipeline.config
            if session_id:
                rp_cfg["overview_path"] = f"index_store/overviews/{session_id}.jsonl"
            rp_cfg.setdefault("retrievers", {}).setdefault("latechunk", {})["enabled"] = True

            # Run agent
            if force_rag:
                result = RAG_AGENT.retrieval_pipeline.run(
                    query,
                    table_name=table_name,
                    window_size_override=context_window_size
                )
            else:
                result = RAG_AGENT.run(
                    query,
                    table_name=table_name,
                    session_id=session_id,
                    retrieval_k=retrieval_k,
                    context_window_size=context_window_size,
                    reranker_top_k=reranker_top_k,
                    search_type=search_type,
                    dense_weight=dense_weight
                )
            
            # Store AI response if session provided
            if session_id and result and result.get("answer"):
                try:
                    db.add_message(session_id, result["answer"], "assistant")
                except Exception as e:
                    logger.warning(f"Failed to store AI response: {e}")
            
            self.send_json_response(result)

        except Exception as e:
            logger.error(f"RAG chat error: {e}")
            self.send_json_response({"error": f"Server error: {str(e)}"}, status_code=500)

    # Additional helper methods for document overviews, file uploads, etc.
    def _load_document_overviews(self, idx_ids: List[str]) -> List[str]:
        """Load document overviews for routing decisions"""
        import json as _json
        
        aggregated = []
        
        # Try per-index files first
        for idx in idx_ids:
            candidate_paths = [
                f"index_store/overviews/{idx}.jsonl",
                f"../index_store/overviews/{idx}.jsonl"
            ]
            for p in candidate_paths:
                if os.path.exists(p):
                    try:
                        with open(p, "r", encoding="utf-8") as f:
                            for line in f:
                                if line.strip():
                                    try:
                                        record = _json.loads(line)
                                        overview = record.get("overview", "").strip()
                                        if overview:
                                            aggregated.append(overview)
                                    except _json.JSONDecodeError:
                                        continue
                        break
                    except Exception as e:
                        logger.warning(f"Error reading {p}: {e}")
                        break
        
        return aggregated[:40]  # Limit for performance

    def _route_using_overviews(self, query: str, overviews: List[str]) -> bool:
        """Use LLM to make routing decisions based on document overviews"""
        if not overviews:
            return False
        
        overviews_block = "\n".join(f"[{i+1}] {ov}" for i, ov in enumerate(overviews))
        
        router_prompt = f"""You are an AI router deciding whether a user question should be answered via:
• "USE_RAG" – search the user's private documents (described below)  
• "DIRECT_LLM" – reply from general knowledge (greetings, public facts, unrelated topics)

RULES:
1. If ANY overview clearly relates to the question → USE_RAG
2. For document operations (summarize, analyze, explain, extract, find) → USE_RAG  
3. For greetings only ("Hi", "Hello", "Thanks") → DIRECT_LLM
4. When in doubt → USE_RAG

DOCUMENT OVERVIEWS:
{overviews_block}

USER QUERY: "{query}"

Respond with exactly one word: USE_RAG or DIRECT_LLM"""

        try:
            response = ollama_client.chat(
                message=router_prompt,
                model="qwen3:0.6b",
                enable_thinking=False
            )
            
            decision = response.strip().upper()
            
            if "USE_RAG" in decision:
                logger.info(f"Overview-based routing: USE_RAG for '{query[:50]}...'")
                return True
            elif "DIRECT_LLM" in decision:
                logger.info(f"Overview-based routing: DIRECT_LLM for '{query[:50]}...'")
                return False
            else:
                logger.warning(f"Unclear routing decision '{decision}', defaulting to RAG")
                return True
                
        except Exception as e:
            logger.error(f"LLM routing failed: {e}")
            return self._simple_pattern_routing(query)

    # File upload and indexing methods
    def handle_file_upload(self, session_id: str):
        """Handle file uploads for a session"""
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={'REQUEST_METHOD': 'POST', 'CONTENT_TYPE': self.headers['Content-Type']}
        )

        uploaded_files = []
        if 'files' in form:
            files = form['files']
            if not isinstance(files, list):
                files = [files]
            
            upload_dir = "shared_uploads"
            os.makedirs(upload_dir, exist_ok=True)

            for file_item in files:
                if file_item.filename:
                    unique_filename = f"{uuid.uuid4()}_{file_item.filename}"
                    file_path = os.path.join(upload_dir, unique_filename)
                    
                    with open(file_path, 'wb') as f:
                        f.write(file_item.file.read())
                    
                    absolute_file_path = os.path.abspath(file_path)
                    db.add_document_to_session(session_id, absolute_file_path)
                    uploaded_files.append({"filename": file_item.filename, "stored_path": absolute_file_path})

        if not uploaded_files:
            self.send_json_response({"error": "No files were uploaded"}, status_code=400)
            return
            
        self.send_json_response({
            "message": f"Successfully uploaded {len(uploaded_files)} files.",
            "uploaded_files": uploaded_files
        })

    def handle_index_documents(self, session_id: str):
        """Index documents for a session using internal RAG pipeline"""
        try:
            file_paths = db.get_documents_for_session(session_id)
            if not file_paths:
                self.send_json_response({"message": "No documents to index for this session."})
                return

            logger.info(f"Indexing {len(file_paths)} documents for session {session_id}")
            
            # Use internal indexing pipeline
            table_name = f"text_pages_{session_id}"
            
            # Configure indexing pipeline
            import copy
            config_override = copy.deepcopy(INDEXING_PIPELINE.config)
            config_override["storage"]["text_table_name"] = table_name
            config_override.setdefault("retrievers", {}).setdefault("dense", {})["lancedb_table_name"] = table_name
            config_override["overview_path"] = f"index_store/overviews/{session_id}.jsonl"
            config_override.setdefault("retrievers", {}).setdefault("latechunk", {})["enabled"] = True
            
            # Create temporary pipeline with session-specific config
            temp_pipeline = INDEXING_PIPELINE.__class__(
                config_override, 
                INDEXING_PIPELINE.llm_client, 
                INDEXING_PIPELINE.ollama_config
            )
            
            # Run indexing
            temp_pipeline.run(file_paths)
            
            # Update index metadata
            try:
                idx_meta = {
                    "session_linked": True,
                    "retrieval_mode": "hybrid",
                }
                db.update_index_metadata(session_id, idx_meta)
            except Exception as e:
                logger.warning(f"Failed to update index metadata: {e}")
            
            self.send_json_response({
                "message": f"Successfully indexed {len(file_paths)} documents",
                "table_name": table_name
            })

        except Exception as e:
            logger.error(f"Indexing error: {e}")
            self.send_json_response({"error": f"Indexing failed: {str(e)}"}, status_code=500)

    # Model and system info methods
    def handle_get_models(self):
        """Get available models grouped by capability"""
        try:
            generation_models = []
            embedding_models = []
            
            # Get Ollama models if available
            if ollama_client.is_ollama_running():
                all_ollama_models = ollama_client.list_models()
                
                ollama_embedding_models = [m for m in all_ollama_models if any(k in m for k in ['embed','bge','embedding','text'])]
                ollama_generation_models = [m for m in all_ollama_models if m not in ollama_embedding_models]
                
                generation_models.extend(ollama_generation_models)
                embedding_models.extend(ollama_embedding_models)
            
            # Add supported HuggingFace embedding models
            huggingface_embedding_models = [
                "Qwen/Qwen3-Embedding-0.6B",
                "Qwen/Qwen3-Embedding-4B", 
                "Qwen/Qwen3-Embedding-8B"
            ]
            embedding_models.extend(huggingface_embedding_models)
            
            generation_models.sort()
            embedding_models.sort()
            
            self.send_json_response({
                "generation_models": generation_models,
                "embedding_models": embedding_models
            })
        except Exception as e:
            self.send_json_response({
                "error": f"Could not list models: {str(e)}"
            }, status_code=500)

    # Utility methods
    def send_json_response(self, data, status_code: int = 200):
        """Send JSON response with CORS headers"""
        try:
            self.send_response(status_code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
            self.send_header('Access-Control-Allow-Credentials', 'true')
            self.end_headers()
        
            response_bytes = json.dumps(data, indent=2).encode('utf-8')
            self.wfile.write(response_bytes)
        except BrokenPipeError:
            logger.warning("Client disconnected during response")
        except Exception as e:
            logger.error(f"Error sending response: {e}")

    # Additional session and index management methods would go here...
    # (Abbreviated for space - would include all remaining methods from original servers)

    # Missing handler methods - adding them now
    def handle_get_session(self, session_id: str):
        """Get a specific session with its messages"""
        try:
            session = db.get_session(session_id)
            if not session:
                self.send_json_response({
                    "error": "Session not found"
                }, status_code=404)
                return
            
            messages = db.get_messages(session_id)
            
            self.send_json_response({
                "session": session,
                "messages": messages
            })
        except Exception as e:
            self.send_json_response({
                "error": f"Failed to get session: {str(e)}"
            }, status_code=500)

    def handle_get_session_documents(self, session_id: str):
        """Return documents and basic info for a session."""
        try:
            session = db.get_session(session_id)
            if not session:
                self.send_json_response({"error": "Session not found"}, status_code=404)
                return

            docs = db.get_documents_for_session(session_id)
            filenames = [os.path.basename(p).split('_', 1)[-1] if '_' in os.path.basename(p) else os.path.basename(p) for p in docs]

            self.send_json_response({
                "session": session,
                "files": filenames,
                "file_count": len(docs)
            })
        except Exception as e:
            self.send_json_response({"error": f"Failed to get documents: {str(e)}"}, status_code=500)

    def handle_cleanup_sessions(self):
        """Clean up empty sessions"""
        try:
            cleanup_count = db.cleanup_empty_sessions()
            self.send_json_response({
                "message": f"Cleaned up {cleanup_count} empty sessions",
                "cleanup_count": cleanup_count
            })
        except Exception as e:
            self.send_json_response({
                "error": f"Failed to cleanup sessions: {str(e)}"
            }, status_code=500)

    def handle_delete_session(self, session_id: str):
        """Delete a session and its messages"""
        try:
            deleted = db.delete_session(session_id)
            if deleted:
                self.send_json_response({'deleted': deleted})
            else:
                self.send_json_response({'error': 'Session not found'}, status_code=404)
        except Exception as e:
            self.send_json_response({'error': str(e)}, status_code=500)

    def handle_rename_session(self, session_id: str):
        """Rename an existing session title"""
        try:
            session = db.get_session(session_id)
            if not session:
                self.send_json_response({"error": "Session not found"}, status_code=404)
                return

            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self.send_json_response({"error": "Request body required"}, status_code=400)
                return

            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            new_title: str = data.get('title', '').strip()

            if not new_title:
                self.send_json_response({"error": "Title cannot be empty"}, status_code=400)
                return

            db.update_session_title(session_id, new_title)
            updated_session = db.get_session(session_id)

            self.send_json_response({
                "message": "Session renamed successfully",
                "session": updated_session
            })

        except json.JSONDecodeError:
            self.send_json_response({"error": "Invalid JSON"}, status_code=400)
        except Exception as e:
            self.send_json_response({"error": f"Failed to rename session: {str(e)}"}, status_code=500)

    # Index management methods
    def handle_get_indexes(self):
        """Get all indexes"""
        try:
            data = db.list_indexes()
            self.send_json_response({'indexes': data, 'total': len(data)})
        except Exception as e:
            self.send_json_response({'error': str(e)}, status_code=500)

    def handle_get_index(self, index_id: str):
        """Get a specific index"""
        try:
            data = db.get_index(index_id)
            if not data:
                self.send_json_response({'error': 'Index not found'}, status_code=404)
                return
            self.send_json_response(data)
        except Exception as e:
            self.send_json_response({'error': str(e)}, status_code=500)

    def handle_create_index(self):
        """Create a new index"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            name = data.get('name')
            description = data.get('description')
            metadata = data.get('metadata', {})
            if not name:
                self.send_json_response({'error': 'Name required'}, status_code=400)
                return
            idx_id = db.create_index(name, description, metadata)
            self.send_json_response({'index_id': idx_id}, status_code=201)
        except Exception as e:
            self.send_json_response({'error': str(e)}, status_code=500)

    def handle_index_file_upload(self, index_id: str):
        """Handle file upload for an index"""
        form = cgi.FieldStorage(
            fp=self.rfile, 
            headers=self.headers, 
            environ={'REQUEST_METHOD':'POST', 'CONTENT_TYPE': self.headers['Content-Type']}
        )
        uploaded_files = []
        if 'files' in form:
            files = form['files']
            if not isinstance(files, list):
                files = [files]
            upload_dir = 'shared_uploads'
            os.makedirs(upload_dir, exist_ok=True)
            for f in files:
                if f.filename:
                    unique = f"{uuid.uuid4()}_{f.filename}"
                    path = os.path.join(upload_dir, unique)
                    with open(path,'wb') as out: 
                        out.write(f.file.read())
                    db.add_document_to_index(index_id, f.filename, os.path.abspath(path))
                    uploaded_files.append({'filename':f.filename,'stored_path':os.path.abspath(path)})
        if not uploaded_files:
            self.send_json_response({'error':'No files uploaded'}, status_code=400)
            return
        self.send_json_response({'message':f"Uploaded {len(uploaded_files)} files","uploaded_files":uploaded_files})

    def handle_build_index(self, index_id: str):
        """Build/index documents for a specific index"""
        try:
            index = db.get_index(index_id)
            if not index:
                self.send_json_response({'error':'Index not found'}, status_code=404)
                return
            file_paths = [d['stored_path'] for d in index.get('documents',[])]
            if not file_paths:
                self.send_json_response({'error':'No documents to index'}, status_code=400)
                return

            # Parse request body for optional configuration
            latechunk = False
            docling_chunk = False
            chunk_size = 512
            chunk_overlap = 64
            retrieval_mode = 'hybrid'
            window_size = 2
            enable_enrich = True
            embedding_model = None
            enrich_model = None
            batch_size_embed = 50
            batch_size_enrich = 25
            overview_model = None
            
            if 'Content-Length' in self.headers and int(self.headers['Content-Length']) > 0:
                try:
                    length = int(self.headers['Content-Length'])
                    body = self.rfile.read(length)
                    opts = json.loads(body.decode('utf-8'))
                    latechunk = bool(opts.get('latechunk', False))
                    docling_chunk = bool(opts.get('doclingChunk', False))
                    chunk_size = int(opts.get('chunkSize', 512))
                    chunk_overlap = int(opts.get('chunkOverlap', 64))
                    retrieval_mode = str(opts.get('retrievalMode', 'hybrid'))
                    window_size = int(opts.get('windowSize', 2))
                    enable_enrich = bool(opts.get('enableEnrich', True))
                    embedding_model = opts.get('embeddingModel')
                    enrich_model = opts.get('enrichModel')
                    batch_size_embed = int(opts.get('batchSizeEmbed', 50))
                    batch_size_enrich = int(opts.get('batchSizeEnrich', 25))
                    overview_model = opts.get('overviewModel')
                except Exception:
                    pass  # Keep defaults on parse error

            # Use internal indexing pipeline
            table_name = f"text_pages_{index_id}"
            
            # Configure indexing pipeline
            import copy
            config_override = copy.deepcopy(INDEXING_PIPELINE.config)
            config_override["storage"]["text_table_name"] = table_name
            config_override.setdefault("retrievers", {}).setdefault("dense", {})["lancedb_table_name"] = table_name
            config_override["overview_path"] = f"index_store/overviews/{index_id}.jsonl"
            
            # Configure options
            if latechunk:
                config_override.setdefault("retrievers", {}).setdefault("latechunk", {})["enabled"] = True
            if docling_chunk:
                config_override["chunker_mode"] = "docling"
            
            config_override.setdefault("contextual_enricher", {})
            config_override["contextual_enricher"]["enabled"] = enable_enrich
            config_override["contextual_enricher"]["window_size"] = window_size
            
            config_override.setdefault("indexing", {})
            config_override["indexing"]["embedding_batch_size"] = batch_size_embed
            config_override["indexing"]["enrichment_batch_size"] = batch_size_enrich
            
            config_override.setdefault("chunking", {})
            config_override["chunking"]["chunk_size"] = chunk_size
            config_override["chunking"]["chunk_overlap"] = chunk_overlap
            
            if embedding_model:
                config_override["embedding_model_name"] = embedding_model
            if enrich_model:
                config_override["enrich_model"] = enrich_model
            if overview_model:
                config_override["overview_model_name"] = overview_model

            # Create temporary pipeline with overridden config
            temp_pipeline = INDEXING_PIPELINE.__class__(
                config_override, 
                INDEXING_PIPELINE.llm_client, 
                INDEXING_PIPELINE.ollama_config
            )
            temp_pipeline.run(file_paths)

            # Update metadata
            meta_updates = {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "retrieval_mode": retrieval_mode,
                "window_size": window_size,
                "enable_enrich": enable_enrich,
                "latechunk": latechunk,
                "docling_chunk": docling_chunk,
            }
            if embedding_model:
                meta_updates["embedding_model"] = embedding_model
            if enrich_model:
                meta_updates["enrich_model"] = enrich_model
            if overview_model:
                meta_updates["overview_model"] = overview_model
            
            try:
                db.update_index_metadata(index_id, meta_updates)
            except Exception as e:
                logger.warning(f"Failed to update index metadata: {e}")

            self.send_json_response({
                "message": f"Successfully indexed {len(file_paths)} documents",
                "table_name": table_name,
                **meta_updates
            })

        except Exception as e:
            logger.error(f"Index building error: {e}")
            self.send_json_response({"error": f"Index building failed: {str(e)}"}, status_code=500)

    def handle_link_index_to_session(self, session_id: str, index_id: str):
        """Link an index to a session"""
        try:
            db.link_index_to_session(session_id, index_id)
            self.send_json_response({'message':'Index linked to session'})
        except Exception as e:
            self.send_json_response({'error':str(e)}, status_code=500)

    def handle_get_session_indexes(self, session_id: str):
        """Get indexes linked to a session"""
        try:
            idx_ids = db.get_indexes_for_session(session_id)
            indexes = [db.get_index(i) for i in idx_ids if db.get_index(i)]
            self.send_json_response({'indexes': indexes, 'total': len(indexes)})
        except Exception as e:
            self.send_json_response({'error': str(e)}, status_code=500)

    def handle_delete_index(self, index_id: str):
        """Delete an index"""
        try:
            deleted = db.delete_index(index_id)
            if deleted:
                self.send_json_response({'message': 'Index deleted successfully', 'index_id': index_id})
            else:
                self.send_json_response({'error': 'Index not found'}, status_code=404)
        except Exception as e:
            self.send_json_response({'error': str(e)}, status_code=500)

    # RAG-specific endpoints
    def handle_rag_chat_stream(self):
        """Handle streaming RAG chat"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            query = data.get('query')
            session_id = data.get('session_id')
            
            if not query:
                self.send_json_response({"error": "Query is required"}, status_code=400)
                return

            # Prepare response headers for SSE
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'keep-alive')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            def emit(event_type: str, payload):
                """Send a single SSE event."""
                try:
                    data_str = json.dumps({"type": event_type, "data": payload})
                    self.wfile.write(f"data: {data_str}\n\n".encode('utf-8'))
                    self.wfile.flush()
                except BrokenPipeError:
                    raise

            # Configure and run RAG with streaming
            table_name = data.get('table_name')
            if not table_name and session_id:
                table_name = f"text_pages_{session_id}"

            try:
                if session_id:
                    idx_ids = db.get_indexes_for_session(session_id)
                    self._apply_index_embedding_model(idx_ids)
                    RAG_AGENT.load_overviews_for_indexes(idx_ids)

                # Configure pipeline
                rp_cfg = RAG_AGENT.retrieval_pipeline.config
                if session_id:
                    rp_cfg["overview_path"] = f"index_store/overviews/{session_id}.jsonl"
                rp_cfg.setdefault("retrievers", {}).setdefault("latechunk", {})["enabled"] = True

                # Run with streaming
                final_result = RAG_AGENT.run(
                    query,
                    table_name=table_name,
                    session_id=session_id,
                    event_callback=emit
                )

                emit("complete", final_result)
                
            except BrokenPipeError:
                logger.info("Client disconnected from SSE stream.")
            except Exception as e:
                try:
                    emit("error", {"error": str(e)})
                finally:
                    logger.error(f"Stream error: {e}")

        except Exception as e:
            self.send_json_response({"error": f"Server error: {str(e)}"}, status_code=500)

    def handle_rag_index(self):
        """Handle RAG indexing endpoint"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            file_paths = data.get('file_paths')
            session_id = data.get('session_id')
            
            if not file_paths or not isinstance(file_paths, list):
                self.send_json_response({
                    "error": "A 'file_paths' list is required."
                }, status_code=400)
                return

            # Extract configuration
            enable_latechunk = bool(data.get("enable_latechunk", False))
            enable_docling_chunk = bool(data.get("enable_docling_chunk", False))
            chunk_size = int(data.get("chunk_size", 512))
            chunk_overlap = int(data.get("chunk_overlap", 64))
            retrieval_mode = data.get("retrieval_mode", "hybrid")
            window_size = int(data.get("window_size", 2))
            enable_enrich = bool(data.get("enable_enrich", True))
            embedding_model = data.get('embedding_model')
            enrich_model = data.get('enrich_model')
            overview_model = data.get('overview_model_name')
            batch_size_embed = int(data.get("batch_size_embed", 50))
            batch_size_enrich = int(data.get("batch_size_enrich", 25))

            # Configure table name
            table_name = data.get('table_name')
            if not table_name and session_id:
                table_name = f"text_pages_{session_id}"

            # Configure indexing pipeline
            import copy
            config_override = copy.deepcopy(INDEXING_PIPELINE.config)
            if table_name:
                config_override["storage"]["text_table_name"] = table_name
                config_override.setdefault("retrievers", {}).setdefault("dense", {})["lancedb_table_name"] = table_name
            
            # Apply configuration options
            if enable_latechunk:
                config_override.setdefault("retrievers", {}).setdefault("latechunk", {})["enabled"] = True
            if enable_docling_chunk:
                config_override["chunker_mode"] = "docling"
            
            config_override.setdefault("contextual_enricher", {})
            config_override["contextual_enricher"]["enabled"] = enable_enrich
            config_override["contextual_enricher"]["window_size"] = window_size
            
            config_override.setdefault("indexing", {})
            config_override["indexing"]["embedding_batch_size"] = batch_size_embed
            config_override["indexing"]["enrichment_batch_size"] = batch_size_enrich
            
            config_override.setdefault("chunking", {})
            config_override["chunking"]["chunk_size"] = chunk_size
            config_override["chunking"]["chunk_overlap"] = chunk_overlap
            
            if embedding_model:
                config_override["embedding_model_name"] = embedding_model
            if enrich_model:
                config_override["enrich_model"] = enrich_model
            if overview_model:
                config_override["overview_model_name"] = overview_model
            
            if session_id:
                config_override["overview_path"] = f"index_store/overviews/{session_id}.jsonl"

            # Create temporary pipeline and run indexing
            temp_pipeline = INDEXING_PIPELINE.__class__(
                config_override, 
                INDEXING_PIPELINE.llm_client, 
                INDEXING_PIPELINE.ollama_config
            )
            temp_pipeline.run(file_paths)

            self.send_json_response({
                "message": f"Indexing process for {len(file_paths)} file(s) completed successfully.",
                "table_name": table_name or "default_text_table",
                "indexing_config": {
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "retrieval_mode": retrieval_mode,
                    "window_size": window_size,
                    "enable_enrich": enable_enrich,
                    "embedding_model": embedding_model,
                    "enrich_model": enrich_model,
                    "batch_size_embed": batch_size_embed,
                    "batch_size_enrich": batch_size_enrich
                }
            })

        except Exception as e:
            logger.error(f"RAG indexing error: {e}")
            self.send_json_response({"error": f"Failed to start indexing: {str(e)}"}, status_code=500)

    def handle_legacy_chat(self):
        """Handle legacy chat requests (without sessions)"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            message = data.get('message', '')
            model = data.get('model', 'qwen3:8b')
            conversation_history = data.get('conversation_history', [])
            
            if not message:
                self.send_json_response({
                    "error": "Message is required"
                }, status_code=400)
                return
            
            # Check if Ollama is running
            if not ollama_client.is_ollama_running():
                self.send_json_response({
                    "error": "Ollama is not running. Please start Ollama first."
                }, status_code=503)
                return
            
            # Get response from Ollama
            response = ollama_client.chat(message, model, conversation_history)
            
            self.send_json_response({
                "response": response,
                "model": model,
                "message_count": len(conversation_history) + 1
            })
            
        except json.JSONDecodeError:
            self.send_json_response({
                "error": "Invalid JSON"
            }, status_code=400)
        except Exception as e:
            self.send_json_response({
                "error": f"Server error: {str(e)}"
            }, status_code=500)

def main():
    """Main function to start the consolidated server"""
    PORT = int(os.getenv("PORT", 8000))
    
    try:
        logger.info("🔧 Initializing database...")
        
        # Cleanup empty sessions on startup
        cleanup_count = db.cleanup_empty_sessions()
        if cleanup_count > 0:
            logger.info(f"✨ Cleaned up {cleanup_count} empty sessions")

        # Start the server
        with ReusableTCPServer(("", PORT), ConsolidatedHandler) as httpd:
            logger.info(f"🚀 Starting Consolidated RAG Server on port {PORT}")
            logger.info(f"📍 Frontend API: http://localhost:{PORT}")
            logger.info(f"🔍 Health check: http://localhost:{PORT}/health")
            logger.info(f"💬 RAG Chat: http://localhost:{PORT}/rag/chat")
            
            # Test Ollama connection
            if ollama_client.is_ollama_running():
                models = ollama_client.list_models()
                logger.info(f"✅ Ollama running with {len(models)} models")
            else:
                logger.warning("⚠️ Ollama not running. Install from https://ollama.ai")
            
            logger.info("🌐 Ready for deployment!")
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server startup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 