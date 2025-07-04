import json
import http.server
import socketserver
import cgi
import os
import uuid
from urllib.parse import urlparse, parse_qs
import requests  # 🆕 Import requests for making HTTP calls
from ollama_client import OllamaClient
from database import db, generate_session_title
import simple_pdf_processor as pdf_module
from simple_pdf_processor import initialize_simple_pdf_processor
from typing import List, Dict, Any
import re

# 🆕 Reusable TCPServer with address reuse enabled
class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True

class ChatHandler(http.server.BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.ollama_client = OllamaClient()
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
        
        if parsed_path.path == '/health':
            self.send_json_response({
                "status": "ok",
                "ollama_running": self.ollama_client.is_ollama_running(),
                "available_models": self.ollama_client.list_models(),
                "database_stats": db.get_stats()
            })
        elif parsed_path.path == '/sessions':
            self.handle_get_sessions()
        elif parsed_path.path == '/sessions/cleanup':
            self.handle_cleanup_sessions()
        elif parsed_path.path == '/models':
            self.handle_get_models()
        elif parsed_path.path == '/indexes':
            self.handle_get_indexes()
        elif parsed_path.path.startswith('/indexes/') and parsed_path.path.count('/') == 2:
            index_id = parsed_path.path.split('/')[-1]
            self.handle_get_index(index_id)
        elif parsed_path.path.startswith('/sessions/') and parsed_path.path.endswith('/documents'):
            session_id = parsed_path.path.split('/')[-2]
            self.handle_get_session_documents(session_id)
        elif parsed_path.path.startswith('/sessions/') and parsed_path.path.endswith('/indexes'):
            session_id = parsed_path.path.split('/')[-2]
            self.handle_get_session_indexes(session_id)
        elif parsed_path.path.startswith('/sessions/') and parsed_path.path.count('/') == 2:
            session_id = parsed_path.path.split('/')[-1]
            self.handle_get_session(session_id)
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        """Handle POST requests"""
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/chat':
            self.handle_chat()
        elif parsed_path.path == '/sessions':
            self.handle_create_session()
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
        elif parsed_path.path.startswith('/sessions/') and parsed_path.path.endswith('/messages'):
            session_id = parsed_path.path.split('/')[-2]
            self.handle_session_chat(session_id)
        elif parsed_path.path.startswith('/sessions/') and parsed_path.path.endswith('/upload'):
            session_id = parsed_path.path.split('/')[-2]
            self.handle_file_upload(session_id)
        elif parsed_path.path.startswith('/sessions/') and parsed_path.path.endswith('/index'):
            session_id = parsed_path.path.split('/')[-2]
            self.handle_index_documents(session_id)
        else:
            self.send_response(404)
            self.end_headers()

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
            self.send_response(404)
            self.end_headers()
    
    def handle_chat(self):
        """Handle legacy chat requests (without sessions)"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            message = data.get('message', '')
            model = data.get('model', 'llama3.2:latest')
            conversation_history = data.get('conversation_history', [])
            
            if not message:
                self.send_json_response({
                    "error": "Message is required"
                }, status_code=400)
                return
            
            # Check if Ollama is running
            if not self.ollama_client.is_ollama_running():
                self.send_json_response({
                    "error": "Ollama is not running. Please start Ollama first."
                }, status_code=503)
                return
            
            # Get response from Ollama
            response = self.ollama_client.chat(message, model, conversation_history)
            
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

            # Extract original filenames from stored paths
            filenames = [os.path.basename(p).split('_', 1)[-1] if '_' in os.path.basename(p) else os.path.basename(p) for p in docs]

            self.send_json_response({
                "session": session,
                "files": filenames,
                "file_count": len(docs)
            })
        except Exception as e:
            self.send_json_response({"error": f"Failed to get documents: {str(e)}"}, status_code=500)
    
    def handle_create_session(self):
        """Create a new chat session"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            title = data.get('title', 'New Chat')
            model = data.get('model', 'llama3.2:latest')
            
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
        """
        Handle chat within a specific session.
        Intelligently routes between direct LLM (fast) and RAG pipeline (document-aware).
        """
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
            
            if session['message_count'] == 0:
                title = generate_session_title(message)
                db.update_session_title(session_id, title)
            
            # 🎯 SMART ROUTING: Decide between direct LLM vs RAG
            idx_ids = db.get_indexes_for_session(session_id)
            force_rag = bool(data.get("force_rag", False))
            use_rag = True if force_rag else self._should_use_rag(message, idx_ids)
            
            if use_rag:
                # 🔍 --- Use RAG Pipeline for Document-Related Queries ---
                print(f"🔍 Using RAG pipeline for document query: '{message[:50]}...'")
                response_text, source_docs = self._handle_rag_query(session_id, message, data, idx_ids)
            else:
                # ⚡ --- Use Direct LLM for General Queries (FAST) ---
                print(f"⚡ Using direct LLM for general query: '{message[:50]}...'")
                response_text, source_docs = self._handle_direct_llm_query(session_id, message, session)

            # Add AI response to database
            ai_message_id = db.add_message(session_id, response_text, "assistant")
            
            updated_session = db.get_session(session_id)
            
            # Send response with proper error handling
            self.send_json_response({
                "response": response_text,
                "session": updated_session,
                "source_documents": source_docs,
                "used_rag": use_rag
            })
            
        except BrokenPipeError:
            # Client disconnected - this is normal for long queries, just log it
            print(f"⚠️  Client disconnected during RAG processing for query: '{message[:30]}...'")
        except json.JSONDecodeError:
            self.send_json_response({
                "error": "Invalid JSON"
            }, status_code=400)
        except Exception as e:
            print(f"❌ Server error in session chat: {str(e)}")
            try:
                self.send_json_response({
                    "error": f"Server error: {str(e)}"
                }, status_code=500)
            except BrokenPipeError:
                print(f"⚠️  Client disconnected during error response")
    
    def _should_use_rag(self, message: str, idx_ids: List[str]) -> bool:
        """
        🧠 ENHANCED: Determine if a query should use RAG pipeline using document overviews.
        
        Args:
            message: The user's query
            idx_ids: List of index IDs associated with the session
            
        Returns:
            bool: True if should use RAG, False for direct LLM
        """
        # No indexes = definitely no RAG needed
        if not idx_ids:
            return False

        # Load document overviews for intelligent routing
        try:
            doc_overviews = self._load_document_overviews()
            if doc_overviews:
                return self._route_using_overviews(message, doc_overviews)
        except Exception as e:
            print(f"⚠️ Overview-based routing failed, falling back to simple routing: {e}")
        
        # Fallback to simple pattern matching if overviews unavailable
        return self._simple_pattern_routing(message, idx_ids)

    def _load_document_overviews(self) -> List[str]:
        """Load document overviews from the index store."""
        import json
        import os
        
        # Fix path: backend server runs from backend/ directory, so we need to go up one level
        overviews_path = "../index_store/overviews/overviews.jsonl"
        if not os.path.exists(overviews_path):
            # Try alternative paths
            alt_paths = [
                "index_store/overviews/overviews.jsonl",  # If running from project root
                "./index_store/overviews/overviews.jsonl",
                "../index_store/overviews/overviews.jsonl"
            ]
            
            for path in alt_paths:
                if os.path.exists(path):
                    overviews_path = path
                    break
            else:
                print(f"⚠️ Could not find overviews.jsonl in any of: {alt_paths}")
                return []
        
        print(f"📖 Loading overviews from: {overviews_path}")
        
        overviews = []
        try:
            with open(overviews_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        overview = data.get('overview', '').strip()
                        if overview:
                            overviews.append(overview)
            
            print(f"✅ Loaded {len(overviews)} document overviews")
            
        except Exception as e:
            print(f"⚠️ Error loading overviews: {e}")
            return []
        
        return overviews[:40]  # Limit to first 40 for performance

    def _route_using_overviews(self, query: str, overviews: List[str]) -> bool:
        """
        🎯 Use document overviews and LLM to make intelligent routing decisions.
        
        Returns True if RAG should be used, False for direct LLM.
        """
        if not overviews:
            return False
        
        # Format overviews for the routing prompt
        overviews_block = "\n".join(f"[{i+1}] {ov}" for i, ov in enumerate(overviews))
        
        router_prompt = f"""You are an AI router deciding whether a user question should be answered via:
• "USE_RAG" – search the user's private documents (described below)  
• "DIRECT_LLM" – reply from general knowledge (greetings, public facts, unrelated topics)

CRITICAL PRINCIPLE: When documents exist in the KB, strongly prefer USE_RAG unless the query is purely conversational or completely unrelated to any possible document content.

RULES:
1. If ANY overview clearly relates to the question (entities, numbers, addresses, dates, amounts, companies, technical terms) → USE_RAG
2. For document operations (summarize, analyze, explain, extract, find) → USE_RAG  
3. For greetings only ("Hi", "Hello", "Thanks") → DIRECT_LLM
4. For pure math/world knowledge clearly unrelated to documents → DIRECT_LLM
5. When in doubt → USE_RAG

DOCUMENT OVERVIEWS:
{overviews_block}

DECISION EXAMPLES:
• "What invoice amounts are mentioned?" → USE_RAG (document-specific)
• "Who is PromptX AI LLC?" → USE_RAG (entity in documents)  
• "What is the DeepSeek model?" → USE_RAG (mentioned in documents)
• "Summarize the research paper" → USE_RAG (document operation)
• "What is 2+2?" → DIRECT_LLM (pure math)
• "Hi there" → DIRECT_LLM (greeting only)

USER QUERY: "{query}"

Respond with exactly one word: USE_RAG or DIRECT_LLM"""

        try:
            # Use Ollama to make the routing decision
            response = self.ollama_client.chat(
                message=router_prompt,
                model="qwen3:0.6b",  # Fast model for routing
                enable_thinking=False  # Fast routing
            )
            
            # The response is directly the text, not a dict
            decision = response.strip().upper()
            
            # Parse decision
            if "USE_RAG" in decision:
                print(f"🎯 Overview-based routing: USE_RAG for query: '{query[:50]}...'")
                return True
            elif "DIRECT_LLM" in decision:
                print(f"⚡ Overview-based routing: DIRECT_LLM for query: '{query[:50]}...'")
                return False
            else:
                print(f"⚠️ Unclear routing decision '{decision}', defaulting to RAG")
                return True  # Default to RAG when uncertain
                
        except Exception as e:
            print(f"❌ LLM routing failed: {e}, falling back to pattern matching")
            return self._simple_pattern_routing(query, [])

    def _simple_pattern_routing(self, message: str, idx_ids: List[str]) -> bool:
        """
        📝 FALLBACK: Simple pattern-based routing (original logic).
        """
        message_lower = message.lower()
        
        # Always use Direct LLM for greetings and casual conversation
        greeting_patterns = [
            'hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening',
            'how are you', 'how do you do', 'nice to meet', 'pleasure to meet',
            'thanks', 'thank you', 'bye', 'goodbye', 'see you', 'talk to you later',
            'test', 'testing', 'check', 'ping', 'just saying', 'nevermind',
            'ok', 'okay', 'alright', 'got it', 'understood', 'i see'
        ]
        
        # Check for greeting patterns
        for pattern in greeting_patterns:
            if pattern in message_lower:
                return False  # Use Direct LLM for greetings
        
        # Keywords that strongly suggest document-related queries
        rag_indicators = [
            'document', 'doc', 'file', 'pdf', 'text', 'content', 'page',
            'according to', 'based on', 'mentioned', 'states', 'says',
            'what does', 'summarize', 'summary', 'analyze', 'analysis',
            'quote', 'citation', 'reference', 'source', 'evidence',
            'explain from', 'extract', 'find in', 'search for'
        ]
        
        # Check for strong RAG indicators
        for indicator in rag_indicators:
            if indicator in message_lower:
                return True
        
        # Question words + substantial length might benefit from RAG
        question_words = ['what', 'how', 'when', 'where', 'why', 'who', 'which']
        starts_with_question = any(message_lower.startswith(word) for word in question_words)
        
        if starts_with_question and len(message) > 40:
            return True
        
        # Very short messages - use direct LLM
        if len(message.strip()) < 20:
            return False
        
        # Default to Direct LLM unless there's clear indication of document query
        return False
    
    def _handle_direct_llm_query(self, session_id: str, message: str, session: dict):
        """
        Handle query using direct Ollama client with thinking disabled for speed.
        
        Returns:
            tuple: (response_text, empty_source_docs)
        """
        try:
            # Get conversation history for context
            conversation_history = db.get_conversation_history(session_id)
            
            # Use the session's model or default
            model = session.get('model', 'qwen3:8b')  # Default to fast model
            
            # Direct Ollama call with thinking disabled for speed
            response_text = self.ollama_client.chat(
                message=message,
                model=model,
                conversation_history=conversation_history,
                enable_thinking=False  # ⚡ DISABLE THINKING FOR SPEED
            )
            
            return response_text, []  # No source docs for direct LLM
            
        except Exception as e:
            print(f"❌ Direct LLM error: {e}")
            return f"Error processing query: {str(e)}", []
    
    def _handle_rag_query(self, session_id: str, message: str, data: dict, idx_ids: List[str]):
        """
        Handle query using the full RAG pipeline.
        
        Returns:
            tuple: (response_text, source_documents)
        """
        response_text = ""
        source_docs: List[dict] = []

        try:
            # The advanced RAG server runs on port 8001
            rag_api_url = "http://localhost:8001/chat"

            # Determine vector table: prefer last linked index if exists
            table_name = None
            if idx_ids:
                table_name = f"text_pages_{idx_ids[-1]}"

            payload: Dict[str, Any] = {"query": message, "session_id": session_id}
            if table_name:
                payload["table_name"] = table_name

            # Extract RAG configuration parameters from the incoming data
            compose_flag = data.get("compose_sub_answers")
            decomp_flag = data.get("query_decompose")
            ai_rerank_flag = data.get("ai_rerank")
            ctx_expand_flag = data.get("context_expand")
            verify_flag = data.get("verify")

            # ✨ NEW RETRIEVAL PARAMETERS (all optional)
            retrieval_k = data.get("retrieval_k")
            context_window_size = data.get("context_window_size")
            reranker_top_k = data.get("reranker_top_k")
            search_type = data.get("search_type")
            dense_weight = data.get("dense_weight")
            provence_prune = data.get("provence_prune")
            provence_threshold = data.get("provence_threshold")

            # Add feature flags to payload if provided
            if compose_flag is not None:
                payload["compose_sub_answers"] = bool(compose_flag)
            if decomp_flag is not None:
                payload["query_decompose"] = bool(decomp_flag)
            if ai_rerank_flag is not None:
                payload["ai_rerank"] = bool(ai_rerank_flag)
            if ctx_expand_flag is not None:
                payload["context_expand"] = bool(ctx_expand_flag)
            if verify_flag is not None:
                payload["verify"] = bool(verify_flag)

            # Add retrieval parameters if provided
            if retrieval_k is not None:
                payload["retrieval_k"] = int(retrieval_k)
            if context_window_size is not None:
                payload["context_window_size"] = int(context_window_size)
            if reranker_top_k is not None:
                payload["reranker_top_k"] = int(reranker_top_k)
            if search_type is not None:
                payload["search_type"] = str(search_type)
            if dense_weight is not None:
                payload["dense_weight"] = float(dense_weight)

            # 🌿 Provence pruning
            if provence_prune is not None:
                payload["provence_prune"] = bool(provence_prune)
            if provence_threshold is not None:
                payload["provence_threshold"] = float(provence_threshold)

            rag_response = requests.post(rag_api_url, json=payload)

            if rag_response.status_code == 200:
                rag_data = rag_response.json()
                response_text = rag_data.get("answer", "No answer found in RAG response.")
                source_docs = rag_data.get("source_documents", [])
            else:
                error_info = rag_response.text
                response_text = f"Error from RAG API: {error_info}"
                print(f"❌ RAG API error ({rag_response.status_code}): {error_info}")

            # 🧹 Clean up any thinking markers that might sneak through
            response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL | re.IGNORECASE)
            response_text = re.sub(r'<thinking>.*?</thinking>', '', response_text, flags=re.DOTALL | re.IGNORECASE)
            response_text = response_text.strip()

            if rag_response.status_code == 200:
                print(f"✅ Received RAG response with {len(source_docs)} source docs.")

        except requests.exceptions.ConnectionError:
            response_text = "Could not connect to the RAG API server. Please ensure it is running."
            print("❌ Connection to RAG API failed. Is the server running on port 8001?")
        except Exception as e:
            response_text = f"Error processing RAG query: {str(e)}"
            print(f"❌ RAG processing error: {e}")

        return response_text, source_docs

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
    
    def handle_file_upload(self, session_id: str):
        """Handle file uploads, save them, and associate with the session."""
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
                    # Create a unique filename to avoid overwrites
                    unique_filename = f"{uuid.uuid4()}_{file_item.filename}"
                    file_path = os.path.join(upload_dir, unique_filename)
                    
                    with open(file_path, 'wb') as f:
                        f.write(file_item.file.read())
                    
                    # Store the absolute path for the indexing service
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
        """Triggers indexing for all documents in a session."""
        print(f"🔥 Received request to index documents for session {session_id[:8]}...")
        try:
            file_paths = db.get_documents_for_session(session_id)
            if not file_paths:
                self.send_json_response({"message": "No documents to index for this session."}, status_code=200)
                return

            print(f"Found {len(file_paths)} documents to index. Sending to RAG API...")
            
            rag_api_url = "http://localhost:8001/index"
            rag_response = requests.post(rag_api_url, json={"file_paths": file_paths, "session_id": session_id})

            if rag_response.status_code == 200:
                print("✅ RAG API successfully indexed documents.")
                self.send_json_response(rag_response.json())
            else:
                error_info = rag_response.text
                print(f"❌ RAG API indexing failed ({rag_response.status_code}): {error_info}")
                self.send_json_response({"error": f"Indexing failed: {error_info}"}, status_code=500)

        except Exception as e:
            print(f"❌ Exception during indexing: {str(e)}")
            self.send_json_response({"error": f"An unexpected error occurred: {str(e)}"}, status_code=500)
            
    def handle_pdf_upload(self, session_id: str):
        """
        Processes PDF files: extracts text and stores it in the database.
        DEPRECATED: This is the old method. Use handle_file_upload instead.
        """
        # This function is now deprecated in favor of the new indexing workflow
        # but is kept for potential legacy/compatibility reasons.
        # For new functionality, it should not be used.
        self.send_json_response({
            "warning": "This upload method is deprecated. Use the new file upload and indexing flow.",
            "message": "No action taken."
        }, status_code=410) # 410 Gone

    def handle_get_models(self):
        """Get available models from both Ollama and HuggingFace, grouped by capability"""
        try:
            generation_models = []
            embedding_models = []
            
            # Get Ollama models if available
            if self.ollama_client.is_ollama_running():
                all_ollama_models = self.ollama_client.list_models()
                
                # Very naive classification - same logic as RAG API server
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
            
            # Sort models for consistent ordering
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

    def handle_get_indexes(self):
        try:
            data = db.list_indexes()
            self.send_json_response({'indexes': data, 'total': len(data)})
        except Exception as e:
            self.send_json_response({'error': str(e)}, status_code=500)
    
    def handle_get_index(self, index_id: str):
        try:
            data = db.get_index(index_id)
            if not data:
                self.send_json_response({'error': 'Index not found'}, status_code=404)
                return
            self.send_json_response(data)
        except Exception as e:
            self.send_json_response({'error': str(e)}, status_code=500)
    
    def handle_create_index(self):
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
        """Reuse file upload logic but store docs under index."""
        form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={'REQUEST_METHOD':'POST', 'CONTENT_TYPE': self.headers['Content-Type']})
        uploaded_files=[]
        if 'files' in form:
            files=form['files']
            if not isinstance(files, list):
                files=[files]
            upload_dir='shared_uploads'
            os.makedirs(upload_dir, exist_ok=True)
            for f in files:
                if f.filename:
                    unique=f"{uuid.uuid4()}_{f.filename}"
                    path=os.path.join(upload_dir, unique)
                    with open(path,'wb') as out: out.write(f.file.read())
                    db.add_document_to_index(index_id, f.filename, os.path.abspath(path))
                    uploaded_files.append({'filename':f.filename,'stored_path':os.path.abspath(path)})
        if not uploaded_files:
            self.send_json_response({'error':'No files uploaded'}, status_code=400); return
        self.send_json_response({'message':f"Uploaded {len(uploaded_files)} files","uploaded_files":uploaded_files})
    
    def handle_build_index(self, index_id: str):
        try:
            index=db.get_index(index_id)
            if not index:
                self.send_json_response({'error':'Index not found'}, status_code=404); return
            file_paths=[d['stored_path'] for d in index.get('documents',[])]
            if not file_paths:
                self.send_json_response({'error':'No documents to index'}, status_code=400); return

            # Parse request body for optional flags and configuration
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
                except Exception:
                    # Keep defaults on parse error
                    pass

            # Delegate to advanced RAG API same as session indexing
            rag_api_url = "http://localhost:8001/index"
            import requests, json as _json
            payload = {
                "file_paths": file_paths, 
                "session_id": index_id,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "retrieval_mode": retrieval_mode,
                "window_size": window_size,
                "enable_enrich": enable_enrich,
                "batch_size_embed": batch_size_embed,
                "batch_size_enrich": batch_size_enrich
            }
            if latechunk:
                payload["enable_latechunk"] = True
            if docling_chunk:
                payload["enable_docling_chunk"] = True
            if embedding_model:
                payload["embedding_model"] = embedding_model
            if enrich_model:
                payload["enrich_model"] = enrich_model
                
            rag_resp = requests.post(rag_api_url, json=payload)
            if rag_resp.status_code==200:
                self.send_json_response({
                    "response": rag_resp.json(),
                    "latechunk": latechunk,
                    "docling_chunk": docling_chunk,
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
            else:
                self.send_json_response({"error":f"RAG indexing failed: {rag_resp.text}"}, status_code=500)
        except Exception as e:
            self.send_json_response({'error':str(e)}, status_code=500)
    
    def handle_link_index_to_session(self, session_id: str, index_id: str):
        try:
            db.link_index_to_session(session_id, index_id)
            self.send_json_response({'message':'Index linked to session'})
        except Exception as e:
            self.send_json_response({'error':str(e)}, status_code=500)

    def handle_get_session_indexes(self, session_id: str):
        try:
            idx_ids = db.get_indexes_for_session(session_id)
            indexes = [db.get_index(i) for i in idx_ids if db.get_index(i)]
            self.send_json_response({'indexes': indexes, 'total': len(indexes)})
        except Exception as e:
            self.send_json_response({'error': str(e)}, status_code=500)

    def handle_delete_index(self, index_id: str):
        """Remove an index, its documents, links, and the underlying LanceDB table."""
        try:
            deleted = db.delete_index(index_id)
            if deleted:
                self.send_json_response({'message': 'Index deleted successfully', 'index_id': index_id})
            else:
                self.send_json_response({'error': 'Index not found'}, status_code=404)
        except Exception as e:
            self.send_json_response({'error': str(e)}, status_code=500)

    def send_json_response(self, data, status_code=200):
        """Send a JSON response with proper error handling"""
        try:
            self.send_response(status_code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
            self.end_headers()
            
            response = json.dumps(data, indent=2)
            self.wfile.write(response.encode('utf-8'))
        except BrokenPipeError:
            # Client disconnected - log but don't crash
            print("⚠️  Client disconnected during response (this is normal for long RAG queries)")
        except Exception as e:
            print(f"❌ Error sending response: {e}")
    
    def log_message(self, format, *args):
        """Custom log format"""
        print(f"[{self.date_time_string()}] {format % args}")

def main():
    """Main function to initialize and start the server"""
    PORT = 8000  # 🆕 Define port
    try:
        # Initialize the database
        print("✅ Database initialized successfully")

        # Initialize the PDF processor
        try:
            pdf_module.initialize_simple_pdf_processor()
            print("📄 Initializing simple PDF processing...")
            if pdf_module.simple_pdf_processor:
                print("✅ Simple PDF processor initialized")
            else:
                print("⚠️ PDF processing could not be initialized.")
        except Exception as e:
            print(f"❌ Error initializing PDF processor: {e}")
            print("⚠️ PDF processing disabled - server will run without RAG functionality")

        # Set a global reference to the initialized processor if needed elsewhere
        global pdf_processor
        pdf_processor = pdf_module.simple_pdf_processor
        if pdf_processor:
            print("✅ Global PDF processor initialized")
        else:
            print("⚠️ PDF processing disabled - server will run without RAG functionality")
        
        # Cleanup empty sessions on startup
        print("🧹 Cleaning up empty sessions...")
        cleanup_count = db.cleanup_empty_sessions()
        if cleanup_count > 0:
            print(f"✨ Cleaned up {cleanup_count} empty sessions")
        else:
            print("✨ No empty sessions to clean up")

        # Start the server
        with ReusableTCPServer(("", PORT), ChatHandler) as httpd:
            print(f"🚀 Starting localGPT backend server on port {PORT}")
            print(f"📍 Chat endpoint: http://localhost:{PORT}/chat")
            print(f"🔍 Health check: http://localhost:{PORT}/health")
            
            # Test Ollama connection
            client = OllamaClient()
            if client.is_ollama_running():
                models = client.list_models()
                print(f"✅ Ollama is running with {len(models)} models")
                print(f"📋 Available models: {', '.join(models[:3])}{'...' if len(models) > 3 else ''}")
            else:
                print("⚠️  Ollama is not running. Please start Ollama:")
                print("   Install: https://ollama.ai")
                print("   Run: ollama serve")
            
            print(f"\n🌐 Frontend should connect to: http://localhost:{PORT}")
            print("💬 Ready to chat!\n")
            
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")

if __name__ == "__main__":
    main() 