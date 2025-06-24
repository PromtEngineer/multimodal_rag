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
        This now delegates RAG queries to the advanced RAG API server.
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
            
            # 🆕 --- Delegate to Advanced RAG API ---
            print(f"🤖 Delegating query to Advanced RAG API: '{message}'")
            response_text = ""
            source_docs = []
            try:
                # The advanced RAG server runs on port 8001
                rag_api_url = "http://localhost:8001/chat"
                conversation_history = db.get_conversation_history(session_id)
                # Determine vector table: prefer last linked index if exists
                idx_ids = db.get_indexes_for_session(session_id)
                table_name = None
                if idx_ids:
                    table_name = f"text_pages_{idx_ids[-1]}"
                payload={"query": message, "session_id": session_id}
                if table_name:
                    payload["table_name"] = table_name
                compose_flag = data.get("compose_sub_answers")
                decomp_flag = data.get("query_decompose")
                ai_rerank_flag = data.get("ai_rerank")
                ctx_expand_flag = data.get("context_expand")
                if compose_flag is not None:
                    payload["compose_sub_answers"] = bool(compose_flag)
                if decomp_flag is not None:
                    payload["query_decompose"] = bool(decomp_flag)
                if ai_rerank_flag is not None:
                    payload["ai_rerank"] = bool(ai_rerank_flag)
                if ctx_expand_flag is not None:
                    payload["context_expand"] = bool(ctx_expand_flag)
                rag_response = requests.post(rag_api_url, json=payload)
                
                if rag_response.status_code == 200:
                    rag_data = rag_response.json()
                    # Extract the final answer from the agent's response
                    response_text = rag_data.get("answer", "No answer found in RAG response.")
                    source_docs = rag_data.get("source_documents", [])
                    print(f"✅ Received response from Advanced RAG API with {len(source_docs)} source docs.")
                else:
                    error_info = rag_response.text
                    response_text = f"Error from Advanced RAG API: {error_info}"
                    source_docs = []
                    print(f"❌ Error from Advanced RAG API ({rag_response.status_code}): {error_info}")

            except requests.exceptions.ConnectionError:
                response_text = "Could not connect to the Advanced RAG API server. Please ensure it is running."
                print("❌ Connection to Advanced RAG API failed. Is the server running on port 8001?")
            # 🆕 --- End Delegation ---

            # Add AI response to database
            ai_message_id = db.add_message(session_id, response_text, "assistant")
            
            updated_session = db.get_session(session_id)
            
            self.send_json_response({
                "response": response_text,
                "session": updated_session,
                "user_message_id": user_message_id,
                "ai_message_id": ai_message_id,
                "source_documents": source_docs
            })
            
        except json.JSONDecodeError:
            self.send_json_response({"error": "Invalid JSON"}, status_code=400)
        except Exception as e:
            self.send_json_response({"error": f"Server error: {str(e)}"}, status_code=500)

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

            # Parse request body for optional flags
            latechunk = False
            docling_chunk = False
            if 'Content-Length' in self.headers and int(self.headers['Content-Length']) > 0:
                try:
                    length = int(self.headers['Content-Length'])
                    body = self.rfile.read(length)
                    opts = json.loads(body.decode('utf-8'))
                    latechunk = bool(opts.get('latechunk'))
                    docling_chunk = bool(opts.get('doclingChunk'))
                except Exception:
                    latechunk = False
                    docling_chunk = False

            # Delegate to advanced RAG API same as session indexing
            rag_api_url = "http://localhost:8001/index"
            import requests, json as _json
            payload = {"file_paths": file_paths, "session_id": index_id}
            if latechunk:
                payload["enable_latechunk"] = True
            if docling_chunk:
                payload["enable_docling_chunk"] = True
            rag_resp = requests.post(rag_api_url, json=payload)
            if rag_resp.status_code==200:
                self.send_json_response({
                    "response": rag_resp.json(),
                    "latechunk": latechunk,
                    "docling_chunk": docling_chunk
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
        """Send JSON response with CORS headers"""
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
        response = json.dumps(data, indent=2)
        self.wfile.write(response.encode('utf-8'))
    
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