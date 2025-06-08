import json
import http.server
import socketserver
import cgi
from urllib.parse import urlparse, parse_qs
from ollama_client import OllamaClient
from database import db, generate_session_title
import simple_pdf_processor as pdf_module
from simple_pdf_processor import initialize_simple_pdf_processor

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
        elif parsed_path.path.startswith('/sessions/'):
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
        elif parsed_path.path.startswith('/sessions/') and parsed_path.path.endswith('/messages'):
            session_id = parsed_path.path.split('/')[-2]
            self.handle_session_chat(session_id)
        elif parsed_path.path.startswith('/sessions/') and parsed_path.path.endswith('/upload'):
            session_id = parsed_path.path.split('/')[-2]
            self.handle_pdf_upload(session_id)
        else:
            self.send_response(404)
            self.end_headers()

    def do_DELETE(self):
        """Handle DELETE requests"""
        parsed_path = urlparse(self.path)
        
        if parsed_path.path.startswith('/sessions/') and parsed_path.path.count('/') == 2:
            session_id = parsed_path.path.split('/')[-1]
            self.handle_delete_session(session_id)
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
        """Handle chat within a specific session"""
        try:
            # Check if session exists
            session = db.get_session(session_id)
            if not session:
                self.send_json_response({
                    "error": "Session not found"
                }, status_code=404)
                return
            
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            message = data.get('message', '')
            model = data.get('model', session['model_used'])
            
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
            
            # Add user message to database
            user_message_id = db.add_message(session_id, message, "user")
            
            # Auto-generate title from first message
            if session['message_count'] == 0:
                title = generate_session_title(message)
                db.update_session_title(session_id, title)
            
            # Get conversation history
            conversation_history = db.get_conversation_history(session_id)
            
            # Get all PDF content for this session if available
            pdf_context = ""
            if message.strip() and pdf_module.simple_pdf_processor:
                session_documents = pdf_module.simple_pdf_processor.get_session_documents(session_id)
                if session_documents:
                    print(f"📄 Found {len(session_documents)} documents for context")
                    context_parts = []
                    for doc in session_documents:
                        # Get the full document content from database
                        doc_content = pdf_module.simple_pdf_processor.get_document_content(doc['id'])
                        if doc_content:
                            context_parts.append(f"=== Document: {doc['filename']} ===\n{doc_content}")
                    
                    if context_parts:
                        pdf_context = "\n\n".join(context_parts)
                        print(f"📄 Added {len(pdf_context)} characters of PDF context")
            
            # Augment message with full document context if available
            augmented_message = message
            if pdf_context:
                augmented_message = f"""Based on the following document(s), please answer the question:

DOCUMENT CONTENT:
{pdf_context}

QUESTION: {message}

Please answer based on the provided document content."""
                
                # Replace the last message in conversation history with augmented version
                if conversation_history and conversation_history[-1]['role'] == 'user':
                    conversation_history[-1]['content'] = augmented_message
            
            # Get response from Ollama
            response = self.ollama_client.chat(augmented_message, model, conversation_history[:-1])  # Exclude the just-added user message
            
            # Add AI response to database
            ai_message_id = db.add_message(session_id, response, "assistant")
            
            # Get updated session info
            updated_session = db.get_session(session_id)
            
            self.send_json_response({
                "response": response,
                "session": updated_session,
                "user_message_id": user_message_id,
                "ai_message_id": ai_message_id
            })
            
        except json.JSONDecodeError:
            self.send_json_response({
                "error": "Invalid JSON"
            }, status_code=400)
        except Exception as e:
            self.send_json_response({
                "error": f"Server error: {str(e)}"
            }, status_code=500)

    def handle_delete_session(self, session_id: str):
        """Delete a chat session and all its messages"""
        try:
            # Check if session exists
            session = db.get_session(session_id)
            if not session:
                self.send_json_response({
                    "error": "Session not found"
                }, status_code=404)
                return
            
            # Delete the session (will cascade delete messages)
            success = db.delete_session(session_id)
            
            if success:
                self.send_json_response({
                    "message": "Session deleted successfully",
                    "deleted_session_id": session_id
                })
            else:
                self.send_json_response({
                    "error": "Failed to delete session"
                }, status_code=500)
                
        except Exception as e:
            self.send_json_response({
                "error": f"Failed to delete session: {str(e)}"
            }, status_code=500)
    
    def handle_pdf_upload(self, session_id: str):
        """Handle PDF file upload for a session"""
        try:
            # Check if PDF processor is available
            print(f"🔍 PDF Upload Debug - simple_pdf_processor: {pdf_module.simple_pdf_processor}")
            print(f"🔍 PDF Upload Debug - type: {type(pdf_module.simple_pdf_processor)}")
            if not pdf_module.simple_pdf_processor:
                print("❌ PDF processor is None/False - returning 503")
                self.send_json_response({
                    "error": "PDF processing is not available. Please check server logs."
                }, status_code=503)
                return
            
            # Check if session exists
            session = db.get_session(session_id)
            if not session:
                self.send_json_response({
                    "error": "Session not found"
                }, status_code=404)
                return
            
            # Parse multipart form data
            content_type = self.headers.get('Content-Type', '')
            print(f"📤 Upload Content-Type: {content_type}")
            if not content_type.startswith('multipart/form-data'):
                self.send_json_response({
                    "error": "Expected multipart/form-data"
                }, status_code=400)
                return
            
            # Parse the form data  
            print("📤 Starting form data parsing...")
            try:
                # Set a larger max file size (50MB)
                import tempfile
                tempfile.tempdir = '/tmp'  # Ensure we have space
                
                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={
                        'REQUEST_METHOD': 'POST',
                        'CONTENT_TYPE': self.headers['Content-Type'],
                    },
                    # Increase file size limits
                    keep_blank_values=True,
                )
                print(f"📤 Form parsed successfully, found {len(form.keys())} fields")
            except Exception as parse_error:
                print(f"❌ Form parsing failed: {str(parse_error)}")
                self.send_json_response({
                    "error": f"Failed to parse form data: {str(parse_error)}"
                }, status_code=400)
                return
            
            uploaded_files = []
            processing_results = []
            
            # Process each uploaded file
            print(f"📤 Processing {len(form.keys())} form fields...")
            for field_name in form.keys():
                print(f"📤 Processing field: {field_name}")
                field = form[field_name]
                print(f"📤 Field has filename attr: {hasattr(field, 'filename')}")
                if hasattr(field, 'filename'):
                    print(f"📤 Filename: {field.filename}")
                if hasattr(field, 'filename') and field.filename:
                    # Check if it's a PDF file
                    if not field.filename.lower().endswith('.pdf'):
                        processing_results.append({
                            "filename": field.filename,
                            "success": False,
                            "error": "Only PDF files are supported"
                        })
                        continue
                    
                    # Read file content
                    print(f"📤 Field.file type: {type(field.file)}")
                    print(f"📤 Field.file available methods: {[m for m in dir(field.file) if not m.startswith('_')]}")
                    
                    file_content = field.file.read()
                    print(f"📤 File content size: {len(file_content)} bytes")
                    
                    # Try to get more info about the file
                    if hasattr(field.file, 'tell'):
                        print(f"📤 File position after read: {field.file.tell()}")
                    if hasattr(field.file, 'seek'):
                        field.file.seek(0)  # Reset to beginning
                        file_content = field.file.read()  # Try reading again
                        print(f"📤 File content size after seek/re-read: {len(file_content)} bytes")
                    
                    # Try getvalue() for BytesIO objects
                    if hasattr(field.file, 'getvalue'):
                        buffer_content = field.file.getvalue()
                        print(f"📤 Buffer content via getvalue(): {len(buffer_content)} bytes")
                        if buffer_content and not file_content:
                            file_content = buffer_content
                            print(f"📤 Using buffer content instead")
                    
                    if not file_content:
                        processing_results.append({
                            "filename": field.filename,
                            "success": False,
                            "error": "Empty file - frontend may not be sending file content properly"
                        })
                        continue
                    
                    # Process the PDF
                    print(f"📄 Starting PDF processing for {field.filename} ({len(file_content)} bytes)")
                    try:
                        result = pdf_module.simple_pdf_processor.process_pdf(file_content, field.filename, session_id)
                        print(f"📄 PDF processing result: {result}")
                        processing_results.append(result)
                    except Exception as pdf_error:
                        print(f"❌ PDF processing failed: {str(pdf_error)}")
                        processing_results.append({
                            "filename": field.filename,
                            "success": False,
                            "error": f"PDF processing failed: {str(pdf_error)}"
                        })
                    
                    if result.get("success", False):
                        uploaded_files.append({
                            "filename": field.filename,
                            "file_id": result.get("file_id", ""),
                            "chunks": result.get("chunks", 0),
                            "text_length": result.get("text_length", 0)
                        })
            
            if not processing_results:
                self.send_json_response({
                    "error": "No PDF files found in upload"
                }, status_code=400)
                return
            
            # Get updated session documents
            session_documents = pdf_module.simple_pdf_processor.get_session_documents(session_id)
            
            print(f"✅ PDF processing complete. Attempting to send response...")
            try:
                self.send_json_response({
                    "message": f"Processed {len(uploaded_files)} PDF files",
                    "uploaded_files": uploaded_files,
                    "processing_results": processing_results,
                    "session_documents": session_documents,
                    "total_session_documents": len(session_documents)
                })
                print(f"✅ Response sent successfully")
            except Exception as response_error:
                print(f"❌ Failed to send response: {str(response_error)}")
                # The processing was successful even if response sending failed
                print(f"📊 Processing summary: {len(uploaded_files)} files uploaded, {len(session_documents)} total docs")
                pass
            
        except Exception as e:
            print(f"❌ Error in PDF upload: {str(e)}")
            try:
                self.send_json_response({
                    "error": f"Failed to process PDF upload: {str(e)}"
                }, status_code=500)
            except:
                # Ignore broken pipe errors when sending error response
                print("❌ Failed to send error response (broken pipe)")
                pass
    
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
    PORT = 8000
    
    print(f"✅ Database initialized successfully")
    
    # Initialize simple PDF processor
    print("📄 Initializing simple PDF processing...")
    pdf_initialized = initialize_simple_pdf_processor()
    if pdf_initialized:
        print("✅ Simple PDF processing ready")
    else:
        print("⚠️ PDF processing disabled - server will run without RAG functionality")
    
    # Clean up any empty sessions on startup
    print("🧹 Cleaning up empty sessions...")
    cleanup_count = db.cleanup_empty_sessions()
    if cleanup_count == 0:
        print("✨ No empty sessions to clean up")
    
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
    
    with socketserver.TCPServer(("", PORT), ChatHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Server stopped")

if __name__ == "__main__":
    main() 