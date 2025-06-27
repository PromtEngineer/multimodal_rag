import json
import http.server
import socketserver
from urllib.parse import urlparse, parse_qs
import os
import requests

# Import the core logic from the new factory script
from rag_system.factory import get_agent, get_indexing_pipeline

# Get the desired agent mode from environment variables, defaulting to 'default'
# This allows us to easily switch between 'default', 'fast', 'react', etc.
AGENT_MODE = os.getenv("RAG_CONFIG_MODE", "default")
RAG_AGENT = get_agent(AGENT_MODE)
INDEXING_PIPELINE = get_indexing_pipeline(AGENT_MODE)

# --- Global Singleton for the RAG Agent ---
# The agent is initialized once when the server starts.
# This avoids reloading all the models on every request.
print("🧠 Initializing RAG Agent with MAXIMUM ACCURACY... (This may take a moment)")
if RAG_AGENT is None:
    print("❌ Critical error: RAG Agent could not be initialized. Exiting.")
    exit(1)
print("✅ RAG Agent initialized successfully with MAXIMUM ACCURACY.")
# ---

class AdvancedRagApiHandler(http.server.BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        """Handle CORS preflight requests for frontend integration."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        """Handle POST requests for chat and indexing."""
        parsed_path = urlparse(self.path)

        if parsed_path.path == '/chat':
            self.handle_chat()
        elif parsed_path.path == '/chat/stream':
            self.handle_chat_stream()
        elif parsed_path.path == '/index':
            self.handle_index()
        else:
            self.send_json_response({"error": "Not Found"}, status_code=404)

    def do_GET(self):
        parsed_path = urlparse(self.path)

        if parsed_path.path == '/models':
            self.handle_models()
        else:
            self.send_json_response({"error": "Not Found"}, status_code=404)

    def handle_chat(self):
        """Handles a chat query by calling the agentic RAG pipeline."""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            query = data.get('query')
            session_id = data.get('session_id')
            compose_flag = data.get('compose_sub_answers')
            decomp_flag = data.get('query_decompose')
            ai_rerank_flag = data.get('ai_rerank')
            ctx_expand_flag = data.get('context_expand')
            if not query:
                self.send_json_response({"error": "Query is required"}, status_code=400)
                return

            # Allow explicit table_name override
            table_name = data.get('table_name')
            if not table_name and session_id:
                table_name = f"text_pages_{session_id}"

            # Use the single, persistent agent instance to run the query
            result = RAG_AGENT.run(query, table_name=table_name, session_id=session_id, compose_sub_answers=compose_flag, query_decompose=decomp_flag, ai_rerank=ai_rerank_flag, context_expand=ctx_expand_flag)
            
            # The result is a dict, so we need to dump it to a JSON string
            self.send_json_response(result)

        except json.JSONDecodeError:
            self.send_json_response({"error": "Invalid JSON"}, status_code=400)
        except Exception as e:
            self.send_json_response({"error": f"Server error: {str(e)}"}, status_code=500)

    def handle_chat_stream(self):
        """Stream internal phases and final answer using SSE (text/event-stream)."""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))

            query = data.get('query')
            session_id = data.get('session_id')
            compose_flag = data.get('compose_sub_answers')
            decomp_flag = data.get('query_decompose')
            ai_rerank_flag = data.get('ai_rerank')
            ctx_expand_flag = data.get('context_expand')

            if not query:
                self.send_json_response({"error": "Query is required"}, status_code=400)
                return

            # Allow explicit table_name override
            table_name = data.get('table_name')
            if not table_name and session_id:
                table_name = f"text_pages_{session_id}"

            # Prepare response headers for SSE
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            # Keep connection alive for SSE; no manual chunked encoding (Python http.server
            # does not add chunk sizes automatically, so declaring it breaks clients).
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
                    # Client disconnected
                    raise

            # Run the agent synchronously, emitting checkpoints
            try:
                final_result = RAG_AGENT.run(
                    query,
                    table_name=table_name,
                    session_id=session_id,
                    compose_sub_answers=compose_flag,
                    query_decompose=decomp_flag,
                    ai_rerank=ai_rerank_flag,
                    context_expand=ctx_expand_flag,
                    event_callback=emit,
                )

                # Ensure the final answer is sent (in case callback missed it)
                emit("complete", final_result)
            except BrokenPipeError:
                print("🔌 Client disconnected from SSE stream.")
            except Exception as e:
                # Send error event then close
                error_payload = {"error": str(e)}
                try:
                    emit("error", error_payload)
                finally:
                    print(f"❌ Stream error: {e}")

        except json.JSONDecodeError:
            self.send_json_response({"error": "Invalid JSON"}, status_code=400)
        except Exception as e:
            self.send_json_response({"error": f"Server error: {str(e)}"}, status_code=500)

    def handle_index(self):
        """Triggers the document indexing pipeline for specific files."""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            file_paths = data.get('file_paths')
            session_id = data.get('session_id')
            enable_latechunk = bool(data.get("enable_latechunk"))
            enable_docling_chunk = bool(data.get("enable_docling_chunk"))
            config_overrides = data.get('config_overrides', {})

            if not file_paths or not isinstance(file_paths, list):
                self.send_json_response({
                    "error": "A 'file_paths' list is required."
                }, status_code=400)
                return

            table_name = data.get('table_name')
            if not table_name and session_id:
                table_name = f"text_pages_{session_id}"

            print(f"--- Received index request for table: {table_name} ---")
            print(f"--- Files: {file_paths} ---")
            if config_overrides:
                print(f"--- Configuration overrides: {config_overrides} ---")
            
            try:
                if table_name:
                    import copy
                    config_override = copy.deepcopy(INDEXING_PIPELINE.config)
                    config_override["storage"]["text_table_name"] = table_name
                    config_override.setdefault("retrievers", {}).setdefault("dense", {})["lancedb_table_name"] = table_name
                    
                    # Apply configuration overrides from frontend
                    self._apply_config_overrides(config_override, config_overrides)
                    
                    if enable_latechunk:
                        config_override["retrievers"].setdefault("latechunk", {})["enabled"] = True
                    else:
                        config_override["retrievers"].setdefault("latechunk", {})["enabled"] = False
                    if enable_docling_chunk:
                        config_override["chunker_mode"] = "docling"
                    
                    print("--- Initializing temporary pipeline with overridden config ---")
                    temp_pipeline = INDEXING_PIPELINE.__class__(
                        config_override, 
                        INDEXING_PIPELINE.llm_client, 
                        INDEXING_PIPELINE.ollama_config
                    )
                    temp_pipeline.run(file_paths)
                    print("--- Temporary pipeline run complete ---")
                else:
                    print("--- Running default pipeline ---")
                    if enable_latechunk:
                        INDEXING_PIPELINE.config.setdefault("retrievers", {}).setdefault("latechunk", {})["enabled"] = True
                    if enable_docling_chunk:
                        INDEXING_PIPELINE.config["chunker_mode"] = "docling"
                    INDEXING_PIPELINE.run(file_paths)
                    print("--- Default pipeline run complete ---")

                self.send_json_response({
                    "message": f"Indexing process for {len(file_paths)} file(s) completed successfully.",
                    "table_name": table_name or "default_text_table",
                    "latechunk": enable_latechunk,
                    "docling_chunk": enable_docling_chunk
                })

            except Exception as e:
                import traceback
                print("!!!!!! ERROR DURING INDEXING PIPELINE EXECUTION !!!!!!")
                traceback.print_exc()
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                self.send_json_response({"error": f"Pipeline Error: {str(e)}"}, status_code=500)

        except json.JSONDecodeError:
            self.send_json_response({"error": "Invalid JSON"}, status_code=400)
        except Exception as e:
            import traceback
            print("!!!!!! ERROR IN /index HANDLER !!!!!!")
            traceback.print_exc()
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            self.send_json_response({"error": f"Failed to start indexing: {str(e)}"}, status_code=500)

    def handle_models(self):
        """Return a list of locally installed Ollama models grouped by capability."""
        try:
            resp = requests.get(f"{RAG_AGENT.ollama_config['host']}/api/tags", timeout=5)
            resp.raise_for_status()
            data = resp.json()

            all_models = [m.get('name') for m in data.get('models', [])]

            # Very naive classification
            embedding_models = [m for m in all_models if any(k in m for k in ['embed','bge','embedding','text'])]
            generation_models = [m for m in all_models if m not in embedding_models]

            self.send_json_response({
                "generation_models": generation_models,
                "embedding_models": embedding_models
            })
        except Exception as e:
            self.send_json_response({"error": f"Could not list models: {e}"}, status_code=500)

    def _apply_config_overrides(self, config, overrides):
        """Apply configuration overrides from frontend to the pipeline config."""
        
        # Apply chunking configuration
        if 'chunking' in overrides:
            chunking_config = overrides['chunking']
            if 'max_chunk_size' in chunking_config:
                # Store in a place where MarkdownRecursiveChunker can use it
                config.setdefault('chunking', {})['max_chunk_size'] = chunking_config['max_chunk_size']
            if 'chunk_overlap' in chunking_config:
                config.setdefault('chunking', {})['chunk_overlap'] = chunking_config['chunk_overlap']
        
        # Apply contextual enricher configuration
        if 'contextual_enricher' in overrides:
            config.setdefault('contextual_enricher', {}).update(overrides['contextual_enricher'])
        
        # Apply indexing batch sizes
        if 'indexing' in overrides:
            config.setdefault('indexing', {}).update(overrides['indexing'])
        
        # Apply embedding model
        if 'embedding_model_name' in overrides:
            config['embedding_model_name'] = overrides['embedding_model_name']
        
        # Apply retriever configuration
        if 'retrievers' in overrides:
            config.setdefault('retrievers', {}).update(overrides['retrievers'])
        
        print(f"✅ Applied configuration overrides to pipeline config")

    def send_json_response(self, data, status_code=200):
        """Utility to send a JSON response with CORS headers."""
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        response = json.dumps(data, indent=2)
        self.wfile.write(response.encode('utf-8'))

def start_server(port=8001):
    """Starts the API server."""
    # Use a reusable TCP server to avoid "address in use" errors on restart
    class ReusableTCPServer(socketserver.TCPServer):
        allow_reuse_address = True

    with ReusableTCPServer(("", port), AdvancedRagApiHandler) as httpd:
        print(f"🚀 Starting Advanced RAG API server on port {port}")
        print(f"💬 Chat endpoint: http://localhost:{port}/chat")
        print(f"✨ Indexing endpoint: http://localhost:{port}/index")
        httpd.serve_forever()

if __name__ == "__main__":
    # To run this server: python -m rag_system.api_server
    start_server() 