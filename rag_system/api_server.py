import json
import http.server
import socketserver
from urllib.parse import urlparse, parse_qs

# Import the core logic from the main RAG system script
from rag_system.main import run_indexing, run_chat

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
        elif parsed_path.path == '/index':
            self.handle_index()
        else:
            self.send_json_response({"error": "Not Found"}, status_code=404)

    def handle_chat(self):
        """Handles a chat query by calling the agentic RAG pipeline."""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            query = data.get('query')
            if not query:
                self.send_json_response({"error": "Query is required"}, status_code=400)
                return

            # run_chat now returns a JSON string
            result_json_string = run_chat(query)
            
            # The result is already a JSON string, so no need to dump it again
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(result_json_string.encode('utf-8'))

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
            if not file_paths or not isinstance(file_paths, list):
                self.send_json_response({
                    "error": "A 'file_paths' list is required."
                }, status_code=400)
                return

            # run_indexing is synchronous and prints progress to console
            run_indexing(file_paths=file_paths)

            self.send_json_response({
                "message": f"Indexing process for {len(file_paths)} file(s) completed successfully."
            })
        except json.JSONDecodeError:
            self.send_json_response({"error": "Invalid JSON"}, status_code=400)
        except Exception as e:
            self.send_json_response({"error": f"Failed to start indexing: {str(e)}"}, status_code=500)
    
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