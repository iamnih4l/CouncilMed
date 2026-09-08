import json
from http.server import BaseHTTPRequestHandler, HTTPServer

class MockOllamaHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        print("Received Ollama request:", post_data.decode('utf-8'))
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        # Ollama /api/chat mock response
        response = {
            "model": "llama3:8b-instruct-fp16",
            "message": {
                "role": "assistant",
                "content": "This is a mock Ollama explanation of the report. The primary findings show a high-risk indication that requires further clinical review. The structural features in the MRI confirm this analysis."
            },
            "done": True
        }
        self.wfile.write(json.dumps(response).encode('utf-8'))
        
    def log_message(self, format, *args):
        pass

def run(server_class=HTTPServer, handler_class=MockOllamaHandler, port=11434):
    server_address = ('127.0.0.1', port)
    httpd = server_class(server_address, handler_class)
    print(f'Starting mock Ollama server on port {port}...')
    httpd.serve_forever()

if __name__ == '__main__':
    run()
