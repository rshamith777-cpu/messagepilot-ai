import os
import sys
import webbrowser
import http.server
import socketserver

PORT = 8080
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))

class QuietHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DASHBOARD_DIR, **kwargs)
    
    def log_message(self, format, *args):
        # Quiet server logs
        pass

def serve():
    url = f"http://localhost:{PORT}/index.html"
    print("\n==========================================================================")
    print("      MESSAGEPILOT AI — INTERACTIVE EXPLAINABILITY DASHBOARD SERVER        ")
    print("==========================================================================")
    print(f"Opening dashboard in your web browser: {url}")
    print("Press Ctrl+C to stop the dashboard server.\n")

    try:
        webbrowser.open(url)
        with socketserver.TCPServer(("", PORT), QuietHTTPRequestHandler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard server stopped successfully.")

if __name__ == "__main__":
    serve()
