from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import time

from tests import run_tests

PORT = 9385

class SimpleHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/run_tests':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)

            # Parse JSON data if needed
            data = json.loads(post_data.decode('utf-8'))
            entry = data.get('entry')
            model_patch = data.get('model_patch')
            use_test_patch = data.get('use_test_patch')
            model_name_or_path = data.get('model_name_or_path')
            test_directives = data.get('test_directives')

            print(f'Running tests for instance: {entry['instance_id']}')
            passed, log_text = run_tests(entry, model_patch, use_test_patch, model_name_or_path, test_directives, test_server_host=None)

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {'passed': passed, 'log_text': log_text}
            self.wfile.write(json.dumps(response).encode('utf-8'))
        else:
            self.send_response(404)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write('Not found'.encode('utf-8'))

def run(server_class=HTTPServer, handler_class=SimpleHandler, port=PORT):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f'Starting server on port {port}...')
    httpd.serve_forever()

if __name__ == '__main__':
    run()