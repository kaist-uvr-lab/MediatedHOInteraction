import os
from http.server import SimpleHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs
import base64
import io

class CustomRequestHandler(SimpleHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        content_type = self.headers.get('Content-Type', '')

        if content_length > 0 and content_type == 'image':
            raw_data = self.rfile.read(content_length)
            self.save_image(raw_data)
            response_message = 'Successfully received and saved image.'
        else:
            response_message = 'No image received.'

        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(response_message.encode())

    def save_image(self, image_data):
        image_name = 'received_image.png'
        with open(image_name, 'wb') as image_file:
            image_file.write(image_data)

if __name__ == '__main__':
    server_address = ('', 8000)
    httpd = HTTPServer(server_address, CustomRequestHandler)
    print('Starting server on port 8000...')

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass

    httpd.server_close()
    print('Server stopped.')
