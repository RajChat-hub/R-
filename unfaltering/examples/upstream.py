"""Loopback-only demonstration app; not part of the native server runtime."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json

class App(BaseHTTPRequestHandler):
    protocol_version='HTTP/1.1'
    def do_GET(self):
        self.reply(b'')
    def do_POST(self):
        data=bytearray()
        if self.headers.get('Transfer-Encoding','').lower()=='chunked':
            while True:
                n=int(self.rfile.readline(),16)
                if not n:
                    self.rfile.read(2)
                    break
                data.extend(self.rfile.read(n))
                if self.rfile.read(2)!=b'\r\n': raise ValueError('bad framing')
        else:
            data.extend(self.rfile.read(int(self.headers.get('Content-Length','0'))))
        self.reply(bytes(data))
    def reply(self,body):
        value=json.dumps(dict(method=self.command,path=self.path,bytes=len(body),sha256=hashlib.sha256(body).hexdigest(),host=self.headers.get('Host'),forwarded_for=self.headers.get('X-Forwarded-For'))).encode()
        self.send_response(200)
        self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(value)))
        self.end_headers()
        self.wfile.write(value)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=9000)
    args=parser.parse_args()
    ThreadingHTTPServer(('127.0.0.1',args.port),App).serve_forever()
