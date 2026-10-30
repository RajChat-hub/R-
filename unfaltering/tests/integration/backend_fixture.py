"""Independent standard-library backend used for balancing/health failure injection."""
import hashlib,json,queue,socket,ssl,struct,threading,time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
BLOCK=b'0123456789abcdef'*1024
class BackendServer(ThreadingHTTPServer):
    daemon_threads=True
    request_queue_size=64
    allow_reuse_address=True
    def handle_error(self,request,address):pass # Expected fault-injection disconnects.
class Handler(BaseHTTPRequestHandler):
    protocol_version='HTTP/1.1'
    def log_message(self,*args):pass
    def do_GET(self):self.handle_request()
    def do_POST(self):self.handle_request()
    def handle_request(self):
        self.connection.settimeout(5);b=self.server.backend
        if self.path=='/health':
            with b.lock:b.probes+=1
            if b.health_delay:time.sleep(b.health_delay)
            self.reply(200 if b.healthy else 503,b'healthy' if b.healthy else b'unhealthy');return
        digest=hashlib.sha256();count=0
        def consume(n):
            nonlocal count
            while n:
                part=self.rfile.read(min(n,16384))
                if not part:raise EOFError()
                digest.update(part);count+=len(part);n-=len(part)
        if self.headers.get('Transfer-Encoding','').lower()=='chunked':
            while True:
                n=int(self.rfile.readline(),16)
                if not n:self.rfile.read(2);break
                consume(n);assert self.rfile.read(2)==b'\r\n'
        else:consume(int(self.headers.get('Content-Length','0')))
        b.observed.put({'method':self.command,'path':self.path,'operation':self.headers.get('X-Operation-ID'),'length':count})
        if b.mode=='reset':
            self.connection.setsockopt(socket.SOL_SOCKET,socket.SO_LINGER,struct.pack('ii',1,0));self.close_connection=True;return
        if b.mode=='short':
            self.connection.sendall(b'HTTP/1.1 200 OK\r\nContent-Length: 100\r\nConnection: close\r\n\r\nx');self.close_connection=True;return
        if self.path.endswith('/slow'):time.sleep(.4)
        if self.path.endswith('/large') or self.path.endswith('/memory'):
            n=(128 if self.path.endswith('/memory') else 8)*1024*1024
            self.send_response(200);self.send_header('Content-Length',str(n));self.send_header('Connection','close');self.end_headers()
            for _ in range(n//len(BLOCK)):self.wfile.write(BLOCK)
            self.close_connection=True;return
        if self.path.endswith('/chunk'):
            self.connection.sendall(b'HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n3\r\nabc\r\n4\r\ndefg\r\n0\r\n\r\n');self.close_connection=True;return
        value=json.dumps({'backend':b.name,'method':self.command,'length':count,'sha256':digest.hexdigest(),'headers':dict((k.lower(),v) for k,v in self.headers.items())}).encode()
        self.reply(200,value)
    def reply(self,status,body):
        self.send_response(status);self.send_header('Content-Length',str(len(body)));self.send_header('Connection','close');self.end_headers();self.wfile.write(body);self.close_connection=True
class Backend:
    def __init__(self,name,certs=None,port=0):
        self.name=name;self.healthy=True;self.health_delay=0;self.mode='ok';self.probes=0;self.lock=threading.Lock();self.observed=queue.Queue()
        self.server=BackendServer(('127.0.0.1',port),Handler);self.server.backend=self
        if certs:
            ctx=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);ctx.minimum_version=ssl.TLSVersion.TLSv1_2
            ctx.load_cert_chain(certs/'server.crt',certs/'server.key');self.server.socket=ctx.wrap_socket(self.server.socket,server_side=True)
        self.port=self.server.server_address[1];self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
    def close(self):self.server.shutdown();self.server.server_close();self.thread.join(timeout=4)
