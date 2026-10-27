"""Raw independent upstream and real native proxy acceptance tests."""
import hashlib
import json
import os
from pathlib import Path
import queue
import resource
import socket
import socketserver
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest

BINARY=str(Path(sys.argv.pop(1)).resolve())
BLOCK=b'0123456789abcdef'*2048

class Reader:
    def __init__(self,sock): self.sock=sock;self.buf=b''
    def exact(self,n):
        while len(self.buf)<n:
            part=self.sock.recv(65536)
            if not part: raise EOFError('incomplete framed response')
            self.buf+=part
        data,self.buf=self.buf[:n],self.buf[n:]
        return data
    def until(self,end):
        while end not in self.buf:
            part=self.sock.recv(65536)
            if not part: raise EOFError('missing delimiter')
            self.buf+=part
        data,self.buf=self.buf.split(end,1)
        return data
    def response(self,head=False,digest=False):
        lines=self.until(b'\r\n\r\n').decode('latin1').split('\r\n')
        status=int(lines[0].split()[1]);headers={}
        for line in lines[1:]:
            k,v=line.split(':',1);headers[k.lower()]=v.strip()
        result=hashlib.sha256() if digest else bytearray();count=0
        def add(data):
            nonlocal count
            count+=len(data)
            result.update(data) if digest else result.extend(data)
        if not head and status not in (204,304):
            if headers.get('transfer-encoding')=='chunked':
                while True:
                    n=int(self.until(b'\r\n'),16)
                    if not n:
                        if self.exact(2)!=b'\r\n': raise ValueError('bad end chunk')
                        break
                    while n:
                        size=min(n,32768);add(self.exact(size));n-=size
                    if self.exact(2)!=b'\r\n': raise ValueError('bad delimiter')
            else:
                n=int(headers['content-length'])
                while n:
                    size=min(n,32768);add(self.exact(size));n-=size
        return status,headers,(count,result.hexdigest()) if digest else bytes(result)

class Upstream(socketserver.ThreadingTCPServer):
    allow_reuse_address=True
    daemon_threads=True
    def __init__(self):
        self.observed=queue.Queue();self.streaming=threading.Event()
        super().__init__(('127.0.0.1',0),Handler)

class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        try: self.serve()
        except (OSError,EOFError,ValueError): pass
    def serve(self):
        self.connection.settimeout(8)
        line=self.rfile.readline(8192).decode().strip()
        if not line: return
        method,path,version=line.split();headers={}
        while True:
            line=self.rfile.readline(8192)
            if line==b'\r\n': break
            if not line: raise EOFError()
            k,v=line.decode('latin1').split(':',1);headers[k.lower()]=v.strip()
        if path.endswith('/early'):
            self.connection.sendall(b'HTTP/1.1 413 Payload Too Large\r\nContent-Length: 0\r\n\r\n');return
        if path.endswith('/noread'):
            time.sleep(1.5);return
        body=bytearray()
        if headers.get('transfer-encoding')=='chunked':
            while True:
                n=int(self.rfile.readline(),16)
                if not n:
                    if self.rfile.read(2)!=b'\r\n': raise ValueError()
                    break
                body+=self.rfile.read(n)
                if self.rfile.read(2)!=b'\r\n': raise ValueError()
        else:
            n=int(headers.get('content-length','0'))
            while n:
                if path.endswith('/slow-upload'): time.sleep(.001)
                chunk=self.rfile.read(min(16384,n))
                if not chunk: raise EOFError()
                body+=chunk;n-=len(chunk)
                if len(body)>=32768: self.server.streaming.set()
        self.server.observed.put((method,path,headers,len(body)))
        if path.endswith('/timeout'):
            time.sleep(1);return
        if path.endswith('/reset'):
            self.connection.setsockopt(socket.SOL_SOCKET,socket.SO_LINGER,struct.pack('ii',1,0));return
        malformed={
            '/bad-cl':b'Content-Length: 1\r\nContent-Length: 2\r\n',
            '/bad-tecl':b'Content-Length: 1\r\nTransfer-Encoding: chunked\r\n',
            '/bad-fold':b'Content-Length: 0\r\n X: y\r\n',
            '/bad-coding':b'Transfer-Encoding: gzip, chunked\r\n'}
        for suffix,value in malformed.items():
            if path.endswith(suffix):
                self.connection.sendall(b'HTTP/1.1 200 OK\r\n'+value+b'\r\nX');return
        if path.endswith('/short'):
            self.connection.sendall(b'HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\nshort');return
        if path.endswith('/extra'):
            self.connection.sendall(b'HTTP/1.1 200 OK\r\nContent-Length: 1\r\n\r\nXY');return
        if path.endswith('/bad-chunk'):
            self.connection.sendall(b'HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\nZ\r\n');return
        if path.endswith('/trailer'):
            self.connection.sendall(b'HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n0\r\nX: y\r\n\r\n');return
        if path.endswith('/interim'):
            self.connection.sendall(b'HTTP/1.1 103 Early Hints\r\nLink: </x>\r\n\r\n')
        if path.endswith('/eof'):
            self.connection.sendall(b'HTTP/1.0 200 OK\r\nContent-Type: text/plain\r\n\r\neof-body');return
        if path.endswith('/chunk'):
            wire=b'HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close, X-Hop\r\nX-Hop: secret\r\n\r\n3\r\nabc\r\n4\r\ndefg\r\n0\r\n\r\n'
            for i in range(0,len(wire),3): self.connection.sendall(wire[i:i+3])
            return
        if path.endswith('/large') or path.endswith('/memory'):
            n=(128 if path.endswith('/memory') else 8)*1024*1024
            self.connection.sendall(b'HTTP/1.1 200 OK\r\nContent-Length: '+str(n).encode()+b'\r\n\r\n')
            for _ in range(n//len(BLOCK)): self.connection.sendall(BLOCK)
            return
        if path.endswith('/204'):
            self.connection.sendall(b'HTTP/1.1 204 No Content\r\n\r\n');return
        result=json.dumps(dict(method=method,path=path,headers=headers,length=len(body),sha256=hashlib.sha256(body).hexdigest())).encode()
        self.connection.sendall(b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: '+str(len(result)).encode()+b'\r\nConnection: close\r\n\r\n')
        if method!='HEAD': self.connection.sendall(result)

class ProxyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='unfaltering-proxy-');cls.root=Path(cls.temp.name)
        (cls.root/'index.html').write_text('static-alongside-proxy')
        cls.up=Upstream();cls.up_thread=threading.Thread(target=cls.up.serve_forever,daemon=True)
        cls.up_thread.start();up_port=cls.up.server_address[1]
        cls.blackhole=socket.socket();cls.blackhole.bind(('127.0.0.1',0));cls.blackhole.listen(1);cls.fill=[]
        for _ in range(2):
            s=socket.socket();s.settimeout(.1);s.connect(cls.blackhole.getsockname());cls.fill.append(s)
        with socket.socket() as s: s.bind(('127.0.0.1',0));cls.port=s.getsockname()[1]
        with socket.socket() as s: s.bind(('127.0.0.1',0));unused=s.getsockname()[1]
        cls.config=cls.root/'proxy.ufc'
        cls.config.write_text(f'''server {{
 listen "127.0.0.1:{cls.port}"; root "."; max_connections 32;
 request_ms 10000; idle_ms 1000; write_ms 2000; shutdown_ms 500;
 body_bytes 16777216; buffer_bytes 65536; memory_budget_bytes 33554432;
 upstream app {{ address "127.0.0.1:{up_port}"; host "app.internal"; response_ms 15000; }}
 upstream fast {{ address "127.0.0.1:{up_port}"; response_ms 200; write_ms 200; }}
 upstream noforward {{ address "127.0.0.1:{up_port}"; forwarding off; }}
 upstream refused {{ address "127.0.0.1:{unused}"; connect_ms 200; }}
 upstream stalled {{ address "127.0.0.1:{cls.blackhole.getsockname()[1]}"; connect_ms 200; response_ms 2000; }}
 route prefix "/proxy" {{ proxy app; }}
 route prefix "/timeout" {{ proxy fast; }}
 route prefix "/off" {{ proxy noforward; }}
 route exact "/refused" {{ proxy refused; }}
 route exact "/connect-timeout" {{ proxy stalled; }}
}}''')
        cls.log=open(cls.root/'server.log','wb')
        cls.proc=subprocess.Popen([BINARY,'serve','--config',str(cls.config)],stderr=cls.log)
        for _ in range(200):
            if cls.proc.poll() is not None: raise RuntimeError((cls.root/'server.log').read_text())
            try:
                with socket.create_connection(('127.0.0.1',cls.port),timeout=.1): return
            except OSError: time.sleep(.01)
        raise RuntimeError('proxy did not start')
    @classmethod
    def tearDownClass(cls):
        if cls.proc.poll() is None: cls.proc.terminate()
        code=cls.proc.wait(timeout=5);cls.log.close()
        cls.up.shutdown();cls.up.server_close();cls.up_thread.join(timeout=3)
        for s in cls.fill: s.close()
        cls.blackhole.close()
        log=(cls.root/'server.log').read_text()
        if code or 'runtime error:' in log or 'AddressSanitizer' in log: raise AssertionError(log)
        entries=[json.loads(line) for line in log.splitlines() if line.startswith('{')]
        stats=[e for e in entries if e.get('event')=='stopped']
        if not stats or stats[-1]['peak_proxy_queue_bytes']>4*65536: raise AssertionError('wire queue bound failed')
        peak=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        print(f'Child peak RSS {peak} KiB; peak checked proxy queue bytes {stats[-1]["peak_proxy_queue_bytes"]}')
        if os.environ.get('UNFALTERING_INSTRUMENTED')!='ON' and peak>64*1024: raise AssertionError('noninstrumented child exceeds 64 MiB RSS threshold')
        cls.temp.cleanup()
    def connect(self):
        s=socket.create_connection(('127.0.0.1',self.port),timeout=4);s.settimeout(20)
        self.addCleanup(s.close);return s
    def query(self,path='/proxy/echo',method='GET',body=b'',headers=b''):
        s=self.connect()
        s.sendall(method.encode()+b' '+path.encode()+b' HTTP/1.1\r\nHost: public.test:8080\r\nContent-Length: '+str(len(body)).encode()+b'\r\n'+headers+b'\r\n'+body)
        return Reader(s).response(head=method=='HEAD')
    def test_get_post_and_static(self):
        for method,body in [('GET',b''),('POST',b'hello-body')]:
            status,h,b=self.query(method=method,body=body)
            self.assertEqual(status,200);v=json.loads(b)
            self.assertEqual(v['method'],method);self.assertEqual(v['length'],len(body))
            self.assertEqual(v['sha256'],hashlib.sha256(body).hexdigest())
        self.assertEqual(self.query('/')[2],b'static-alongside-proxy')
    def test_forwarding_and_connection_named_headers(self):
        forged=b'Forwarded: for=evil\r\nX-Forwarded-For: evil\r\nX-Forwarded-Proto: https\r\nX-Forwarded-Host: evil\r\nX-Real-IP: evil\r\nConnection: keep-alive, X-Private\r\nX-Private: secret\r\nTE: trailers\r\nUpgrade: websocket\r\n'
        h=json.loads(self.query(headers=forged)[2])['headers']
        self.assertEqual(h['host'],'app.internal');self.assertEqual(h['x-forwarded-for'],'127.0.0.1')
        self.assertEqual(h['x-forwarded-proto'],'http');self.assertEqual(h['x-forwarded-host'],'public.test:8080')
        for name in ['forwarded','x-real-ip','x-private','te','upgrade']: self.assertNotIn(name,h)
        self.assertEqual(h['connection'],'close')
        h=json.loads(self.query('/off/echo',headers=forged)[2])['headers']
        for name in ['forwarded','x-real-ip','x-forwarded-for','x-forwarded-proto','x-forwarded-host']: self.assertNotIn(name,h)
    def test_chunked_request_and_response(self):
        s=self.connect();s.sendall(b'POST /proxy/echo HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n')
        for p in [b'3\r\nabc\r\n',b'2\r\nde\r\n',b'0\r\n\r\n']: s.sendall(p)
        status,h,b=Reader(s).response();self.assertEqual(status,200)
        self.assertEqual(json.loads(b)['sha256'],hashlib.sha256(b'abcde').hexdigest())
        status,h,b=self.query('/proxy/chunk');self.assertEqual(b,b'abcdefg')
        self.assertNotIn('x-hop',h);self.assertEqual(h['transfer-encoding'],'chunked')
    def test_persistent_pipeline(self):
        s=self.connect();r=Reader(s)
        s.sendall(b'GET /proxy/echo HTTP/1.1\r\nHost: x\r\n\r\nGET / HTTP/1.1\r\nHost: x\r\n\r\nGET /proxy/chunk HTTP/1.1\r\nHost: x\r\n\r\n')
        self.assertEqual(r.response()[0],200);self.assertEqual(r.response()[2],b'static-alongside-proxy')
        self.assertEqual(r.response()[2],b'abcdefg')
    def test_head_no_content_and_informational(self):
        self.assertEqual(self.query(method='HEAD')[2],b'')
        self.assertEqual(self.query('/proxy/204')[0],204);self.assertEqual(self.query('/proxy/interim')[0],200)
    def test_close_delimited_response(self):
        status,h,b=self.query('/proxy/eof');self.assertEqual(status,200)
        self.assertEqual(b,b'eof-body');self.assertEqual(h['transfer-encoding'],'chunked')
    def test_fragmented_request(self):
        s=self.connect()
        for b in b'GET /proxy/echo HTTP/1.1\r\nHost: x\r\n\r\n': s.sendall(bytes([b]))
        self.assertEqual(Reader(s).response()[0],200)
    def test_gateway_errors_and_timeout(self):
        self.assertEqual(self.query('/refused')[0],502)
        start=time.monotonic();self.assertEqual(self.query('/timeout/timeout')[0],504)
        self.assertLess(time.monotonic()-start,2);self.assertEqual(self.query('/connect-timeout')[0],504)
        self.assertEqual(self.query('/proxy/reset',method='POST',body=b'one-operation')[0],502)
    def test_invalid_upstream_header_framing(self):
        for path in ['bad-cl','bad-tecl','bad-fold','bad-coding']:
            with self.subTest(path=path): self.assertEqual(self.query('/proxy/'+path)[0],502)
    def test_invalid_upstream_body_framing(self):
        for path in ['bad-chunk','trailer','short']:
            try: self.assertEqual(self.query('/proxy/'+path)[0],502)
            except EOFError: pass
    def test_boundary_overflow_never_reaches_client(self):
        try:
            status,h,b=self.query('/proxy/extra')
            self.assertIn(status,[200,502])
            if status==200: self.assertEqual(b,b'X')
        except EOFError: pass
        self.assertEqual(self.query()[0],200)
    def test_smuggling_is_rejected(self):
        for framing in [b'Content-Length: 1\r\nTransfer-Encoding: chunked',b'Content-Length: 1\r\nContent-Length: 1',b'Transfer-Encoding: gzip, chunked']:
            s=self.connect();s.sendall(b'POST /proxy/echo HTTP/1.1\r\nHost: x\r\n'+framing+b'\r\n\r\nX')
            self.assertEqual(Reader(s).response()[0],400)
    def test_large_stream_and_backpressure(self):
        s=self.connect();s.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,65536)
        s.sendall(b'GET /proxy/large HTTP/1.1\r\nHost: x\r\n\r\n');time.sleep(.15)
        status,h,result=Reader(s).response(digest=True)
        self.assertEqual(status,200);self.assertEqual(result[0],8*1024*1024)
        self.assertEqual(result[1],hashlib.sha256(BLOCK*256).hexdigest())
    def test_memory_bound_with_128_mib_stream(self):
        s=self.connect();s.sendall(b'GET /proxy/memory HTTP/1.1\r\nHost: x\r\n\r\n')
        status,h,result=Reader(s).response(digest=True)
        self.assertEqual(status,200);self.assertEqual(result[0],128*1024*1024)
        digest=hashlib.sha256()
        for _ in range(4096): digest.update(BLOCK)
        self.assertEqual(result[1],digest.hexdigest())
    def test_request_upload_streams_before_completion(self):
        self.up.streaming.clear();s=self.connect()
        s.sendall(b'POST /proxy/echo HTTP/1.1\r\nHost: x\r\nContent-Length: 131072\r\n\r\n'+b'x'*65536)
        self.assertTrue(self.up.streaming.wait(timeout=3),'upstream must receive before upload completion')
        s.sendall(b'x'*65536);self.assertEqual(json.loads(Reader(s).response()[2])['length'],131072)
    def test_slow_upload_and_early_rejection(self):
        body=b'z'*(2*1024*1024)
        self.assertEqual(json.loads(self.query('/proxy/slow-upload',method='POST',body=body)[2])['length'],len(body))
        s=self.connect();s.sendall(b'POST /proxy/early HTTP/1.1\r\nHost: x\r\nContent-Length: 100000\r\n\r\n')
        status,h,b=Reader(s).response();self.assertEqual(status,413);self.assertEqual(h['connection'],'close')
    def test_non_idempotent_failure_is_not_retried(self):
        self.assertEqual(self.query('/proxy/reset',method='POST',body=b'operation',headers=b'X-Operation-ID: unique-no-retry\r\n')[0],502)
        with self.up.observed.mutex: attempts=list(self.up.observed.queue)
        self.assertEqual(sum(h.get('x-operation-id')=='unique-no-retry' for m,p,h,n in attempts),1)
    def test_client_disconnect_and_recovery(self):
        for _ in range(24):
            with socket.create_connection(('127.0.0.1',self.port),timeout=2) as s: s.sendall(b'GET /proxy/large HTTP/1.1\r\nHost: x\r\n\r\n')
        time.sleep(.1);self.assertEqual(self.query()[0],200)
    def test_half_closed_upload_still_receives_response(self):
        s=self.connect();s.sendall(b'POST /proxy/echo HTTP/1.1\r\nHost: x\r\nContent-Length: 3\r\n\r\nabc')
        s.shutdown(socket.SHUT_WR);self.assertEqual(Reader(s).response()[0],200)
    def test_shutdown_with_active_proxy(self):
        with socket.socket() as sock: sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        config=self.root/'shutdown.ufc'
        config.write_text(self.config.read_text().replace(f'127.0.0.1:{self.port}',f'127.0.0.1:{port}'))
        with open(self.root/'shutdown.log','wb') as log:
            p=subprocess.Popen([BINARY,'serve','--config',str(config)],stderr=log)
            try:
                for _ in range(100):
                    try: s=socket.create_connection(('127.0.0.1',port),timeout=.1);break
                    except OSError: time.sleep(.01)
                else: self.fail('shutdown fixture startup')
                with s:
                    s.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,1024)
                    s.sendall(b'GET /proxy/memory HTTP/1.1\r\nHost: x\r\n\r\n');time.sleep(.1)
                    start=time.monotonic();p.terminate()
                    self.assertEqual(p.wait(timeout=3),0);self.assertLess(time.monotonic()-start,2)
            finally:
                if p.poll() is None: p.kill();p.wait()
    def test_configuration_rejects_bad_declarations(self):
        for extra in ['route prefix "/a" { proxy missing; }','upstream app { address "http://example.com"; }','upstream app { address "127.0.0.1:80"; bogus 1; }','upstream app { address "127.0.0.1:80"; forwarding trust; }','buffer_bytes 16384;']:
            path=self.root/'invalid.ufc';path.write_text(f'server {{ listen "127.0.0.1:8080"; root "."; {extra} }}')
            result=subprocess.run([BINARY,'--check','--config',str(path)],capture_output=True)
            self.assertNotEqual(result.returncode,0)
    def test_no_tunnelling_or_arbitrary_destination(self):
        self.assertEqual(self.query(method='CONNECT')[0],405)
        s=self.connect();s.sendall(b'GET http://evil.invalid/ HTTP/1.1\r\nHost: x\r\n\r\n')
        self.assertEqual(Reader(s).response()[0],400)

if __name__=='__main__': unittest.main(verbosity=2)
