"""Real loopback socket tests; no external network or third-party test packages."""
import concurrent.futures
import contextlib
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest

BINARY = str(Path(sys.argv.pop(1)).resolve())

class ResponseReader:
    def __init__(self, sock):
        self.sock, self.buffer = sock, b''

    def response(self, head=False):
        while b'\r\n\r\n' not in self.buffer:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise EOFError(self.buffer)
            self.buffer += chunk
        header, self.buffer = self.buffer.split(b'\r\n\r\n', 1)
        lines = header.decode('ascii').split('\r\n')
        status = int(lines[0].split()[1])
        headers = dict(line.lower().split(': ', 1) for line in lines[1:])
        n = 0 if head else int(headers['content-length'])
        while len(self.buffer) < n:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise EOFError('truncated body')
            self.buffer += chunk
        body, self.buffer = self.buffer[:n], self.buffer[n:]
        return status, headers, body

class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='unfaltering-net-')
        cls.root = Path(cls.temp.name)
        (cls.root/'one').mkdir(); (cls.root/'two').mkdir()
        (cls.root/'one/index.html').write_bytes(b'hello-one')
        (cls.root/'two/index.html').write_bytes(b'hello-two')
        (cls.root/'two/assets').mkdir(); (cls.root/'two/assets/file.txt').write_bytes(b'prefix-route')
        (cls.root/'one/assets').mkdir(); (cls.root/'one/assets/special.txt').write_bytes(b'exact-route')
        cls.payload = bytes(range(256)) * 16384
        (cls.root/'one/large.bin').write_bytes(cls.payload)
        os.symlink('/etc/passwd', cls.root/'one/secret')
        os.symlink('/etc', cls.root/'one/escape')
        with socket.socket() as s:
            s.bind(('127.0.0.1', 0)); cls.port = s.getsockname()[1]
        cls.config = cls.root/'test.ufc'
        cls.config.write_text(f'''server {{
 listen "127.0.0.1:{cls.port}"; root "one";
 host "two.test" {{ root "two"; }}
 route prefix "/assets" {{ root "two"; }}
 route exact "/assets/special.txt" {{ root "one"; }}
 max_connections 32; max_requests 5; header_bytes 2048; header_count 16;
 body_bytes 4096; idle_ms 600; request_ms 400; write_ms 1500; shutdown_ms 500;
}}''')
        cls.logs = open(cls.root/'log', 'wb')
        cls.proc = subprocess.Popen([BINARY, 'serve', '--config', str(cls.config)], stdout=cls.logs, stderr=cls.logs)
        for _ in range(150):
            if cls.proc.poll() is not None:
                raise RuntimeError((cls.root/'log').read_text())
            try:
                with socket.create_connection(('127.0.0.1', cls.port), timeout=.1):
                    return
            except OSError:
                time.sleep(.02)
        raise RuntimeError('server did not start')

    @classmethod
    def tearDownClass(cls):
        if cls.proc.poll() is None:
            cls.proc.send_signal(signal.SIGTERM)
        cls.proc.wait(timeout=5)
        cls.logs.close()
        log = (cls.root/'log').read_text()
        if cls.proc.returncode or 'AddressSanitizer' in log or 'runtime error:' in log:
            raise AssertionError(log)
        cls.temp.cleanup()

    def connect(self):
        s = socket.create_connection(('127.0.0.1', self.port), timeout=3)
        s.settimeout(3)
        self.addCleanup(s.close)
        return s

    def query(self, request, head=False):
        s = self.connect(); s.sendall(request)
        return ResponseReader(s).response(head)

    def test_get_head(self):
        for method in [b'GET', b'HEAD']:
            status, headers, body = self.query(method+b' / HTTP/1.1\r\nHost: one.test\r\n\r\n', method==b'HEAD')
            self.assertEqual(status, 200); self.assertEqual(headers['content-length'], '9')
            self.assertEqual(body, b'' if method==b'HEAD' else b'hello-one')
            self.assertEqual(headers['x-content-type-options'], 'nosniff')

    def test_head_error_has_no_body(self):
        status,headers,body=self.query(b'HEAD /missing HTTP/1.1\r\nHost: x\r\n\r\n',True)
        self.assertEqual(status,404);self.assertEqual(body,b'')
        s=self.connect();s.sendall(b'HEAD / HTTP/1.1\r\nHost: x\r\nHost: y\r\n\r\n')
        r=ResponseReader(s);self.assertEqual(r.response(True)[0],400)
        self.assertEqual(r.buffer,b'');self.assertEqual(s.recv(1),b'')

    def test_virtual_host(self):
        self.assertEqual(self.query(b'GET / HTTP/1.1\r\nHost: TWO.TEST:8080\r\n\r\n')[2], b'hello-two')

    def test_static_routes(self):
        for path, expected in [(b'/assets/file.txt',b'prefix-route'),(b'/assets/special.txt?x=1',b'exact-route')]:
            self.assertEqual(self.query(b'GET '+path+b' HTTP/1.1\r\nHost: x\r\n\r\n')[2],expected)
        self.assertEqual(self.query(b'GET /assetsXYZ HTTP/1.1\r\nHost: x\r\n\r\n')[0],404)

    def test_connection_limit(self):
        held=[]
        try:
            for _ in range(32):
                held.append(socket.create_connection(('127.0.0.1',self.port),timeout=1))
            time.sleep(.05)
            with socket.create_connection(('127.0.0.1',self.port),timeout=1) as extra:
                extra.settimeout(1)
                self.assertEqual(extra.recv(1),b'')
        finally:
            for s in held:s.close()
        time.sleep(.05)
        self.assertEqual(self.query(b'GET / HTTP/1.1\r\nHost: x\r\n\r\n')[0],200)

    def test_fragmented(self):
        s=self.connect(); raw=b'GET / HTTP/1.1\r\nHost: one.test\r\n\r\n'
        for c in raw: s.sendall(bytes([c]))
        self.assertEqual(ResponseReader(s).response()[2], b'hello-one')

    def test_pipeline_and_limit(self):
        s=self.connect(); raw=b'GET / HTTP/1.1\r\nHost: one.test\r\n\r\n'; s.sendall(raw*5)
        r=ResponseReader(s)
        for _ in range(4): self.assertEqual(r.response()[0], 200)
        self.assertEqual(r.response()[1]['connection'], 'close'); self.assertEqual(s.recv(1), b'')

    def test_keep_alive_body_boundaries(self):
        s=self.connect(); r=ResponseReader(s)
        for body, header in [(b'abc',b'Content-Length: 3'),(b'3\r\nabc\r\n0\r\n\r\n',b'Transfer-Encoding: chunked')]:
            s.sendall(b'POST / HTTP/1.1\r\nHost: one.test\r\n'+header+b'\r\n\r\n'+body+b'GET / HTTP/1.1\r\nHost: one.test\r\n\r\n')
            self.assertEqual(r.response()[0],405); self.assertEqual(r.response()[2],b'hello-one')

    def test_half_close(self):
        s=self.connect(); s.sendall(b'GET / HTTP/1.1\r\nHost: x\r\n\r\n'); s.shutdown(socket.SHUT_WR)
        self.assertEqual(ResponseReader(s).response()[0],200)

    def test_half_close_with_backpressure(self):
        s=self.connect();s.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,65536)
        s.sendall(b'GET /large.bin HTTP/1.1\r\nHost: x\r\n\r\n');s.shutdown(socket.SHUT_WR)
        time.sleep(.1)
        self.assertEqual(ResponseReader(s).response()[2],self.payload)

    def test_malformed(self):
        for headers in [b'Host: x\r\nContent-Length: 1\r\nTransfer-Encoding: chunked',b'Host: x\r\nContent-Length: 1\r\nContent-Length: 2',b'Host: x\r\nHost: y',b'Host: x\r\n X: y', b'Host : x',b'Host: x\r\nX: a\nb']:
            with self.subTest(headers=headers):
                self.assertEqual(self.query(b'GET / HTTP/1.1\r\n'+headers+b'\r\n\r\n')[0],400)

    def test_limits(self):
        for header,status in [(b'Host: x\r\nX: '+b'a'*2100,431),(b'Host: x\r\nContent-Length: 4097',413),(b'Host: x\r\n'+b'X: y\r\n'*17,431),(b'Host: x\r\nExpect: 100-continue',417)]:
            self.assertEqual(self.query(b'GET / HTTP/1.1\r\n'+header+b'\r\n\r\n')[0],status)

    def test_paths(self):
        for path in [b'/../etc/passwd',b'/%2e%2e/etc/passwd',b'/%252e%252e/etc/passwd',b'/%00',b'/secret',b'/escape/passwd',b'/a\\b']:
            with self.subTest(path=path): self.assertEqual(self.query(b'GET '+path+b' HTTP/1.1\r\nHost: x\r\n\r\n')[0],403)
        self.assertEqual(self.query(b'GET /missing HTTP/1.1\r\nHost: x\r\n\r\n')[0],404)

    def test_large_partial_writes(self):
        s=self.connect(); s.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,65536)
        s.sendall(b'GET /large.bin HTTP/1.1\r\nHost: x\r\n\r\n'); time.sleep(.05)
        self.assertEqual(ResponseReader(s).response()[2],self.payload)

    def test_request_deadline(self):
        s=self.connect(); s.sendall(b'GET / HTTP/1.1\r\nHo'); time.sleep(.15);s.sendall(b's');time.sleep(.35)
        self.assertEqual(ResponseReader(s).response()[0],408)

    def test_body_deadline(self):
        s=self.connect();s.sendall(b'POST / HTTP/1.1\r\nHost: x\r\nContent-Length: 3\r\n\r\na');time.sleep(.5)
        self.assertEqual(ResponseReader(s).response()[0],408)

    def test_idle_timeout(self):
        s=self.connect();time.sleep(.8);self.assertEqual(s.recv(1),b'')

    def test_concurrent_reset_and_reuse(self):
        def one(i):
            with socket.create_connection(('127.0.0.1',self.port),timeout=3) as s:
                s.sendall(b'GET / HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n')
                if i%4==0:return
                self.assertEqual(ResponseReader(s).response()[0],200)
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:list(pool.map(one,range(120)))
        self.assertEqual(self.query(b'GET / HTTP/1.1\r\nHost: x\r\n\r\n')[0],200)

    def test_blocked_writer_deadline(self):
        s=self.connect();s.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,65536)
        s.sendall(b'GET /large.bin HTTP/1.1\r\nHost: x\r\n\r\n')
        time.sleep(1.8);s.settimeout(2)
        received=b''
        try:
            while True:
                chunk=s.recv(65536)
                if not chunk:break
                received+=chunk
        except ConnectionResetError:pass
        self.assertIn(b'HTTP/1.1 200',received)
        self.assertLess(len(received),len(self.payload))

    def test_graceful_shutdown_with_blocked_client(self):
        with socket.socket() as s:
            s.bind(('127.0.0.1',0));port=s.getsockname()[1]
        cfg=self.root/'shutdown.ufc';cfg.write_text(self.config.read_text().replace(str(self.port),str(port)))
        with open(self.root/'shutdown.log','wb') as log:
            p=subprocess.Popen([BINARY,'serve','--config',str(cfg)],stderr=log,stdout=log)
            try:
                for _ in range(100):
                    try: client=socket.create_connection(('127.0.0.1',port),timeout=.1);break
                    except OSError:time.sleep(.01)
                else: self.fail('shutdown fixture did not start')
                with client:
                    client.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,1024)
                    client.sendall(b'GET /large.bin HTTP/1.1\r\nHost: x\r\n\r\n')
                    time.sleep(.05);start=time.monotonic();p.send_signal(signal.SIGTERM)
                    self.assertEqual(p.wait(timeout=3),0)
                    self.assertLess(time.monotonic()-start,2)
            finally:
                if p.poll() is None:p.kill();p.wait()
        self.assertNotIn('runtime error:',(self.root/'shutdown.log').read_text())

    def test_descriptor_exhaustion_recovery(self):
        import resource
        with socket.socket() as s:
            s.bind(('127.0.0.1',0));port=s.getsockname()[1]
        cfg=self.root/'exhaust.ufc';cfg.write_text(self.config.read_text().replace(str(self.port),str(port)))
        def restrict():resource.setrlimit(resource.RLIMIT_NOFILE,(24,24))
        with open(self.root/'exhaust.log','wb') as log:
            p=subprocess.Popen([BINARY,'serve','--config',str(cfg)],stderr=log,stdout=log,preexec_fn=restrict)
            held=[]
            try:
                for _ in range(100):
                    try: held.append(socket.create_connection(('127.0.0.1',port),timeout=.1));break
                    except OSError:time.sleep(.01)
                for _ in range(36):
                    try:held.append(socket.create_connection(('127.0.0.1',port),timeout=.1))
                    except OSError:pass
                time.sleep(.1)
                self.assertIsNone(p.poll())
                for s in held:s.close()
                held.clear();time.sleep(.8)
                with socket.create_connection(('127.0.0.1',port),timeout=2) as s:
                    s.settimeout(2);s.sendall(b'GET / HTTP/1.1\r\nHost: x\r\n\r\n')
                    self.assertEqual(ResponseReader(s).response()[0],200)
            finally:
                for s in held:s.close()
                if p.poll() is None:p.send_signal(signal.SIGTERM)
                self.assertEqual(p.wait(timeout=3),0)

    def test_config_rejection(self):
        bad=self.root/'bad.ufc';bad.write_text('server { root "one"; unknown 1; }')
        r=subprocess.run([BINARY,'--check','--config',str(bad)],capture_output=True)
        self.assertNotEqual(r.returncode,0);self.assertIn(b'unknown',r.stderr)

if __name__=='__main__':unittest.main(verbosity=2)
