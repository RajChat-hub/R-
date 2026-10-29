"""Real verified TLS sockets; ephemeral credentials, no network downloads or verification bypass."""
import concurrent.futures, contextlib, hashlib, importlib.util, json, os, pathlib, resource
import socket, ssl, subprocess, sys, tempfile, threading, time, unittest
import proxy_test as p
BINARY=p.BINARY
spec=importlib.util.spec_from_file_location('certgen',pathlib.Path(__file__).parents[2]/'examples/make-test-certs.py')
certgen=importlib.util.module_from_spec(spec);spec.loader.exec_module(certgen)
def free_port():
    with socket.socket() as s:s.bind(('127.0.0.1',0));return s.getsockname()[1]
class SecureUpstream(p.Upstream):
    request_queue_size=64
    def get_request(self):
        raw,addr=super().get_request();raw.settimeout(3)
        try:
            secure=self.context.wrap_socket(raw,server_side=True)
            TlsTests.negotiated.append(secure.version())
            return secure,addr
        except Exception:raw.close();raise
    def handle_error(self,request,addr):pass # Expected peer resets/failure injection.
class TlsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=pathlib.Path(cls.temp.name)
        certgen.generate(cls.root/'certs');cls.certs=cls.root/'certs'
        (cls.root/'index.html').write_bytes(b'https-static')
        (cls.root/'large.bin').write_bytes(b'a'*(8*1024*1024))
        cls.services=[];cls.sni=[];cls.negotiated=[]
        for cert in ['server.crt','expired.crt']:
            up=SecureUpstream()
            up.context=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            up.context.minimum_version=ssl.TLSVersion.TLSv1_2
            up.context.load_cert_chain(cls.certs/cert,cls.certs/'server.key')
            up.context.set_servername_callback(lambda sock,name,ctx:cls.sni.append(name))
            t=threading.Thread(target=up.serve_forever,daemon=True);t.start();cls.services.append((up,t))
        cls.blackhole=socket.socket();cls.blackhole.bind(('127.0.0.1',0));cls.blackhole.listen(20)
        cls.port=free_port();cls.config=cls.root/'tls.ufc'
        cls.text=f'''server {{ listen "127.0.0.1:{cls.port}"; root ".";
 tls {{ certificate "certs/server.crt"; private_key "certs/server.key"; handshake_ms 400; shutdown_ms 300; }}
 max_connections 32; memory_budget_bytes 67108864; body_bytes 16777216;
 write_ms 2000; request_ms 3000; idle_ms 1000; shutdown_ms 500;
 upstream secure {{ address "127.0.0.1:{cls.services[0][0].server_address[1]}"; host "app.test"; connect_ms 2000; response_ms 15000;
   tls {{ verify_name "app.test"; sni "app.test"; ca_file "certs/ca.crt"; }} }}
 upstream ip {{ address "127.0.0.1:{cls.services[0][0].server_address[1]}";
   tls {{ verify_name "127.0.0.1"; sni off; ca_file "certs/ca.crt"; }} }}
 upstream wrong {{ address "127.0.0.1:{cls.services[0][0].server_address[1]}";
   tls {{ verify_name "wrong.test"; sni "wrong.test"; ca_file "certs/ca.crt"; }} }}
 upstream expired {{ address "127.0.0.1:{cls.services[1][0].server_address[1]}";
   tls {{ verify_name "app.test"; sni "app.test"; ca_file "certs/ca.crt"; }} }}
 upstream untrusted {{ address "127.0.0.1:{cls.services[0][0].server_address[1]}";
   tls {{ verify_name "app.test"; sni "app.test"; }} }}
 upstream stall {{ address "127.0.0.1:{cls.blackhole.getsockname()[1]}"; connect_ms 200;
   tls {{ verify_name "app.test"; sni "app.test"; ca_file "certs/ca.crt"; }} }}
 route prefix "/proxy" {{ proxy secure; }}
 route prefix "/ip" {{ proxy ip; }}
 route prefix "/wrong" {{ proxy wrong; }}
 route prefix "/expired" {{ proxy expired; }}
 route prefix "/untrusted" {{ proxy untrusted; }}
 route prefix "/stall" {{ proxy stall; }}
}}'''
        cls.config.write_text(cls.text);cls.log=open(cls.root/'server.log','wb')
        cls.proc=subprocess.Popen([BINARY,'serve','--config',str(cls.config)],stderr=cls.log)
        for _ in range(300):
            if cls.proc.poll() is not None:raise RuntimeError((cls.root/'server.log').read_text())
            try:
                with socket.create_connection(('127.0.0.1',cls.port),timeout=.1):break
            except OSError:time.sleep(.01)
        else:raise RuntimeError('TLS startup timeout')
        cls.context=ssl.create_default_context(cafile=str(cls.certs/'ca.crt'));cls.context.set_alpn_protocols(['http/1.1'])

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate();code=cls.proc.wait(timeout=5);cls.log.close()
        log=(cls.root/'server.log').read_text()
        for up,t in cls.services:up.shutdown();up.server_close();t.join(timeout=4)
        cls.blackhole.close()
        if code or 'AddressSanitizer' in log or 'runtime error:' in log:raise AssertionError(log)
        stats=[json.loads(x) for x in log.splitlines() if '"event":"stopped"' in x]
        if not stats:raise AssertionError('missing shutdown resource diagnostic')
        cap=(768 if os.environ.get('UNFALTERING_INSTRUMENTED')=='ON' else 192)*1024
        if stats[-1]['rss_kib']>=cap or stats[-1]['peak_network_endpoints']>64:raise AssertionError(stats[-1])
        print('TLS native resource high-water:',stats[-1],'RSS test ceiling KiB:',cap)
        print('TLS diagnostics:' ,json.dumps([json.loads(x) for x in log.splitlines() if 'tls_error' in x][:6]))
        cls.temp.cleanup()
    def connect(self,context=None):
        raw=socket.create_connection(('127.0.0.1',self.port),timeout=4)
        try:s=(context or self.context).wrap_socket(raw,server_hostname='localhost');s.settimeout(15);return s
        except Exception:raw.close();raise
    def query(self,path='/',method='GET',body=b'',headers=b''):
        with self.connect() as s:
            s.sendall(method.encode()+b' '+path.encode()+b' HTTP/1.1\r\nHost: localhost\r\nContent-Length: '+str(len(body)).encode()+b'\r\n'+headers+b'\r\n'+body)
            return p.Reader(s).response()
    def test_protocol_versions_and_alpn(self):
        for version in [ssl.TLSVersion.TLSv1_2,ssl.TLSVersion.TLSv1_3]:
            c=ssl.create_default_context(cafile=str(self.certs/'ca.crt'));c.minimum_version=c.maximum_version=version;c.set_alpn_protocols(['http/1.1'])
            with self.connect(c) as s:
                self.assertEqual(s.version(),'TLSv1.2' if version==ssl.TLSVersion.TLSv1_2 else 'TLSv1.3')
                self.assertEqual(s.selected_alpn_protocol(),'http/1.1');s.sendall(b'GET / HTTP/1.1\r\nHost: x\r\n\r\n');self.assertEqual(p.Reader(s).response()[2],b'https-static')
    def test_verified_https_proxy_and_forwarding(self):
        for method,body in [('GET',b''),('POST',b'payload'*40000)]:
            status,h,b=self.query('/proxy/echo',method,body,b'X-Forwarded-Proto: forged\r\nForwarded: for=evil\r\n')
            self.assertEqual(status,200);v=json.loads(b)
            self.assertEqual(v['sha256'],hashlib.sha256(body).hexdigest())
            self.assertEqual(v['headers']['x-forwarded-proto'],'https');self.assertNotIn('forwarded',v['headers'])
        self.assertIn('app.test',self.sni)
    def test_upstream_protocols_and_ip_identity(self):
        ctx=self.services[0][0].context
        for version in [ssl.TLSVersion.TLSv1_2,ssl.TLSVersion.TLSv1_3]:
            ctx.minimum_version=ctx.maximum_version=version
            self.assertEqual(self.query('/proxy/echo')[0],200)
            self.assertEqual(self.negotiated[-1],'TLSv1.2' if version==ssl.TLSVersion.TLSv1_2 else 'TLSv1.3')
        ctx.minimum_version=ssl.TLSVersion.TLSv1_2
        self.assertEqual(self.query('/ip/echo')[0],200);self.assertIn(None,self.sni)
    def test_unsupported_alpn_rejected(self):
        c=ssl.create_default_context(cafile=str(self.certs/'ca.crt'));c.set_alpn_protocols(['h2'])
        with self.assertRaises(ssl.SSLError):self.connect(c)
    def test_truncated_tls_eof_is_not_success(self):
        with self.connect() as s:
            s.sendall(b'GET /proxy/eof HTTP/1.1\r\nHost: x\r\n\r\n')
            try:
                status,_,body=p.Reader(s).response();self.assertEqual(status,502)
            except EOFError:pass # No false final HTTP chunk after missing TLS close_notify.
        self.assertEqual(self.query()[0],200)
    def test_chunked_fragmented_and_pipeline(self):
        with self.connect() as s:
            request=b'POST /proxy/echo HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n3\r\nabc\r\n0\r\n\r\n'
            for i in range(0,len(request),2):s.sendall(request[i:i+2])
            r=p.Reader(s);self.assertEqual(json.loads(r.response()[2])['length'],3)
            s.sendall(b'GET /proxy/chunk HTTP/1.1\r\nHost: x\r\n\r\nGET / HTTP/1.1\r\nHost: x\r\n\r\n')
            self.assertEqual(r.response()[2],b'abcdefg');self.assertEqual(r.response()[2],b'https-static')
    def test_verification_failures_and_cleanup(self):
        for path in ['/wrong/echo','/expired/echo','/untrusted/echo']:
            for _ in range(3):self.assertEqual(self.query(path)[0],502)
        self.assertEqual(self.query('/proxy/echo')[0],200)
        time.sleep(.5)
    def test_listener_certificate_verification(self):
        c=ssl.create_default_context()
        with self.assertRaises(ssl.SSLError):self.connect(c)
        with socket.create_connection(('127.0.0.1',self.port),timeout=2) as raw:
            with self.assertRaises(ssl.CertificateError):self.context.wrap_socket(raw,server_hostname='wrong.test')
    def test_upstream_handshake_timeout(self):
        start=time.monotonic();self.assertEqual(self.query('/stall/echo')[0],504);self.assertLess(time.monotonic()-start,2)
    def test_incomplete_and_slow_handshake_deadline(self):
        for data in [b'',b'\x16\x03\x01\x00\xff\x01']:
            with socket.create_connection(('127.0.0.1',self.port),timeout=2) as s:
                if data:s.sendall(data)
                time.sleep(.65);s.settimeout(1)
                try:self.assertEqual(s.recv(1024),b'')
                except ConnectionResetError:pass
        self.assertEqual(self.query()[0],200)
    def test_abrupt_disconnects_and_resets(self):
        for _ in range(20):
            s=self.connect();s.sendall(b'GET /large.bin HTTP/1.1\r\nHost: x\r\n\r\n');fd=s.detach();os.close(fd)
        self.assertEqual(self.query('/proxy/reset',method='POST',body=b'one')[0],502)
        self.assertEqual(self.query()[0],200)
    def test_streaming_backpressure_and_memory(self):
        with self.connect() as s:
            s.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,65536)
            s.sendall(b'GET /proxy/memory HTTP/1.1\r\nHost: x\r\n\r\n');time.sleep(.15)
            status,h,result=p.Reader(s).response(digest=True)
            digest=hashlib.sha256()
            for _ in range(4096):digest.update(p.BLOCK)
            self.assertEqual((status,result),(200,(128*1024*1024,digest.hexdigest())))
    def test_large_static_and_clean_close_notify(self):
        with self.connect() as s:
            s.sendall(b'GET /large.bin HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n')
            self.assertEqual(p.Reader(s).response(digest=True)[2][0],8*1024*1024)
            raw=s.unwrap();raw.close() # Requires peer close_notify, not bare TCP EOF.
    def test_invalid_framing_still_rejected(self):
        self.assertEqual(self.query('/proxy/echo','POST',b'x',b'Transfer-Encoding: chunked\r\n')[0],400)
        for path in ['bad-cl','bad-tecl','bad-fold','bad-coding']:
            self.assertEqual(self.query('/proxy/'+path)[0],502)
    def test_startup_validation(self):
        (self.certs/'bad.crt').write_text('not a certificate')
        variants=[self.text.replace('certs/server.crt','certs/bad.crt'),self.text.replace('certs/server.crt','certs/expired.crt'),
          self.text.replace('certs/server.key','certs/ca.key'),
          self.text.replace('certs/server.crt','certs/missing.crt'),
          self.text.replace('sni "app.test"','sni "wrong.test"'),
          self.text.replace('verify_name "app.test"','verify_name "*.test"'),
          self.text.replace('handshake_ms 400','handshake_ms 0'),
          self.text.replace('ca_file "certs/ca.crt"','ca_file "certs/server.key"')]
        for text in variants:
            path=self.root/'invalid.ufc';path.write_text(text)
            self.assertNotEqual(subprocess.run([BINARY,'--check','--config',str(path)],capture_output=True,timeout=3).returncode,0)
    def test_concurrent_tls_and_admission(self):
        def client(i):
            for _ in range(12):self.assertEqual(self.query('/proxy/chunk')[2],b'abcdefg')
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as e:list(e.map(client,range(8)))
        held=[]
        try:
            for _ in range(32):held.append(self.connect())
            with self.assertRaises((ssl.SSLError,OSError,TimeoutError)):self.connect()
            self.assertIsNone(self.proc.poll())
        finally:
            for s in held:s.close()
    def test_descriptor_exhaustion_recovery_under_tls(self):
        port=free_port();path=self.root/'fds.ufc';path.write_text(self.text.replace(f'127.0.0.1:{self.port}',f'127.0.0.1:{port}'))
        def limit():resource.setrlimit(resource.RLIMIT_NOFILE,(32,32))
        with open(self.root/'fds.log','wb') as log:
            proc=subprocess.Popen([BINARY,'serve','--config',str(path)],stderr=log,preexec_fn=limit)
            held=[]
            try:
                time.sleep(.1);rejected=False
                for _ in range(36):
                    try:
                        raw=socket.create_connection(('127.0.0.1',port),timeout=.5)
                        try:held.append(self.context.wrap_socket(raw,server_hostname='localhost'))
                        except Exception:raw.close();raise
                    except (ssl.SSLError,OSError):rejected=True;break
                self.assertTrue(rejected,'descriptor/admission bound must refuse overload')
                for s in held:s.close()
                held.clear();time.sleep(.25)
                with socket.create_connection(('127.0.0.1',port),timeout=2) as raw:
                    with self.context.wrap_socket(raw,server_hostname='localhost') as s:
                        s.sendall(b'GET / HTTP/1.1\r\nHost: x\r\n\r\n');self.assertEqual(p.Reader(s).response()[2],b'https-static')
            finally:
                for s in held:s.close()
                proc.terminate();self.assertEqual(proc.wait(timeout=3),0)
    def test_shutdown_with_active_tls(self):
        port=free_port();path=self.root/'shutdown.ufc';path.write_text(self.text.replace(f'127.0.0.1:{self.port}',f'127.0.0.1:{port}'))
        with open(self.root/'shutdown.log','wb') as log:
            proc=subprocess.Popen([BINARY,'serve','--config',str(path)],stderr=log)
            try:
                for _ in range(100):
                    try:raw=socket.create_connection(('127.0.0.1',port),timeout=.1);break
                    except OSError:time.sleep(.02)
                else:self.fail('startup')
                with self.context.wrap_socket(raw,server_hostname='localhost') as s:
                    s.sendall(b'GET /proxy/memory HTTP/1.1\r\nHost: x\r\n\r\n');time.sleep(.1)
                    start=time.monotonic();proc.terminate();self.assertEqual(proc.wait(timeout=3),0);self.assertLess(time.monotonic()-start,2)
            finally:
                if proc.poll() is None:proc.kill();proc.wait()
if __name__=='__main__':unittest.main(verbosity=2)
