"""Gate A: multiple original epoll listeners, admission, startup rollback."""
import concurrent.futures, pathlib, socket, ssl, subprocess, tempfile, time, unittest
import tls_test as t
BINARY=t.BINARY
class Listeners(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=pathlib.Path(cls.tmp.name)
        t.certgen.generate(cls.root/'certs');(cls.root/'index.html').write_text('multi-listener')
        cls.http=t.free_port();cls.https=t.free_port()
        cls.text=f'''server {{ root "."; max_connections 32;
 listener plain {{ address "127.0.0.1:{cls.http}"; max_connections 2; }}
 listener secure {{ address "127.0.0.1:{cls.https}"; tls {{ certificate "certs/server.crt"; private_key "certs/server.key"; handshake_ms 400; }} }}
}}'''
        cls.cfg=cls.root/'multi.ufc';cls.cfg.write_text(cls.text)
        cls.log=open(cls.root/'server.log','wb');cls.proc=subprocess.Popen([BINARY,'serve','--config',str(cls.cfg)],stderr=cls.log)
        cls.ctx=ssl.create_default_context(cafile=str(cls.root/'certs/ca.crt'))
        for port in [cls.http,cls.https]:
            for _ in range(200):
                if cls.proc.poll() is not None:raise RuntimeError((cls.root/'server.log').read_text())
                try:
                    with socket.create_connection(('127.0.0.1',port),timeout=.1):break
                except OSError:time.sleep(.01)
            else:raise RuntimeError('multi listener startup')
        time.sleep(.05)
    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate();assert cls.proc.wait(timeout=4)==0;cls.log.close()
        log=(cls.root/'server.log').read_text();assert 'AddressSanitizer' not in log and 'runtime error:' not in log
        cls.tmp.cleanup()
    def query(self,tls):
        raw=socket.create_connection(('127.0.0.1',self.https if tls else self.http),timeout=3)
        with (self.ctx.wrap_socket(raw,server_hostname='localhost') if tls else raw) as s:
            s.sendall(b'GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n')
            return t.p.Reader(s).response()
    def test_independent_http_https_concurrent(self):
        # Per-listener admission is two for plain; avoid interpreting intentional
        # overload rejection as a failed request in this distribution check.
        def task(i):
            for _ in range(12):self.assertEqual(self.query(i%2)[2],b'multi-listener')
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as e:list(e.map(task,range(2)))
    def test_per_listener_admission(self):
        held=[socket.create_connection(('127.0.0.1',self.http),timeout=2) for _ in range(2)]
        try:
            time.sleep(.1)
            with socket.create_connection(('127.0.0.1',self.http),timeout=2) as s:s.settimeout(1);self.assertEqual(s.recv(1),b'')
            self.assertEqual(self.query(True)[0],200)
        finally:
            for s in held:s.close()
    def test_duplicate_and_conflicting_bindings(self):
        for text in [self.text.replace(f'127.0.0.1:{self.https}',f'127.0.0.1:{self.http}'),self.text.replace(f'127.0.0.1:{self.https}',f'0.0.0.0:{self.http}'),self.text.replace('listener secure','listener plain'),self.text.replace('root ".";',f'root "."; listen "127.0.0.1:{t.free_port()}";')]:
            path=self.root/'invalid.ufc';path.write_text(text)
            self.assertNotEqual(subprocess.run([BINARY,'--check','--config',str(path)],capture_output=True).returncode,0)
    def test_partial_initialization_rolls_back(self):
        first=t.free_port()
        with socket.socket() as occupied:
            occupied.bind(('127.0.0.1',0));occupied.listen();second=occupied.getsockname()[1]
            cfg=self.root/'rollback.ufc';cfg.write_text(f'server {{ root "."; listener first {{ address "127.0.0.1:{first}"; }} listener second {{ address "127.0.0.1:{second}"; }} }}')
            result=subprocess.run([BINARY,'serve','--config',str(cfg)],capture_output=True,timeout=3)
            self.assertNotEqual(result.returncode,0)
            with socket.socket() as freed:freed.bind(('127.0.0.1',first))
    def test_shutdown_closes_all_listener_sockets(self):
        ports=[t.free_port(),t.free_port()];cfg=self.root/'shutdown.ufc'
        cfg.write_text(f'server {{ root "."; listener a {{ address "127.0.0.1:{ports[0]}"; }} listener b {{ address "127.0.0.1:{ports[1]}"; }} }}')
        proc=subprocess.Popen([BINARY,'serve','--config',str(cfg)],stderr=subprocess.DEVNULL)
        try:
            for _ in range(100):
                try:
                    with socket.create_connection(('127.0.0.1',ports[1]),timeout=.1):break
                except OSError:time.sleep(.01)
            else:self.fail('startup')
            proc.terminate();self.assertEqual(proc.wait(timeout=3),0)
            for port in ports:
                with self.assertRaises(OSError):socket.create_connection(('127.0.0.1',port),timeout=.2)
        finally:
            if proc.poll() is None:proc.kill();proc.wait()
if __name__=='__main__':unittest.main(verbosity=2)
