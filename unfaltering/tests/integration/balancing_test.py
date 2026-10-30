"""Gate C: real independent HTTP/HTTPS services, deterministic selection and tracking."""
import concurrent.futures,hashlib,json,pathlib,socket,ssl,subprocess,tempfile,time,unittest
import tls_test as t
from backend_fixture import Backend,BLOCK
BINARY=t.BINARY
class Balancing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=pathlib.Path(cls.tmp.name);t.certgen.generate(cls.root/'certs');(cls.root/'index.html').write_text('pool-static')
        cls.services=[Backend('a'),Backend('b',cls.root/'certs'),Backend('c')]
        cls.port=t.free_port();a,b,c=cls.services
        cls.text=f'''server {{listen "127.0.0.1:{cls.port}"; root "."; body_bytes 16777216; request_ms 10000;
 upstream a {{address "127.0.0.1:{a.port}"; host "app.test";}}
 upstream b {{address "127.0.0.1:{b.port}"; host "app.test";tls {{verify_name "app.test";sni "app.test";ca_file "certs/ca.crt";}} }}
 upstream c {{address "127.0.0.1:{c.port}";host "app.test";}}
 upstream limited {{address "127.0.0.1:{a.port}";max_connections 1;}}
 pool apps {{algorithm round_robin;member a;member b;member c;}}
 pool single {{member limited;}}
 route prefix "/pool" {{proxy apps;}}
 route prefix "/single" {{proxy single;}}
 route prefix "/direct" {{proxy a;}}
}}'''
        cls.cfg=cls.root/'pool.ufc';cls.cfg.write_text(cls.text);cls.log=open(cls.root/'native.log','wb');cls.proc=subprocess.Popen([BINARY,'serve','--config',str(cls.cfg)],stderr=cls.log)
        for _ in range(200):
            if cls.proc.poll() is not None:raise RuntimeError((cls.root/'native.log').read_text())
            try:
                with socket.create_connection(('127.0.0.1',cls.port),timeout=.1):break
            except OSError:time.sleep(.01)
        else:raise RuntimeError('pool startup')
    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate();assert cls.proc.wait(timeout=4)==0;cls.log.close()
        for b in cls.services:b.close()
        log=(cls.root/'native.log').read_text();assert 'AddressSanitizer' not in log and 'runtime error:' not in log
        cls.tmp.cleanup()
    def connect(self):return socket.create_connection(('127.0.0.1',self.port),timeout=4)
    def query(self,path='/pool/echo',body=b'',headers=b''):
        with self.connect() as s:
            s.settimeout(12);s.sendall((b'POST' if body else b'GET')+b' '+path.encode()+b' HTTP/1.1\r\nHost: public.test\r\nContent-Length: '+str(len(body)).encode()+b'\r\n'+headers+b'\r\n'+body)
            return t.p.Reader(s).response()
    def test_a_deterministic_http_https_round_robin(self):
        values=[json.loads(self.query()[2])['backend'] for _ in range(12)]
        self.assertEqual(values,['a','b','c']*4)
    def test_post_forwarding_and_chunked(self):
        body=b'payload'*(1024*30)
        status,h,wire=self.query(body=body,headers=b'Forwarded: evil\r\nX-Forwarded-For: evil\r\nConnection: keep-alive, X-Hop\r\nX-Hop: secret\r\n')
        value=json.loads(wire);self.assertEqual(status,200);self.assertEqual(value['sha256'],hashlib.sha256(body).hexdigest());self.assertEqual(value['headers']['host'],'app.test')
        for field in ['forwarded','x-hop']:self.assertNotIn(field,value['headers'])
        with self.connect() as s:
            s.sendall(b'POST /pool/echo HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n3\r\nabc\r\n0\r\n\r\n')
            self.assertEqual(json.loads(t.p.Reader(s).response()[2])['length'],3)
        self.assertEqual(self.query('/pool/chunk')[2],b'abcdefg')
    def test_per_backend_connection_cap_and_recovery(self):
        with self.connect() as held:
            held.sendall(b'GET /single/slow HTTP/1.1\r\nHost: x\r\n\r\n');time.sleep(.1)
            self.assertEqual(self.query('/single/echo')[0],503)
            self.assertEqual(t.p.Reader(held).response()[0],200)
        self.assertEqual(self.query('/single/echo')[0],200)
    def test_concurrent_and_large_stream(self):
        def task(i):
            for _ in range(8):self.assertEqual(self.query()[0],200)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as e:list(e.map(task,range(8)))
        with self.connect() as s:
            s.sendall(b'GET /pool/memory HTTP/1.1\r\nHost: x\r\n\r\n');time.sleep(.1)
            status,h,value=t.p.Reader(s).response(digest=True)
            digest=hashlib.sha256()
            for _ in range(128*1024*1024//len(BLOCK)):digest.update(BLOCK)
            self.assertEqual((status,value),(200,(128*1024*1024,digest.hexdigest())))
    def test_no_retry_on_failed_post(self):
        self.services[0].mode='reset'
        try:self.assertEqual(self.query('/direct/echo',b'one-operation',b'X-Operation-ID: never-replay\r\n')[0],502)
        finally:self.services[0].mode='ok'
        with self.services[0].observed.mutex:records=list(self.services[0].observed.queue)
        self.assertEqual(sum(r['operation']=='never-replay' for r in records),1)
        self.assertEqual(self.query('/direct/echo')[0],200)
    def test_invalid_pool_config(self):
        for extra in ['pool x {}','pool x {member missing;}','pool a {member b;}','pool x {member a;member a;}','pool x {member a;algorithm least_active;}','pool x {member a;member limited;}']:
            path=self.root/'invalid.ufc';path.write_text(self.text.replace('pool apps {',extra+' pool apps {'))
            self.assertNotEqual(subprocess.run([BINARY,'--check','--config',str(path)],capture_output=True).returncode,0)
if __name__=='__main__':unittest.main(verbosity=2)
