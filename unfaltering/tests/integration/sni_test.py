"""Gate B: independent certificates, strict authority isolation and explicit fallback."""
import hashlib,pathlib,socket,ssl,subprocess,tempfile,time,unittest
import tls_test as t
BINARY=t.BINARY
class Sni(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=pathlib.Path(cls.tmp.name);t.certgen.generate(cls.root/'certs')
        def run(*args):return subprocess.run(['openssl',*args],cwd=cls.root/'certs',check=True,capture_output=True).stdout
        for index,name in enumerate(['one.test','two.test']):
            key=name+'.key';cert=name+'.crt';csr=name+'.csr'
            run('req','-new','-newkey','rsa:2048','-nodes','-keyout',key,'-out',csr,'-subj','/CN='+name)
            (cls.root/'certs/ext.cnf').write_text(f'basicConstraints=critical,CA:FALSE\nextendedKeyUsage=serverAuth\nsubjectAltName=DNS:{name}\n')
            run('x509','-req','-in',csr,'-CA','ca.crt','-CAkey','ca.key','-set_serial',str(index+20),'-days','1','-out',cert,'-extfile','ext.cnf')
            (cls.root/name).mkdir();(cls.root/name/'index.html').write_text(name)
        cls.strict=t.free_port();cls.fallback=t.free_port()
        tls='''certificate "certs/one.test.crt"; private_key "certs/one.test.key"; default_name "one.test";
 site "two.test" { certificate "certs/two.test.crt"; private_key "certs/two.test.key"; }'''
        cls.text=f'''server {{root "one.test";
 host "one.test" {{root "one.test";}} host "two.test" {{root "two.test";}}
 listener strict {{address "127.0.0.1:{cls.strict}"; tls {{{tls}}} }}
 listener fallback {{address "127.0.0.1:{cls.fallback}"; tls {{{tls} unknown_sni default; no_sni default; host_sni off;}} }}
}}'''
        cls.cfg=cls.root/'sni.ufc';cls.cfg.write_text(cls.text)
        cls.log=open(cls.root/'native.log','wb');cls.proc=subprocess.Popen([BINARY,'serve','--config',str(cls.cfg)],stderr=cls.log)
        cls.ctx=ssl.create_default_context(cafile=str(cls.root/'certs/ca.crt'))
        for _ in range(200):
            if cls.proc.poll() is not None:raise RuntimeError((cls.root/'native.log').read_text())
            try:
                with socket.create_connection(('127.0.0.1',cls.strict),timeout=.1):break
            except OSError:time.sleep(.01)
        else:raise RuntimeError('SNI startup')
    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate();assert cls.proc.wait(timeout=4)==0;cls.log.close()
        log=(cls.root/'native.log').read_text();assert 'AddressSanitizer' not in log and 'runtime error:' not in log
        cls.tmp.cleanup()
    def connect(self,name='one.test',fallback=False,ctx=None):
        raw=socket.create_connection(('127.0.0.1',self.fallback if fallback else self.strict),timeout=3)
        try:return (ctx or self.ctx).wrap_socket(raw,server_hostname=name)
        except Exception:raw.close();raise
    def request(self,s,host):
        s.sendall(b'GET / HTTP/1.1\r\nHost: '+host.encode()+b'\r\n\r\n');return t.p.Reader(s).response()
    def test_correct_distinct_certificate_and_root(self):
        for name in ['one.test','two.test','ONE.TEST']:
            with self.connect(name) as s:
                cert=s.getpeercert();self.assertIn(('DNS',name.lower()),cert['subjectAltName'])
                other='two.test' if name.lower()=='one.test' else 'one.test';self.assertNotIn(('DNS',other),cert['subjectAltName'])
                self.assertEqual(self.request(s,name)[2],name.lower().encode())
    def test_host_sni_mismatch_is_421_before_routing(self):
        with self.connect('one.test') as s:
            self.assertEqual(self.request(s,'one.test')[2],b'one.test')
            status,h,body=self.request(s,'two.test');self.assertEqual(status,421);self.assertEqual(h['connection'],'close');self.assertNotIn(b'two.test',body)
        with self.connect('two.test') as s:self.assertEqual(self.request(s,'one.test')[0],421)
    def test_unknown_and_missing_sni_rejected(self):
        c=ssl.create_default_context(cafile=str(self.root/'certs/ca.crt'));c.check_hostname=False
        for name in ['unknown.test',None]:
            with self.assertRaises(ssl.SSLError):self.connect(name,ctx=c)
    def test_explicit_default_and_optout(self):
        c=ssl.create_default_context(cafile=str(self.root/'certs/ca.crt'));c.check_hostname=False
        for name in ['unknown.test',None]:
            with self.connect(name,True,c) as s:
                self.assertIn(('DNS','one.test'),s.getpeercert()['subjectAltName'])
                self.assertEqual(self.request(s,'two.test')[2],b'two.test')
    def test_malformed_sni_wire(self):
        for replacement in [b'one\x00test',b'one..tes',b'*.testxx']:
            incoming,outgoing=ssl.MemoryBIO(),ssl.MemoryBIO();conn=self.ctx.wrap_bio(incoming,outgoing,server_hostname='one.test')
            try:conn.do_handshake()
            except ssl.SSLWantReadError:pass
            hello=outgoing.read();self.assertIn(b'one.test',hello);hello=hello.replace(b'one.test',replacement,1)
            with socket.create_connection(('127.0.0.1',self.strict),timeout=2) as raw:
                raw.sendall(hello);raw.settimeout(2);result=raw.recv(1024);self.assertTrue(not result or result[0]==21)
    def test_certificate_mapping_preflight(self):
        for text in [self.text.replace('certs/one.test.crt','certs/two.test.crt'),self.text.replace('certs/one.test.key','certs/two.test.key'),self.text.replace('certs/one.test.crt','certs/expired.crt'),self.text.replace('site "two.test"','site "one.test"'),self.text.replace('default_name "one.test"','default_name "*.test"'),self.text.replace('site "two.test"','site "two..test"')]:
            path=self.root/'invalid.ufc';path.write_text(text);result=subprocess.run([BINARY,'--check','--config',str(path)],capture_output=True,timeout=3);self.assertNotEqual(result.returncode,0)
if __name__=='__main__':unittest.main(verbosity=2)
