"""Gate D: native health probes, verified TLS, hysteresis, passive failure and recovery."""
import concurrent.futures,json,os,pathlib,resource,socket,ssl,subprocess,tempfile,time,unittest
import tls_test as t
from backend_fixture import Backend
BINARY=t.BINARY
class Health(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=pathlib.Path(cls.tmp.name);t.certgen.generate(cls.root/'certs');(cls.root/'index.html').write_text('healthy-static')
        cls.services=[Backend('a'),Backend('b',cls.root/'certs'),Backend('c'),Backend('slow')]
        cls.services[3].health_delay=1.2
        cls.port=t.free_port();cls.secure=t.free_port();cls.admin=t.free_port()
        a,b,c,slow=cls.services
        health='health {interval_ms 200;timeout_ms 500;failures 2;recoveries 2;}'
        cls.text=f'''server {{root ".";max_connections 48;body_bytes 16777216;memory_budget_bytes 100663296;shutdown_ms 500;
 probe_concurrency 2;probe_spacing_ms 50;
 listener app {{address "127.0.0.1:{cls.port}";}}
 listener secure {{address "127.0.0.1:{cls.secure}";tls {{certificate "certs/server.crt";private_key "certs/server.key";}} }}
 listener admin {{address "127.0.0.1:{cls.admin}";metrics "/_uf/metrics";}}
 upstream a {{address "127.0.0.1:{a.port}";{health}}}
 upstream b {{address "127.0.0.1:{b.port}";host "app.test";{health}tls {{verify_name "app.test";sni "app.test";ca_file "certs/ca.crt";}} }}
 upstream c {{address "127.0.0.1:{c.port}";{health}}}
 upstream badtls {{address "127.0.0.1:{b.port}";{health}tls {{verify_name "wrong.test";sni "wrong.test";ca_file "certs/ca.crt";}} }}
 upstream slow {{address "127.0.0.1:{slow.port}";{health}}}
 upstream passive {{address "127.0.0.1:{c.port}";health {{interval_ms 1000;timeout_ms 500;failures 2;recoveries 1;}} }}
 pool apps {{member a;member b;member c;}}
 pool verified {{member badtls;}}
 route prefix "/app" {{proxy apps;}}
 route prefix "/bad" {{proxy verified;}}
 route prefix "/passive" {{proxy passive;}}
}}'''
        cls.cfg=cls.root/'health.ufc';cls.cfg.write_text(cls.text);cls.log=open(cls.root/'native.log','wb');cls.proc=subprocess.Popen([BINARY,'serve','--config',str(cls.cfg)],stderr=cls.log)
        cls.ctx=ssl.create_default_context(cafile=str(cls.root/'certs/ca.crt'))
        for _ in range(200):
            if cls.proc.poll() is not None:raise RuntimeError((cls.root/'native.log').read_text())
            try:
                with socket.create_connection(('127.0.0.1',cls.admin),timeout=.1):break
            except OSError:time.sleep(.01)
        else:raise RuntimeError('health startup')
        cls.case=cls();cls.case.wait(lambda states:all(states[n]['healthy'] for n in ['a','b','c','passive']))
    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate();assert cls.proc.wait(timeout=4)==0;cls.log.close()
        for b in cls.services:b.close()
        log=(cls.root/'native.log').read_text();assert 'AddressSanitizer' not in log and 'runtime error:' not in log
        stats=[json.loads(x) for x in log.splitlines() if '"event":"stopped"' in x]
        assert stats and stats[-1]['peak_probes']<=2 and stats[-1]['peak_network_endpoints']<=2*48+2
        cap=(768 if os.environ.get('UNFALTERING_INSTRUMENTED')=='ON' else 192)*1024
        assert stats[-1]['rss_kib']<cap,stats[-1]
        print('Health/native resource high-water:',stats[-1]);cls.tmp.cleanup()
    def query(self,path='/app/echo',body=b'',secure=False,port=None):
        raw=socket.create_connection(('127.0.0.1',port or (self.secure if secure else self.port)),timeout=4)
        with (self.ctx.wrap_socket(raw,server_hostname='localhost') if secure else raw) as s:
            s.settimeout(10);s.sendall((b'POST' if body else b'GET')+b' '+path.encode()+b' HTTP/1.1\r\nHost: localhost\r\nContent-Length: '+str(len(body)).encode()+b'\r\n\r\n'+body)
            return t.p.Reader(s).response()
    def stats(self):return json.loads(self.query('/_uf/metrics',port=self.admin)[2])
    def states(self):return {x['name']:x for x in self.stats()['backends']}
    def wait(self,predicate,seconds=8):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            state=self.states()
            if predicate(state):return state
            time.sleep(.02)
        self.fail('health transition timeout: '+str(self.states()))
    def restore(self):
        for b in self.services[:3]:b.healthy=True;b.mode='ok'
        self.wait(lambda s:all(s[n]['healthy'] for n in ['a','b','c']))
    def test_a_verified_health_and_round_robin(self):
        self.assertEqual([json.loads(self.query()[2])['backend'] for _ in range(9)],['a','b','c']*3)
        self.assertEqual(self.query(secure=True)[0],200)
    def test_one_backend_fails_recovers_and_reenters(self):
        self.services[1].healthy=False
        try:
            self.wait(lambda s:not s['b']['healthy'])
            values=[json.loads(self.query()[2])['backend'] for _ in range(8)];self.assertNotIn('b',values);self.assertEqual(set(values),{'a','c'})
            self.services[1].healthy=True;self.wait(lambda s:s['b']['healthy']);self.assertIn('b',[json.loads(self.query()[2])['backend'] for _ in range(6)])
        finally:self.restore()
    def test_all_unhealthy_and_partial_recovery(self):
        for b in self.services[:3]:b.healthy=False
        try:
            self.wait(lambda s:not any(s[n]['healthy'] for n in ['a','b','c']))
            self.assertEqual(self.query()[0],503)
            self.services[2].healthy=True;self.wait(lambda s:s['c']['healthy'])
            self.assertEqual([json.loads(self.query()[2])['backend'] for _ in range(3)],['c']*3)
        finally:self.restore()
    def test_slow_and_bad_tls_probes_remain_unavailable(self):
        self.wait(lambda s:s['badtls']['failures']>=2 and s['slow']['failures']>=2)
        state=self.states();self.assertFalse(state['badtls']['healthy']);self.assertFalse(state['slow']['healthy']);self.assertTrue(state['b']['healthy'])
        self.assertEqual(self.query('/bad/echo')[0],503)
        stats=self.stats();self.assertLessEqual(stats['active_probes'],2);self.assertLessEqual(stats['peak_probes'],2)
        self.assertEqual(self.query('/')[2],b'healthy-static')
    def test_passive_reset_without_retry_and_active_recovery(self):
        self.wait(lambda s:s['passive']['healthy'])
        # Wait for a probe completion so the next active probe cannot reset the
        # passive counter between the two intentionally adjacent failures.
        self.wait(lambda s:s['passive']['healthy'] and not s['passive']['probing'])
        self.services[2].healthy=False;self.services[2].mode='reset'
        try:
            for _ in range(2):self.assertEqual(self.query('/passive/echo',b'one-effect')[0],502)
            self.wait(lambda s:not s['passive']['healthy'])
            self.assertEqual(self.query('/passive/echo')[0],503)
            self.services[2].healthy=True;self.services[2].mode='ok';self.wait(lambda s:s['passive']['healthy']);self.assertEqual(self.query('/passive/echo')[0],200)
        finally:self.services[2].mode='ok';self.services[2].healthy=True;self.restore()
    def test_application_errors_are_not_transport_failures(self):
        self.wait(lambda s:s['passive']['healthy'])
        self.services[2].mode='app-error'
        try:
            before=self.states()['passive']['transport_errors']
            for _ in range(3):self.assertEqual(self.query('/passive/echo')[0],503)
            self.assertEqual(self.states()['passive']['transport_errors'],before);self.assertTrue(self.states()['passive']['healthy'])
        finally:self.services[2].mode='ok'
    def test_concurrent_traffic_during_state_changes_and_client_aborts(self):
        def traffic(index):
            good=0
            for i in range(30):
                if i%10==9:
                    with socket.create_connection(('127.0.0.1',self.port),timeout=3) as s:s.sendall(b'GET /app/large HTTP/1.1\r\nHost: x\r\n\r\n')
                else:self.assertEqual(self.query()[0],200);good+=1
            return good
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as e:
                tasks=[e.submit(traffic,n) for n in range(6)];self.services[0].healthy=False;time.sleep(.4);self.services[0].healthy=True
                self.assertEqual(sum(x.result() for x in tasks),162)
        finally:self.restore()
    def test_metrics_are_loopback_read_only_and_bounded(self):
        status,h,body=self.query('/_uf/metrics',port=self.admin);self.assertEqual(status,200);self.assertLess(len(body),65536);self.assertIn('pools',json.loads(body))
        self.assertEqual(self.query('/_uf/metrics',b'mutate',port=self.admin)[0],405)
        for text in [self.text.replace(f'127.0.0.1:{self.admin}',f'0.0.0.0:{self.admin}'),self.text.replace('failures 2','failures 0'),self.text.replace('recoveries 2','recoveries 0'),self.text.replace('probe_concurrency 2','probe_concurrency 17'),self.text.replace('interval_ms 200','interval_ms 1'),self.text.replace('root ".";','root "."; tls_context_budget_bytes 2097152;')]:
            path=self.root/'invalid.ufc';path.write_text(text);self.assertNotEqual(subprocess.run([BINARY,'--check','--config',str(path)],capture_output=True).returncode,0)
    def test_tcp_service_failure_and_restart(self):
        old=self.services[0];port=old.port;old.close()
        try:
            self.wait(lambda states:not states['a']['healthy'])
            self.assertNotIn('a',[json.loads(self.query()[2])['backend'] for _ in range(6)])
        finally:self.services[0]=Backend('a',port=port)
        self.wait(lambda states:states['a']['healthy'])
        self.assertIn('a',[json.loads(self.query()[2])['backend'] for _ in range(6)])
    def test_descriptor_exhaustion_during_health_checks(self):
        ports=[t.free_port() for _ in range(3)];text=self.text
        for old,new in zip([self.port,self.secure,self.admin],ports):text=text.replace(f'127.0.0.1:{old}',f'127.0.0.1:{new}')
        path=self.root/'fd-health.ufc';path.write_text(text)
        def limits():resource.setrlimit(resource.RLIMIT_NOFILE,(48,48))
        with open(self.root/'fd-health.log','wb') as log:
            proc=subprocess.Popen([BINARY,'serve','--config',str(path)],stderr=log,preexec_fn=limits);held=[]
            try:
                for _ in range(200):
                    try:
                        with socket.create_connection(('127.0.0.1',ports[0]),timeout=.1):break
                    except OSError:time.sleep(.01)
                else:self.fail('fd fixture startup')
                for _ in range(44):
                    try:held.append(socket.create_connection(('127.0.0.1',ports[0]),timeout=.2))
                    except OSError:break
                time.sleep(.15);self.assertIsNone(proc.poll())
                for sock in held:sock.close()
                held.clear();time.sleep(.3)
                self.assertEqual(self.query('/',port=ports[0])[2],b'healthy-static')
                stats=json.loads(self.query('/_uf/metrics',port=ports[2])[2]);self.assertLessEqual(stats['active_probes'],2)
            finally:
                for sock in held:sock.close()
                proc.terminate();self.assertEqual(proc.wait(timeout=3),0)
    def test_metrics_rejects_untrusted_host_authority(self):
        with socket.create_connection(('127.0.0.1',self.admin),timeout=3) as sock:
            sock.sendall(b'GET /_uf/metrics HTTP/1.1\r\nHost: attacker.test\r\n\r\n')
            status,headers,body=t.p.Reader(sock).response()
            self.assertEqual(status,403);self.assertNotIn(b'backends',body)
    def test_shutdown_with_probes_and_multiple_active_listeners(self):
        ports=[t.free_port() for _ in range(3)];text=self.text
        for old,new in zip([self.port,self.secure,self.admin],ports):text=text.replace(f'127.0.0.1:{old}',f'127.0.0.1:{new}')
        path=self.root/'shutdown.ufc';path.write_text(text)
        with open(self.root/'shutdown.log','wb') as log:
            proc=subprocess.Popen([BINARY,'serve','--config',str(path)],stderr=log)
            try:
                for _ in range(200):
                    try:
                        status,_,body=self.query('/_uf/metrics',port=ports[2]);stats=json.loads(body)
                        if stats['active_probes']:break
                    except OSError:pass
                    time.sleep(.01)
                else:self.fail('active probe startup')
                start=time.monotonic();proc.terminate();self.assertEqual(proc.wait(timeout=3),0);self.assertLess(time.monotonic()-start,2)
                for port in ports:
                    with self.assertRaises(OSError):socket.create_connection(('127.0.0.1',port),timeout=.2)
            finally:
                if proc.poll() is None:proc.kill();proc.wait()
if __name__=='__main__':unittest.main(verbosity=2)
