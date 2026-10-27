"""Phase A adversarial checks, in addition to all foundation tests."""
import contextlib
import socket
import subprocess
import time
import unittest
import http_test as base

class Hardening(unittest.TestCase):
    setUpClass=classmethod(base.ServerTests.setUpClass.__func__)
    tearDownClass=classmethod(base.ServerTests.tearDownClass.__func__)
    connect=base.ServerTests.connect
    query=base.ServerTests.query

    @contextlib.contextmanager
    def fixture(self,extra=''):
        with socket.socket() as s:
            s.bind(('127.0.0.1',0));port=s.getsockname()[1]
        config=self.root/'adversarial.ufc'
        config.write_text(f'server {{ listen "127.0.0.1:{port}"; root "one"; {extra} }}')
        p=subprocess.Popen([base.BINARY,'serve','--config',str(config)],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        try:
            for _ in range(100):
                try:
                    with socket.create_connection(('127.0.0.1',port),timeout=.1):break
                except OSError:time.sleep(.02)
            else:self.fail('fixture startup failed')
            time.sleep(.05)
            yield p,port
        finally:
            if p.poll() is None:p.terminate()
            self.assertEqual(p.wait(timeout=3),0)
            p.stderr.close()

    def test_tiny_chunk_fragmentation(self):
        s=self.connect();s.sendall(b'POST / HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n')
        for _ in range(4000):s.sendall(b'1\r\nx\r\n')
        s.sendall(b'0\r\n\r\nGET / HTTP/1.1\r\nHost: x\r\n\r\n')
        reader=base.ResponseReader(s)
        self.assertEqual(reader.response()[0],405)
        self.assertEqual(reader.response()[2],b'hello-one')

    def test_blocked_log_pipe_does_not_stall_network_or_shutdown(self):
        with self.fixture() as (p,port):
            # Deliberately never read stderr; more records than a pipe can hold.
            for _ in range(1200):
                with socket.create_connection(('127.0.0.1',port),timeout=2) as s:
                    s.settimeout(2);s.sendall(b'GET / HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n')
                    self.assertEqual(base.ResponseReader(s).response()[0],200)
            self.assertIsNone(p.poll())

    def test_memory_reservation_caps_connection_admission(self):
        with self.fixture('memory_budget_bytes 1048576;') as (p,port):
            held=[]
            try:
                # Default reservation is >400KiB. Two reservations fit, a third
                # is refused rather than allocating up to max_connections=256.
                for _ in range(2):held.append(socket.create_connection(('127.0.0.1',port),timeout=1))
                time.sleep(.05)
                with socket.create_connection(('127.0.0.1',port),timeout=1) as extra:
                    extra.settimeout(1);self.assertEqual(extra.recv(1),b'')
                self.assertIsNone(p.poll())
            finally:
                for s in held:s.close()

if __name__=='__main__':unittest.main(verbosity=2)
