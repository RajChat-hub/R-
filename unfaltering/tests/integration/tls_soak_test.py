"""Timed verified HTTPS concurrency and deliberate disconnect drill."""
import concurrent.futures,json,os,time
import tls_test as t
t.TlsTests.setUpClass()
try:
    fixture=t.TlsTests();duration=float(os.environ.get('UF_TLS_SOAK_SECONDS','30'))
    start=time.monotonic();end=start+duration
    def client(i):
        n=resets=0
        while time.monotonic()<end:
            if n%11==10:
                s=fixture.connect();s.sendall(b'GET /proxy/large HTTP/1.1\r\nHost: x\r\n\r\n');fd=s.detach();os.close(fd);resets+=1
            else:
                with fixture.connect() as s:
                    wire=b'GET /proxy/chunk HTTP/1.1\r\nHost: x\r\n\r\n'
                    if i%2:
                        for k in range(0,len(wire),3):s.sendall(wire[k:k+3])
                    else:s.sendall(wire)
                    if n%7==6:time.sleep(.01)
                    status,_,body=t.p.Reader(s).response()
                    assert status==200 and body==b'abcdefg',(status,body)
            n+=1
        return n,resets
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as e:results=list(e.map(client,range(8)))
    print(json.dumps({'tls_seconds':time.monotonic()-start,'clients':8,'transactions':sum(x[0] for x in results),'injected_disconnects':sum(x[1] for x in results)}))
finally:t.TlsTests.tearDownClass()
