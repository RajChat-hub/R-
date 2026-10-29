"""Bounded sustained concurrency/failure drill; duration is configurable, never a capacity claim."""
import concurrent.futures, json, os, socket, time
import proxy_test as p
p.ProxyTests.setUpClass()
try:
    duration=float(os.environ.get('UF_SOAK_SECONDS','10'))
    start=time.monotonic();end=start+duration
    def client(i):
        completed=resets=0
        while time.monotonic()<end:
            with socket.create_connection(('127.0.0.1',p.ProxyTests.port),timeout=3) as s:
                s.settimeout(5)
                if completed%9==8:
                    s.sendall(b'GET /proxy/large HTTP/1.1\r\nHost: x\r\n\r\n');resets+=1
                else:
                    request=b'GET /proxy/chunk HTTP/1.1\r\nHost: x\r\n\r\n'
                    if i%2:
                        for k in range(0,len(request),3):s.sendall(request[k:k+3])
                    else:s.sendall(request)
                    status,_,body=p.Reader(s).response()
                    assert status==200 and body==b'abcdefg'
                completed+=1
        return completed,resets
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:results=list(ex.map(client,range(8)))
    print(json.dumps({'seconds':time.monotonic()-start,'clients':8,'transactions':sum(x[0] for x in results),'injected_disconnects':sum(x[1] for x in results)}))
finally:p.ProxyTests.tearDownClass()
