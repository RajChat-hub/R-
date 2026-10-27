"""Small reproducible loopback sample, not an Apache/Nginx comparison."""
import concurrent.futures
import json
import hashlib
import math
import os
import statistics
from pathlib import Path
import platform
import resource
import socket
import subprocess
import sys
import tempfile
import time

binary=str(Path(sys.argv[1]).resolve())
with tempfile.TemporaryDirectory(prefix='unfaltering-bench-') as temp:
    root=Path(temp);(root/'index.html').write_bytes(b'x'*1024)
    with socket.socket() as s:
        s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    cfg=root/'server.ufc';cfg.write_text(f'server {{ listen "127.0.0.1:{port}"; root "."; }}')
    before=resource.getrusage(resource.RUSAGE_CHILDREN)
    with open(root/'log','wb') as log:
        p=subprocess.Popen([binary,'serve','--config',str(cfg)],stderr=log)
        try:
            for _ in range(200):
                try:
                    with socket.create_connection(('127.0.0.1',port),timeout=.1):break
                except OSError:time.sleep(.01)
            else:raise RuntimeError('startup timeout')
            def request(_):
                start=time.perf_counter()
                with socket.create_connection(('127.0.0.1',port),timeout=2) as s:
                    s.sendall(b'GET / HTTP/1.1\r\nHost: bench.test\r\nConnection: close\r\n\r\n')
                    data=b''
                    while True:
                        chunk=s.recv(65536)
                        if not chunk:break
                        data+=chunk
                if not data.startswith(b'HTTP/1.1 200') or len(data.split(b'\r\n\r\n',1)[1])!=1024:raise RuntimeError('bad response')
                return (time.perf_counter()-start)*1000
            start=time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:latencies=sorted(pool.map(request,range(512)))
            elapsed=time.perf_counter()-start
        finally:
            p.terminate();p.wait(timeout=5)
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    result=dict(os=platform.platform(),machine=platform.machine(),visible_cpu_count=os.cpu_count(),binary_sha256=hashlib.sha256(Path(binary).read_bytes()).hexdigest(),python=platform.python_version(),requests=512,clients=8,payload_bytes=1024,tls=False,connection_reuse=False,elapsed_seconds=elapsed,requests_per_second=512/elapsed,median_ms=statistics.median(latencies),p95_ms=latencies[math.ceil(len(latencies)*.95)-1],p99_ms=latencies[math.ceil(len(latencies)*.99)-1],child_cpu_seconds=(after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime),child_peak_rss_kib=after.ru_maxrss,error_count=0,scope='Python client + loopback + warm 1KiB file; development sample, no comparative performance claim')
    print(json.dumps(result,indent=2))
