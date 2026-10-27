"""Execute the shipped example configuration against its independent demo app."""
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

binary=str(Path(sys.argv[1]).resolve())
examples=Path(sys.argv[2]).resolve()
def port():
    with socket.socket() as s:
        s.bind(('127.0.0.1',0))
        return s.getsockname()[1]
def ready(number,process):
    for _ in range(150):
        if process.poll() is not None: raise RuntimeError('demo process exited')
        try:
            with socket.create_connection(('127.0.0.1',number),timeout=.1):return
        except OSError:time.sleep(.02)
    raise RuntimeError('demo readiness timeout')
with tempfile.TemporaryDirectory(prefix='unfaltering-demo-') as temp:
    root=Path(temp);shutil.copytree(examples/'www',root/'www')
    app_port,server_port=port(),port()
    config=root/'proxy.ufc'
    config.write_text((examples/'proxy.ufc').read_text().replace('127.0.0.1:9000',f'127.0.0.1:{app_port}').replace('127.0.0.1:8080',f'127.0.0.1:{server_port}'))
    subprocess.run([binary,'--check','--config',str(config)],check=True)
    app=subprocess.Popen([sys.executable,str(examples/'upstream.py'),'--port',str(app_port)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    server=None
    try:
        ready(app_port,app)
        server=subprocess.Popen([binary,'serve','--config',str(config)],stderr=subprocess.DEVNULL)
        ready(server_port,server)
        with urllib.request.urlopen(f'http://127.0.0.1:{server_port}/',timeout=3) as r:
            if b'UNFALTERING' not in r.read(): raise AssertionError('static example')
        for method,body in [('GET',None),('POST',b'example body')]:
            request=urllib.request.Request(f'http://127.0.0.1:{server_port}/app/hello',data=body,method=method)
            with urllib.request.urlopen(request,timeout=3) as r:value=json.load(r)
            if value['method']!=method or value['sha256']!=hashlib.sha256(body or b'').hexdigest(): raise AssertionError(value)
            if value['host']!='app.local' or value['forwarded_for']!='127.0.0.1': raise AssertionError('example header policy')
        print('Shipped configuration + static site + native proxied demo GET/POST passed')
    finally:
        if server:
            server.terminate()
            if server.wait(timeout=4)!=0:raise AssertionError('native demo shutdown')
        app.terminate();app.wait(timeout=4)
