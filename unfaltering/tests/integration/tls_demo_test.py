"""Run the shipped HTTPS example with the shipped independent TLS demo app."""
import importlib.util,json,pathlib,socket,ssl,subprocess,sys,tempfile,time
binary=pathlib.Path(sys.argv[1]).resolve();examples=pathlib.Path(sys.argv[2]).resolve()
spec=importlib.util.spec_from_file_location('certgen',examples/'make-test-certs.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def port():
    with socket.socket() as s:s.bind(('127.0.0.1',0));return s.getsockname()[1]
with tempfile.TemporaryDirectory() as temp:
    root=pathlib.Path(temp);m.generate(root/'tls-local');down,up=port(),port()
    text=(examples/'https.ufc').read_text().replace('127.0.0.1:8443',f'127.0.0.1:{down}').replace('127.0.0.1:9443',f'127.0.0.1:{up}').replace('root "www"',f'root "{examples / "www"}"')
    config=root/'https.ufc';config.write_text(text)
    ctx=ssl.create_default_context(cafile=str(root/'tls-local/ca.crt'))
    processes=[]
    try:
        with open(root/'up.log','wb') as uplog,open(root/'native.log','wb') as native:
            processes.append(subprocess.Popen([sys.executable,str(examples/'upstream.py'),'--port',str(up),'--certificate',str(root/'tls-local/server.crt'),'--private-key',str(root/'tls-local/server.key')],stderr=uplog))
            subprocess.run([str(binary),'--check','--config',str(config)],check=True)
            processes.append(subprocess.Popen([str(binary),'serve','--config',str(config)],stderr=native))
            for target in [up,down]:
                for _ in range(200):
                    try:
                        with socket.create_connection(('127.0.0.1',target),timeout=.1):break
                    except OSError:time.sleep(.01)
                else:raise RuntimeError('demo startup timeout')
            import http.client
            for path,body in [('/',None),('/app/hello',None),('/app/echo',b'hello-tls')]:
                c=http.client.HTTPSConnection('localhost',down,context=ctx,timeout=4)
                c.request('POST' if body else 'GET',path,body=body);r=c.getresponse();data=r.read();assert r.status==200
                if path.startswith('/app'):
                    value=json.loads(data);assert value['bytes']==len(body or b'') and value['host']=='app.test'
                else:assert b'UNFALTERING' in data
                c.close()
            print('Shipped HTTPS config + verified static/GET/POST via shipped HTTPS upstream passed')
    finally:
        for proc in processes:
            if proc.poll() is None:proc.terminate()
        for proc in processes:
            if proc.wait(timeout=4)!=0 and proc is processes[-1]:raise RuntimeError('native demo failed')
