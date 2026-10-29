"""Distinguish an unsupported TSan runtime from a real race/test failure."""
import json,pathlib,subprocess,sys,time
build=pathlib.Path(sys.argv[1]).resolve();start=time.monotonic()
probe=subprocess.run([str(build/'uf_concurrency')],capture_output=True,text=True)
print(probe.stdout+probe.stderr,flush=True)
if probe.returncode:
    if 'FATAL: ThreadSanitizer: unexpected memory mapping' in probe.stderr:
        print('::warning::ThreadSanitizer runtime unsupported: unexpected memory mapping; race result INCONCLUSIVE')
        print(json.dumps({'tsan':'unsupported-inconclusive','seconds':time.monotonic()-start}))
        raise SystemExit(0)
    raise SystemExit(probe.returncode)
result=subprocess.run(['ctest','--test-dir',str(build),'-R','concurrency_unit|sustained_concurrency|tls_partial_io','--output-on-failure','--verbose'])
print(json.dumps({'tsan':'passed' if result.returncode==0 else 'failed','seconds':time.monotonic()-start}))
raise SystemExit(result.returncode)
