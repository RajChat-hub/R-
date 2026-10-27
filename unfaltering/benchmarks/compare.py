"""Small paired development benchmark; not commercial capacity certification."""
import json
import os
from pathlib import Path
import subprocess
import sys

baseline,current=sys.argv[1:3]
script=Path(__file__).with_name('loopback.py')
results=[]
for repeat in range(3):
    for label,binary in [('foundation',baseline),('phase2',current)]:
        p=subprocess.run([sys.executable,str(script),binary],capture_output=True,text=True,check=True)
        results.append(dict(repeat=repeat,label=label,metrics=json.loads(p.stdout)))
print(json.dumps(dict(baseline_commit='549c531e9ce2eff7fa5d5c64efc5d8fefce89c00',phase2_commit=os.environ.get('GITHUB_SHA'),compiler='GCC; CMake Release -O3 -DNDEBUG',method='three paired short loopback runs; 512 new-connection 1KiB requests, eight clients; warm files; no TLS; Python client overhead; no Nginx comparison',runs=results),indent=2))
