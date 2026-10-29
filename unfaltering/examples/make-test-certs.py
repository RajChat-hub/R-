#!/usr/bin/env python3
"""Generate disposable LOCAL TEST credentials only, in an operator-selected directory.
Uses the installed openssl command, no Python crypto dependency. Never add its output to Git.
"""
import os, pathlib, subprocess, sys

def generate(directory):
    root=pathlib.Path(directory).resolve();root.mkdir(mode=0o700,parents=True,exist_ok=True)
    def run(*args):
        subprocess.run(['openssl',*args],cwd=root,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    run('req','-x509','-newkey','rsa:2048','-nodes','-keyout','ca.key','-out','ca.crt','-days','2','-subj','/CN=UNFALTERING Local Test CA','-addext','basicConstraints=critical,CA:TRUE','-addext','keyUsage=critical,keyCertSign,cRLSign')
    (root/'extensions.cnf').write_text('basicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature,keyEncipherment\nextendedKeyUsage=serverAuth\nsubjectAltName=DNS:localhost,DNS:app.test,IP:127.0.0.1\n')
    run('req','-new','-newkey','rsa:2048','-nodes','-keyout','server.key','-out','server.csr','-subj','/CN=localhost')
    run('x509','-req','-in','server.csr','-CA','ca.crt','-CAkey','ca.key','-set_serial','2','-out','server.crt','-days','1','-extfile','extensions.cnf')
    (root/'index.txt').write_text('');(root/'serial').write_text('03\n')
    (root/'ca.cnf').write_text('[ca]\ndefault_ca=test\n[test]\ndatabase=index.txt\nserial=serial\nnew_certs_dir=.\ncertificate=ca.crt\nprivate_key=ca.key\ndefault_md=sha256\npolicy=policy\n[policy]\ncommonName=supplied\n')
    run('ca','-batch','-config','ca.cnf','-in','server.csr','-out','expired.crt','-startdate','20000101000000Z','-enddate','20010101000000Z','-extfile','extensions.cnf')
    for key in root.glob('*.key'):os.chmod(key,0o600)
    return root
if __name__=='__main__':
    if len(sys.argv)!=2:raise SystemExit('Usage: make-test-certs.py DIRECTORY (local tests only)')
    print(generate(sys.argv[1]))
