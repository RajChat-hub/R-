"""Temporary certificate fixture for the native TLS partial-IO test."""
import importlib.util,pathlib,subprocess,sys,tempfile
spec=importlib.util.spec_from_file_location('certgen',pathlib.Path(__file__).parents[2]/'examples/make-test-certs.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
with tempfile.TemporaryDirectory() as root:
    m.generate(root);subprocess.run([sys.argv[1],root],check=True,timeout=20)
