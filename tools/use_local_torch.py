"""Use the user-downloaded training wheel without downloading it again."""
import hashlib
from pathlib import Path
from common import ROOT

FILENAME='torch-2.14.1+cu126-cp312-cp312-manylinux_2_28_x86_64.whl'
SHA256='75727245346d8f92224a860bf40c56bd681edcfa0456d01d7dfcb0bf6f0cf86c'

def main():
    wheel=ROOT/'data/downloads/wheels'/FILENAME
    if not wheel.is_file():
        print('No local training wheel; using configured package source.');return
    digest=hashlib.sha256()
    with wheel.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):digest.update(chunk)
    if digest.hexdigest()!=SHA256:raise SystemExit('Training torch wheel checksum mismatch: '+str(wheel))
    replacements=[
        (ROOT/'envs/training/pyproject.toml','torch = { index = "pytorch-cu126" }'),
        (ROOT/'third_party/SuperMimic-Reproduced/pyproject.toml', '''torch = { index = "pytorch-cu126", marker = "sys_platform != 'darwin'" }'''),
    ]
    new='torch = { path = "../../data/downloads/wheels/'+FILENAME+'" }'
    for path,old in replacements:
        text=path.read_text()
        if new in text:continue
        if text.count(old)!=1:raise SystemExit('Unexpected source configuration; inspect '+str(path))
        path.write_text(text.replace(old,new,1))
    print('Verified local training wheel selected:',wheel)

if __name__=='__main__':main()
