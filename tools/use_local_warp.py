"""Select the verified CUDA 12 Warp wheel for the current NVIDIA driver."""
import hashlib
from common import ROOT

FILENAME = 'warp_lang-1.18.0+cu12-py3-none-manylinux_2_28_x86_64.whl'
SHA256 = '6dda3c163dde4edaa7d46ade9fff3ddb1126ee7b147d9d890cd38cc70dafd656'
URL = 'https://github.com/NVIDIA/warp/releases/download/v1.18.0/warp_lang-1.18.0%2Bcu12-py3-none-manylinux_2_28_x86_64.whl'

def main():
    wheel = ROOT / 'data/downloads/wheels' / FILENAME
    if not wheel.is_file():
        raise SystemExit(f'Missing CUDA 12 Warp wheel. Download {URL} to {wheel}. See docs/DOWNLOADS.md')
    digest = hashlib.sha256()
    with wheel.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(chunk)
    if digest.hexdigest() != SHA256:
        raise SystemExit(f'Warp wheel checksum mismatch: {wheel}')
    targets = [ROOT / 'envs/training/pyproject.toml',
               ROOT / 'third_party/SuperMimic-Reproduced/pyproject.toml']
    new = f'warp-lang = {{ path = "../../data/downloads/wheels/{FILENAME}" }}'
    for path in targets:
        content = path.read_text()
        sources = [line for line in content.splitlines() if line.startswith('warp-lang = ')]
        if len(sources) != 1:
            raise SystemExit(f'Unexpected Warp source configuration: {path}')
        content = content.replace(sources[0], new)
        content = content.replace('warp-lang>=1.15.0', 'warp-lang==1.18.0+cu12')
        if content != path.read_text():
            path.write_text(content)
    print('Verified CUDA 12 Warp wheel selected:', wheel)

if __name__ == '__main__':
    main()
