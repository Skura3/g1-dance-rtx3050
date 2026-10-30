"""Fetch pinned upstream source; never overwrite existing checkouts or fetch model weights."""
import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def download(url, target):
    partial = target.with_suffix(target.suffix + '.partial')
    for attempt in range(2):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'g1-dance-rtx3050'})
            with urllib.request.urlopen(req, timeout=35) as src, partial.open('wb') as dst:
                shutil.copyfileobj(src, dst)
            partial.replace(target)
            return True
        except Exception as exc:
            print(f'Attempt {attempt + 1}/2: {type(exc).__name__}: {exc}', flush=True)
    print(f'MANUAL DOWNLOAD: {url}\nSAVE TO: {target}', flush=True)
    return False


def unpack(archive, destination):
    with tempfile.TemporaryDirectory(prefix='source_', dir=destination.parent) as temp:
        temp = Path(temp)
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                parts = PurePosixPath(member.name).parts
                if not parts or member.name.startswith('/') or '..' in parts:
                    raise ValueError('Unsafe archive path')
                path = temp.joinpath(*parts)
                if member.isdir():
                    path.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(member) as src, path.open('wb') as dst:
                        shutil.copyfileobj(src, dst)
                    path.chmod(member.mode & 0o777)
                elif member.issym() or member.islnk():
                    raise ValueError(f'Archive link needs manual inspection: {member.name}')
            roots = list(temp.iterdir())
            if len(roots) != 1 or not roots[0].is_dir():
                raise ValueError('Expected one top-level source directory')
        roots[0].rename(destination)


def patch_sources():
    expected = json.loads((ROOT / 'configs/patched_sources_sha256.json').read_text())
    for repo in ('GVHMR', 'SuperMimic-Reproduced'):
        rows = [x for x in expected if x['path'].startswith(repo + '/')]
        if not (ROOT / 'third_party' / repo).is_dir():
            continue
        if all((ROOT / 'third_party' / x['path']).is_file() and
               digest(ROOT / 'third_party' / x['path']) == x['sha256'] for x in rows):
            print(f'{repo}: exact patched source already present')
            continue
        patch = ROOT / 'configs/upstream_changes.patch'
        cmd = ['git', 'apply', '--unsafe-paths', '--include=' + repo + '/*', str(patch)]
        subprocess.run(cmd[:2] + ['--check'] + cmd[2:], cwd=ROOT / 'third_party', check=True)
        subprocess.run(cmd, cwd=ROOT / 'third_party', check=True)
        for row in rows:
            if digest(ROOT / 'third_party' / row['path']) != row['sha256']:
                raise ValueError(f'Patched source differs from tested file: {row["path"]}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--only', nargs='+', help='Select upstream names, e.g. GVHMR GMR')
    p.add_argument('--list', action='store_true', help='Print planned sources without downloads or writes')
    args = p.parse_args()
    sources = json.loads((ROOT / 'configs/upstream_sources.json').read_text())
    if args.only:
        unknown = set(args.only) - {x['name'] for x in sources}
        if unknown:
            p.error('Unknown sources: ' + ', '.join(sorted(unknown)))
        sources = [x for x in sources if x['name'] in args.only]
    if args.list:
        print(json.dumps(sources, indent=2))
        return
    downloads = ROOT / 'data/downloads/sources'
    downloads.mkdir(parents=True, exist_ok=True)
    (ROOT / 'third_party').mkdir(exist_ok=True)
    failed = []
    for entry in sources:
        name = entry['name']; target = ROOT / 'third_party' / name
        if target.exists():
            print(f'KEEP existing {target}; no checkout replacement')
            continue
        if 'bundled_source_archive' in entry:
            archive = ROOT / entry['bundled_source_archive']
            if digest(archive) != entry['bundled_source_sha256']:
                raise ValueError('Bundled source checksum mismatch')
        else:
            url = entry.get('source_archive')
            if not url:
                path = entry['repository'].removeprefix('https://github.com/')
                url = 'https://codeload.github.com/' + path + '/tar.gz/' + entry['commit']
            archive = downloads / (name + '-' + entry['commit'] + '.tar.gz')
            if not archive.exists() and not download(url, archive):
                failed.append(name); continue
        try:
            unpack(archive, target)
        except Exception as exc:
            failed.append(name)
            print(f'Inspect/extract manually: {archive} -> {target}: {exc}')
    patch_sources()
    if failed:
        raise SystemExit('Sources still needed: ' + ', '.join(failed))
    print('Pinned source restored. Models and environments must be prepared separately.')


if __name__ == '__main__':
    main()
