"""Verify manually downloaded release assets and prepare portable demo/resume paths."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', type=Path, default=ROOT / 'data/releases/v1.0.0')
    p.add_argument('--mode', choices=('display', 'resume', 'all'), default='display')
    args = p.parse_args()
    folder = args.assets.resolve()
    manifest = json.loads((ROOT / 'configs/release_assets.json').read_text())
    required = {'checkpoint_429526.pt', 'g1_scene.mjb'} if args.mode == 'display' else {'checkpoint_final_452376.pt'}
    if args.mode == 'all':
        required = {x['filename'] for x in manifest['assets']}
    for row in manifest['assets']:
        if row['filename'] not in required:
            continue
        path = folder / row['filename']
        if not path.is_file():
            raise FileNotFoundError(f'Download release asset to {path}')
        if digest(path) != row['sha256']:
            raise ValueError(f'Wrong or incomplete asset: {path}')
    out = ROOT / 'outputs/public_demo'
    out.mkdir(parents=True, exist_ok=True)
    motion = ROOT / 'data/preview/reference_full_contact.npz'
    motion.parent.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'examples/reference_dance/reference_full_contact.npz'
    if motion.exists() and digest(motion) != digest(source):
        raise FileExistsError(f'Keep existing different motion; relocate it yourself: {motion}')
    if not motion.exists():
        shutil.copyfile(source, motion)
    if args.mode in ('display', 'all'):
        report = json.loads((ROOT / 'results/rtx3050/evaluation_429526.json').read_text())
        report['checkpoint'] = str(folder / 'checkpoint_429526.pt')
        report['clips'][0]['states'] = str(ROOT / 'examples/reference_dance/policy_429526_states.npz')
        (out / 'replay_report.json').write_text(json.dumps(report, indent=2) + chr(10))
    if args.mode in ('resume', 'all'):
        settings = json.loads((ROOT / 'configs/g1_3050_long_training.json').read_text())
        settings['criteria_file'] = 'configs/dance_acceptance.json'
        settings.pop('first_evaluation_iteration', None)
        (out / 'resume_supervision.json').write_text(json.dumps(settings, indent=2) + chr(10))
    print(json.dumps({'verified': sorted(required), 'output': str(out),
                      'training_started': False, 'viewer_started': False}, indent=2))


if __name__ == '__main__':
    main()
