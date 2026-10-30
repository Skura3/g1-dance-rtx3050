"""Prepare new tutorial configs only; never start training or overwrite a run."""
import argparse
import copy
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name', required=True)
    p.add_argument('--motion', type=Path, required=True)
    p.add_argument('--batch', type=int, choices=(32, 128, 256, 512, 1024), default=32)
    p.add_argument('--use-original-focus', action='store_true',
                   help='Only for the exact original reference_full_contact motion')
    args = p.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', args.name):
        p.error('Use only letters, digits, underscores and hyphens for the name')
    motion = (ROOT / args.motion).resolve()
    if not motion.is_file() or motion.suffix != '.npz':
        p.error('Motion must be an existing .npz file')
    try:
        relative = motion.relative_to(ROOT)
    except ValueError:
        p.error('Place the motion inside this project first')
    digest = hashlib.sha256(motion.read_bytes()).hexdigest()
    original = ROOT / 'data/preview/reference_full_contact.npz'
    if args.use_original_focus and (motion.stem != original.stem or not original.is_file()
                                   or digest != hashlib.sha256(original.read_bytes()).hexdigest()):
        p.error('Original focus windows require the original clip name and exact contents')
    out = ROOT / 'outputs/tutorial_runs' / args.name
    # Check every potential collision before writing anything.
    targets = [out] + [ROOT / 'logs/train' / (args.name + suffix)
                       for suffix in ('_warmup', '_long', '_random', '_random_smoke')]
    if any(x.exists() for x in targets):
        p.error('This experiment name already exists; choose a new one')
    cfg = json.loads((ROOT / 'configs/g1_3050_nju_anchor.json').read_text())
    cfg['data'].update(npz_dir=str(relative.parent), clips=[motion.stem],
                       library=None, manifest=None, limit=None)
    cfg['wm_trainer']['batch_size'] = args.batch
    cfg['policy_trainer']['batch_size'] = args.batch
    if not args.use_original_focus:
        cfg['env'].update(fail_reset_frac=0.0, fail_reset_list=None)
    gate = json.loads((ROOT / 'configs/g1_3050_long_training.json').read_text())
    gate.pop('first_evaluation_iteration', None)
    gate['criteria_file'] = str((out / 'acceptance.json').relative_to(ROOT))
    gate['notes'] = 'Fresh tutorial run. Full evaluation every 4000 additional iterations. No early-fall stop.'
    acceptance = json.loads((ROOT / 'configs/dance_acceptance.json').read_text())
    acceptance['reference'] = str(relative)
    acceptance['status'] = 'Copied engineering targets; not a proven attainable or theoretical minimum'
    scratch = json.loads((ROOT / 'configs/g1_3050.json').read_text())
    scratch['run_name'] = args.name + '_random'
    scratch['data'] = copy.deepcopy(cfg['data'])
    out.mkdir(parents=True)
    for filename, value in [('finetune.json', cfg), ('supervision.json', gate),
                            ('acceptance.json', acceptance), ('random_init.json', scratch),
                            ('session.json', {'name': args.name, 'motion': str(relative),
                             'motion_sha256': digest, 'batch': args.batch,
                             'original_focus': args.use_original_focus, 'training_started': False})]:
        (out / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'created': str(out), 'motion': str(relative),
                      'next': 'Follow tutorial chapter 04; no process has been started'},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
