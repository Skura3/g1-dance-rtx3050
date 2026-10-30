"""Validate local assets on CPU, one process per model; never download files.

Checkpoint files must come from sources trusted by the user: upstream loaders
deserialize Python pickle. Tests use synthetic inputs, not dance inference.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from common import ROOT, GVHMR, CHECKPOINTS, add_sources
from use_local_warp import FILENAME as WARP_FILENAME, SHA256 as WARP_SHA256

ASSETS = {
    'yolo': CHECKPOINTS / 'yolo/yolov8x.pt',
    'vitpose': CHECKPOINTS / 'vitpose/vitpose-h-multi-coco.pth',
    'hmr2': CHECKPOINTS / 'hmr2/epoch=10-step=25000.ckpt',
    'gvhmr': CHECKPOINTS / 'gvhmr/gvhmr_siga24_release.ckpt',
    'smplx': CHECKPOINTS / 'body_models/smplx/SMPLX_NEUTRAL.npz',
    'torch_wheel': ROOT / 'data/downloads/wheels/torch-2.14.1+cu126-cp312-cp312-manylinux_2_28_x86_64.whl',
    'warp_wheel': ROOT / 'data/downloads/wheels' / WARP_FILENAME,
}
TORCH_SHA256 = '75727245346d8f92224a860bf40c56bd681edcfa0456d01d7dfcb0bf6f0cf86c'
DEST = ROOT / 'outputs/download_audit'

def file_info(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(chunk)
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f'Archive CRC failed: {bad}')
        members = len(archive.infolist())
    return dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size,
                sha256=digest.hexdigest(), archive_crc_ok=True, archive_members=members,
                is_symlink=path.is_symlink())

def validate(name, info):
    path = ASSETS[name]
    if name in ('torch_wheel', 'warp_wheel'):
        from email.parser import Parser
        with zipfile.ZipFile(path) as archive:
            meta = next(n for n in archive.namelist() if n.endswith('.dist-info/METADATA'))
            wheel = next(n for n in archive.namelist() if n.endswith('.dist-info/WHEEL'))
            metadata = Parser().parsestr(archive.read(meta).decode())
            tags = [line for line in archive.read(wheel).decode().splitlines() if line.startswith('Tag:')]
        if name == 'torch_wheel':
            assert metadata['Name'] == 'torch' and metadata['Version'] == '2.14.1+cu126'
            assert 'Tag: cp312-cp312-manylinux_2_28_x86_64' in tags
            assert info['sha256'] == TORCH_SHA256, 'SHA256 differs from training lock'
            info['matches_lock'] = True
        else:
            assert metadata['Name'].replace('_', '-') == 'warp-lang' and metadata['Version'] == '1.18.0+cu12'
            assert 'Tag: py3-none-manylinux_2_28_x86_64' in tags
            assert info['sha256'] == WARP_SHA256, 'SHA256 differs from official CUDA 12 release'
            info['matches_official_release'] = True
        info.update(version=metadata['Version'], wheel_tags=tags)
        return
    add_sources()
    os.chdir(GVHMR)
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(0)
    if name == 'smplx':
        import numpy as np
        import smplx
        # Official NPZ files can contain object-valued auxiliary metadata. Read
        # headers for inspection; the SMPL-X loader handles its required arrays.
        info['array_shapes'] = {}
        with zipfile.ZipFile(path) as archive:
            for member in archive.namelist():
                with archive.open(member) as stream:
                    version = np.lib.format.read_magic(stream)
                    shape, _, _ = np.lib.format._read_array_header(stream, version)
                    info['array_shapes'][member.removesuffix('.npy')] = list(shape)
        with np.load(path, allow_pickle=False) as archive:
            assert archive['v_template'].shape == (10475, 3)
            assert archive['J_regressor'].shape == (55, 10475)
        model = smplx.create(str(CHECKPOINTS / 'body_models'), 'smplx',
                             gender='neutral', use_pca=False).eval()
        with torch.inference_mode():
            output = model(return_full_pose=True)
        assert torch.isfinite(output.vertices).all() and torch.isfinite(output.joints).all()
        info.update(vertices=list(output.vertices.shape), joints=list(output.joints.shape),
                    forward_ok=True, version='SMPL-X NPZ, neutral filename, 10475 vertices / 55 skeletal joints',
                    provenance_note='No publisher checksum or embedded release/gender metadata; compatibility verified, origin not independently authenticated.')
        return
    if name == 'yolo':
        from ultralytics import YOLO
        model = YOLO(str(path)).model.float().cpu().eval()
        assert model.names[0] == 'person' and len(model.names) == 80
        with torch.inference_mode():
            output = model(torch.zeros(1, 3, 256, 256))[0]
        assert torch.isfinite(output).all()
        info.update(version='YOLOv8x, COCO80', model_yaml=str(model.yaml.get('yaml_file')),
                    parameters=sum(p.numel() for p in model.parameters()),
                    classes=len(model.names), forward_shape=list(output.shape), forward_ok=True)
        return
    checkpoint = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    state = checkpoint['state_dict']
    info['checkpoint_tensors'] = len(state)
    info['checkpoint_metadata'] = {key: checkpoint[key] for key in
        ('epoch', 'global_step', 'pytorch-lightning_version') if key in checkpoint}
    if name == 'vitpose':
        from hmr4d.utils.preproc.vitpose_pytorch import build_model
        model = build_model('ViTPose_huge_coco_256x192')
        info.update(version='ViTPose-H, COCO17, input 256x192',
                    mmpose_version=checkpoint.get('meta', {}).get('mmpose_version'))
        inputs = torch.zeros(1, 3, 256, 192)
    elif name == 'hmr2':
        from hmr4d.network.hmr2 import HMR2
        from hmr4d.network.hmr2.configs import get_config
        cfg = get_config(str(GVHMR / 'hmr4d/network/hmr2/configs/model_config.yaml'))
        cfg.defrost(); cfg.MODEL.BBOX_SHAPE = [192, 256]; cfg.freeze()
        model = HMR2(cfg)
        state = {key: value for key, value in state.items() if key.split('.')[0] in ('backbone', 'smpl_head')}
        info.update(version='HMR2.0a, epoch 10 / step 25000', selected_tensors=len(state),
                    filtering='Same backbone/smpl_head filtering as upstream load_hmr2')
        assert checkpoint['epoch'] == 10 and checkpoint['global_step'] == 25000
        inputs = {'img': torch.zeros(1, 3, 256, 256)}
    else:
        import hydra
        from hydra import initialize_config_module, compose
        import hmr4d.model.gvhmr.gvhmr_pl_demo
        import hmr4d.model.gvhmr.utils.endecoder
        import hmr4d.network.gvhmr.relative_transformer
        with initialize_config_module(version_base='1.3', config_module='hmr4d.configs'):
            cfg = compose(config_name='demo', overrides=['video_name=asset_audit'])
        model = hydra.utils.instantiate(cfg.model, _recursive_=False)
        info['version'] = 'GVHMR SIGGRAPH Asia 2024 release architecture'
        inputs = None
    incompatible = model.load_state_dict(state, strict=False)
    info.update(missing_keys=list(incompatible.missing_keys), unexpected_keys=list(incompatible.unexpected_keys))
    assert not incompatible.missing_keys and not incompatible.unexpected_keys, 'Model/state keys mismatch'
    for key, value in state.items():
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f'Non-finite weight: {key}')
    info['all_weights_finite'] = True
    info['strict_architecture_match'] = True
    model.eval()
    with torch.inference_mode():
        if inputs is not None:
            output = model(inputs)
            assert torch.isfinite(output).all()
            info.update(forward_ok=True, forward_shape=list(output.shape))
        else:
            # Use the model's loaded body layer to check its auxiliary data, too.
            skeleton = model.pipeline.endecoder.smplx_model.get_skeleton(torch.zeros(1, 10))
            assert torch.isfinite(skeleton).all()
            info.update(body_layer_ok=True, skeleton_shape=list(skeleton.shape),
                        forward_scope='Checkpoint and body layer only; no video sequence inference')

def check_one(name):
    started = time.monotonic()
    info = {'name': name, 'path': str(ASSETS[name].relative_to(ROOT)), 'ok': False}
    try:
        info.update(file_info(ASSETS[name]))
        validate(name, info)
        info['ok'] = True
    except Exception as error:
        info.update(error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
    info['seconds'] = round(time.monotonic() - started, 2)
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / f'{name}.json').write_text(json.dumps(info, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)
    return info['ok']

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset', choices=ASSETS)
    parser.add_argument('--collect-only', action='store_true',
                        help='Aggregate saved checks; reject files whose size or mtime has changed')
    args = parser.parse_args()
    if args.asset:
        raise SystemExit(0 if check_one(args.asset) else 1)
    DEST.mkdir(parents=True, exist_ok=True)
    assets = {}
    for name in ASSETS:
        result_path = DEST / f'{name}.json'
        if not args.collect_only:
            print(f'Validating {name} on CPU...', flush=True)
            if result_path.exists():
                result_path.unlink()
            with (DEST / f'{name}.log').open('w') as log:
                subprocess.run([sys.executable, str(Path(__file__).resolve()), '--asset', name],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                    env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
        assets[name] = json.loads(result_path.read_text()) if result_path.exists() else {
            'ok': False, 'error': f'No result; see {name}.log'}
        if args.collect_only and assets[name]['ok']:
            path = ASSETS[name]
            if not path.is_file() or path.stat().st_size != assets[name]['bytes'] or path.stat().st_mtime_ns > result_path.stat().st_mtime_ns:
                assets[name] = {'ok': False, 'error': 'Asset changed or absent; rerun its validation'}
        print(f"{name}: {'PASS' if assets[name]['ok'] else 'FAIL'}", flush=True)
    report = dict(checked_at=datetime.now(timezone.utc).isoformat(),
                  all_ok=all(item['ok'] for item in assets.values()), assets=assets,
                  scope='File integrity, architecture/load compatibility and synthetic CPU smoke checks; not dance inference or training.')
    (DEST / 'assets.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    raise SystemExit(0 if report['all_ok'] else 1)

if __name__ == '__main__':
    main()
