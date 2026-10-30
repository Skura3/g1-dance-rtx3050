"""Resumable GVHMR stages. Each stage uses a separate process to release GPU memory."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from common import ROOT, GVHMR, CHECKPOINTS, add_sources, require_file

STAGES = ('track', 'pose', 'features', 'camera', 'predict')
OUTPUTS = {'track':'preprocess/bbx.pt','pose':'preprocess/vitpose.pt',
           'features':'preprocess/vit_features.pt','camera':'preprocess/slam_results.pt',
           'predict':'hmr4d_results.pt'}
WEIGHTS = {'track':['yolo/yolov8x.pt'], 'pose':['vitpose/vitpose-h-multi-coco.pth'],
           'features':['hmr2/epoch=10-step=25000.ckpt'], 'camera':[],
           'predict':['gvhmr/gvhmr_siga24_release.ckpt','body_models/smplx/SMPLX_NEUTRAL.npz']}

def run_stage(a, out):
    target=out/OUTPUTS[a.stage]
    if target.exists(): print(f'Cached: {target}'); return
    for name in WEIGHTS[a.stage]: require_file(CHECKPOINTS/name,'checkpoint')
    add_sources()
    os.chdir(GVHMR)
    os.environ['GVHMR_BATCH_SIZE']=str(a.batch_size)
    os.environ['GVHMR_PREPROC_DEVICE']=a.preproc_device
    os.environ['GVHMR_PREPROC_PRECISION']=a.preproc_precision
    os.environ['GVHMR_FLIP_TEST']='1' if a.flip_test else '0'
    import torch
    torch.set_num_threads(a.threads)
    target.parent.mkdir(parents=True,exist_ok=True)
    result=None
    if a.stage=='track':
        from hmr4d.utils.preproc import Tracker
        from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy
        tracker=Tracker();xyxy=tracker.get_one_track(str(a.video)).float()
        (target.parent/'tracking_diagnostics.json').write_text(json.dumps(tracker.tracking_diagnostics,indent=2)+'\n')
        result={'bbx_xyxy':xyxy,'bbx_xys':get_bbx_xys_from_xyxy(xyxy,base_enlarge=1.2).float()}
    elif a.stage in ('pose','features'):
        box=torch.load(require_file(out/OUTPUTS['track']),map_location='cpu')['bbx_xys']
        if a.stage=='pose':
            from hmr4d.utils.preproc import VitPoseExtractor
            result=VitPoseExtractor().extract(str(a.video),box)
        else:
            from hmr4d.utils.preproc import Extractor
            result=Extractor().extract_video_features(str(a.video),box)
    elif a.stage=='camera':
        from hmr4d.utils.video_io_utils import get_video_lwh
        if a.static_camera:
            result=torch.eye(4).repeat(get_video_lwh(str(a.video))[0],1,1).numpy()
        else:
            from hmr4d.utils.preproc import SimpleVO
            result=SimpleVO(str(a.video),scale=0.5,step=8,method='sift',f_mm=a.f_mm).compute()
    elif a.stage=='predict':
        import hydra
        from hydra import initialize_config_module,compose
        # Only register inference components; the upstream all-purpose store imports
        # training datasets, metrics and renderers which this headless pipeline does not use.
        import hmr4d.model.gvhmr.gvhmr_pl_demo
        import hmr4d.model.gvhmr.utils.endecoder
        import hmr4d.network.gvhmr.relative_transformer
        from hmr4d.utils.video_io_utils import get_video_lwh
        from hmr4d.utils.geo.hmr_cam import estimate_K,create_camera_sensor
        from hmr4d.utils.geo_transform import compute_cam_angvel
        from hmr4d.utils.net_utils import detach_to_cpu
        length,width,height=get_video_lwh(str(a.video))
        traj=torch.as_tensor(torch.load(require_file(out/OUTPUTS['camera']),map_location='cpu'))
        K=(create_camera_sensor(width,height,a.f_mm)[2] if a.f_mm else estimate_K(width,height)).repeat(length,1,1)
        data={'length':torch.tensor(length),
              'bbx_xys':torch.load(require_file(out/OUTPUTS['track']),map_location='cpu')['bbx_xys'],
              'kp2d':torch.load(require_file(out/OUTPUTS['pose']),map_location='cpu'),
              'f_imgseq':torch.load(require_file(out/OUTPUTS['features']),map_location='cpu'),
              'K_fullimg':K,'cam_angvel':compute_cam_angvel(traj[:,:3,:3])}
        with initialize_config_module(version_base='1.3',config_module='hmr4d.configs'):
            cfg=compose(config_name='demo',overrides=[f'video_name={a.video.stem}'])
        model=hydra.utils.instantiate(cfg.model,_recursive_=False)
        model.load_pretrained_model(str(CHECKPOINTS/'gvhmr/gvhmr_siga24_release.ckpt'))
        model=model.eval().cuda()
        with torch.inference_mode(): result=detach_to_cpu(model.predict(data,static_cam=a.static_camera))
    temp=target.with_suffix('.partial')
    torch.save(result,temp);temp.replace(target)
    gc.collect()
    print(f'Saved {target}',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video',type=Path,default=ROOT/'data/video/reference_pilot.mp4')
    p.add_argument('--output',type=Path)
    p.add_argument('--stage',choices=('all',)+STAGES,default='all')
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--preproc-device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--preproc-precision',choices=['fp32','fp16'],default='fp32',help='FP16 saves VRAM; CUDA only, output caches remain float32')
    p.add_argument('--flip-test',action='store_true',help='Optional doubled ViTPose inference; off in low-memory profile')
    p.add_argument('--static-camera',action='store_true',help='Use only after confirming a fixed camera; default estimates motion')
    p.add_argument('--f-mm',type=int)
    a=p.parse_args();a.video=require_file(a.video).resolve()
    if a.batch_size<1: p.error('batch-size must be positive')
    if a.preproc_precision=='fp16' and a.preproc_device!='cuda': p.error('fp16 preprocessing requires CUDA')
    out=(a.output or ROOT/'outputs/gvhmr'/a.video.stem).resolve();out.mkdir(parents=True,exist_ok=True)
    manifest={'video':str(a.video),'sha256':hashlib.sha256(a.video.read_bytes()).hexdigest(),
              'static_camera':a.static_camera,'f_mm':a.f_mm,'flip_test':a.flip_test}
    mp=out/'input_manifest.json'
    if mp.exists() and json.loads(mp.read_text())!=manifest:
        raise RuntimeError('Input/config changed. Use a new output directory; cached results are not interchangeable.')
    mp.write_text(json.dumps(manifest,indent=2)+'\n')
    if a.stage in ('pose','features'):
        pp=out/'preprocess'/(a.stage+'_runtime.json')
        provenance={'device':a.preproc_device,'precision':a.preproc_precision,'batch_size':a.batch_size}
        if (out/OUTPUTS[a.stage]).exists() and pp.exists() and json.loads(pp.read_text())!=provenance:
            raise RuntimeError('Preprocessing config differs from cached output; use a separate output directory.')
        pp.parent.mkdir(parents=True,exist_ok=True)
        if not (out/OUTPUTS[a.stage]).exists():pp.write_text(json.dumps(provenance,indent=2)+'\n')
    if a.stage!='all': run_stage(a,out);return
    missing=[str(CHECKPOINTS/x) for stage in STAGES for x in WEIGHTS[stage] if not (CHECKPOINTS/x).is_file()]
    if missing:
        raise SystemExit('Missing model files (see docs/DOWNLOADS.md):\n'+'\n'.join(missing))
    for stage in STAGES:
        cmd=[sys.executable,str(Path(__file__).resolve()),'--video',str(a.video),'--output',str(out),
             '--stage',stage,'--batch-size',str(a.batch_size),'--threads',str(a.threads),'--preproc-device',a.preproc_device,'--preproc-precision',a.preproc_precision]
        if a.static_camera:cmd+=['--static-camera']
        if a.flip_test:cmd+=['--flip-test']
        if a.f_mm:cmd+=['--f-mm',str(a.f_mm)]
        subprocess.run(cmd,check=True,cwd=ROOT)

if __name__=='__main__':main()

