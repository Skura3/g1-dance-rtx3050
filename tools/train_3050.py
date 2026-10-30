"""Small-memory SuperMimic launcher; use a separate compatible training environment."""
import argparse
import dataclasses
import json
import sys
from pathlib import Path
from common import ROOT, add_sources

def update_config(obj,values):
    allowed={f.name for f in dataclasses.fields(obj)}
    for key,value in values.items():
        if key not in allowed:raise ValueError(f'Unknown configuration field: {type(obj).__name__}.{key}')
        old=getattr(obj,key)
        if dataclasses.is_dataclass(old):update_config(old,value)
        else:setattr(obj,key,tuple(value) if isinstance(old,tuple) and value is not None else value)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/g1_3050.json')
    p.add_argument('--check-config',action='store_true',help='Parse JSON only; does not validate runtime compatibility')
    p.add_argument('--smoke',action='store_true',help='80 iterations including model and actor updates')
    p.add_argument('--resume',type=Path)
    p.add_argument('--threads',type=int,default=2,help='PyTorch CPU worker threads; benchmark on this computer')
    p.add_argument('--precision',choices=['highest','high'],default='highest')
    p.add_argument('--fast-fk',action='store_true',help='Use validated depth-batched G1 forward kinematics')
    a=p.parse_args();values=json.loads(a.config.read_text())
    if a.check_config:print(json.dumps(values,indent=2));return
    if sys.version_info<(3,12):
        raise SystemExit('Training upstream requires Python 3.12; the motion environment uses 3.10. See docs/TRAINING.md')
    add_sources()
    from supermimic.config import TrainConfig
    from supermimic.training.loop import Trainer
    import torch
    if a.threads<1:p.error('--threads must be positive')
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision(a.precision)
    if a.fast_fk:
        from fast_g1_fk import enable_layered_fk
        enable_layered_fk()
    print('Runtime:',json.dumps({'cpu_threads':a.threads,'matmul_precision':a.precision,'kinematics':'layered' if a.fast_fk else 'original'}))
    cfg=TrainConfig();update_config(cfg,values)
    cfg.data.npz_dir=ROOT/cfg.data.npz_dir;cfg.log_root=ROOT/'logs/train'
    if not any(cfg.data.npz_dir.glob('*.npz')):raise SystemExit(f'No motion data in {cfg.data.npz_dir}')
    if not torch.cuda.is_available():raise SystemExit('CUDA unavailable')
    if a.smoke:
        cfg.schedule.total_iterations=80;cfg.schedule.max_hours=0.1;cfg.schedule.log_interval=10
        cfg.schedule.checkpoint_interval=80;cfg.run_name+='_smoke'
    if a.resume:cfg.resume=a.resume.resolve()
    log_dir=cfg.log_root/cfg.run_name
    log_dir.mkdir(parents=True,exist_ok=True)
    (log_dir/'runtime.json').write_text(json.dumps({'cpu_threads':a.threads,'matmul_precision':a.precision,'kinematics':'layered' if a.fast_fk else 'original','config_file':str(a.config.resolve())},indent=2))
    torch.cuda.reset_peak_memory_stats();Trainer(cfg).run()
    print('Peak torch allocated MiB:',torch.cuda.max_memory_allocated()/1024**2)
    print('Peak torch reserved MiB:',torch.cuda.max_memory_reserved()/1024**2)
    print('Monitor nvidia-smi too: Warp/graphics memory is outside PyTorch accounting.')

if __name__=='__main__':main()
