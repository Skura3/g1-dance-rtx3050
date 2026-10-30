"""Validate actual training imports and a compiled Warp GPU kernel."""
import json
import sys
import traceback
from datetime import datetime, timezone
from importlib.metadata import version
from common import ROOT,add_sources

def main():
    add_sources();report={'python':sys.version,'ready':False,
                         'checked_at':datetime.now(timezone.utc).isoformat(),
                         'scope':'Imports, training configuration, Torch CUDA arithmetic and a compiled Warp kernel'}
    try:
        import torch
        import warp as wp
        import mujoco
        import mujoco_warp
        import mjlab
        from supermimic.config import TrainConfig
        from train_3050 import update_config
        cfg=TrainConfig();update_config(cfg,json.loads((ROOT/'configs/g1_3050.json').read_text()))
        report.update(torch=torch.__version__,torch_cuda=torch.version.cuda,warp=wp.__version__,mujoco=mujoco.__version__)
        report['warp_distribution']=version('warp-lang')
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        x=torch.arange(16,device='cuda',dtype=torch.float32);value=float((x*x).sum())
        if value!=1240:raise RuntimeError('CUDA arithmetic mismatch')
        report['torch_cuda_arithmetic_ok']=True
        wp.init()
        if not wp.is_cuda_available():raise RuntimeError('Warp CUDA unavailable; check CUDA 12 Warp wheel in docs/DOWNLOADS.md')
        # Compiling a real kernel catches driver/toolchain mismatches that import alone misses.
        from training_kernel import increment
        arr=wp.zeros(16,dtype=wp.float32,device='cuda:0')
        wp.launch(increment,dim=16,inputs=[arr],device='cuda:0');wp.synchronize()
        if not (arr.numpy()==1).all():raise RuntimeError('Warp kernel mismatch')
        report['warp_cuda_kernel_ok']=True
        report.update(ready=True,gpu=torch.cuda.get_device_name(),config_applied=True,
            note='Imports, torch CUDA and a Warp kernel only; no dance training or full simulation verified.')
    except Exception as e:
        report.update(error=f'{type(e).__name__}: {e}',traceback=traceback.format_exc())
    (ROOT/'outputs/training_readiness.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if report['ready'] else 1)

if __name__=='__main__':main()

