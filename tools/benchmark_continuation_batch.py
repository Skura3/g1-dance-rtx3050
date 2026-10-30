"""Profile batch sizes from one saved checkpoint; never publish probe weights."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time

from common import ROOT,add_sources


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batches',type=int,nargs='+',default=[32,64,128,256])
    a=p.parse_args()
    os.environ['G1_SUPERMIMIC_SOURCE']='nju';add_sources()
    import torch
    from supermimic.config import TrainConfig
    from supermimic.training.loop import Trainer
    from train_3050 import update_config
    from fast_g1_fk import enable_layered_fk
    torch.set_num_threads(1);torch.set_num_interop_threads(1);enable_layered_fk()
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    cfg=TrainConfig();update_config(cfg,ck['config'])
    cfg.log_root=a.output.parent;cfg.run_name='disposable_batch_profile';cfg.resume=a.checkpoint.resolve()
    cfg.resume_reset_sampler=False
    class ProbeTrainer(Trainer):
        def load(self,path):
            for key in ('wm','policy','critic'):getattr(self,key).load_state_dict(ck[key],strict=True)
            self.wm_trainer.opt.load_state_dict(ck['wm_opt'])
            self.policy_trainer.opt.load_state_dict(ck['policy_opt'])
            self.policy_trainer.critic_opt.load_state_dict(ck['critic_opt'])
            self.command.load_sampler_state(ck)
            self.iteration=int(ck['iteration']);self.num_updates=int(ck['num_updates'])
        def save(self,name='ckpt_latest.pt'):
            raise RuntimeError('Disposable profiling must not save model weights')
    report={'checkpoint':str(a.checkpoint.resolve()),'checkpoint_iteration':int(ck['iteration']),
        'checkpoint_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        'scope':'Timing/memory/finite checks only; these few updates do not assess dance quality and probe weights are discarded.',
        'num_envs':cfg.env.num_envs,'horizon':cfg.policy_trainer.horizon,'results':[]}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    trainer=None
    try:
        trainer=ProbeTrainer(cfg);obs,_=trainer.env.reset();times=[]
        for i in range(96):
            t=time.perf_counter();obs,*_=trainer.env.step(trainer.act(obs));torch.cuda.synchronize()
            if i>=64:times.append(time.perf_counter()-t)
        report['environment_step_median_seconds']=statistics.median(times)
        # Keep replay and the same random starts across size tests, but restore all learned state.
        for batch in a.batches:
            row={'batch':batch,'ok':False};stop=threading.Event();memory=[]
            def monitor():
                while not stop.is_set():
                    try:
                        s=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,temperature.gpu,utilization.gpu','--format=csv,noheader,nounits'],text=True)
                        memory.append([float(x) for x in s.strip().split(',')])
                    except (ValueError,OSError,subprocess.SubprocessError):pass
                    stop.wait(.1)
            thread=None
            try:
                for key in ('wm','policy','critic'):getattr(trainer,key).requires_grad_(True)
                trainer.load(a.checkpoint);trainer.apply_lr(trainer.iteration)
                trainer.wm_trainer.cfg.batch_size=batch;trainer.policy_trainer.cfg.batch_size=batch
                torch.manual_seed(101);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
                thread=threading.Thread(target=monitor,daemon=True);thread.start();updates=[]
                for i in range(5):
                    t=time.perf_counter();wm=trainer.wm_trainer.update(trainer.buffer);torch.cuda.synchronize();wm_s=time.perf_counter()-t
                    t=time.perf_counter();pi=trainer.policy_trainer.update(trainer.buffer);torch.cuda.synchronize();pi_s=time.perf_counter()-t
                    if not all(math.isfinite(float(x)) for x in [*wm.values(),*pi.values()]):raise FloatingPointError('Nonfinite probe update')
                    updates.append({'wm_s':wm_s,'policy_critic_s':pi_s})
                steady=updates[2:];update_s=statistics.median(x['wm_s']+x['policy_critic_s'] for x in steady)
                expected=report['environment_step_median_seconds']+cfg.schedule.updates_per_env_step*update_s
                row.update(ok=True,updates=updates,steady_update_seconds=update_s,
                    expected_iteration_seconds=expected,imagined_trajectories_per_wall_second=batch*cfg.schedule.updates_per_env_step/expected,
                    torch_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                    torch_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                    free_cuda_mib=torch.cuda.mem_get_info()[0]/2**20)
            except torch.cuda.OutOfMemoryError as exc:
                row['error']='CUDA OOM: '+str(exc)[:400]
            finally:
                stop.set()
                if thread:thread.join(timeout=2)
                if memory:row.update(total_gpu_peak_mib=max(x[0] for x in memory),temperature_peak_c=max(x[1] for x in memory))
                report['results'].append(row)
                a.output.write_text(json.dumps(report,indent=2)+chr(10))
                print(json.dumps(row),flush=True)
            if not row['ok'] or row.get('total_gpu_peak_mib',0)>3200:break
    finally:
        if trainer is not None:
            if trainer.writer:trainer.writer.close()
            trainer.env.close()
    print('Disposable benchmark complete; checkpoint files were not modified.',flush=True)


if __name__=='__main__':main()
