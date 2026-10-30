"""Single-dance SuperMimic fine-tuning with optional continuous supervision."""
import argparse
import dataclasses
import hashlib
import json
import math
import os
import signal
from pathlib import Path
import subprocess
import threading
import time

from common import ROOT, add_sources


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/g1_3050_nju_finetune.json')
    p.add_argument('--checkpoint',type=Path,default=ROOT/'data/downloads/nju/e13i_0291000/ckpt_e13i_0291000.pt')
    p.add_argument('--steps',type=int,default=600)
    p.add_argument('--run-name',default='reference_contact_e13i_ft1')
    p.add_argument('--keep-sampler',action='store_true',help='Only for continuing the same single-dance library')
    p.add_argument('--until-targets',action='store_true',help='Continuous learning until repeated full-clip targets pass; no short step/time budget')
    p.add_argument('--supervision',type=Path,default=ROOT/'configs/g1_3050_long_training.json')
    a=p.parse_args()
    if a.steps<80:p.error('Use at least 80 steps including replay warm-up')
    os.environ['G1_SUPERMIMIC_SOURCE']='nju';add_sources()
    import numpy as np
    import torch
    import supermimic
    from supermimic.config import TrainConfig
    from supermimic.training.loop import Trainer
    from train_3050 import update_config
    from fast_g1_fk import enable_layered_fk
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    enable_layered_fk()
    cfg=TrainConfig()
    # Keep all trained network/state normalization definitions. Clear foreign data paths below.
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    update_config(cfg,ck['config']);initial_iteration=int(ck['iteration']);del ck
    update_config(cfg,json.loads(a.config.read_text()))
    cfg.run_name=a.run_name;cfg.log_root=ROOT/'logs/train'
    cfg.data.npz_dir=ROOT/cfg.data.npz_dir
    cfg.data.library=None;cfg.data.manifest=None
    if cfg.env.fail_reset_list is not None:cfg.env.fail_reset_list=ROOT/cfg.env.fail_reset_list
    cfg.resume=a.checkpoint.resolve();cfg.resume_reset_sampler=not a.keep_sampler
    cfg.schedule.total_iterations=initial_iteration+a.steps
    supervision_settings=None
    if a.until_targets:
        supervision_settings=json.loads(a.supervision.read_text())
        if not a.keep_sampler:p.error('--until-targets requires --keep-sampler for this continuation')
        cfg.schedule.max_hours=None
        cfg.schedule.total_iterations=2**62
        cfg.schedule.checkpoint_interval=int(supervision_settings['recovery_checkpoint_interval_steps'])
        cfg.schedule.log_interval=100
    out=cfg.log_root/cfg.run_name
    if (out/'ckpt_latest.pt').exists():raise FileExistsError(f'Choose a fresh run name to preserve {out}')
    out.mkdir(parents=True,exist_ok=True)
    motion=cfg.data.npz_dir/(cfg.data.clips[0]+'.npz')
    if not motion.is_file():raise FileNotFoundError(motion)

    class LocalTrainer(Trainer):
        def load(self,path):
            # Avoid copying the old multi-library sampler and every checkpoint tensor to GPU.
            state=torch.load(path,map_location='cpu',weights_only=False)
            for name in ('wm','policy','critic'):getattr(self,name).load_state_dict(state[name],strict=True)
            self.wm_trainer.opt.load_state_dict(state['wm_opt'])
            self.policy_trainer.opt.load_state_dict(state['policy_opt'])
            self.policy_trainer.critic_opt.load_state_dict(state['critic_opt'])
            if self.cfg.resume_reset_sampler:self.command.reset_sampler_state()
            else:self.command.load_sampler_state(state)
            self.iteration=int(state['iteration']);self.num_updates=int(state['num_updates'])
            if a.until_targets:
                def same(left,right):
                    if isinstance(right,torch.Tensor):return isinstance(left,torch.Tensor) and torch.equal(left.detach().cpu(),right.detach().cpu())
                    if isinstance(right,dict):return isinstance(left,dict) and left.keys()==right.keys() and all(same(left[k],v) for k,v in right.items())
                    if isinstance(right,(tuple,list)):return isinstance(left,(tuple,list)) and len(left)==len(right) and all(same(x,y) for x,y in zip(left,right))
                    return left==right
                loaded={'iteration':self.iteration,'num_updates':self.num_updates,
                    'networks_exact':{name:same(getattr(self,name).state_dict(),state[name]) for name in ('wm','policy','critic')},
                    'optimizers_exact':{name:same(opt.state_dict(),state[name]) for name,opt in [('wm_opt',self.wm_trainer.opt),('policy_opt',self.policy_trainer.opt),('critic_opt',self.policy_trainer.critic_opt)]},
                    'sampler_exact':{name:same(getattr(self.command,name),state[name]) for name in ('bin_return','bin_memory','bin_visits')} if not self.cfg.resume_reset_sampler else {},
                    'replay_reinitialized':True}
                if not all(v for key in ('networks_exact','optimizers_exact','sampler_exact') for v in loaded[key].values()):
                    raise RuntimeError('Checkpoint continuation validation failed')
                from training_supervisor import atomic_json
                atomic_json(self.log_dir/'resume_validation.json',loaded)
            print(f'Loaded all models/optimizers at {self.iteration}; sampler_reset={self.cfg.resume_reset_sampler}',flush=True)

        def save(self,name='ckpt_latest.pt'):
            if not self.is_main:return
            # Long runs retain latest, previous, best and only two evaluated snapshots.
            if a.until_targets and name!='ckpt_latest.pt':return
            for model_name in ('wm','policy','critic'):
                if not all(bool(torch.isfinite(v).all()) for v in getattr(self,model_name).state_dict().values() if v.is_floating_point()):
                    raise FloatingPointError(f'{model_name}: nonfinite checkpoint parameters; previous checkpoint preserved')
            from training_supervisor import link_checkpoint
            destination=self.log_dir/name;temporary=self.log_dir/(name+'.tmp')
            super().save(temporary.name)
            with temporary.open('rb') as f:os.fsync(f.fileno())
            if a.until_targets and destination.exists():link_checkpoint(destination,self.log_dir/'ckpt_previous.pt')
            os.replace(temporary,destination)

    report={'ok':False,'checkpoint':str(cfg.resume),'source_module':str(supermimic.__file__),
        'initial_iteration':initial_iteration,'requested_additional_steps':None if a.until_targets else a.steps,'config':dataclasses.asdict(cfg),
        'motion_sha256':hashlib.sha256(motion.read_bytes()).hexdigest(),'scope':'Continuous fixed-configuration training with low-frequency full-rollout checks' if a.until_targets else 'Bounded single-dance training'}
    resources={'samples':0,'memory_peak_mib':0.,'temperature_max_c':0.};stop=threading.Event()
    def monitor():
        while not stop.is_set():
            try:
                s=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,temperature.gpu,utilization.gpu','--format=csv,noheader,nounits'],text=True)
                sample=[float(v) for v in s.split(',')]
                resources.update(current_memory_mib=sample[0],current_temperature_c=sample[1],current_utilization_percent=sample[2])
                resources['samples']+=1
                resources['memory_peak_mib']=max(resources['memory_peak_mib'],sample[0])
                resources['temperature_max_c']=max(resources['temperature_max_c'],sample[1])
            except (OSError,ValueError,subprocess.CalledProcessError):pass
            stop.wait(5.)
    t=time.perf_counter();thread=threading.Thread(target=monitor,daemon=True);thread.start();trainer=None;supervisor=None
    metrics_file=(out/'updates.jsonl').open('w')
    try:
        torch.cuda.reset_peak_memory_stats();trainer=LocalTrainer(cfg)
        trainer.resource_snapshot=lambda:dict(resources,
            torch_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            torch_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
        for model_name in ('wm','policy','critic'):
            if not all(bool(torch.isfinite(v).all()) for v in getattr(trainer,model_name).state_dict().values() if v.is_floating_point()):
                raise FloatingPointError(f'{model_name}: nonfinite loaded parameters')
        # Bound logging diagnostics as well as the training batch; leave training math unchanged.
        online=trainer.wm_trainer.online_error
        trainer.wm_trainer.online_error=lambda buffer,n=1024,k=8:online(buffer,n=min(n,64),k=k)
        for name,obj in [('wm',trainer.wm_trainer),('policy',trainer.policy_trainer)]:
            original=obj.update
            def checked(*args,_fn=original,_name=name,**kwargs):
                nonlocal metrics_file
                result=_fn(*args,**kwargs)
                if not all(math.isfinite(float(v)) for v in result.values()):raise FloatingPointError(f'{_name}: nonfinite optimizer metrics')
                if metrics_file.tell()>32*2**20:
                    metrics_file.close()
                    previous=out/'updates.previous.jsonl';previous.unlink(missing_ok=True)
                    os.replace(out/'updates.jsonl',previous)
                    metrics_file=(out/'updates.jsonl').open('w')
                metrics_file.write(json.dumps({'iteration':trainer.iteration,'component':_name,**result})+'\n');metrics_file.flush()
                return result
            obj.update=checked
        start_updates=trainer.num_updates
        stop_reason='step_or_time_budget'
        if a.until_targets:
            from training_supervisor import TrainingSupervisor,TrainingStopped
            supervisor=TrainingSupervisor(trainer,supervision_settings,motion,initial_iteration)
            original_act=trainer.act
            def supervised_act(obs):
                supervisor.before_step()
                return original_act(obs)
            trainer.act=supervised_act
            def request_stop(signum,frame):supervisor.stop_requested=True
            signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
            try:trainer.run()
            except TrainingStopped as exc:stop_reason=str(exc)
        else:trainer.run()
        finite=all(bool(torch.isfinite(v).all()) for v in trainer.policy.state_dict().values() if v.is_floating_point())
        if not finite:raise FloatingPointError('Saved policy has nonfinite values')
        report.update(ok=True,final_iteration=trainer.iteration,additional_steps=trainer.iteration-initial_iteration,
            stop_reason=stop_reason,operational_targets_met=stop_reason=='targets_met_video_review_pending',
            optimizer_updates=trainer.num_updates-start_updates,saved_checkpoint=str(out/'ckpt_latest.pt'),
            policy_finite=True,torch_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            torch_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
    except Exception as exc:
        report['error']=repr(exc)
        if supervisor is not None:supervisor.publish('failed_runtime',error=repr(exc))
        raise
    finally:
        metrics_file.close();stop.set();thread.join(timeout=2.)
        if trainer is not None:
            if trainer.writer is not None:trainer.writer.close()
            trainer.env.close()
        report['wall_seconds']=time.perf_counter()-t
        if resources['samples']:
            report.update(gpu_total_memory_peak_mib=resources['memory_peak_mib'],gpu_temperature_max_c=resources['temperature_max_c'])
        (out/'run_report.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k!='config'},indent=2),flush=True)


if __name__=='__main__':main()
