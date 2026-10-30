"""Disposable SuperMimic pipeline benchmark: collect data and time real updates."""
import argparse,json,time,subprocess,threading,traceback
from pathlib import Path
from common import ROOT,add_sources

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--envs',type=int,default=128);p.add_argument('--batch',type=int,default=64);p.add_argument('--threads',type=int,default=2);p.add_argument('--precision',choices=['highest','high'],default='highest');p.add_argument('--fast-fk',action='store_true');p.add_argument('--updates',type=int,default=3);p.add_argument('--full-size',action='store_true',help='Use upstream full network dimensions, not local reduced networks');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if min(a.envs,a.batch,a.threads)<1:p.error('envs, batch and threads must be positive')
    if a.updates<3:p.error('--updates must be at least 3')
    add_sources();import numpy as np,torch
    from supermimic.config import TrainConfig
    from supermimic.training.loop import Trainer
    from train_3050 import update_config
    if a.fast_fk:
        from fast_g1_fk import enable_layered_fk
        enable_layered_fk()
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);torch.set_float32_matmul_precision(a.precision)
    cfg=TrainConfig();update_config(cfg,json.loads((ROOT/'configs/g1_3050.json').read_text()))
    if a.full_size:
        defaults=TrainConfig();cfg.world_model=defaults.world_model;cfg.policy=defaults.policy;cfg.critic=defaults.critic
    cfg.data.npz_dir=ROOT/'data/preview';cfg.data.clips=['reference_full_contact'];cfg.log_root=ROOT/'outputs/performance/disposable';cfg.run_name=f'env{a.envs}_batch{a.batch}_t{a.threads}_{a.precision}_'+('layered' if a.fast_fk else 'original')
    cfg.run_name+='_full' if a.full_size else '_small'
    cfg.env.num_envs=a.envs;cfg.wm_trainer.batch_size=a.batch;cfg.policy_trainer.batch_size=a.batch
    cfg.env.sampling_mode='uniform';cfg.wm_trainer.batches_per_update=2
    samples=[];stop=threading.Event()
    def monitor():
        while not stop.is_set():
            raw=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,temperature.gpu,utilization.gpu','--format=csv,noheader,nounits'],text=True)
            samples.append([float(x.strip()) for x in raw.split(',')]);stop.wait(.25)
    import supermimic
    thread=threading.Thread(target=monitor,daemon=True);thread.start();report={'ok':False,'source_module':str(supermimic.__file__),'num_envs':a.envs,'batch':a.batch,'cpu_threads':a.threads,'matmul_precision':a.precision,'kinematics':'layered' if a.fast_fk else 'original','scope':f'Disposable performance run, {a.updates} WM+policy+critic updates, no checkpoint, no convergence claim'}
    trainer=None
    report['network_dimensions']={'world_model':list(cfg.world_model.hidden),'policy_encoder':list(cfg.policy.encoder_hidden),'policy_decoder':list(cfg.policy.decoder_hidden)}
    try:
        torch.cuda.reset_peak_memory_stats();t=time.perf_counter();trainer=Trainer(cfg);obs,_=trainer.env.reset();torch.cuda.synchronize();report['initialize_s']=time.perf_counter()-t
        for _ in range(12):obs,*_=trainer.env.step(trainer.act(obs))
        torch.cuda.synchronize();times=[]
        for _ in range(32):
            t=time.perf_counter();obs,*_=trainer.env.step(trainer.act(obs));torch.cuda.synchronize();times.append(time.perf_counter()-t)
        report['environment_step_median_s']=float(np.median(times));report['aggregate_environment_steps_per_s']=a.envs/float(np.median(times));report['sampleable_windows']=int(trainer.buffer.window_mask(8).sum())
        updates=[]
        for i in range(a.updates):
            t=time.perf_counter();wm=trainer.wm_trainer.update(trainer.buffer);torch.cuda.synchronize();wm_s=time.perf_counter()-t
            t=time.perf_counter();pi=trainer.policy_trainer.update(trainer.buffer);torch.cuda.synchronize();pi_s=time.perf_counter()-t
            if not all(np.isfinite(float(v)) for v in list(wm.values())+list(pi.values())):raise RuntimeError('Nonfinite optimizer metrics')
            updates.append({'world_model_s':wm_s,'policy_critic_s':pi_s,'world_model_loss':wm.get('loss'),'policy_metrics':pi});print('update',i,wm_s,pi_s,flush=True)
        report.update(ok=True,updates=updates,torch_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,torch_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,gpu_free_mib=torch.cuda.mem_get_info()[0]/2**20,checkpoint_saved=False)
        report['timing_warmup_updates_excluded']=2 if a.updates>=5 else 1
        steady=updates[report['timing_warmup_updates_excluded']:]
        report['estimated_iteration_s_at_utd_0_4']=report['environment_step_median_s']+.4*float(np.median([r['world_model_s']+r['policy_critic_s'] for r in steady]))
    except Exception as e:report.update(error=str(e),traceback=traceback.format_exc());print(report['traceback'],flush=True)
    finally:
        if trainer is not None and trainer.writer is not None:trainer.writer.close()
        stop.set();thread.join(timeout=2.)
    arr=np.asarray(samples);report.update(gpu_total_memory_used_peak_mib=float(arr[:,0].max()),gpu_temperature_max_c=float(arr[:,1].max()),gpu_utilization_peak_percent=float(arr[:,2].max()))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));raise SystemExit(0 if report['ok'] else 1)

if __name__=='__main__':main()
