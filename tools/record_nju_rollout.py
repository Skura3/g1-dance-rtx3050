"""Record released-policy physical rollouts; retain terminal states, never hide resets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time

from common import ROOT, add_sources


def orientation_quality(qpos, qref, dt):
    """Sign-invariant geodesic and wrapped heading errors, with wxyz quaternions."""
    import numpy as np
    qa=np.asarray(qpos,dtype=float)[:,3:7].copy()
    qr=np.asarray(qref,dtype=float)[:,3:7].copy()
    qa/=np.linalg.norm(qa,axis=-1,keepdims=True)
    qr/=np.linalg.norm(qr,axis=-1,keepdims=True)
    geodesic=2*np.arccos(np.clip(np.abs((qa*qr).sum(-1)),0,1))*180/np.pi
    def yaw(q):
        w,x,y,z=q.T
        return np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
    difference=yaw(qa)-yaw(qr)
    heading=np.arctan2(np.sin(difference),np.cos(difference))*180/np.pi
    return {'orientation_error_max_deg':float(geodesic.max()),
            'yaw_error_mean_abs_deg':float(np.abs(heading).mean()),
            'yaw_error_max_abs_deg':float(np.abs(heading).max()),
            'yaw_error_last_deg':float(heading[-1]),
            'yaw_error_peak_time_s':float(np.argmax(np.abs(heading))*dt)}


def local_joint_quality(qpos, qref, joint_names, dt):
    """Catch upper-limb and localized errors hidden by a whole-clip mean."""
    import numpy as np
    actual=np.asarray(qpos,dtype=float);reference=np.asarray(qref,dtype=float)
    if actual.ndim!=2 or actual.shape!=reference.shape or actual.shape[1]!=7+len(joint_names):
        raise ValueError('Mismatched joint trajectory or names')
    if not np.isfinite(actual).all() or not np.isfinite(reference).all() or not np.isfinite(dt) or dt<=0:
        raise ValueError('Invalid joint trajectory or sample interval')
    arms=[i for i,name in enumerate(joint_names) if any(part in name for part in ('shoulder','elbow','wrist'))]
    window=int(round(2.0/dt))
    if not arms or window<1 or len(actual)<window:
        raise ValueError('Missing arm joints or a full two-second window')
    squared=(actual[:,7:]-reference[:,7:])**2
    frame_mse=squared.mean(axis=1)
    cumulative=np.concatenate(([0.0],np.cumsum(frame_mse)))
    windows=(cumulative[window:]-cumulative[:-window])/window
    peak=int(np.argmax(windows))
    return {'arms_joint_rmse_rad':float(np.sqrt(squared[:,arms].mean())),
        'worst_2s_joint_rmse_rad':float(np.sqrt(max(0.0,windows[peak]))),
        'worst_2s_window_start_s':peak*dt,'joint_quality_window_samples':window}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--motions', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, default=ROOT/'outputs/nju_validation/rollouts')
    p.add_argument('--no-render', action='store_true', help='Record physics and metrics without video rendering')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--reset-noise', action='store_true', help='Use upstream initial-state perturbations for robustness checks')
    a = p.parse_args()
    os.environ['G1_SUPERMIMIC_SOURCE'] = 'nju'
    os.environ.setdefault('MUJOCO_GL', 'egl')
    add_sources()
    import numpy as np
    import torch
    import mujoco
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.policy.nets import Policy, PolicyCfg
    from mjlab.utils.lab_api.math import quat_apply_inverse

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    checkpoint_hash=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    checkpoint_iteration=int(ck.get('iteration',-1))
    cfg = ck['config']; ec = cfg['env']
    policy = Policy(PolicyCfg(**{k:tuple(v) if isinstance(v,list) else v for k,v in cfg['policy'].items()}))
    policy.load_state_dict(ck['policy'], strict=True)
    del ck
    policy.to('cuda:0').eval()
    args = SuperMimicEnvCfgArgs(npz_files=[str(x.resolve()) for x in a.motions],
        num_envs=len(a.motions), future_frames=ec['future_frames'], episode_length_s=1e5,
        action_noise_std=0., obs_noise=False, reset_noise=a.reset_noise, domain_randomization=False,
        sampling_mode='start', record_transitions=False, seed=a.seed,
        anchor_obs=ec.get('anchor_obs',False), lock_waist=ec.get('lock_waist',False),
        lock_waist_refs=ec.get('lock_waist_refs','feasible'), locked_action_weight=0.,
        waist_coupling=ec.get('waist_coupling',False), waist_coupling_k=ec.get('waist_coupling_k',1.4),
        waist_coupling_c=ec.get('waist_coupling_c',.95), waist_roll_range=ec.get('waist_roll_range'))
    if args.lock_waist:
        raise ValueError('This recorder requires the released 29-joint unlocked model')
    a.output.mkdir(parents=True,exist_ok=True)
    samples=[];stop=threading.Event()
    def monitor():
        while not stop.is_set():
            try:
                raw=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,temperature.gpu','--format=csv,noheader,nounits'],text=True)
                samples.append([float(x) for x in raw.strip().split(',')])
            except (OSError,ValueError,subprocess.CalledProcessError):
                pass
            stop.wait(.5)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    env=None;t0=time.perf_counter()
    try:
        env=make_env(args,device='cuda:0')
        env.cfg.auto_reset=False
        cmd=env.command_manager.get_term('motion');lib=cmd.library
        assert [x.name for x in lib.clips]==[x.stem for x in a.motions], 'Clip order differs'
        model=env.sim.mj_model
        if not a.no_render:mujoco.mj_saveModel(model,str(a.output/'g1_scene.mjb'),None)
        cmd.forced_frame[:]=lib.clip_start
        obs,_=env.reset()
        origins=env.scene.env_origins.detach().cpu().numpy()
        records=[{'qpos':[],'qref':[],'frame':[],'metrics':[],'action':[]} for _ in a.motions]
        active=np.ones(len(records),dtype=bool);final=[None]*len(records)
        def capture():
            q=env.sim.data.qpos.detach().cpu().numpy().copy();q[:,:3]-=origins
            ref=torch.cat([cmd.root_pos_w,cmd.root_quat_w,cmd.joint_pos],-1).cpu().numpy().copy();ref[:,:3]-=origins
            grav=cmd.robot.data.gravity_vec_w
            ga=quat_apply_inverse(cmd.robot_root_quat_w,grav);gr=quat_apply_inverse(cmd.root_quat_w,grav)
            tilt=torch.rad2deg(torch.acos(((ga*gr).sum(-1)/(ga.norm(dim=-1)*gr.norm(dim=-1))).clamp(-1,1))).cpu().numpy()
            body=(cmd.robot_body_pos_w-cmd.body_pos_relative_w).norm(dim=-1).mean(-1).cpu().numpy()
            absolute_body=(cmd.robot_body_pos_w-cmd.body_pos_w).norm(dim=-1).mean(-1).cpu().numpy()
            frames=(cmd.frame-lib.clip_start).cpu().numpy()
            for i in np.flatnonzero(active):
                assert np.isfinite(q[i]).all() and np.isfinite(ref[i]).all()
                records[i]['qpos'].append(q[i]);records[i]['qref'].append(ref[i]);records[i]['frame'].append(int(frames[i]))
                records[i]['metrics'].append([float(np.linalg.norm(q[i,:3]-ref[i,:3])),float(abs(q[i,2]-ref[i,2])),float(tilt[i]),float(np.sqrt(np.mean((q[i,7:]-ref[i,7:])**2))),float(body[i]),float(absolute_body[i])])
        with torch.no_grad():
            capture()
            max_steps=max(x.length for x in lib.clips)+2
            ret=np.zeros(len(records))
            for step in range(1,max_steps+1):
                act=policy(obs['policy'],obs['goal'])
                aa=act.cpu().numpy()
                for i in np.flatnonzero(active):records[i]['action'].append(aa[i].copy())
                obs,rew,term,tout,_=env.step(act)
                ret+=rew.cpu().numpy()*active
                capture()
                done=(term|tout).cpu().numpy() & active
                for i in np.flatnonzero(done):
                    final[i]={'steps':step,'failed':bool(term[i]),'timeout':bool(tout[i]),'norm_return':float(ret[i]/(.1*step))}
                active[done]=False
                if step%100==0 or done.any():print(f'physical step {step}/{max_steps}, active {int(active.sum())}',flush=True)
                if not active.any():break
                if done.any():
                    # Only finished environments reset; their trajectories have ended and receive no more samples.
                    obs,_=env.reset(env_ids=torch.as_tensor(np.flatnonzero(done),device='cuda:0'))
            if active.any():raise RuntimeError('A clip did not terminate within its frame count')
        dt=float(env.step_dt);joint_names=list(cmd.robot.joint_names)
        report={'checkpoint':str(a.checkpoint.resolve()),'checkpoint_sha256':checkpoint_hash,'checkpoint_iteration':checkpoint_iteration,
            'seed':a.seed,'reset_noise':a.reset_noise,
            'control_dt_s':dt,'scope':'Learned-policy free-base physics; every recorded clip runs until its first terminal state, with no mid-clip resets',
            'external_forces':False,'gravity':list(model.opt.gravity),'resets_during_each_recorded_clip':0,
            'metric_timing':'Post-step actual state versus refreshed reference; may differ slightly from upstream reward/tilt measurement timing',
            'clips':[]}
        for i,(path,rec) in enumerate(zip(a.motions,records)):
            arr={k:np.asarray(v) for k,v in rec.items()};metrics=arr['metrics']
            dst=a.output/(path.stem+'_states.npz')
            np.savez_compressed(dst,**arr,fps=np.array([1/dt]),joint_names=np.array(joint_names),
                metric_names=np.array(['root_position_error_m','root_height_error_m','tilt_error_deg','joint_rmse_rad','aligned_body_error_m','absolute_body_error_m']))
            row=dict(final[i],name=path.stem,states=str(dst),simulated_seconds=final[i]['steps']*dt,
                root_error_mean_m=float(metrics[:,0].mean()),root_error_max_m=float(metrics[:,0].max()),
                height_error_max_m=float(metrics[:,1].max()),tilt_mean_deg=float(metrics[:,2].mean()),tilt_max_deg=float(metrics[:,2].max()),
                tilt_peak_time_s=float(np.argmax(metrics[:,2])*dt),tilt_above_20deg_frames=int((metrics[:,2]>20).sum()),
                joint_rmse_rad=float(np.sqrt(np.mean((arr['qpos'][:,7:]-arr['qref'][:,7:])**2))),
                aligned_body_error_mean_m=float(metrics[:,4].mean()),absolute_body_error_mean_m=float(metrics[:,5].mean()))
            row.update(orientation_quality(arr['qpos'],arr['qref'],dt))
            if final[i]['steps']*dt>=2.0:
                row.update(local_joint_quality(arr['qpos'],arr['qref'],joint_names,dt))
            report['clips'].append(row)
        report['reference_timing_note']='Upstream reset initializes the robot at frame 0 and advances command to frame 1 before first observation. Recorded metrics use that refreshed command; no retiming applied.'
        report['acceptance_note']='No termination is not a fidelity pass: upstream projected-gravity/height checks do not constrain absolute yaw or horizontal drift.'
        report['physics_wall_seconds']=time.perf_counter()-t0
        env.close();env=None
        stop.set();thread.join(timeout=2.)
        if samples:
            report['gpu_total_memory_peak_mib']=max(x[0] for x in samples);report['gpu_temperature_max_c']=max(x[1] for x in samples)
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if a.no_render:
            print(json.dumps(report,indent=2),flush=True)
            return
        from validate_g1 import render_frames
        for path,rec in zip(a.motions,records):
            poses=np.asarray(rec['qpos'])[::2]
            output=a.output/(path.stem+'_policy.mp4')
            print('Rendering '+str(output),flush=True)
            render_frames(model,poses,output,1/dt/2,'LEARNED POLICY PHYSICS | '+a.checkpoint.parent.name+' | No mid-clip reset')
        print(json.dumps(report,indent=2),flush=True)
    finally:
        stop.set();thread.join(timeout=2.)
        if env is not None:env.close()


if __name__=='__main__':main()
