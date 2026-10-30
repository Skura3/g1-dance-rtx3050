"""Interactive reference versus live best-policy physics; no video recording."""
import argparse
import hashlib
import os
from pathlib import Path
import time
from common import ROOT, add_sources


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--expected-sha256',required=True)
    parser.add_argument('--motion',type=Path,default=ROOT/'data/preview/reference_full_contact.npz')
    parser.add_argument('--status',type=Path,required=True)
    args=parser.parse_args()
    os.environ['G1_SUPERMIMIC_SOURCE']='nju';os.environ['MUJOCO_GL']='glfw';add_sources()
    import mujoco
    import numpy as np
    import torch
    from mjlab.viewer import NativeMujocoViewer
    from mjlab.viewer.viewer_config import ViewerConfig
    from supermimic.envs import SuperMimicEnvCfgArgs,make_env
    from supermimic.policy.nets import Policy,PolicyCfg
    from training_supervisor import atomic_json
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    digest=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if digest!=args.expected_sha256:raise ValueError('Pinned best checkpoint does not match expected hash')
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    cfg=checkpoint['config'];iteration=int(checkpoint['iteration']);ec=cfg['env']
    policy=Policy(PolicyCfg(**{k:tuple(v) if isinstance(v,list) else v for k,v in cfg['policy'].items()}))
    policy.load_state_dict(checkpoint['policy'],strict=True);policy.to('cuda:0').eval();del checkpoint
    def act(obs):
        with torch.no_grad():return policy(obs['policy'],obs['goal'])
    if ec.get('lock_waist',False):raise ValueError('Expected unlocked 29-joint checkpoint')
    env=make_env(SuperMimicEnvCfgArgs(npz_files=[str(args.motion.resolve())],num_envs=1,
        future_frames=ec['future_frames'],episode_length_s=1e5,action_noise_std=0.,
        obs_noise=False,reset_noise=False,domain_randomization=False,sampling_mode='start',
        record_transitions=False,seed=0,anchor_obs=ec.get('anchor_obs',False),
        lock_waist=False,locked_action_weight=0.,debug_vis=False,
        waist_coupling=ec.get('waist_coupling',False),waist_coupling_k=ec.get('waist_coupling_k',1.4),
        waist_coupling_c=ec.get('waist_coupling_c',.95),waist_roll_range=ec.get('waist_roll_range')),device='cuda:0')
    env.cfg.auto_reset=False
    cmd=env.command_manager.get_term('motion');cmd.forced_frame[:]=cmd.library.clip_start
    obs,_=env.reset();dt=float(env.step_dt)
    q0=cmd.root_quat_w[0].cpu().numpy();w,x,y,z=q0
    heading=float(np.degrees(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))))
    azimuth=heading+155.;rad=np.radians(azimuth)
    offset=.95*np.array([np.sin(rad),-np.cos(rad),0.])
    env.cfg.viewer.origin_type=ViewerConfig.OriginType.WORLD
    env.cfg.viewer.distance=4.6;env.cfg.viewer.azimuth=azimuth;env.cfg.viewer.elevation=-12.
    env.cfg.viewer.lookat=(0.,0.,.85);env.cfg.viewer.fovy=45.

    class ComparisonViewer(NativeMujocoViewer):
        def __init__(self):
            super().__init__(env,act,frame_rate=30.,enable_perturbations=False)
            self.obs=obs;self.clip_step=0;self.loops=0;self.pending_reset=None
            self.clip_failed=False;self.last_publish=0.;self.reference_data=None

        def setup(self):
            super().setup()
            self.reference_data=mujoco.MjData(self.mjm)
            self.reference_option=mujoco.MjvOption();self.reference_option.geomgroup[3:]=0
            self.reference_perturb=mujoco.MjvPerturb()
            self.viewer.opt.geomgroup[3:]=0;self.viewer.opt.frame=mujoco.mjtFrame.mjFRAME_NONE
            self.publish('running')
            print(f'LIVE COMPARISON READY | checkpoint {iteration} | LEFT reference, RIGHT live policy',flush=True)
            print('Space: pause/resume | Enter: restart clip | +/-: speed | Mouse: rotate/zoom | Close window: exit preview only',flush=True)

        def publish(self,state):
            atomic_json(args.status,{'status':state,'pid':os.getpid(),'checkpoint_iteration':iteration,
                'checkpoint_sha256':digest,'mode':'live_policy_inference_and_mjlab_physics',
                'reference_mode':'kinematic target geometry only, no policy, no collision with live robot',
                'policy_inference_device':'cuda:0','physics_device':'cuda:0',
                'clip_step':self.clip_step,'clip_seconds':self.clip_step*dt,'completed_loops':self.loops,
                'clip_failed':self.clip_failed,'paused':self._is_paused,'error':self._last_error,
                'actual_realtime':self.actual_realtime,'updated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                'training_restarted':False,'records_video':False})

        def reset_environment(self):
            cmd.forced_frame[:]=cmd.library.clip_start
            self.obs,_=env.reset();self.clip_step=0;self.pending_reset=None;self.clip_failed=False
            self._sim_budget=0.;self._last_error=None;self._last_tick_time=time.perf_counter()
            self._clear_histories()

        def _execute_step(self):
            if self.pending_reset is not None:
                if time.monotonic()<self.pending_reset:return True
                self.reset_environment()
            try:
                with torch.no_grad():
                    action=act(self.obs)
                    self.obs,_,terminated,timeout,_=env.step(action)
                self.clip_step+=1;self._step_count+=1;self._stats_steps+=1
                if bool(terminated[0]):
                    self.clip_failed=True;self.pause();self.publish('paused_on_physical_failure')
                    print('Physical failure: paused, not hidden by an automatic mid-clip reset. Enter to restart.',flush=True)
                elif bool(timeout[0]):
                    self.loops+=1;self.pending_reset=time.monotonic()+.7
                    self.publish('clip_complete')
                    print(f'Completed full live clip {self.loops}, {self.clip_step} steps; looping.',flush=True)
                return True
            except Exception as exc:
                self._last_error=repr(exc);self.pause();self.publish('paused_on_error')
                print(f'Preview error: {exc!r}',flush=True);return False

        def _sync_env_state_to_mjdata(self,target_data,sim_data,env_idx):
            super()._sync_env_state_to_mjdata(target_data,sim_data,env_idx)
            # Display-only translation: physics and observations remain untouched.
            target_data.qpos[:3]+=offset

        def sync_viewer_to_env(self):
            # Mouse rotates/zooms without applying physical forces.
            pass

        def _update_debug_visualizers(self,v):
            v.user_scn.ngeom=0
            qref=torch.cat([cmd.root_pos_w[0],cmd.root_quat_w[0],cmd.joint_pos[0]]).cpu().numpy()
            self.reference_data.qpos[:]=qref;self.reference_data.qpos[:3]-=offset
            mujoco.mj_forward(self.mjm,self.reference_data)
            mujoco.mjv_addGeoms(self.mjm,self.reference_data,self.reference_option,self.reference_perturb,
                mujoco.mjtCatBit.mjCAT_DYNAMIC.value,v.user_scn)
            v.cam.lookat[:]=[qref[0],qref[1],.85]
            for label,position,color in [('REFERENCE / NO POLICY',qref[:3]-offset,(.25,.75,1.,1.)),
                (f'LIVE POLICY / {iteration}',self.mjd.qpos[:3],(1.,.85,.3,1.))]:
                if v.user_scn.ngeom>=v.user_scn.maxgeom:break
                geom=v.user_scn.geoms[v.user_scn.ngeom];v.user_scn.ngeom+=1
                mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_LABEL.value,np.zeros(3),
                    np.asarray(position)+[0,0,1.05],np.eye(3).ravel(),np.array(color,dtype=np.float32))
                geom.label=label

        def _set_status_overlay(self,v):
            status=self.get_status();state='PAUSED' if status.paused else 'PLAYING'
            if self.clip_failed:state='FAILED - PAUSED; Enter restarts'
            titles=chr(10).join(['G1 live comparison','Checkpoint','Time / clip','State','Speed / actual','Controls'])
            values=chr(10).join(['LEFT reference | RIGHT policy',str(iteration),
                f'{self.clip_step*dt:.2f} / {cmd.library.clips[0].length*dt:.2f} s',state,
                f'{status.speed_label} / {status.actual_realtime:.2f}x','Space pause | Enter restart | +/- speed'])
            v.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100.value,mujoco.mjtGridPos.mjGRID_TOPLEFT.value,titles,values))
            if time.monotonic()-self.last_publish>3:
                self.publish('paused' if self._is_paused else 'running');self.last_publish=time.monotonic()

    viewer=ComparisonViewer()
    try:viewer.run()
    finally:
        viewer.publish('closed');env.close();time.sleep(.5)


if __name__=='__main__':main()
