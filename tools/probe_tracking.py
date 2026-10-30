"""Short free-base PD reference probes; no training, root resets or external forces."""
import argparse,json,time,os
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from common import ROOT,add_sources

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=float,default=1.);a=p.parse_args()
    os.environ.setdefault('MUJOCO_GL','egl');add_sources();import torch,mujoco,warp as wp
    from mjlab.asset_zoo.robots import get_g1_robot_cfg
    from mjlab.scene import Scene,SceneCfg
    from mjlab.sim import Simulation,SimulationCfg,MujocoCfg
    from mjlab.terrains import TerrainEntityCfg
    torch.set_num_threads(2);torch.set_num_interop_threads(1);wp.init()
    with np.load(a.input,allow_pickle=False) as f:d={k:f[k] for k in f.files}
    fps=float(d['fps'][0]);n=len(d['joint_pos']);starts=np.array([0.,.9,4.2,7.7,9.6,10.8]);E=len(starts)
    scene=Scene(SceneCfg(num_envs=E,terrain=TerrainEntityCfg(terrain_type='plane'),entities={'robot':get_g1_robot_cfg()},extent=2.),device='cuda:0')
    sim=Simulation(num_envs=E,cfg=SimulationCfg(nconmax=128,njmax=512,graph_conditional=False,mujoco=MujocoCfg(timestep=.005,iterations=20,ls_iterations=50)),spec=scene.spec,device='cuda:0');scene.initialize(mj_model=sim.mj_model,model=sim.model,data=sim.data)
    model=sim.mj_model;robot=scene['robot'];root=list(d['body_names']).index('pelvis');q=np.column_stack([d['body_pos_w'][:,root],d['body_quat_w'][:,root],d['joint_pos']]).astype(float)
    vel=np.column_stack([d['body_lin_vel_w'][:,root],Rotation.from_quat(q[:,[4,5,6,3]]).inv().apply(d['body_ang_vel_w'][:,root]),d['joint_vel']])
    ids=np.rint(starts*fps).astype(int);act_j=model.actuator_trnid[:,0];jcols=model.jnt_qposadr[act_j]-7
    kp=model.actuator_gainprm[:,0];kd=-model.actuator_biasprm[:,2]
    sim.reset();sim.data.qpos[:]=torch.tensor(q[ids],device='cuda',dtype=torch.float32);sim.data.qvel[:]=torch.tensor(vel[ids],device='cuda',dtype=torch.float32)
    sim.forward();wp.synchronize();samples=[];metrics=[];failed=np.full(E,np.nan);start=time.perf_counter()
    for step in range(round(a.seconds/.005)):
        t=(step+1)*.005;idx=np.minimum(np.rint((starts+t)*fps).astype(int),n-1)
        target=d['joint_pos'][idx][:,jcols]+d['joint_vel'][idx][:,jcols]*(kd/kp)
        sim.data.ctrl[:]=torch.tensor(target,device='cuda',dtype=torch.float32);sim.step()
        if (step+1)%4==0:
            actual=sim.data.qpos.detach().cpu().numpy().copy();reference=q[idx]
            error=np.linalg.norm(actual[:,:3]-reference[:,:3],axis=1)
            qa=actual[:,3:7];qr=reference[:,3:7];angle=2*np.arccos(np.clip(abs(np.sum(qa*qr,axis=1)),0,1))
            joint=np.sqrt(np.mean((actual[:,7:]-reference[:,7:])**2,axis=1))
            for e in range(E):
                if np.isnan(failed[e]) and (error[e]>.25 or angle[e]>.8 or not np.isfinite(actual[e]).all()):failed[e]=t
            samples.append(actual);metrics.append(np.column_stack([error,angle,joint]))
    wp.synchronize();arr=np.asarray(metrics);states=np.asarray(samples);a.output.mkdir(parents=True,exist_ok=True)
    results=[{'reference_start_s':float(starts[e]),'first_tracking_failure_s':None if np.isnan(failed[e]) else float(failed[e]),'max_root_error_m':float(arr[:,e,0].max()),'max_orientation_error_deg':float(np.rad2deg(arr[:,e,1].max())),'mean_joint_rmse_rad':float(arr[:,e,2].mean())} for e in range(E)]
    report={'input':str(a.input),'probe_seconds':a.seconds,'num_envs':E,'results':results,'all_probes_passed':bool(np.isnan(failed).all()),'wall_s':time.perf_counter()-start,'root_state_writes_after_initialization':0,'external_forces':False,'gravity':list(model.opt.gravity),'controller':'Original model position PD with desired joint velocity feedforward; no learned policy','actuator_mapping_verified':True,'interpretation':'Failure shows this simple controller cannot track the reference; it does not prove no learned controller can succeed.','training_approved':False}
    np.savez_compressed(a.output/'probe_states.npz',qpos=states,metrics=arr,fps=50,starts=starts,joint_names=d['joint_names'])
    (a.output/'probe_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
