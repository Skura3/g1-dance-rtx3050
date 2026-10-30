"""Instantaneous contact-force and torque feasibility screening, not a controller."""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation
from common import ROOT,add_sources
from refine_contacts import geometry_bottom

def contact_force_jacobian(model,data,geom):
    """Material contact point Jacobian, including the sphere/capsule radius.

    geometry_bottom's Jacobian differentiates the *height envelope*. Its
    horizontal rows describe the centerline endpoint and cannot map friction
    forces to joint torques. Recompute at the actual surface contact point.
    """
    import mujoco
    point,_=geometry_bottom(model,data,geom)
    jac=np.zeros((3,model.nv))
    mujoco.mj_jac(model,data,jac,None,point,int(model.geom_bodyid[geom]))
    return point,jac

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--contact-distance',type=float,default=.01);p.add_argument('--friction',type=float,default=.6);a=p.parse_args()
    add_sources();import torch,mujoco
    from mjlab.asset_zoo.robots import get_g1_robot_cfg
    from mjlab.scene import Scene,SceneCfg
    from mjlab.terrains import TerrainEntityCfg
    torch.set_num_threads(1)
    scene=Scene(SceneCfg(num_envs=1,terrain=TerrainEntityCfg(terrain_type='plane'),entities={'robot':get_g1_robot_cfg()},extent=2.),device='cpu')
    model=scene.spec.compile();data=mujoco.MjData(model)
    with np.load(a.input,allow_pickle=False) as f:d={k:f[k] for k in f.files}
    names=[model.joint(i).name.removeprefix('robot/') for i in range(1,model.njnt)]
    if names!=list(d['joint_names']):raise ValueError('Model joint order changed')
    fps=float(d['fps'][0]);n=len(d['joint_pos']);root=list(d['body_names']).index('pelvis')
    q=np.column_stack([d['body_pos_w'][:,root],d['body_quat_w'][:,root],d['joint_pos']]).astype(float)
    rot=Rotation.from_quat(q[:,[4,5,6,3]])
    vel=np.column_stack([d['body_lin_vel_w'][:,root],rot.inv().apply(d['body_ang_vel_w'][:,root]),d['joint_vel']]).astype(float)
    acc=np.gradient(vel,1/fps,axis=0);actj=model.actuator_trnid[:,0];dofs=model.jnt_dofadr[actj]
    B=np.zeros((model.nv,model.nu));B[dofs,np.arange(model.nu)]=model.actuator_gear[:,0]
    bounds=model.actuator_forcerange.copy();limits=np.maximum(abs(bounds[:,0]),abs(bounds[:,1]))
    if not np.all(model.actuator_forcelimited) or np.any(limits<=0):raise ValueError('Expected finite actuator torque limits')
    joint_limits=np.empty(model.nv-6);joint_limits[dofs-6]=limits
    mass=float(model.body_mass.sum());bw=mass*9.81;scale=np.r_[np.full(3,bw),np.full(3,bw*.5),joint_limits]
    geoms=[i for i in range(model.ngeom) if model.geom(i).name.startswith('robot/') and (model.geom_contype[i] or model.geom_conaffinity[i])]
    rays=np.array([[a.friction,0,1],[-a.friction,0,1],[0,a.friction,1],[0,-a.friction,1]]).T
    metrics=[];torques=[];residuals=[];counts=[];unloaded=[];start=time.perf_counter();failures=[]
    for t in range(n):
        data.qpos[:]=q[t];data.qvel[:]=vel[t];mujoco.mj_forward(model,data)
        needed=np.zeros(model.nv);mujoco.mj_mulM(model,data,needed,acc[t]);needed+=data.qfrc_bias-data.qfrc_passive
        points=[]
        for g in geoms:
            pt,jac=contact_force_jacobian(model,data,g)
            if pt[2]<=a.contact_distance:points.append((pt,jac,g))
        C=np.concatenate([jac.T@rays for pt,jac,g in points],axis=1) if points else np.zeros((model.nv,0))
        E=np.column_stack([B,C]);weights=1/scale
        reg=np.diag(np.r_[.002/limits,np.full(C.shape[1],.0001/bw)])
        mat=np.vstack([E*weights[:,None],reg]);target=np.r_[needed*weights,np.zeros(E.shape[1])]
        lo=np.r_[bounds[:,0],np.zeros(C.shape[1])];hi=np.r_[bounds[:,1],np.full(C.shape[1],np.inf)]
        fit=lsq_linear(mat,target,bounds=(lo,hi),method='bvls',tol=1e-8,max_iter=300)
        error=E@fit.x-needed;torque=fit.x[:model.nu]
        force=float(np.linalg.norm(error[:3])/bw);moment=float(np.linalg.norm(error[3:6])/(bw*.5));joint=float(np.max(abs(error[6:])/joint_limits))
        feasible=force<.05 and moment<.05 and joint<.05 and fit.success
        metrics.append([force,moment,joint,float(np.max(abs(torque)/limits)),int(feasible)])
        torques.append(torque);residuals.append(error);counts.append(len(points));unloaded.append(needed)
        if not fit.success:failures.append(t)
        if t%100==0:print(f'Dynamics {t}/{n}',flush=True)
    metrics=np.asarray(metrics);a.output.mkdir(parents=True,exist_ok=True)
    report={'source':str(a.input),'frames':n,'fps':fps,'mass_kg':mass,'friction_coefficient_assumed':a.friction,'friction_cone':'Conservative four-ray pyramid, no adhesion','candidate_contact_height_m':a.contact_distance,
        'instantaneously_feasible_frames':int(metrics[:,4].sum()),'infeasible_frames':int(n-metrics[:,4].sum()),'frames_without_geometric_contact':int(np.sum(np.array(counts)==0)),
        'normalized_base_force_residual_median':float(np.median(metrics[:,0])),'normalized_base_force_residual_p95':float(np.percentile(metrics[:,0],95)),
        'normalized_joint_residual_p95':float(np.percentile(metrics[:,2],95)),'solver_failure_frames':failures,'wall_s':time.perf_counter()-start,
        'joint_torque_limits_nm':dict(zip(names,joint_limits.tolist())),'training_approved':False,
        'limitations':['Instantaneous force fitting cannot establish trajectory realizability or feedback stability.','Contact candidates within 1cm are optimistic and may still slide; sticking/complementarity are not enforced.','Finite differences amplify reconstruction noise. Friction is an explicit analysis assumption, not measured grass friction.','No-contact frames still require ballistic free-base dynamics; they are not automatically considered valid.']}
    np.savez_compressed(a.output/'dynamics_series.npz',metrics=metrics,torque=torques,residual=residuals,contact_count=counts,required_generalized_force=unloaded,fps=fps)
    (a.output/'dynamics_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
