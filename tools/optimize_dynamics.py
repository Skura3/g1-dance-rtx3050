"""Sequential sparse inverse-dynamics trajectory repair on the actual mjlab model."""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsmr
from scipy.spatial.transform import Rotation
from common import ROOT,add_sources
from repair_physics import export
from track_wbc import load_reference
from refine_contacts import geometry_bottom


def trajectory_derivatives(q,fps):
    v=np.zeros((len(q),35));v[:,:3]=np.gradient(q[:,:3],1/fps,axis=0);v[:,6:]=np.gradient(q[:,7:],1/fps,axis=0)
    rot=Rotation.from_quat(q[:,[4,5,6,3]])
    world=(rot[2:]*rot[:-2].inv()).as_rotvec()*fps/2
    world=np.vstack([world[0],world,world[-1]])
    v[:,3:6]=rot.inv().apply(world)
    return v,np.gradient(v,1/fps,axis=0)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--name',default='reference_full_dynamic');p.add_argument('--iterations',type=int,default=30);p.add_argument('--speed',type=float,default=1.);p.add_argument('--weight',type=float,default=50.);a=p.parse_args()
    add_sources();import mujoco
    out=ROOT/'outputs/physics_repair'/a.name;out.mkdir(parents=True,exist_ok=True)
    q,_,_,fps,meta=load_reference(a.input);fps*=a.speed;n=len(q);nv=35
    model=mujoco.MjModel.from_binary_path(str(ROOT/'outputs/physics_repair/g1_scene.mjb'));data=mujoco.MjData(model)
    qref=q.copy();dofs=model.jnt_dofadr[model.actuator_trnid[:,0]];bounds=model.actuator_forcerange;limits=np.empty(29);limits[dofs-6]=np.maximum(abs(bounds[:,0]),abs(bounds[:,1]))
    bw=model.body_mass.sum()*9.81;scale=np.r_[np.full(3,bw),np.full(3,bw*.5),limits]
    geoms=[g for g in range(model.ngeom) if (model.geom(g).name or '').startswith(('robot/left_foot','robot/right_foot','robot/left_hand','robot/right_hand')) and (model.geom_contype[g] or model.geom_conaffinity[g])]
    # Seed compliant physical contact; gap-free geometry alone produces no support force.
    for i in range(n):
        data.qpos[:]=q[i];mujoco.mj_forward(model,data)
        z=min(geometry_bottom(model,data,g)[0][2] for g in geoms)
        if z<.012:q[i,2]-=z+.003
    D=sparse.lil_matrix((n,n))
    D[0,0]=-fps;D[0,1]=fps;D[-1,-2]=-fps;D[-1,-1]=fps
    for i in range(1,n-1):D[i,i-1]=-fps/2;D[i,i+1]=fps/2
    D=D.tocsr();V=sparse.kron(D,sparse.eye(nv),format='csr');A=V@V
    d2=sparse.diags([np.ones(n-2),-2*np.ones(n-2),np.ones(n-2)],[0,1,2],shape=(n-2,n),format='csr')
    pose_weights=np.r_[np.full(3,8.),np.full(3,2.),np.full(29,.7)]
    R=sparse.kron(sparse.eye(n),sparse.diags(pose_weights),format='csr')
    smooth=sparse.kron(d2,sparse.diags(np.r_[np.full(3,20.),np.full(3,5.),np.ones(29)*3]),format='csr')
    trace=[];start=time.perf_counter()
    def evaluate(q,derivatives=False):
        vel,acc=trajectory_derivatives(q,fps);forces=[];res=[];jq=[];jv=[];ja=[];pose=np.zeros((n,nv));violations=[]
        for i in range(n):
            data.qpos[:]=q[i];data.qvel[:]=vel[i];data.qacc[:]=acc[i];mujoco.mj_inverse(model,data);f=data.qfrc_inverse.copy();forces.append(f)
            error=f.copy();error[6:]=np.maximum(f[6:]-limits,0)+np.minimum(f[6:]+limits,0)
            factor=a.weight/scale;active=np.r_[np.ones(6),abs(f[6:])>limits];res.extend(error*factor)
            mujoco.mj_differentiatePos(model,pose[i],1.,qref[i],q[i])
            if derivatives:
                dq=np.zeros((nv,nv));dv=dq.copy();da=dq.copy();mujoco.mjd_inverseFD(model,data,1e-6,False,dq,dv,da,None,None,None,None)
                row=factor*active;jq.append(sparse.csr_matrix(dq.T*row[:,None]));jv.append(sparse.csr_matrix(dv.T*row[:,None]));ja.append(sparse.csr_matrix(da.T*row[:,None]))
        f=np.asarray(forces);r=np.r_[res,R@pose.ravel(),smooth@pose.ravel()]
        stats={'cost':float(r@r/2),'median_base_force_ratio':float(np.median(np.linalg.norm(f[:,:3],axis=1)/bw)),'p95_base_force_ratio':float(np.percentile(np.linalg.norm(f[:,:3],axis=1)/bw,95)),'native_inverse_pass_frames':int(np.sum((np.linalg.norm(f[:,:3],axis=1)<.05*bw)&(np.linalg.norm(f[:,3:6],axis=1)<.025*bw)&(np.max(abs(f[:,6:])/limits,axis=1)<1.05))),'max_torque_ratio':float(np.max(abs(f[:,6:])/limits))}
        if not derivatives:return r,stats,f
        J=sparse.block_diag(jq,format='csr')+sparse.block_diag(jv,format='csr')@V+sparse.block_diag(ja,format='csr')@A
        return r,sparse.vstack([J,R,smooth],format='csr'),stats
    for iteration in range(a.iterations):
        r,J,stats=evaluate(q,True)
        norm=np.sqrt(np.asarray(J.power(2).sum(axis=0)).ravel())+1.;S=sparse.diags(1/norm)
        solve=lsmr(J@S,-r,damp=.03,atol=1e-5,btol=1e-5,maxiter=220);step=(solve[0]/norm).reshape(n,nv)
        ratio=max(float(np.max(abs(step[:,:3])/.03)),float(np.max(abs(step[:,3:])/.10)),1.);step/=ratio
        accepted=False
        for fraction in [1.,.5,.25,.1]:
            trial=q.copy()
            for i in range(n):mujoco.mj_integratePos(model,trial[i],step[i],fraction)
            trial[:,7:]=np.clip(trial[:,7:],model.jnt_range[1:,0]+1e-6,model.jnt_range[1:,1]-1e-6)
            _,new,_=evaluate(trial)
            if new['cost']<stats['cost']:q=trial;accepted=True;break
        row={'iteration':iteration,'accepted':accepted,'fraction':fraction,'wall_s':time.perf_counter()-start,**(new if accepted else stats)};trace.append(row);print(json.dumps(row),flush=True)
        if not accepted:break
    _,stats,forces=evaluate(q)
    target=ROOT/'data/preview'/f'{a.name}.npz';conversion=export(q,fps,meta,target,out,a.name)
    np.savez_compressed(out/'optimization_states.npz',qpos=q,forces=forces,fps=fps)
    report={'input':str(a.input),'output':str(target),'speed_multiplier':a.speed,'source_frames':n,'iterations':trace,'final':stats,'wall_s':time.perf_counter()-start,'conversion':conversion,'training_approved':False,'method':'Native compliant-contact inverse dynamics, sparse sequential least squares; no external root forces applied in rollout'}
    (out/'optimization_report.json').write_text(json.dumps(report,indent=2));print('saved',target,flush=True)


if __name__=='__main__':main()
