"""Support/COM-constrained dance adaptation; preserves original and reports deviation."""
import argparse,json,pickle,time
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.optimize import least_squares
from scipy.signal import savgol_filter,medfilt
from scipy.spatial.transform import Rotation
from common import ROOT,add_sources
from refine_contacts import geometry_bottom
from convert_motion import convert


def prepare(source,smooth=9):
    import mujoco
    with np.load(source,allow_pickle=False) as f:d={k:f[k] for k in f.files}
    root=list(d['body_names']).index('pelvis');fps=float(d['fps'][0])
    q=np.c_[d['body_pos_w'][:,root],d['body_quat_w'][:,root],d['joint_pos']].astype(float)
    if smooth>1:
        q[:,:3]=savgol_filter(q[:,:3],smooth,3,axis=0)
        q[:,7:]=savgol_filter(q[:,7:],smooth,3,axis=0)
        quat=q[:,3:7].copy()
        for t in range(1,len(q)):
            if quat[t]@quat[t-1]<0:quat[t]*=-1
        quat=savgol_filter(quat,smooth,3,axis=0);q[:,3:7]=quat/np.linalg.norm(quat,axis=1)[:,None]
    return q,d,fps


def export(q,fps,d,destination,report_dir,name):
    pkl=report_dir/(name+'.pkl')
    with pkl.open('wb') as f:pickle.dump({'fps':fps,'root_pos':q[:,:3],'root_rot':q[:,[4,5,6,3]],'dof_pos':q[:,7:],'joint_names':list(d['joint_names']),'root_quaternion_order':'xyzw'},f)
    return convert(pkl,destination,output_fps=50.,ground_offset=0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=ROOT/'data/preview/reference_full_contact.npz')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/physics_repair')
    p.add_argument('--name',default='reference_full_support')
    p.add_argument('--speed',type=float,default=1.)
    p.add_argument('--smooth',type=int,default=9)
    p.add_argument('--iterations',type=int,default=60)
    p.add_argument('--balance',type=float,default=20.)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    add_sources();import mujoco
    from supermimic.data.motion_convert import load_g1_model
    model=load_g1_model();data=mujoco.MjData(model);q0,d,fps=prepare(a.input,a.smooth);n=len(q0);nv=model.nv
    q0[:,7:]=np.clip(q0[:,7:],model.jnt_range[1:,0]+1e-5,model.jnt_range[1:,1]-1e-5)
    groups=[[g for g in range(model.ngeom) if model.geom(g).name.startswith(side) and (model.geom_contype[g] or model.geom_conaffinity[g])] for side in ('left_foot','right_foot','left_hand','right_hand')]
    bodies=[model.body(name).id for name in ('left_ankle_roll_link','right_ankle_roll_link','left_wrist_yaw_link','right_wrist_yaw_link')]
    geoms=[g for g in range(model.ngeom) if model.geom_contype[g] or model.geom_conaffinity[g]]
    points=np.zeros((n,4,3));heights=np.zeros((n,4));com=np.zeros((n,3))
    for t in range(n):
        data.qpos[:]=q0[t];mujoco.mj_forward(model,data);points[t]=data.xpos[bodies];com[t]=data.subtree_com[1]
        heights[t]=[min(geometry_bottom(model,data,g)[0][2] for g in gs) for gs in groups]
    # This is an explicit adapted support hypothesis, not a claim of video truth.
    side=medfilt(np.argmin(heights[:,:2],axis=1),5).astype(int);mask=np.zeros((n,4),bool);mask[np.arange(n),side]=True
    for s in (0,1):mask[:,s]|=medfilt((abs(heights[:,0]-heights[:,1])<.025).astype(int),5).astype(bool)
    t=np.arange(n)/fps
    flight=((t>=6.10)&(t<=6.20))|((t>=6.46)&(t<=6.58))|((t>=9.02)&(t<=9.38))
    mask[flight]=False
    mask[(t>=7.7)&(t<=8.03),0]=True
    mask[(t>=7.8)&(t<=8.),2]=True;mask[(t>=7.8)&(t<=7.9),3]=True
    anchors=points.copy();segments=[]
    for s in range(4):
        ids=np.flatnonzero(mask[:,s])
        for block in np.split(ids,np.flatnonzero(np.diff(ids)>1)+1):
            if not len(block):continue
            anchors[block,s,:2]=np.median(points[block,s,:2],axis=0)
            segments.append({'effector':s,'start_s':float(block[0]/fps),'end_s':float(block[-1]/fps)})
    # Project original COM toward each support segment; allow 4 cm from its centerline.
    target=com[:,:2].copy()
    for i in range(n):
        supp=anchors[i,mask[i],:2]
        if not len(supp):continue
        if len(supp)==1:near=supp[0]
        else:
            candidates=[]
            for u in supp:
                for v in supp:
                    delta=v-u;alpha=np.clip((com[i,:2]-u)@delta/max(delta@delta,1e-9),0,1);candidates.append(u+alpha*delta)
            near=min(candidates,key=lambda x:np.linalg.norm(x-com[i,:2]))
        delta=com[i,:2]-near;length=np.linalg.norm(delta)
        target[i]=near+delta*min(1.,.04/max(length,1e-9))
    # Sparse temporal penalty preserves original motion while smoothing changes.
    d1=sparse.diags([-np.ones(n-1),np.ones(n-1)],[0,1],shape=(n-1,n),format='csr')
    d2=sparse.diags([np.ones(n-2),-2*np.ones(n-2),np.ones(n-2)],[0,1,2],shape=(n-2,n),format='csr')
    weights=np.r_[np.full(3,4.),np.full(3,2.),np.ones(29)]
    reg=sparse.vstack([sparse.kron(sparse.eye(n),sparse.diags(weights)),sparse.kron(d1,sparse.diags(weights))*2,sparse.kron(d2,sparse.diags(weights))*10],format='csr')
    bounds=np.r_[.4,.4,.35,.25,.25,.25,np.full(29,.7)]
    lower=np.tile(-bounds,(n,1));upper=np.tile(bounds,(n,1));lower[:,6:]=np.maximum(lower[:,6:],model.jnt_range[1:,0]-q0[:,7:]+1e-6);upper[:,6:]=np.minimum(upper[:,6:],model.jnt_range[1:,1]-q0[:,7:]-1e-6)
    cache={};calls=0;start=time.perf_counter()
    def evaluate(v):
        nonlocal calls
        if 'x' in cache and np.array_equal(v,cache['x']):return cache['r'],cache['j']
        delta=v.reshape(n,nv);res=[];rr=[];cc=[];vv=[]
        def add(value,jac,i):
            value=np.atleast_1d(value);jac=np.atleast_2d(jac);row=len(res);res.extend(value.tolist());r,c=np.nonzero(jac);rr.extend((row+r).tolist());cc.extend((i*nv+c).tolist());vv.extend(jac[r,c].tolist())
        for i in range(n):
            q=q0[i].copy();mujoco.mj_integratePos(model,q,delta[i],1.);data.qpos[:]=q;mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data)
            for s in np.flatnonzero(mask[i]):
                j=np.zeros((3,nv));mujoco.mj_jacBody(model,data,j,None,bodies[s]);add((data.xpos[bodies[s],:2]-anchors[i,s,:2])*50,j[:2]*50,i)
                h,j=min((geometry_bottom(model,data,g) for g in groups[s]),key=lambda x:x[0][2]);add((h[2]-.0005)*180,j[2]*180,i)
            if mask[i].any():
                j=np.zeros((3,nv));mujoco.mj_jacSubtreeCom(model,data,j,1);add((data.subtree_com[1,:2]-target[i])*a.balance,j[:2]*a.balance,i)
            for g in geoms:
                h,j=geometry_bottom(model,data,g);active=h[2]<-.001;add(min(0.,h[2]+.001)*240,j[2]*240 if active else np.zeros(nv),i)
        j=sparse.coo_matrix((vv,(rr,cc)),shape=(len(res),n*nv)).tocsr();r=np.r_[reg@v,res];j=sparse.vstack([reg,j],format='csr')
        calls+=1
        if calls%10==0:print('evaluation',calls,'seconds',round(time.perf_counter()-start,1),'cost',round(float(r@r)/2,3),flush=True)
        cache.update(x=v.copy(),r=r,j=j);return r,j
    result=least_squares(lambda x:evaluate(x)[0],np.zeros(n*nv),jac=lambda x:evaluate(x)[1],bounds=(lower.ravel(),upper.ravel()),max_nfev=a.iterations,ftol=1e-4,xtol=1e-4,gtol=1e-3,tr_options={'atol':1e-4,'btol':1e-4})
    q=q0.copy()
    for i,x in enumerate(result.x.reshape(n,nv)):mujoco.mj_integratePos(model,q[i],x,1.)
    path=ROOT/'data/preview'/f'{a.name}.npz';conversion=export(q,fps*a.speed,d,path,a.output,a.name)
    np.savez_compressed(a.output/(a.name+'_support.npz'),mask=mask,anchors=anchors,com_target=target,qpos=q,fps=fps*a.speed,flight=flight)
    report={'input':str(a.input),'output':str(path),'source_fps':fps,'speed_multiplier':a.speed,'frames_before_resample':n,'smoothing_window_frames':a.smooth,'balance_weight':a.balance,'support_segments':segments,'flight_intervals_source_s':[[6.10,6.20],[6.46,6.58],[9.02,9.38]],'support_labels':'Adapted geometric hypothesis; small ambiguous hops may be removed, not verified source contacts','success':bool(result.success),'evaluations':result.nfev,'wall_s':time.perf_counter()-start,'cost':float(result.cost),'max_delta':float(abs(result.x).max()),'conversion':conversion,'training_approved':False}
    (a.output/(a.name+'_report.json')).write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='support_segments'},indent=2))


if __name__=='__main__':main()
