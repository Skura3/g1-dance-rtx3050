"""Sparse contact-constrained kinematic refinement; retains unmodified inputs."""
import argparse,json,pickle,time
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.optimize import least_squares
from common import ROOT,add_sources
from convert_motion import DEFAULT_XML,convert

def geometry_bottom(model,data,g):
    """Exact sphere/capsule lowest point and height Jacobian."""
    import mujoco
    p=data.geom_xpos[g].copy();kind=model.geom_type[g]
    if kind==mujoco.mjtGeom.mjGEOM_CAPSULE:
        axis=data.geom_xmat[g].reshape(3,3)[:,2]
        p-=np.sign(axis[2])*model.geom_size[g,1]*axis
    elif kind!=mujoco.mjtGeom.mjGEOM_SPHERE:raise ValueError('Review changed collision geometry')
    jac=np.zeros((3,model.nv));mujoco.mj_jac(model,data,jac,None,p,int(model.geom_bodyid[g]))
    p[2]-=model.geom_size[g,0]
    return p,jac

def contact_schedule(config,fps,n):
    import torch
    cfg=json.loads(Path(config).read_text());pred=ROOT/'outputs/gvhmr/reference_full'
    static=torch.load(pred/'hmr4d_results.pt',map_location='cpu',weights_only=False)['net_outputs']['static_conf_logits'][0].sigmoid().numpy()
    kp=torch.load(pred/'preprocess/vitpose.pt',map_location='cpu',weights_only=False).numpy()
    contacts=list(cfg['manual_contacts'])
    for side,col,ankle in [('left',1,15),('right',3,16)]:
        mask=(static[:,col]>cfg['automatic_static_probability'])&(kp[:,ankle,2]>cfg['automatic_ankle_confidence'])
        ids=np.flatnonzero(mask)
        for group in np.split(ids,np.flatnonzero(np.diff(ids)>1)+1):
            if len(group)<cfg['automatic_minimum_source_frames']:continue
            contacts.append({'body':side+'_foot','start':float(group[0]/cfg['source_fps']),'end':float(group[-1]/cfg['source_fps']),'source':'GVHMR static-toe + confident ViTPose ankle; inferred, not ground truth'})
    for c in contacts:
        c['frames']=list(range(max(0,int(np.ceil(c['start']*fps-1e-6))),min(n-1,int(np.floor(c['end']*fps+1e-6)))+1))
    return contacts

def refine(source,destination,config,output,max_nfev=100):
    add_sources();import mujoco
    from supermimic.data.motion_convert import load_g1_model,G1_JOINT_NAMES
    if Path(source).resolve()==Path(destination).resolve():raise ValueError('Keep original reference')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with np.load(source,allow_pickle=False) as f:d={k:f[k] for k in f.files}
    fps=float(d['fps'][0]);n=len(d['joint_pos']);root=list(d['body_names']).index('pelvis')
    if list(d['joint_names'])!=list(G1_JOINT_NAMES):raise ValueError('Unexpected G1 joint order')
    q0=np.column_stack([d['body_pos_w'][:,root],d['body_quat_w'][:,root],d['joint_pos']]).astype(float)
    model=load_g1_model(DEFAULT_XML);data=mujoco.MjData(model)
    geoms=[i for i in range(model.ngeom) if model.geom_contype[i] or model.geom_conaffinity[i]]
    columns=np.r_[0:3,6:35];qcols=np.r_[0:3,7:36];k=len(columns)
    contacts=contact_schedule(config,fps,n);mask=np.zeros(n,bool);perframe={}
    for c in contacts:
        a,b=c['frames'][0],c['frames'][-1];mask[max(0,a-18):min(n,b+19)]=True
        c['geoms']=[g for g in geoms if model.geom(g).name.startswith(c['body'])]
        body_name=c['body'].replace('_foot','_ankle_roll_link').replace('_hand','_wrist_yaw_link')
        body=model.body(body_name).id;c['body_id']=body;xyz=[]
        for t in c['frames']:
            data.qpos[:]=q0[t];mujoco.mj_kinematics(model,data);xyz.append(data.xpos[body].copy())
        c['target_xy']=np.median(xyz,axis=0)[:2].tolist()
        for t in c['frames']:perframe.setdefault(t,[]).append(c)
    rows=np.flatnonzero(mask);lookup={t:i for i,t in enumerate(rows)};m=len(rows)
    c1=sparse.lil_matrix((n-1,m));c2=sparse.lil_matrix((n-2,m))
    for t,r in lookup.items():
        if t>0:c1[t-1,r]=1
        if t<n-1:c1[t,r]=-1
        if t>=2:c2[t-2,r]=1
        if 1<=t<n-1:c2[t-1,r]=-2
        if t<n-2:c2[t,r]=1
    weights=np.r_[np.full(3,7.),np.ones(29)]
    reg=sparse.vstack([sparse.kron(sparse.eye(m),sparse.diags(weights)),sparse.kron(c1,sparse.diags(weights))*3,sparse.kron(c2,sparse.diags(weights))*8],format='csr')
    limit=np.r_[.18,.18,.24,np.full(29,.8)];lower=np.tile(-limit,(m,1));upper=np.tile(limit,(m,1))
    lower[:,3:]=np.maximum(lower[:,3:],model.jnt_range[1:,0]-q0[rows,7:]+1e-7)
    upper[:,3:]=np.minimum(upper[:,3:],model.jnt_range[1:,1]-q0[rows,7:]-1e-7)
    x=np.clip(np.zeros(m*k),lower.ravel()+1e-9,upper.ravel()-1e-9);rounds=[];pairs=set();timing={'evaluate_calls':0,'evaluate_seconds':0.}
    for outer in range(3):
        collision_pairs=sorted(pairs);cache={}
        def evaluate(v):
            if 'x' in cache and np.array_equal(v,cache['x']):return cache['r'],cache['j']
            start=time.perf_counter();delta=v.reshape(m,k);res=[];ii=[];jj=[];vv=[]
            def add(value,jac,t):
                value=np.atleast_1d(value);jac=np.atleast_2d(jac)
                first=len(res);res.extend(value.tolist());ridx,cidx=np.nonzero(jac)
                ii.extend((ridx+first).tolist());jj.extend((cidx+lookup[t]*k).tolist());vv.extend(jac[ridx,cidx].tolist())
            for idx,t in enumerate(rows):
                q=q0[t].copy();q[qcols]+=delta[idx];data.qpos[:]=q
                mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data)
                bottoms={g:geometry_bottom(model,data,g) for g in geoms}
                for c in perframe.get(t,[]):
                    body=c['body_id'];point=data.xpos[body];jac=np.zeros((3,model.nv));mujoco.mj_jac(model,data,jac,None,point,body)
                    add((point[:2]-c['target_xy'])*60,jac[:2,columns]*60,t)
                    g=min(c['geoms'],key=lambda g:bottoms[g][0][2]);p,j=bottoms[g]
                    add((p[2]-.001)*180,j[2,columns]*180,t)
                for g,(p,j) in bottoms.items():
                    active=p[2]<-.001;add(min(0.,p[2]+.001)*300,j[2,columns]*300 if active else np.zeros(k),t)
                for ga,gb in collision_pairs:
                    segment=np.zeros(6);dist=mujoco.mj_geomDistance(model,data,ga,gb,.08,segment)
                    if dist<.004:
                        normal=(segment[3:]-segment[:3])/max(abs(dist),1e-8)
                        if dist<0:normal=-normal
                        ja=np.zeros((3,model.nv));jb=ja.copy()
                        mujoco.mj_jac(model,data,ja,None,segment[:3],int(model.geom_bodyid[ga]));mujoco.mj_jac(model,data,jb,None,segment[3:],int(model.geom_bodyid[gb]))
                        add((dist-.004)*240,(normal@(jb-ja))[columns]*240,t)
                    else:add(0.,np.zeros(k),t)
            nonlinear=sparse.coo_matrix((vv,(ii,jj)),shape=(len(res),m*k)).tocsr()
            r=np.r_[reg@v,res];j=sparse.vstack([reg,nonlinear],format='csr')
            cache.update(x=v.copy(),r=r,j=j);timing['evaluate_calls']+=1;timing['evaluate_seconds']+=time.perf_counter()-start
            return r,j
        start=time.perf_counter()
        result=least_squares(lambda v:evaluate(v)[0],x,jac=lambda v:evaluate(v)[1],bounds=(lower.ravel(),upper.ravel()),max_nfev=max_nfev,ftol=2e-5,xtol=2e-5,gtol=1e-4,tr_options={'atol':1e-5,'btol':1e-5})
        if not np.isfinite(result.x).all():raise RuntimeError('Invalid optimization')
        x=result.x;q=q0.copy();q[np.ix_(rows,qcols)]+=x.reshape(m,k)
        new=set();maxdepth=0.
        for t in rows:
            data.qpos[:]=q[t];mujoco.mj_forward(model,data)
            for c in data.contact[:data.ncon]:
                if c.dist<-.0005:new.add(tuple(sorted([int(c.geom1),int(c.geom2)])));maxdepth=max(maxdepth,-float(c.dist))
        row={'round':outer,'success':bool(result.success),'message':result.message,'nfev':result.nfev,'cost':float(result.cost),'wall_s':time.perf_counter()-start,'max_self_penetration_m':maxdepth,'detected_pairs':len(new)}
        rounds.append(row);print(json.dumps(row),flush=True)
        if not new:break
        pairs|=new
    motion={'fps':fps,'root_pos':q[:,:3],'root_rot':q[:,[4,5,6,3]],'dof_pos':q[:,7:],'joint_names':list(G1_JOINT_NAMES),'root_quaternion_order':'xyzw','source':str(source)}
    pkl=output/'contact_refined.pkl'
    with pkl.open('wb') as f:pickle.dump(motion,f)
    conversion=convert(pkl,destination,output_fps=fps,ground_offset=0)
    measured=[]
    for c in contacts:
        values={}
        for label,trajectory in [('before',q0),('after',q)]:
            heights=[];xy=[]
            for t in c['frames']:
                data.qpos[:]=trajectory[t];mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data)
                heights.append(min(geometry_bottom(model,data,g)[0][2] for g in c['geoms']));xy.append(data.xpos[c['body_id'],:2].copy())
            xy=np.asarray(xy);speed=np.linalg.norm(np.diff(xy,axis=0)*fps,axis=1)
            values[label]={'height_max_abs_m':float(abs(np.array(heights)).max()),'height_mean_m':float(np.mean(heights)),'xy_speed_mean_m_s':float(speed.mean()) if len(speed) else 0.,'xy_speed_max_m_s':float(speed.max()) if len(speed) else 0.,'xy_max_anchor_error_m':float(np.linalg.norm(xy-c['target_xy'],axis=1).max())}
        measured.append({**c,**values})
    report={'source':str(source),'output':str(destination),'frames':n,'optimized_frames':m,'variables':m*k,'rounds':rounds,'performance':timing,'contacts':measured,'conversion':conversion,'max_joint_change_rad':float(abs(q[:,7:]-q0[:,7:]).max()),'max_root_translation_change_m':float(np.linalg.norm(q[:,:3]-q0[:,:3],axis=1).max()),'training_approved':False,'scope':'Kinematic contact candidate; inferred labels and physical feasibility require separate checks'}
    (output/'contact_report.json').write_text(json.dumps(report,indent=2));np.savez_compressed(output/'contact_corrections.npz',qpos_before=q0,qpos_after=q,fps=fps,optimized_frames=rows)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output-motion',type=Path,required=True);p.add_argument('--config',type=Path,default=ROOT/'configs/reference_full_contacts.json');p.add_argument('--report-dir',type=Path,default=ROOT/'outputs/contact_validation');p.add_argument('--max-nfev',type=int,default=100)
    a=p.parse_args();r=refine(a.input,a.output_motion,a.config,a.report_dir,a.max_nfev);print(json.dumps({k:v for k,v in r.items() if k!='contacts'},indent=2))
