"""Produce a separate, auditable kinematic candidate; never certify physical feasibility."""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from common import add_sources
from convert_motion import DEFAULT_XML,convert
from audit_motion import bottom

def height_correction(sole,all_bottom,anchor_ids,fps):
    """Smooth sparse-anchor correction, with a floor inequality and no per-frame grounding."""
    n=len(sole);eye=sparse.eye(n,format='csr')
    d1=sparse.diags([-np.ones(n-1),np.ones(n-1)],[0,1],shape=(n-1,n),format='csr')
    d2=sparse.diags([np.ones(n-2),-2*np.ones(n-2),np.ones(n-2)],[0,1,2],shape=(n-2,n),format='csr')
    # Dense smoothness is expressed in physical time, not arbitrary frame counts.
    a=sparse.vstack([eye[anchor_ids]*30,d2*fps**1.5,d1*2,eye*.01],format='csr')
    b=np.r_[(.002-sole[anchor_ids])*30,np.zeros(n-2+n-1+n)]
    lower=-all_bottom-.001
    from qpsolvers import solve_qp
    x=solve_qp((a.T@a).toarray(),-np.asarray(a.T@b).ravel(),lb=lower,ub=np.maximum(0.,lower+1e-8),solver='daqp')
    if x is None or not np.isfinite(x).all() or np.any(x<lower-1e-6):raise RuntimeError('Height correction QP failed')
    return x,{'solver':'DAQP bounded convex least squares','cost':float(np.sum((a@x-b)**2)/2),'anchor_residual_m':(sole[anchor_ids]+x[anchor_ids]-.002).tolist()}

def repair(source,destination,annotations,output):
    add_sources();import mujoco
    from supermimic.data.motion_convert import load_g1_model,G1_JOINT_NAMES
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if Path(source).resolve()==Path(destination).resolve():raise ValueError('Repair must write a separate output, preserving the original')
    with np.load(source,allow_pickle=False) as f:d={k:f[k] for k in f.files}
    cfg=json.loads(Path(annotations).read_text());fps=float(d['fps'][0]);names=list(d['joint_names']);root=list(d['body_names']).index('pelvis')
    if names!=list(G1_JOINT_NAMES):raise ValueError('Joint names/order differ from training G1')
    pos=d['body_pos_w'][:,root].astype(float);quat=d['body_quat_w'][:,root].astype(float);joint=d['joint_pos'].astype(float);original=joint.copy();n=len(joint)
    model=load_g1_model(DEFAULT_XML);data=mujoco.MjData(model);geoms=[i for i in range(model.ngeom) if model.geom_contype[i] or model.geom_conaffinity[i]]
    feet=[j for j,i in enumerate(geoms) if model.geom(i).name.startswith(('left_foot','right_foot'))]
    qpos=np.column_stack([pos,quat,joint]);heights=[]
    for q in qpos:
        data.qpos[:]=q;mujoco.mj_forward(model,data);heights.append([bottom(model,data,i) for i in geoms])
    heights=np.array(heights);anchors=np.rint(np.array(cfg['support_frames'])/cfg['fps']*fps).astype(int)
    if not len(anchors) or np.any(anchors<0) or np.any(anchors>=n):raise ValueError('Support annotations outside motion')
    shift,ground_report=height_correction(heights[:,feet].min(axis=1),heights.min(axis=1),anchors,fps)
    pos[:,2]+=shift;qpos[:,2]+=shift
    # Only the two identified shoulder branch transitions receive a short taper.
    smoothing=[];time=np.arange(n)/fps
    for w in cfg.get('shoulder_transition_windows_s',[]):
        side=w['joint'].split('_')[0];mask=(time>=w['start'])&(time<=w['end']);u=(time[mask]-w['start'])/(w['end']-w['start'])
        weight=np.sin(np.pi*u)**2
        for j,name in enumerate(names):
            if not name.startswith(side+'_shoulder_'):continue
            filtered=savgol_filter(joint[:,j],15,2)
            change=np.clip(filtered[mask]-joint[mask,j],-.35,.35)*weight
            joint[mask,j]+=change;smoothing.append({**w,'joint':name,'max_change_rad':float(abs(change).max())})
    qpos[:,7:]=joint
    arm_cols=np.array([j for j,name in enumerate(names) if any(k in name for k in ('shoulder','elbow','wrist'))])
    rounds=[]
    for round_id in range(3):
        contacts={};active_frames=[]
        for t,q in enumerate(qpos):
            data.qpos[:]=q;mujoco.mj_forward(model,data)
            for c in data.contact[:data.ncon]:
                if c.dist>=-.0005:continue
                pair=tuple(sorted([int(c.geom1),int(c.geom2)]))
                contacts.setdefault(pair,[]).append(t);active_frames.append(t)
        if not active_frames:break
        mask=np.zeros(n,bool)
        for t in active_frames:mask[max(0,t-9):min(n,t+10)]=True
        rows=np.flatnonzero(mask);lookup={t:i for i,t in enumerate(rows)};m=len(rows);k=len(arm_cols)
        pairs=list(contacts);constraints=[]
        for pair,frames in contacts.items():
            check=set()
            for t in frames:check.update(range(max(0,t-9),min(n,t+10)))
            constraints.extend((t,pair) for t in sorted(check))
        rows_base=joint[rows][:,arm_cols].copy();base_q=qpos.copy()
        # Sparse temporal regularization smooths corrections, not the original dance.
        c1=sparse.lil_matrix((n-1,m));c2=sparse.lil_matrix((n-2,m))
        for t,r in lookup.items():
            if t>0:c1[t-1,r]=1
            if t<n-1:c1[t,r]=-1
            if t>=2:c2[t-2,r]=1
            if 1<=t<n-1:c2[t-1,r]=-2
            if t<n-2:c2[t,r]=1
        reg=sparse.vstack([sparse.eye(m*k),sparse.kron(c1,sparse.eye(k))*2,sparse.kron(c2,sparse.eye(k))*5],format='csr')
        pattern=sparse.lil_matrix((len(constraints),m*k))
        for i,(t,pair) in enumerate(constraints):pattern[i,lookup[t]*k:(lookup[t]+1)*k]=1
        jac=sparse.vstack([reg.astype(bool),pattern.tocsr()],format='csr')
        def residual(x):
            delta=x.reshape(m,k);penalties=[];last=-1
            for t,pair in constraints:
                if t!=last:
                    q=base_q[t].copy();q[7+arm_cols]=rows_base[lookup[t]]+delta[lookup[t]]
                    data.qpos[:]=q;mujoco.mj_kinematics(model,data);last=t
                dist=mujoco.mj_geomDistance(model,data,*pair,1.,None)
                penalties.append(max(0.,.006-dist)*180)
            return np.r_[reg@x,penalties]
        lower=np.maximum(model.jnt_range[1+arm_cols,0]-rows_base,-.65).ravel()
        upper=np.minimum(model.jnt_range[1+arm_cols,1]-rows_base,.65).ravel()
        result=least_squares(residual,np.zeros(m*k),bounds=(lower+1e-10,upper-1e-10),jac_sparsity=jac,
            diff_step=1e-4,ftol=1e-4,xtol=1e-4,gtol=1e-3,max_nfev=200)
        if not np.isfinite(result.x).all():raise RuntimeError('Nonfinite collision correction')
        joint[np.ix_(rows,arm_cols)]=rows_base+result.x.reshape(m,k);qpos[:,7:]=joint
        rounds.append({'round':round_id,'frames':m,'evaluations':result.nfev,'success':bool(result.success),'message':result.message,'max_delta_rad':float(abs(result.x).max()),'cost':float(result.cost),
            'pairs':[[model.geom(i).name for i in pair] for pair in pairs]})
        print(json.dumps(rounds[-1]),flush=True)
    motion={'fps':fps,'root_pos':pos,'root_rot':quat[:,[1,2,3,0]],'dof_pos':joint,'joint_names':names,
        'root_quaternion_order':'xyzw','source':str(source),'repair':'sparse_support_drift_and_local_arm_collision_candidate'}
    pkl=output/'repaired_gmr.pkl'
    with pkl.open('wb') as f:pickle.dump(motion,f)
    converted=convert(pkl,destination,output_fps=fps,ground_offset=0)
    report={'source':str(source),'output':str(destination),'annotations':str(annotations),'frames':n,
        'ground':ground_report,'max_vertical_correction_m':float(abs(shift).max()),'vertical_correction_range_m':[float(shift.min()),float(shift.max())],
        'smoothing':smoothing,'collision_solver_rounds':rounds,'max_joint_change_rad':float(abs(joint-original).max()),
        'joint_change_rms_rad':float(np.sqrt(np.mean((joint-original)**2))),'conversion':converted,
        'training_approved':False,'note':'Kinematic candidate only. Sparse manual support observations are approximate; horizontal drift, hand-floor contact and dynamics remain to be checked.'}
    np.savez_compressed(output/'corrections.npz',fps=fps,vertical_shift=shift,joint_delta=joint-original,anchor_frames=anchors)
    (output/'repair_report.json').write_text(json.dumps(report,indent=2));return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output-motion',type=Path,required=True)
    p.add_argument('--annotations',type=Path,required=True);p.add_argument('--report-dir',type=Path,required=True);a=p.parse_args()
    print(json.dumps(repair(a.input,a.output_motion,a.annotations,a.report_dir),indent=2))
