"""Free-base physical reference tracking via torque-limited whole-body QP."""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from common import ROOT
from audit_dynamics import contact_force_jacobian


def load_reference(path):
    with np.load(path,allow_pickle=False) as f:d={k:f[k] for k in f.files}
    r=list(d['body_names']).index('pelvis')
    q=np.c_[d['body_pos_w'][:,r],d['body_quat_w'][:,r],d['joint_pos']].astype(float)
    v=np.c_[d['body_lin_vel_w'][:,r],Rotation.from_quat(q[:,[4,5,6,3]]).inv().apply(d['body_ang_vel_w'][:,r]),d['joint_vel']].astype(float)
    fps=float(d['fps'][0]);acc=np.gradient(v,1/fps,axis=0)
    return q,v,acc,fps,d


class WholeBodyController:
    def __init__(self,model,friction=.6):
        import mujoco
        self.m=model;self.nv=model.nv;self.dofs=model.jnt_dofadr[model.actuator_trnid[:,0]]
        self.qadr=model.jnt_qposadr[model.actuator_trnid[:,0]]
        self.kp=model.actuator_gainprm[:,0];self.kd=-model.actuator_biasprm[:,2]
        self.bounds=model.actuator_forcerange.copy()
        self.geoms=[g for g in range(model.ngeom) if (model.geom(g).name or '').startswith(('robot/left_foot','robot/right_foot','robot/left_hand','robot/right_hand')) and (model.geom_contype[g] or model.geom_conaffinity[g])]
        self.rays=np.array([[friction,0,1],[-friction,0,1],[0,friction,1],[0,-friction,1]]).T
        self.mass=np.zeros((self.nv,self.nv));self.diff=np.zeros(self.nv)

    def control(self,data,qr,vr,ar,mode='wbc'):
        import mujoco,daqp
        m=self.m;nv=self.nv;dofs=self.dofs
        mujoco.mj_differentiatePos(m,self.diff,1.,data.qpos,qr)
        if mode=='pd':return qr[self.qadr]+self.kd/self.kp*vr[dofs],{'solver':1}
        mujoco.mj_fullM(m,data,self.mass);h=data.qfrc_bias-data.qfrc_passive
        desired=ar+np.r_[np.full(3,80),np.full(3,100),np.full(29,120)]*self.diff+np.r_[np.full(3,18),np.full(3,20),np.full(29,22)]*(vr-data.qvel)
        pts=[]
        for c in data.contact[:data.ncon]:
            g1,g2=int(c.geom1),int(c.geom2)
            g=g1 if g1 in self.geoms else (g2 if g2 in self.geoms else None)
            if g is None:continue
            other=g2 if g==g1 else g1
            if (m.geom(other).name or '').startswith('robot/') or c.dist>.001:continue
            point=c.pos.copy();jac=np.zeros((3,nv));mujoco.mj_jac(m,data,jac,None,point,int(m.geom_bodyid[g]))
            pts.append((point,jac,g))
        # Limit contact variables to the geometrically lowest points on each link.
        selected=[]
        for body in sorted({int(m.geom_bodyid[g]) for _,_,g in pts}):
            group=sorted([x for x in pts if m.geom_bodyid[x[2]]==body],key=lambda x:x[0][2])
            selected.extend([x for x in group if x[0][2]<=group[0][0][2]+.003][:4])
        pts=selected;C=np.concatenate([j.T@self.rays for _,j,_ in pts],axis=1) if pts else np.zeros((nv,0));nf=C.shape[1];nx=nv+nf
        weights=np.r_[np.full(3,5.),np.full(3,5.),np.ones(29)]
        A=np.zeros((nv,nx));A[:,:nv]=np.diag(weights);b=desired*weights
        mats=[A];targets=[b]
        for point,jac,g in pts:
            jd=np.zeros((3,nv));mujoco.mj_jacDot(m,data,jd,None,point,int(m.geom_bodyid[g]))
            contact_desired=-jd@data.qvel-30*(jac@data.qvel)
            contact_desired[2]+=-300*point[2]
            A=np.zeros((3,nx));A[:,:nv]=jac*25;mats.append(A);targets.append(contact_desired*25)
        T=np.c_[self.mass[dofs],-C[dofs]]
        mats.append(T*.008);targets.append(-h[dofs]*.008)
        A=np.vstack(mats);b=np.concatenate(targets)
        H=A.T@A+np.diag(np.r_[np.full(nv,1e-5),np.full(nf,1e-5)]);f=-A.T@b
        equality=np.c_[self.mass[:6],-C[:6]]
        scale=np.maximum(np.linalg.norm(equality,axis=1),1.)
        constraints=np.vstack([equality/scale[:,None],T])
        lo=np.r_[np.full(nv,-1500.),np.zeros(nf),-h[:6]/scale,self.bounds[:,0]-h[dofs]]
        hi=np.r_[np.full(nv,1500.),np.full(nf,2000.),-h[:6]/scale,self.bounds[:,1]-h[dofs]]
        sense=np.zeros(nx+6+m.nu,dtype=np.int32);sense[nx:nx+6]=5
        x,obj,flag,info=daqp.solve(np.ascontiguousarray(H),np.ascontiguousarray(f),np.ascontiguousarray(constraints),hi,lo,sense,iter_limit=3000,primal_tol=1e-6)
        if flag!=1 or not np.isfinite(x).all():
            torque=np.clip(h[dofs]+20*self.diff[dofs]+2*(vr[dofs]-data.qvel[dofs]),self.bounds[:,0],self.bounds[:,1])
        else:torque=T@x+h[dofs]
        ctrl=(torque+self.kp*data.qpos[self.qadr]+self.kd*data.qvel[dofs])/self.kp
        limited=m.actuator_ctrllimited.astype(bool);ctrl[limited]=np.clip(ctrl[limited],m.actuator_ctrlrange[limited,0],m.actuator_ctrlrange[limited,1])
        return ctrl,{'solver':int(flag),'contacts':len(pts),'base_equation_error':float(np.max(abs(equality@x+h[:6]))) if flag==1 else None,'max_torque_ratio':float(np.max(abs(torque)/np.maximum(abs(self.bounds[:,0]),abs(self.bounds[:,1]))))}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--starts',type=float,nargs='+',default=[0,.9,4.2,7.7,9.6,10.8]);p.add_argument('--seconds',type=float,default=1.);p.add_argument('--mode',choices=['wbc','pd'],default='wbc');p.add_argument('--stop-on-failure',action='store_true');a=p.parse_args()
    import mujoco
    m=mujoco.MjModel.from_binary_path(str(ROOT/'outputs/physics_repair/g1_scene.mjb'));m.opt.timestep=.005
    q,v,acc,fps,meta=load_reference(a.input);controller=WholeBodyController(m)
    joint_names=[m.joint(i).name.removeprefix('robot/') for i in range(1,m.njnt)]
    if joint_names!=list(meta['joint_names']):raise ValueError('Joint ordering changed')
    results=[];allstates={};allrefs={};start=time.perf_counter()
    def ref(t):
        idx=np.clip(t*fps,0,len(q)-1);i=int(idx);j=min(i+1,len(q)-1);u=idx-i
        qr=q[i]*(1-u)+q[j]*u
        if q[i,3:7]@q[j,3:7]<0:qr[3:7]=q[i,3:7]*(1-u)-q[j,3:7]*u
        qr[3:7]/=np.linalg.norm(qr[3:7]);return qr,v[i]*(1-u)+v[j]*u,acc[i]*(1-u)+acc[j]*u
    for begin in a.starts:
        data=mujoco.MjData(m);qr,vr,_=ref(begin);data.qpos[:]=qr;data.qvel[:]=vr;mujoco.mj_forward(m,data)
        states=[data.qpos.copy()];refs=[qr.copy()];metrics=[];infos=[];failure=None
        for step in range(round(min(a.seconds,(len(q)-1)/fps-begin)/m.opt.timestep)):
            t=begin+step*m.opt.timestep;qr,vr,ar=ref(t)
            data.ctrl[:],info=controller.control(data,qr,vr,ar,a.mode);infos.append(info);mujoco.mj_step(m,data);mujoco.mj_forward(m,data)
            if (step+1)%4==0:
                qr,_,_=ref(t+m.opt.timestep);actual=data.qpos.copy();angle=2*np.arccos(np.clip(abs(actual[3:7]@qr[3:7]),0,1));err=float(np.linalg.norm(actual[:3]-qr[:3]));rmse=float(np.sqrt(np.mean((actual[7:]-qr[7:])**2)))
                states.append(actual);refs.append(qr.copy());metrics.append([err,float(angle),rmse])
                if failure is None and (err>.25 or angle>.8 or not np.isfinite(actual).all()):failure=(step+1)*m.opt.timestep
                if failure is not None and a.stop_on_failure:break
        mat=np.asarray(metrics);result={'start_s':begin,'first_failure_s':failure,'simulated_seconds':len(metrics)*.02,'max_root_error_m':float(mat[:,0].max()),'max_orientation_error_deg':float(np.rad2deg(mat[:,1].max())),'joint_rmse_rad':float(mat[:,2].mean()),'qp_failures':sum(x['solver']!=1 for x in infos),'qp_steps':len(infos),'max_torque_ratio':max((x.get('max_torque_ratio',0) for x in infos),default=0),'max_base_equation_error':max((x.get('base_equation_error') or 0 for x in infos),default=0)}
        results.append(result);allstates[str(begin)]=np.asarray(states);allrefs[str(begin)]=np.asarray(refs);print(result,flush=True)
    a.output.mkdir(parents=True,exist_ok=True)
    report={'input':str(a.input),'mode':a.mode,'backend':'CPU MuJoCo actual mjlab-compiled model','results':results,'wall_s':time.perf_counter()-start,'gravity':m.opt.gravity.tolist(),'root_state_writes_after_initialization':0,'external_forces':False,'failure_thresholds':{'root_error_m':.25,'orientation_error_rad':.8},'training_approved':False}
    (a.output/'tracking_report.json').write_text(json.dumps(report,indent=2));np.savez_compressed(a.output/'tracking_states.npz',**allstates,fps=50,joint_names=meta['joint_names']);np.savez_compressed(a.output/'tracking_references.npz',**allrefs,fps=50)
    print('wall_seconds',report['wall_s'])


if __name__=='__main__':main()
