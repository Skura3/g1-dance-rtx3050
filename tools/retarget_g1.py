"""Headless GVHMR -> GMR conversion; includes frame zero and preserves frame timing.

Optional branch recovery adapted from SuperMimic-Repro-Sim2Real (MIT), f8f60d7.
Copyright (c) 2026 Yuxi Liu; Copyright (c) 2026 Hongyu Ding.
Full license notice: docs/NJU_THIRD_PARTY_NOTICE.txt.
"""
import argparse
import json
import pickle
from pathlib import Path
from common import ROOT, CHECKPOINTS, add_sources, require_file

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=ROOT/'outputs/gvhmr/reference_pilot/hmr4d_results.pt')
    p.add_argument('--output',type=Path,default=ROOT/'data/motions/reference_pilot_gmr.pkl')
    p.add_argument('--body-models',type=Path,default=CHECKPOINTS/'body_models')
    p.add_argument('--body-batch',type=int,default=16)
    p.add_argument('--init-from-human',action='store_true',help='Seed pelvis pose and settle first-frame IK')
    p.add_argument('--restart-tol',type=float,default=0.,help='Optional competing waist/arm IK initialization; zero disables')
    a=p.parse_args();require_file(a.input);require_file(a.body_models/'smplx/SMPLX_NEUTRAL.npz')
    if a.body_batch<1:p.error('body-batch must be positive')
    add_sources()
    import numpy as np
    import torch
    import smplx
    import mujoco
    from smplx.joint_names import JOINT_NAMES
    from scipy.spatial.transform import Rotation
    from general_motion_retargeting import GeneralMotionRetargeting
    torch.set_num_threads(4)
    pred=torch.load(a.input,map_location='cpu',weights_only=False)['smpl_params_global']
    n=len(pred['body_pose']);betas=pred['betas'][:1].float()
    model=smplx.create(str(a.body_models),'smplx',gender='neutral',use_pca=False)
    height=1.66+0.1*float(betas[0,0])
    solver=GeneralMotionRetargeting(src_human='smplx',tgt_robot='unitree_g1',actual_human_height=height,verbose=False)
    names=JOINT_NAMES[:len(model.parents)]
    parents=model.parents.tolist()
    basis=Rotation.from_matrix([[1,0,0],[0,0,-1],[0,1,0]])
    qposes=[];switches=[]
    with torch.inference_mode():
        for start in range(0,n,a.body_batch):
            end=min(n,start+a.body_batch);count=end-start
            out=model(betas=betas.expand(count,-1),body_pose=pred['body_pose'][start:end].float(),
                global_orient=pred['global_orient'][start:end].float(),transl=pred['transl'][start:end].float(),
                expression=torch.zeros(count,model.num_expression_coeffs),
                left_hand_pose=torch.zeros(count,45),right_hand_pose=torch.zeros(count,45),
                jaw_pose=torch.zeros(count,3),leye_pose=torch.zeros(count,3),reye_pose=torch.zeros(count,3),return_full_pose=True)
            poses=out.full_pose.reshape(count,-1,3).numpy();joints=out.joints.numpy()
            for t in range(count):
                rotations=[];human={}
                for j,name in enumerate(names):
                    r=Rotation.from_rotvec(poses[t,j])
                    if parents[j]>=0:r=rotations[parents[j]]*r
                    rotations.append(r)
                    human[name]=(basis.apply(joints[t,j]),(basis*r).as_quat(scalar_first=True))
                if not qposes and a.init_from_human:
                    seed=solver.configuration.data.qpos.copy();seed[:3]=human['pelvis'][0]
                    seed[3:7]=(Rotation.from_quat(human['pelvis'][1],scalar_first=True)*solver.rot_offsets1['pelvis']).as_quat(scalar_first=True)
                    solver.configuration.update(seed)
                    for _ in range(20):solver.retarget(human,offset_to_ground=False)
                best=solver.retarget(human,offset_to_ground=False).copy()
                if a.restart_tol>0:
                    error=solver.error1()+solver.error2();seed=best.copy();seed[19:]=0.;solver.configuration.update(seed)
                    for _ in range(30):candidate=solver.retarget(human,offset_to_ground=False).copy()
                    if solver.error1()+solver.error2()<error-a.restart_tol:
                        best=candidate;switches.append(len(qposes))
                    solver.configuration.update(best)
                qposes.append(best)
            print(f'Retargeted {end}/{n}',flush=True)
    q=np.asarray(qposes)
    if q.shape!=(n,36) or not np.isfinite(q).all():raise RuntimeError('Invalid G1 trajectory')
    raw_q=q.copy()
    for index in switches:
        if index==0:continue
        changed=np.flatnonzero(abs(raw_q[index,7:]-raw_q[index-1,7:])>.3)+7
        left=max(0,index-5);right=min(n-1,index+5)
        weight=(.5-.5*np.cos(np.linspace(0,np.pi,right-left+1)))[:,None]
        q[np.ix_(np.arange(left,right+1),changed)]=(1-weight)*raw_q[left,changed]+weight*raw_q[right,changed]
    joint_names=[mujoco.mj_id2name(solver.model,mujoco.mjtObj.mjOBJ_JOINT,j) for j in range(1,solver.model.njnt)]
    motion={'fps':30.,'root_pos':q[:,:3],'root_rot':q[:,[4,5,6,3]],'dof_pos':q[:,7:],
            'joint_names':joint_names,'root_quaternion_order':'xyzw','source':str(a.input),'ground_alignment':'none'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('wb') as f:pickle.dump(motion,f)
    a.output.with_suffix('.json').write_text(json.dumps({'frames':n,'fps':30,'source':str(a.input),'joint_names':joint_names,'init_from_human':a.init_from_human,'restart_tolerance':a.restart_tol,'restart_frames':switches,'branch_recovery_method_source':'NJU-R-L-Group-Embodied-Lab/SuperMimic-Repro-Sim2Real tools/video2motion/gvhmr_to_supermimic.py, f8f60d7','max_branch_blend_change_rad':float(abs(q[:,7:]-raw_q[:,7:]).max())},indent=2)+'\n')
    print(a.output)

if __name__=='__main__':main()
