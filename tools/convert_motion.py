"""GMR pickle -> training NPZ with fixed ground offset and exact G1 kinematics."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
from common import ROOT, MJLAB, add_sources, require_file

DEFAULT_XML=MJLAB/'src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml'

def capsule_bottoms(model,qpos):
    import mujoco
    ids=[i for i in range(model.ngeom) if model.geom(i).name.startswith(('left_foot','right_foot')) and model.geom(i).name.endswith('_collision')]
    if not ids:raise ValueError('No foot collision geometry')
    if any(model.geom_type[i]!=mujoco.mjtGeom.mjGEOM_CAPSULE for i in ids):raise ValueError('Review changed foot geometry')
    data=mujoco.MjData(model);heights=[]
    for frame in qpos:
        data.qpos[:]=frame;mujoco.mj_kinematics(model,data)
        z=[data.geom_xpos[i,2]-abs(data.geom_xmat[i].reshape(3,3)[2,2])*model.geom_size[i,1]-model.geom_size[i,0] for i in ids]
        heights.append(min(z))
    return np.array(heights)

def align_height(root,offset):
    aligned=np.array(root,dtype=np.float64,copy=True);aligned[:,2]+=float(offset)
    return aligned

def convert(input_path,output_path,xml=DEFAULT_XML,output_fps=50.,ground_offset=None,support_range=None):
    add_sources()
    from supermimic.data.motion_convert import G1_JOINT_NAMES,load_g1_model,resample_motion,forward_kinematics
    if (ground_offset is None)==(support_range is None):raise ValueError('Specify fixed ground offset or verified support interval')
    if output_fps<=0:raise ValueError('FPS must be positive')
    # Load only trusted local GMR results: pickle is not safe for unknown files.
    with require_file(input_path).open('rb') as f:motion=pickle.load(f)
    fps=float(motion['fps']);names=motion.get('joint_names')
    if fps<=0:raise ValueError('Source FPS must be positive')
    if names is None or len(names)!=29 or set(names)!=set(G1_JOINT_NAMES):raise ValueError('Explicit G1 joint_names required; use retarget_g1.py')
    if motion.get('root_quaternion_order')!='xyzw':raise ValueError('Explicit xyzw quaternion metadata required')
    pos=np.asarray(motion['root_pos'],dtype=float);quat=np.asarray(motion['root_rot'],dtype=float)
    joint=np.asarray(motion['dof_pos'],dtype=float)[:,[names.index(n) for n in G1_JOINT_NAMES]]
    n=len(pos)
    if n<3 or pos.shape!=(n,3) or quat.shape!=(n,4) or joint.shape!=(n,29):raise ValueError('Invalid trajectory dimensions')
    if not all(np.isfinite(v).all() for v in (pos,quat,joint)):raise ValueError('Non-finite motion data')
    norm=np.linalg.norm(quat,axis=1)
    if np.any(norm<1e-6):raise ValueError('Zero quaternion')
    quat=quat/norm[:,None];model=load_g1_model(require_file(xml))
    raw=np.column_stack([pos,quat,joint]);qpos=np.column_stack([pos,quat[:,[3,0,1,2]],joint])
    bottoms=capsule_bottoms(model,qpos)
    if support_range is not None:
        start,end=map(float,support_range);t=np.arange(n)/fps
        if not 0<=start<end<=n/fps:raise ValueError('Invalid support interval')
        mask=(t>=start)&(t<end)
        if mask.sum()<3:raise ValueError('Support interval needs three frames')
        ground_offset=-float(np.median(bottoms[mask]))
    raw[:,:3]=align_height(pos,ground_offset)
    rs=resample_motion(raw,fps,output_fps);bs=forward_kinematics(model,rs)
    out=Path(output_path);out.parent.mkdir(parents=True,exist_ok=True)
    if out.suffix!='.npz':raise ValueError('Output must end with .npz')
    arrays=dict(fps=np.array([output_fps],np.float32),joint_pos=rs.joint_pos.astype(np.float32),joint_vel=rs.joint_vel.astype(np.float32),
        body_pos_w=bs.pos_w.astype(np.float32),body_quat_w=bs.quat_w.astype(np.float32),body_lin_vel_w=bs.lin_vel_w.astype(np.float32),
        body_ang_vel_w=bs.ang_vel_w.astype(np.float32),joint_names=np.array(G1_JOINT_NAMES),body_names=np.array(bs.body_names),
        source=np.array(str(Path(input_path).resolve())),input_fps=np.array([fps],np.float32),ground_offset_m=np.array([ground_offset],np.float32),
        model_xml_sha256=np.array(hashlib.sha256(Path(xml).read_bytes()).hexdigest()))
    for key,value in arrays.items():
        if np.issubdtype(value.dtype,np.number) and not np.isfinite(value).all():raise ValueError(f'Non-finite output: {key}')
    lim=model.jnt_range[1:];bad=(joint<lim[:,0]-1e-4)|(joint>lim[:,1]+1e-4)
    report={'source':str(input_path),'output':str(out),'frames_in':n,'frames_out':rs.num_frames,'input_fps':fps,'output_fps':output_fps,
        'duration_s':(rs.num_frames-1)/output_fps,'fixed_ground_offset_m':float(ground_offset),'support_range_s':support_range,
        'foot_min_z_before_m':float(bottoms.min()),'foot_min_z_after_m':float(bottoms.min()+ground_offset),
        'frames_with_foot_penetration_gt_2cm':int(np.sum(bottoms+ground_offset<-.02)),
        'frames_with_joint_limit_violations':int(bad.any(axis=1).sum()),'max_abs_joint_velocity_rad_s':float(np.max(np.abs(rs.joint_vel))),
        'root_height_range_m':float(np.ptp(pos[:,2])),'kind':'kinematic_reference_not_physics_rollout',
        'note':'Inspect slips and contacts. No clipping or per-frame grounding applied.'}
    with out.with_suffix('.partial').open('wb') as f:np.savez(f,**arrays)
    out.with_suffix('.partial').replace(out);out.with_suffix('.report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=ROOT/'data/motions/reference_pilot_gmr.pkl')
    p.add_argument('--output',type=Path,default=ROOT/'data/npz/pilot/reference_pilot.npz')
    p.add_argument('--xml',type=Path,default=DEFAULT_XML);p.add_argument('--fps',type=float,default=50)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--ground-offset',type=float,help='Fixed vertical translation in metres; zero for unaligned preview')
    g.add_argument('--support-range',type=float,nargs=2,metavar=('START','END'),help='Verified support interval, seconds')
    a=p.parse_args();print(json.dumps(convert(a.input,a.output,a.xml,a.fps,a.ground_offset,a.support_range),indent=2))

if __name__=='__main__':main()
