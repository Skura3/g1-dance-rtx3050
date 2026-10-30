"""Audit every frame against the training G1 geometry; screening is not dynamics validation."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import numpy as np
from common import add_sources
from convert_motion import DEFAULT_XML

def intervals(mask,fps):
    ids=np.flatnonzero(mask)
    if not len(ids):return []
    groups=np.split(ids,np.flatnonzero(np.diff(ids)>1)+1)
    return [{'start_frame':int(g[0]),'end_frame':int(g[-1]),'start_s':float(g[0]/fps),'end_s':float(g[-1]/fps),'frames':len(g)} for g in groups]

def bottom(model,data,i):
    import mujoco
    kind=model.geom_type[i];size=model.geom_size[i];rot=data.geom_xmat[i].reshape(3,3)
    if kind==mujoco.mjtGeom.mjGEOM_SPHERE:radius=size[0]
    elif kind==mujoco.mjtGeom.mjGEOM_CAPSULE:radius=size[0]+abs(rot[2,2])*size[1]
    elif kind==mujoco.mjtGeom.mjGEOM_BOX:radius=np.abs(rot[2])@size
    elif kind==mujoco.mjtGeom.mjGEOM_CYLINDER:radius=abs(rot[2,2])*size[1]+size[0]*np.sqrt(max(0.,1-rot[2,2]**2))
    else:raise ValueError(f'Unsupported collision geometry: {model.geom(i).name} type={kind}')
    return data.geom_xpos[i,2]-radius

def audit(source,output,xml=DEFAULT_XML):
    add_sources()
    import mujoco
    from scipy.spatial.transform import Rotation
    from supermimic.data.motion_convert import load_g1_model,G1_JOINT_NAMES
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with np.load(source,allow_pickle=False) as f:motion={k:f[k] for k in f.files}
    fps=float(motion['fps'][0]);joint=motion['joint_pos'];n=len(joint)
    if list(motion['joint_names'])!=list(G1_JOINT_NAMES):raise ValueError('Joint order mismatch')
    if not all(np.isfinite(v).all() for v in motion.values() if np.issubdtype(v.dtype,np.number)):raise ValueError('Nonfinite motion')
    root=list(motion['body_names']).index('pelvis');pos=motion['body_pos_w'][:,root];quat=motion['body_quat_w'][:,root]
    qpos=np.column_stack([pos,quat,joint]);model=load_g1_model(xml);data=mujoco.MjData(model)
    collision=[i for i in range(model.ngeom) if model.geom_contype[i] or model.geom_conaffinity[i]]
    geom_names=[model.geom(i).name for i in collision]
    foot_indices=[[j for j,name in enumerate(geom_names) if name.startswith(side+'_foot')] for side in ('left','right')]
    nonfoot=[j for j in range(len(collision)) if j not in foot_indices[0]+foot_indices[1]]
    heights=np.empty((n,len(collision)));depth=np.zeros(n);pairs=Counter();contact_events=[]
    foot_xyz=np.empty((n,2,3))
    foot_bodies=[model.body(side+'_ankle_roll_link').id for side in ('left','right')]
    for t,q in enumerate(qpos):
        data.qpos[:]=q;mujoco.mj_forward(model,data)
        heights[t]=[bottom(model,data,i) for i in collision]
        foot_xyz[t]=data.xpos[foot_bodies]
        for c in data.contact[:data.ncon]:
            if c.dist>=-.001:continue
            a,b=int(c.geom1),int(c.geom2)
            if model.geom_bodyid[a]==0 or model.geom_bodyid[b]==0:continue
            pair=' / '.join(sorted([model.geom(a).name,model.geom(b).name]))
            pairs[pair]+=1;depth[t]=max(depth[t],-float(c.dist))
            contact_events.append({'frame':t,'time_s':t/fps,'pair':pair,'depth_m':-float(c.dist)})
    feet=np.column_stack([heights[:,i].min(axis=1) for i in foot_indices]);foot_zvel=np.gradient(feet,1/fps,axis=0)
    near=(feet<.025)&(feet>-.02)&(np.abs(foot_zvel)<.15)
    foot_speed=np.linalg.norm(np.gradient(foot_xyz,1/fps,axis=0)[...,:2],axis=-1)
    velocity=motion['joint_vel'];accel=np.gradient(velocity,1/fps,axis=0)
    limits=model.jnt_range[1:];violation=(joint<limits[:,0]-1e-4)|(joint>limits[:,1]+1e-4)
    r=Rotation.from_quat(quat[:,[1,2,3,0]]);angle=np.r_[0,(r[:-1].inv()*r[1:]).magnitude()]
    root_step=np.r_[0,np.linalg.norm(np.diff(pos,axis=0),axis=1)]
    max_vel=np.max(abs(velocity),axis=1);max_accel=np.max(abs(accel),axis=1)
    flags={'joint_limit':violation.any(axis=1),'feet_below_floor_2cm':feet.min(axis=1)<-.02,
           'nonfoot_below_floor_2cm':heights[:,nonfoot].min(axis=1)<-.02,'self_penetration_5mm':depth>.005,
           'speed_above_20rad_s':max_vel>20,'acceleration_above_300rad_s2':max_accel>300,
           'root_step_above_10cm':root_step>.1,'root_rotation_step_above_20deg':angle>np.deg2rad(20),
           'near_floor_foot_speed_above_30cm_s':(near&(foot_speed>.3)).any(axis=1)}
    flat=np.unravel_index(np.argmax(abs(velocity)),velocity.shape)
    report={'source':str(Path(source).resolve()),'frames':n,'fps':fps,'sample_span_s':(n-1)/fps,
      'finite':True,'quaternion_norm_max_error':float(abs(np.linalg.norm(motion['body_quat_w'],axis=-1)-1).max()),
      'foot_bottom_min_m':float(feet.min()),'lowest_foot_max_height_m':float(feet.min(axis=1).max()),
      'nonfoot_bottom_min_m':float(heights[:,nonfoot].min()),'maximum_self_penetration_m':float(depth.max()),
      'self_collision_pairs':dict(pairs),'max_joint_speed_rad_s':float(abs(velocity).max()),
      'max_speed_joint':str(motion['joint_names'][flat[1]]),'max_speed_time_s':flat[0]/fps,
      'max_joint_acceleration_rad_s2':float(abs(accel).max()),'max_root_step_m':float(root_step.max()),
      'max_root_rotation_step_deg':float(np.rad2deg(angle.max())),
      'root_height_min_max_m':[float(pos[:,2].min()),float(pos[:,2].max())],
      'screening_counts':{k:int(v.sum()) for k,v in flags.items()},'screening_intervals':{k:intervals(v,fps) for k,v in flags.items()},
      'training_approved':False,'kind':'kinematic_audit_not_dynamic_feasibility',
      'caveats':['Near-floor/slip masks are geometric heuristics, not verified contacts.',
                 'Speed and acceleration thresholds are review thresholds, not certified motor limits.',
                 'Self collisions use enabled training-model collision pairs, excluding XML exclusions.',
                 'A passing geometric audit does not establish balance, torque feasibility or policy tracking.']}
    np.savez_compressed(output/'quality_series.npz',fps=fps,foot_bottom=feet,foot_speed=foot_speed,near_floor=near,
        geom_bottom=heights,geom_names=np.array(geom_names),self_depth=depth,root_step=root_step,root_angle_step=angle,
        max_joint_speed=max_vel,max_joint_acceleration=max_accel,**flags)
    with (output/'frame_audit.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['frame','time_s','left_foot_bottom_m','right_foot_bottom_m','nonfoot_bottom_m','self_penetration_m','max_joint_speed_rad_s','max_joint_accel_rad_s2']+list(flags))
        for t in range(n):writer.writerow([t,t/fps,*feet[t],heights[t,nonfoot].min(),depth[t],max_vel[t],max_accel[t]]+[int(v[t]) for v in flags.values()])
    (output/'self_collision_events.json').write_text(json.dumps(contact_events,indent=2))
    (output/'quality.json').write_text(json.dumps(report,indent=2))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=audit(a.input,a.output);print(json.dumps({k:v for k,v in r.items() if k!='screening_intervals'},indent=2))
