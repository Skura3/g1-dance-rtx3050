"""Render a reference trajectory. This is explicitly kinematics, never a policy rollout."""
import argparse
import os
from pathlib import Path
from common import ROOT,require_file
from convert_motion import DEFAULT_XML

def render(motion_path,output_path,xml=DEFAULT_XML,fps=30,width=640,height=480):
    os.environ.setdefault('MUJOCO_GL','egl')
    import mujoco
    import numpy as np
    import cv2
    import imageio.v2 as imageio
    data_in=np.load(require_file(motion_path),allow_pickle=False)
    spec=mujoco.MjSpec.from_file(str(require_file(xml)))
    spec.worldbody.add_geom(name='preview_ground',type=mujoco.mjtGeom.mjGEOM_PLANE,size=[20,20,.1],rgba=[.65,.68,.72,1])
    spec.worldbody.add_light(pos=[0,-3,5],dir=[0,.4,-1])
    spec.stat.extent=2.0
    model=spec.compile();data=mujoco.MjData(model)
    names=[model.joint(i).name for i in range(1,model.njnt)]
    if list(data_in['joint_names'])!=names:raise ValueError('Reference and renderer joint order differ')
    body_names=list(data_in['body_names']);root=body_names.index('pelvis')
    n=len(data_in['joint_pos']);source_fps=float(data_in['fps'][0])
    ids=np.minimum((np.arange(int((n-1)/source_fps*fps)+1)*source_fps/fps).round().astype(int),n-1)
    output_path=Path(output_path);output_path.parent.mkdir(parents=True,exist_ok=True)
    camera=mujoco.MjvCamera();camera.distance=3.;camera.azimuth=135.;camera.elevation=-15.
    renderer=mujoco.Renderer(model,height=height,width=width)
    try:
        with imageio.get_writer(str(output_path),fps=fps,codec='libx264',macro_block_size=1) as writer:
            for i in ids:
                data.qpos[:3]=data_in['body_pos_w'][i,root]
                data.qpos[3:7]=data_in['body_quat_w'][i,root]
                data.qpos[7:]=data_in['joint_pos'][i]
                mujoco.mj_forward(model,data)
                camera.lookat[:]=[data.qpos[0],data.qpos[1],.8]
                renderer.update_scene(data,camera=camera);frame=renderer.render().copy()
                cv2.putText(frame,'KINEMATIC REFERENCE - NOT PHYSICS TRACKING',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1)
                cv2.putText(frame,f'{i/source_fps:.2f} s',(8,43),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
                writer.append_data(frame)
    finally:renderer.close();data_in.close()
    return output_path

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--xml',type=Path,default=DEFAULT_XML)
    a=p.parse_args();print(render(a.input,a.output,a.xml))
