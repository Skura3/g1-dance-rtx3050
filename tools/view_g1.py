"""Replay verified G1 states in a local MuJoCo viewer; no policy training."""
import argparse
from pathlib import Path
import time
from common import ROOT, add_sources, require_file
from convert_motion import DEFAULT_XML

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=ROOT/'outputs/environment_validation/stand_states.npz')
    p.add_argument('--once',action='store_true')
    p.add_argument('--speed',type=float,default=1.)
    a=p.parse_args()
    if a.speed<=0:p.error('--speed must be positive')
    import numpy as np
    import mujoco
    import mujoco.viewer
    spec=mujoco.MjSpec.from_file(str(DEFAULT_XML))
    spec.worldbody.add_geom(name='viewer_ground',type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[20,20,.1],rgba=[.3,.38,.45,1.])
    spec.worldbody.add_light(pos=[0,-3,5],dir=[0,.4,-1])
    spec.stat.extent=2.
    model=spec.compile();data=mujoco.MjData(model)
    names=[model.joint(i).name.removeprefix('robot/') for i in range(1,model.njnt)]
    with np.load(require_file(a.input),allow_pickle=False) as d:
        if list(d['joint_names'])!=names:raise ValueError('Reference joint names/order differ')
        fps=float(d['fps'][0])
        if 'qpos' in d:
            qposes=d['qpos'].copy()
            print('Replaying recorded simulation states; this viewer does not rerun control.')
        else:
            root=list(d['body_names']).index('pelvis')
            qposes=np.concatenate([d['body_pos_w'][:,root],d['body_quat_w'][:,root],d['joint_pos']],axis=1)
            print('Kinematic reference playback; not learned-policy tracking.')
    print('Close the viewer window to stop. Mouse drag rotates camera; wheel zooms.',flush=True)
    with mujoco.viewer.launch_passive(model,data) as viewer:
        w,x,y,z=qposes[0,3:7]
        heading=np.degrees(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))
        viewer.cam.distance=2.8;viewer.cam.azimuth=float(heading+155.);viewer.cam.elevation=-12.
        viewer.opt.geomgroup[3:]=0
        start=time.monotonic()
        while viewer.is_running():
            index=int((time.monotonic()-start)*fps*a.speed)
            if a.once and index>=len(qposes):break
            q=qposes[index%len(qposes)];data.qpos[:]=q;mujoco.mj_forward(model,data)
            viewer.cam.lookat[:]=[q[0],q[1],.8];viewer.sync();time.sleep(.005)
    # The passive viewer closes asynchronously; let its render thread finish
    # before Python destroys model and OpenGL objects.
    time.sleep(1.)
    print('Viewer closed cleanly.',flush=True)

if __name__=='__main__':main()
