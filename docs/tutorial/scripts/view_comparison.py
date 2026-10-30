"""Tutorial comparison viewer, generalized from tools/view_best_replay.py.
Uses an explicit scene file and recorded duration, with checkpoint provenance validation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from queue import SimpleQueue
import time

import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tools'))


class ReplayClock:
    def __init__(self, now):
        self.anchor=now;self.position=0.;self.paused=False

    def elapsed(self, now):
        return self.position if self.paused else self.position+now-self.anchor

    def toggle(self, now):
        self.position=self.elapsed(now);self.anchor=now;self.paused=not self.paused

    def reset(self, now):
        self.position=0.;self.anchor=now


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--status',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True,help='g1_scene.mjb from the same robot evaluation')
    a=p.parse_args()
    os.environ['MUJOCO_GL']='glfw'
    import mujoco
    import mujoco.viewer
    import numpy as np
    from training_supervisor import atomic_json

    report=json.loads(a.report.read_text());clip=report['clips'][0]
    digest=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    if digest!=report['checkpoint_sha256']:raise ValueError('Checkpoint hash differs from evaluated rollout')
    if clip['failed']:raise ValueError('This comparison requires a completed physical rollout; inspect failed trials via their rendered video')
    states=Path(clip['states'])
    with np.load(states,allow_pickle=False) as d:
        qpos=d['qpos'].copy();qref=d['qref'].copy();fps=float(d['fps'].item());names=d['joint_names'].tolist()
    if qpos.shape!=qref.shape or not np.isfinite(qpos).all() or not np.isfinite(qref).all():
        raise ValueError('Invalid recorded trajectory')
    duration=(len(qpos)-1)/fps
    if abs(duration-clip['simulated_seconds'])>1e-6:raise ValueError('Duration differs from recorded simulation')
    if len(qpos)-1 != clip['steps'] or not np.isfinite(fps) or fps <= 0:
        raise ValueError('Frame count or FPS does not match the report')
    a.status.parent.mkdir(parents=True,exist_ok=True)
    model=mujoco.MjModel.from_binary_path(str(a.model.resolve()))
    expected=[model.joint(i).name.removeprefix('robot/') for i in range(1,model.njnt)]
    if expected!=names or model.nq!=qpos.shape[1]:raise ValueError('Robot model / trajectory mismatch')
    model.vis.global_.fovy=45.
    data=mujoco.MjData(model);reference=mujoco.MjData(model)
    option=mujoco.MjvOption();option.geomgroup[3:]=0;pert=mujoco.MjvPerturb()
    w,x,y,z=qref[0,3:7]
    heading=float(np.degrees(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))))
    azimuth=heading+155.;angle=np.radians(azimuth)
    offset=.95*np.array([np.sin(angle),-np.cos(angle),0.])
    iteration=report['checkpoint_iteration'];keys=SimpleQueue()
    clock=ReplayClock(time.perf_counter());last_publish=0.;last_loop=0;last_overlay=0.
    shown=0;stats_start=time.perf_counter();render_fps=0.;last_elapsed=0.;last_stats=stats_start;measured_speed=None
    previous_index=None;loops=0;index=0;last_loop_wall=stats_start;last_loop_seconds=None
    source_sha=hashlib.sha256(states.read_bytes()).hexdigest()

    def publish(state):
        atomic_json(a.status,{'status':state,'pid':os.getpid(),'mode':'recorded_physics_trajectory_in_native_mujoco_viewer',
            'checkpoint_iteration':iteration,'checkpoint_sha256':digest,'states_sha256':source_sha,
            'source_report':str(a.report.resolve()),'source_states':str(states.resolve()),
            'recorded_fps':fps,'duration_seconds':duration,'frame':index,'clip_seconds':index/fps,
            'completed_loops':loops,'paused':clock.paused,'playback_speed':1.,'measured_timeline_speed':measured_speed,
            'last_loop_wall_seconds':last_loop_seconds,'render_fps':render_fps,
            'timing':'Wall-clock 1x; original sample timestamps, no retiming; delayed display skips frames instead of slowing down.',
            'live_policy_inference':False,'records_video':False,'training_restarted':False,
            'updated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')})

    print(f'Opening native 1x replay | checkpoint {iteration} | {duration:.2f} seconds per loop',flush=True)
    try:
        with mujoco.viewer.launch_passive(model,data,key_callback=keys.put,show_left_ui=False,show_right_ui=False) as viewer:
            with viewer.lock():
                viewer.cam.type=mujoco.mjtCamera.mjCAMERA_FREE
                viewer.cam.distance=4.6;viewer.cam.azimuth=azimuth;viewer.cam.elevation=-12.
                viewer.opt.geomgroup[3:]=0;viewer.opt.frame=mujoco.mjtFrame.mjFRAME_NONE
            start=time.perf_counter();clock.reset(start);last_loop_wall=start;last_stats=start;stats_start=start
            deadline=start
            while viewer.is_running():
                now=time.perf_counter()
                while not keys.empty():
                    key=keys.get()
                    if key==32:clock.toggle(now)
                    elif key in (257,259):
                        clock.reset(now);last_loop=0;loops=0;last_loop_wall=now;last_loop_seconds=None
                        previous_index=None
                    last_stats=now;last_elapsed=clock.elapsed(now);measured_speed=None
                elapsed=clock.elapsed(now);loops=int(elapsed/duration)
                phase=elapsed%duration;index=min(int(phase*fps),len(qpos)-2)
                if loops!=last_loop:
                    last_loop_seconds=(now-last_loop_wall)/(loops-last_loop)
                    print(f'Loop {loops}: {duration:.2f}s motion / {last_loop_seconds:.3f}s wall; original-speed replay',flush=True)
                    last_loop=loops;last_loop_wall=now
                with viewer.lock():
                    if index!=previous_index:
                        data.qpos[:]=qpos[index];data.qpos[:3]+=offset
                        reference.qpos[:]=qref[index];reference.qpos[:3]-=offset
                        data.qvel[:]=0.;reference.qvel[:]=0.
                        mujoco.mj_forward(model,data);mujoco.mj_forward(model,reference)
                        previous_index=index
                    viewer.user_scn.ngeom=0
                    mujoco.mjv_addGeoms(model,reference,option,pert,mujoco.mjtCatBit.mjCAT_DYNAMIC.value,viewer.user_scn)
                    viewer.cam.lookat[:]=[qref[index,0],qref[index,1],.85]
                    for label,pos,color in [('REFERENCE / NO POLICY',reference.qpos[:3],(.25,.75,1.,1.)),
                        (f'POLICY {iteration} / REPLAY',data.qpos[:3],(1.,.85,.3,1.))]:
                        geom=viewer.user_scn.geoms[viewer.user_scn.ngeom];viewer.user_scn.ngeom+=1
                        mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_LABEL.value,np.zeros(3),
                            np.asarray(pos)+[0,0,1.05],np.eye(3).ravel(),np.asarray(color,dtype=np.float32))
                        geom.label=label
                if now-last_overlay>.15:
                    title=chr(10).join(['ORIGINAL SPEED 1x','Verified physics replay','Controls'])
                    values=chr(10).join([f'{phase:.2f} / {duration:.2f} s | '+('PAUSED' if clock.paused else 'PLAYING'),
                        f'Policy {iteration} | LEFT reference / RIGHT policy','Space pause | Enter restart | Mouse rotate/zoom'])
                    viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100.value,mujoco.mjtGridPos.mjGRID_TOPLEFT.value,title,values))
                    last_overlay=now
                viewer.sync(state_only=True);shown+=1
                if now-stats_start>=2:
                    render_fps=shown/(now-stats_start);shown=0;stats_start=now
                    measured_speed=(elapsed-last_elapsed)/(now-last_stats) if now>last_stats else None
                    last_elapsed=elapsed;last_stats=now
                if now-last_publish>=2:
                    publish('paused' if clock.paused else 'playing');last_publish=now
                deadline+=1/fps
                if deadline<time.perf_counter()-1/fps:deadline=time.perf_counter()
                time.sleep(max(0.,deadline-time.perf_counter()))
    finally:
        publish('closed');time.sleep(.5)


if __name__=='__main__':main()
