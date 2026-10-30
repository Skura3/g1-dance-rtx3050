"""G1 single-world mjlab validation: real PD standing or kinematic reference."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
from common import ROOT, add_sources, require_file

def render_frames(model, qposes, destination, fps, title):
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw
    width, height = 800, 600
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    model.vis.quality.shadowsize = 2048
    renderer = mujoco.Renderer(model, height=height, width=width)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    w,x,y,z=qposes[0,3:7]
    heading=np.degrees(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))
    camera.distance = 3.3; camera.azimuth = float(heading+155.); camera.elevation = -12.
    option = mujoco.MjvOption(); option.geomgroup[3:] = 0
    proc = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}',
        '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(destination)], stdin=subprocess.PIPE)
    try:
        for i, q in enumerate(qposes):
            data.qpos[:] = q; mujoco.mj_forward(model, data)
            camera.lookat[:] = [q[0], q[1], .9]
            renderer.update_scene(data, camera=camera, scene_option=option)
            im = Image.fromarray(renderer.render().copy()); draw = ImageDraw.Draw(im)
            draw.rectangle((0, 0, width, 49), fill=(20, 26, 35))
            draw.text((12, 8), title, fill='white')
            draw.text((12, 28), f'Time: {i/fps:.2f}s | 29 joints | RTX 3050 | mjlab / MuJoCo Warp', fill='white')
            proc.stdin.write(np.asarray(im).tobytes())
            if i in (0, len(qposes)//2, len(qposes)-1):
                im.save(destination.with_name(f'{destination.stem}_{i:04d}.png'))
    finally:
        proc.stdin.close(); renderer.close()
    if proc.wait() != 0: raise RuntimeError('ffmpeg encoding failed')

def run(a, report):
    os.environ.setdefault('MUJOCO_GL', 'egl'); add_sources()
    import numpy as np
    import torch
    import mujoco
    import warp as wp
    from mjlab.asset_zoo.robots import get_g1_robot_cfg
    from mjlab.scene import Scene, SceneCfg
    from mjlab.sim import Simulation, SimulationCfg, MujocoCfg
    from mjlab.terrains import TerrainEntityCfg
    torch.set_num_threads(4); wp.init()
    scene = Scene(SceneCfg(num_envs=1, terrain=TerrainEntityCfg(terrain_type='plane'),
        entities={'robot':get_g1_robot_cfg()}, extent=2.0), device='cuda:0')
    print('Building actual G1 mjlab GPU simulation...', flush=True)
    sim = Simulation(num_envs=1, cfg=SimulationCfg(nconmax=64, njmax=256, graph_conditional=False,
        mujoco=MujocoCfg(timestep=.005, iterations=10, ls_iterations=20)),
        spec=scene.spec, device='cuda:0')
    scene.initialize(mj_model=sim.mj_model, model=sim.model, data=sim.data)
    robot = scene['robot']; model = sim.mj_model
    assert model.nq==36 and model.nv==35 and model.nu==29
    joint_names = list(robot.joint_names)
    report.update(model_loaded=True, nq=model.nq, nv=model.nv, nu=model.nu,
        meshes=model.nmesh, bodies=model.nbody, joints=joint_names,
        backend='mjlab Simulation / MuJoCo Warp on cuda:0', physics_dt_s=.005,
        gravity=list(model.opt.gravity), num_envs=1, cuda_graphs_enabled=sim.use_cuda_graph)
    sim.reset(); scene.reset()
    if a.mode == 'stand':
        host = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, host, model.key('init_state').id)
        mujoco.mj_forward(model, host)
        feet=[i for i in range(model.ngeom) if '_foot' in model.geom(i).name and model.geom(i).name.endswith('_collision')]
        assert len(feet)==14
        bottom=min(host.geom_xpos[i,2]-abs(host.geom_xmat[i].reshape(3,3)[2,2])*model.geom_size[i,1]-model.geom_size[i,0] for i in feet)
        host.qpos[2] += .002-bottom
        from standing_controller import build_controller, control
        qstart,u0,K,controller_info=build_controller(model,host.qpos.copy())
        report['standing_controller']=controller_info
        np.savez(a.output/'standing_controller.npz',q0=qstart,u0=u0,K=K)
        sim.data.qpos[0].copy_(torch.as_tensor(qstart,device='cuda:0',dtype=torch.float32))
        sim.data.qvel.zero_()
        sim.data.ctrl[0].copy_(torch.as_tensor(u0,device='cuda:0',dtype=torch.float32)); sim.forward()
        scene.update(.005)
        samples=[qstart]; metrics=[]
        print('Running free-base standing with equilibrium feedforward and LQR/PD feedback...',flush=True)
        for step in range(round(a.seconds/.005)):
            q=sim.data.qpos[0].detach().cpu().numpy().copy()
            v=sim.data.qvel[0].detach().cpu().numpy().copy()
            command=control(model,q,v,qstart,u0,K)
            sim.data.ctrl[0].copy_(torch.as_tensor(command,device='cuda:0',dtype=torch.float32))
            sim.step(); scene.update(.005)
            if (step+1)%8==0:
                q=sim.data.qpos[0].detach().cpu().numpy().copy()
                v=sim.data.qvel[0].detach().cpu().numpy().copy()
                if not np.isfinite(q).all() or not np.isfinite(v).all(): raise RuntimeError('Non-finite physics state')
                tilt=float(np.degrees(np.arccos(np.clip(1-2*(q[4]**2+q[5]**2),-1,1))))
                samples.append(q); metrics.append([q[2],tilt,float(np.linalg.norm(q[:2]-qstart[:2])),float(np.max(np.abs(v)))])
                if q[2]<.45 or tilt>35: break
        arr=np.asarray(metrics)
        report.update(kind='physics_standing_equilibrium_LQR_PD_not_learned_policy', root_fixed=False,
            gravity_disabled=False, actuator_gravity_feedforward=True, external_forces=False,
            resets_during_rollout=0, root_state_writes_after_start=0,
            elapsed_sim_seconds=len(metrics)*.04, requested_seconds=a.seconds,
            min_pelvis_height_m=float(arr[:,0].min()), max_tilt_deg=float(arr[:,1].max()),
            max_horizontal_drift_m=float(arr[:,2].max()), initial_ground_adjustment_m=float(.002-bottom),
            standing_ok=bool(len(metrics)*.04>=a.seconds-.04 and arr[:,0].min()>.45 and arr[:,1].max()<35))
        output=a.output/'standing.mp4'; title='PHYSICS STANDING | Equilibrium + LQR / PD | No learned policy'
    else:
        from supermimic.envs.mdp.motion_library import MotionLibrary
        from supermimic.envs.tracking_env_cfg import TRACKED_BODIES
        library=MotionLibrary([str(a.input)],TRACKED_BODIES,robot.data.default_joint_pos[0],device='cuda:0')
        report.update(supermimic_motion_library_loaded=True,
                      motion_library_clips=library.num_clips,motion_library_frames=library.num_frames)
        with np.load(require_file(a.input),allow_pickle=False) as motion:
            assert list(motion['joint_names'])==joint_names,'Joint order mismatch'
            root=list(motion['body_names']).index('pelvis'); fps=float(motion['fps'][0])
            n=len(motion['joint_pos']); ids=np.minimum(np.round(np.arange(int((n-1)/fps*25)+1)*fps/25).astype(int),n-1)
            samples=[]; max_error=0.; max_body_error=0.; max_quat_error=0.
            body_indices=[list(robot.body_names).index(name) for name in motion['body_names']]
            for i in range(n):
                q=np.concatenate([motion['body_pos_w'][i,root],motion['body_quat_w'][i,root],motion['joint_pos'][i]])
                sim.data.qpos[0].copy_(torch.as_tensor(q,device='cuda:0',dtype=torch.float32))
                sim.data.qvel.zero_(); sim.forward(); scene.update(0.)
                actual=sim.data.qpos[0].detach().cpu().numpy().copy()
                assert np.isfinite(actual).all()
                max_error=max(max_error,float(np.max(np.abs(actual-q))))
                actual_pos=robot.data.body_link_pos_w[0].detach().cpu().numpy()[body_indices]
                actual_quat=robot.data.body_link_quat_w[0].detach().cpu().numpy()[body_indices]
                ref_quat=motion['body_quat_w'][i]
                max_body_error=max(max_body_error,float(np.linalg.norm(actual_pos-motion['body_pos_w'][i],axis=-1).max()))
                max_quat_error=max(max_quat_error,float(np.minimum(np.linalg.norm(actual_quat-ref_quat,axis=-1),np.linalg.norm(actual_quat+ref_quat,axis=-1)).max()))
                assert np.isfinite(actual_pos).all() and np.isfinite(actual_quat).all()
                samples.append(actual)
            assert max_body_error<1e-4 and max_quat_error<1e-4, 'GPU forward kinematics differ from reference model'
            report.update(kind='kinematic_reference_not_policy_or_physics_tracking',
                reference=str(a.input), reference_frames=n, reference_fps=fps,
                duration_s=(n-1)/fps, frames_validated=len(samples),frames_played=len(ids),
                qpos_roundtrip_max_error=max_error, body_fk_max_position_error_m=max_body_error,
                body_fk_max_quaternion_error=max_quat_error, bodies_validated=len(body_indices), playback_ok=True)
            samples=np.asarray(samples)[ids]
        output=a.output/'reference_playback.mp4'; title='KINEMATIC REFERENCE | Video-derived | Not policy tracking'
    wp.synchronize()
    report['gpu']=torch.cuda.get_device_name()
    report['gpu_free_mib_after_simulation']=torch.cuda.mem_get_info()[0]/1024**2
    print('Rendering verified states...',flush=True)
    samples=np.asarray(samples)
    np.savez(a.output/f'{a.mode}_states.npz',qpos=samples,fps=np.array([25]),joint_names=np.array(joint_names))
    render_frames(model,samples,output,25,title)
    report['video']=str(output); report['ok']=report.get('standing_ok',report.get('playback_ok',False))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['stand','play'],default='stand')
    p.add_argument('--seconds',type=float,default=10.)
    p.add_argument('--input',type=Path)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/environment_validation')
    a=p.parse_args()
    if a.mode=='play' and a.input is None:p.error('--input is required for playback')
    if a.seconds<=0:p.error('--seconds must be positive')
    a.output.mkdir(parents=True,exist_ok=True); report={'ok':False,'mode':a.mode}; start=time.monotonic()
    try:run(a,report)
    except Exception as e:report.update(error=str(e),traceback=traceback.format_exc())
    report['wall_seconds']=time.monotonic()-start
    (a.output/f'{a.mode}_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    raise SystemExit(0 if report['ok'] else 1)

if __name__=='__main__':main()
