"""Render the accepted recorded trajectory after the training process releases CUDA."""
import argparse
import json
import os
from pathlib import Path
from common import ROOT
from training_supervisor import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    a=p.parse_args();status_path=a.run_dir/'status.json'
    if not status_path.exists():return
    state=json.loads(status_path.read_text())
    if state.get('status')!='targets_met_video_review_pending':return
    try:
        os.environ.setdefault('MUJOCO_GL','egl')
        import numpy as np
        import mujoco
        from validate_g1 import render_frames
        recorded=Path(state['last_evaluation']['nominal']['states'])
        with np.load(recorded) as data:
            poses=data['qpos'][::2];reference=data['qref'][::2];fps=float(data['fps'].item())/2
        model=mujoco.MjModel.from_binary_path(str(ROOT/'outputs/nju_validation/rollouts/g1_scene.mjb'))
        if poses.shape[1]!=model.nq:raise ValueError('Recorded poses and render model differ')
        dest=a.run_dir/'accepted_preview';dest.mkdir(exist_ok=True)
        render_frames(model,poses,dest/'policy.mp4',fps,'LEARNED POLICY | Full clip | No mid-clip resets')
        render_frames(model,reference,dest/'reference.mp4',fps,'RETARGETED REFERENCE | Kinematic comparison')
        state.update(policy_video=str(dest/'policy.mp4'),reference_video=str(dest/'reference.mp4'),
            visual_review_complete=False,render_status='complete')
    except Exception as exc:
        state.update(render_status='failed',render_error=repr(exc))
        print(f'Accepted checkpoint remains saved; rendering failed: {exc!r}',flush=True)
    atomic_json(status_path,state)


if __name__=='__main__':main()
