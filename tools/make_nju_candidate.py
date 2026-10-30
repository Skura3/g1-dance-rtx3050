"""Compare NJU rolling-lower-envelope grounding without overwriting prior references.

Algorithm adapted from SuperMimic-Repro-Sim2Real (MIT), f8f60d7:
Copyright (c) 2026 Yuxi Liu; Copyright (c) 2026 Hongyu Ding.
Full notice: docs/NJU_THIRD_PARTY_NOTICE.txt.
"""
import argparse,json,pickle
from pathlib import Path
import numpy as np
from common import ROOT,add_sources
from convert_motion import DEFAULT_XML,capsule_bottoms,convert


def lower_envelope_floor(heights,fps):
    heights=np.asarray(heights,dtype=float)
    width=max(1,int(round(fps)))
    pad=np.pad(heights,width//2,mode='edge')
    envelope=np.array([pad[i:i+width].min() for i in range(len(heights))])
    pad=np.pad(envelope,width//2,mode='edge')
    return np.convolve(pad,np.ones(width)/width,mode='valid')[:len(heights)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=ROOT/'data/motions/reference_full_nju_gmr.pkl')
    p.add_argument('--name',default='reference_full_nju_drift')
    a=p.parse_args();add_sources()
    from supermimic.data.motion_convert import load_g1_model,G1_JOINT_NAMES
    with a.input.open('rb') as f:d=pickle.load(f)
    if d['root_quaternion_order']!='xyzw' or list(d['joint_names'])!=list(G1_JOINT_NAMES):raise ValueError('Input convention changed')
    model=load_g1_model(DEFAULT_XML);j=np.asarray(d['dof_pos']);clipped=np.clip(j,model.jnt_range[1:,0],model.jnt_range[1:,1])
    q=np.c_[d['root_pos'],np.asarray(d['root_rot'])[:,[3,0,1,2]],clipped];before=capsule_bottoms(model,q)
    floor=lower_envelope_floor(before,d['fps']);q[:,2]-=floor;q[:,:2]-=q[0,:2]
    d.update(root_pos=q[:,:3],dof_pos=clipped)
    out=ROOT/'outputs/nju_validation';out.mkdir(parents=True,exist_ok=True)
    pkl=out/(a.name+'.pkl')
    with pkl.open('wb') as f:pickle.dump(d,f)
    report=convert(pkl,ROOT/'data/preview'/(a.name+'.npz'),ground_offset=0.)
    report.update(method='NJU one-second rolling lower envelope followed by one-second smoothing',clipped_values=int(np.sum(j!=clipped)),max_joint_clip_rad=float(abs(j-clipped).max()),floor_range_m=[float(floor.min()),float(floor.max())],training_approved=False,warning='Geometric heuristic only; long flights and hand support need independent checks. Competing candidate, not automatically preferred.')
    (out/(a.name+'_report.json')).write_text(json.dumps(report,indent=2));np.savez_compressed(out/(a.name+'_floor.npz'),floor=floor,foot_before=before,foot_after=before-floor)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
