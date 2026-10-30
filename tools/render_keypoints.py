"""Overlay cached ViTPose COCO-17 keypoints on the matching input video."""
import argparse
from pathlib import Path
from common import ROOT, require_file

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video',type=Path,default=ROOT/'data/video/reference_pilot.mp4')
    p.add_argument('--keypoints',type=Path,default=ROOT/'outputs/gvhmr/reference_pilot/preprocess/vitpose.pt')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/reference_pilot_keypoints.mp4')
    p.add_argument('--confidence',type=float,default=.5)
    a=p.parse_args()
    import torch,cv2,numpy as np,imageio.v2 as imageio
    kp=torch.load(require_file(a.keypoints),map_location='cpu',weights_only=False).numpy()
    if kp.ndim!=3 or kp.shape[1:]!=(17,3):raise ValueError('Expected (frames,17,3) COCO keypoints')
    cap=cv2.VideoCapture(str(require_file(a.video)));fps=cap.get(cv2.CAP_PROP_FPS)
    if int(cap.get(cv2.CAP_PROP_FRAME_COUNT))!=len(kp):raise ValueError('Video and keypoint frame counts differ')
    edges=[(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
    a.output.parent.mkdir(parents=True,exist_ok=True)
    try:
        with imageio.get_writer(str(a.output),fps=fps,codec='libx264',macro_block_size=1) as writer:
            for frame_kp in kp:
                ok,frame=cap.read()
                if not ok:raise RuntimeError('Video decoding ended before keypoints')
                valid=np.isfinite(frame_kp).all(axis=1)&(frame_kp[:,2]>=a.confidence)
                for u,v in edges:
                    if valid[u] and valid[v]:cv2.line(frame,tuple(frame_kp[u,:2].astype(int)),tuple(frame_kp[v,:2].astype(int)),(0,255,0),3)
                for point in frame_kp[valid]:cv2.circle(frame,tuple(point[:2].astype(int)),4,(0,255,255),-1)
                writer.append_data(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    finally:cap.release()
    print(a.output)

if __name__=='__main__':main()
