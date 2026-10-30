"""SMPL-X reprojection sanity check against ViTPose; not an independent accuracy benchmark."""
import argparse,json
from pathlib import Path
import numpy as np
from common import CHECKPOINTS

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reconstruction',type=Path,required=True);p.add_argument('--video',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    import torch,smplx,cv2
    from PIL import Image,ImageDraw
    torch.set_num_threads(3);d=torch.load(a.reconstruction,weights_only=False,map_location='cpu');pred=d['smpl_params_incam'];g=d['smpl_params_global']
    prep=a.reconstruction.parent/'preprocess';kp=torch.load(prep/'vitpose.pt',weights_only=False).numpy();box=torch.load(prep/'bbx.pt',weights_only=False)['bbx_xys'].numpy()
    model=smplx.create(str(CHECKPOINTS/'body_models'),'smplx',gender='neutral',use_pca=False)
    # Vertex-selected face landmarks plus SMPL-X body joints approximate COCO.
    ids=[55,57,56,59,58,16,17,18,19,20,21,1,2,4,5,7,8];projected=[];world=[]
    n=len(kp)
    with torch.inference_mode():
        for start in range(0,n,16):
            end=min(n,start+16);count=end-start
            fixed=dict(expression=torch.zeros(count,model.num_expression_coeffs),left_hand_pose=torch.zeros(count,45),right_hand_pose=torch.zeros(count,45),jaw_pose=torch.zeros(count,3),leye_pose=torch.zeros(count,3),reye_pose=torch.zeros(count,3))
            for name,x in [('camera',pred),('world',g)]:
                result=model(**{k:v[start:end].float() for k,v in x.items()},**fixed)
                if name=='camera':
                    joints=result.joints[:,ids];pix=torch.einsum('bij,bkj->bki',d['K_fullimg'][start:end],joints);projected.append((pix[...,:2]/pix[...,2:]).numpy())
                else:world.append(result.joints[:,:22].numpy())
    proj=np.concatenate(projected);wj=np.concatenate(world);error=np.linalg.norm(proj-kp[...,:2],axis=-1);valid=kp[...,2]>=.5;normalized=error/box[:,None,2]
    frame_median=np.array([np.median(error[i,valid[i]]) for i in range(n)]);worst=np.argsort(frame_median)[-10:][::-1]
    report={'frames':n,'finite':bool(np.isfinite(proj).all()),'valid_keypoint_threshold':.5,
        'reprojection_px_median':float(np.median(error[valid])),'reprojection_px_p95':float(np.percentile(error[valid],95)),
        'normalized_by_bbox_median':float(np.median(normalized[valid])),'normalized_by_bbox_p95':float(np.percentile(normalized[valid],95)),
        'worst_frames':[{'frame':int(i),'time_s':float(i/30),'median_error_px':float(frame_median[i])} for i in worst],
        'per_joint_median_px':[float(np.median(error[:,j][valid[:,j]])) for j in range(17)],
        'note':'Not ground truth: compares dependent monocular models with approximate COCO/SMPL-X joint correspondence. Low reprojection error cannot establish depth, scale or contact accuracy.'}
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'reconstruction_quality.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(a.output/'reconstruction_series.npz',projected_coco=proj,keypoints=kp,reprojection_error=error,valid=valid,world_joints_y_up=wj)
    frames=[0,90,180,225,235,245,255,265,275,285,330,369];canvas=Image.new('RGB',(4*216,3*408));cap=cv2.VideoCapture(str(a.video))
    edges=[(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
    for j,i in enumerate(frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,frame=cap.read();assert ok
        for u,v in edges:
            cv2.line(frame,tuple(proj[i,u].astype(int)),tuple(proj[i,v].astype(int)),(0,140,255),4)
        for point in kp[i][valid[i]]:cv2.circle(frame,tuple(point[:2].astype(int)),5,(0,255,0),-1)
        im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).resize((216,384));canvas.paste(im,((j%4)*216,(j//4)*408+24));ImageDraw.Draw(canvas).text(((j%4)*216+4,(j//4)*408+5),f'f{i} {i/30:.2f}s | {frame_median[i]:.1f}px',fill='white')
    cap.release();canvas.save(a.output/'reprojection_contact_sheet.jpg');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
