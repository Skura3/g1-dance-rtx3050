"""Render a saved physics evaluation while the independent trainer keeps running."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from common import ROOT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    os.environ.setdefault('MUJOCO_GL','egl')
    import mujoco
    import numpy as np
    from PIL import Image,ImageDraw

    report=json.loads(args.report.read_text());clip=report['clips'][0]
    states=Path(clip['states']);iteration=int(report['checkpoint_iteration'])
    if args.output.exists():raise FileExistsError('Use a fresh preview directory')
    with np.load(states) as a:
        qpos=a['qpos'].copy();qref=a['qref'].copy();names=a['joint_names'].tolist()
        input_fps=float(a['fps'].item())
    if not np.isfinite(qpos).all() or not np.isfinite(qref).all():raise ValueError('Nonfinite trajectory')
    model=mujoco.MjModel.from_binary_path(str(ROOT/'outputs/nju_validation/rollouts/g1_scene.mjb'))
    actual_names=[model.joint(i).name.removeprefix('robot/') for i in range(1,model.njnt)]
    if names!=actual_names or qpos.shape!=qref.shape or qpos.shape[1]!=model.nq:
        raise ValueError('Saved trajectory and model joint ordering differ')
    # Fixed sampling of recorded physical timestamps, no learned-motion retiming.
    stride=2;fps=input_fps/stride;indices=np.arange(0,len(qpos)-1,stride)
    args.output.mkdir(parents=True)
    width,height=640,480
    model.vis.global_.offwidth=width;model.vis.global_.offheight=height
    model.vis.quality.shadowsize=1024
    renderer=mujoco.Renderer(model,height=height,width=width);data=mujoco.MjData(model)
    camera=mujoco.MjvCamera();w,x,y,z=qref[0,3:7]
    heading=float(np.degrees(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))))
    camera.distance=3.3;camera.azimuth=heading+155.;camera.elevation=-12.
    option=mujoco.MjvOption();option.geomgroup[3:]=0
    started=time.monotonic();encoders={}
    titles={'reference':'RETARGETED TARGET | Kinematic reference',
            'policy':f'LEARNED POLICY | iteration {iteration} | Recorded physics'}
    stills=[]
    try:
        for name in ('reference','policy'):
            encoders[name]=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-nostdin',
                '-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r',str(fps),'-i','-',
                '-an','-c:v','libx264','-threads','2','-preset','veryfast','-crf','20','-pix_fmt','yuv420p',
                '-movflags','+faststart',str(args.output/(name+'.mp4'))],stdin=subprocess.PIPE)
        for frame,index in enumerate(indices):
            pair=[]
            for name,poses in [('reference',qref),('policy',qpos)]:
                data.qpos[:]=poses[index];mujoco.mj_forward(model,data)
                # Both panels use precisely the same reference-following camera.
                camera.lookat[:]=[qref[index,0],qref[index,1],.85]
                renderer.update_scene(data,camera=camera,scene_option=option)
                im=Image.fromarray(renderer.render().copy());draw=ImageDraw.Draw(im)
                draw.rectangle((0,0,width,45),fill=(20,26,35))
                draw.text((10,7),titles[name],fill='white')
                draw.text((10,26),f't = {index/input_fps:.2f}s | Same camera | No mid-clip reset',fill='white')
                encoders[name].stdin.write(np.asarray(im).tobytes());pair.append(im)
            if frame in (0,100,200,225,250,len(indices)-1):
                still=Image.new('RGB',(2*width,height));still.paste(pair[0],(0,0));still.paste(pair[1],(width,0))
                still.save(args.output/f'comparison_{index:04d}.jpg',quality=90)
                stills.append(still.resize((640,240)))
            if frame%100==0:print(f'Rendered {frame+1}/{len(indices)} frames; training continues independently.',flush=True)
    finally:
        renderer.close()
        for proc in encoders.values():
            if proc.stdin:proc.stdin.close()
        for proc in encoders.values():
            if proc.wait()!=0:raise RuntimeError('Video encoder failed')
    duration=len(indices)/fps
    graph=(f'[0:v]fps={fps},scale=270:480,setsar=1,setpts=PTS-STARTPTS,'
        'drawbox=x=0:y=0:w=iw:h=45:color=0x141a23:t=fill,'
        "drawtext=text='SOURCE VIDEO':x=10:y=15:fontsize=16:fontcolor=white[src];"
        '[1:v]setpts=PTS-STARTPTS[ref];[2:v]setpts=PTS-STARTPTS[pol];'
        '[src][ref][pol]hstack=inputs=3:shortest=1[out]')
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-threads','1',
        '-i',str(ROOT/'reference_video.mp4'),'-i',str(args.output/'reference.mp4'),
        '-i',str(args.output/'policy.mp4'),'-filter_complex_threads','1','-filter_complex',graph,
        '-map','[out]','-map','0:a?','-t',str(duration),'-c:v','libx264','-threads','2',
        '-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-c:a','aac','-b:a','128k',
        '-movflags','+faststart',str(args.output/'source_target_policy.mp4')],check=True)
    sheet=Image.new('RGB',(1280,240*((len(stills)+1)//2)))
    for i,im in enumerate(stills):sheet.paste(im,((i%2)*640,(i//2)*240))
    sheet.save(args.output/'contact_sheet.jpg',quality=90)
    metadata={'checkpoint_iteration':iteration,'checkpoint_sha256':report['checkpoint_sha256'],
        'source_report':str(args.report.resolve()),'states':str(states.resolve()),
        'states_sha256':hashlib.sha256(states.read_bytes()).hexdigest(),
        'scope':'Playback of previously recorded learned-policy physics, not a live training feed or kinematic target posing.',
        'training_configuration_changed':False,'training_restarted':False,
        'same_reference_following_camera':True,'frame_stride':stride,'fps':fps,'duration_seconds':duration,
        'timing_note':report.get('reference_timing_note'),'source_video_offset_seconds':0,
        'source_video_note':'Shared zero start and playback rate; reconstructed video and control samples are not manually realigned.',
        'render_seconds':time.monotonic()-started,'files':['source_target_policy.mp4','policy.mp4','reference.mp4','contact_sheet.jpg']}
    (args.output/'preview.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2),flush=True)


if __name__=='__main__':main()
