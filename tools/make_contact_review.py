"""Create review videos/stills; these display kinematics, not policy tracking."""
import subprocess
from pathlib import Path
from common import ROOT


def main():
    out=ROOT/'outputs/contact_validation'
    source=ROOT/'data/video/reference_full.mp4'
    old=ROOT/'outputs/full_validation/reference_playback.mp4'
    new=out/'reference_playback.mp4'
    for p in (source,old,new):
        if not p.is_file():raise FileNotFoundError(p)
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(source),'-i',str(new),
        '-filter_complex','[0:v]fps=25,scale=338:600,setsar=1[s];[1:v]setsar=1[r];[s][r]hstack=inputs=2:shortest=1[v]',
        '-map','[v]','-map','0:a?','-c:v','libx264','-threads','2','-pix_fmt','yuv420p','-c:a','aac','-shortest','-movflags','+faststart',str(out/'source_vs_g1.mp4')],check=True)
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(old),'-i',str(new),
        '-filter_complex',"[0:v]drawtext=text='PREVIOUS REPAIR':x=12:y=64:fontcolor=white:fontsize=22:box=1:boxcolor=black@0.6[l];[1:v]drawtext=text='CONTACT CORRECTION':x=12:y=64:fontcolor=white:fontsize=22:box=1:boxcolor=black@0.6[r];[l][r]hstack=inputs=2:shortest=1[v]",
        '-map','[v]','-an','-c:v','libx264','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'repaired_vs_contact.mp4')],check=True)
    import cv2
    from PIL import Image,ImageDraw
    times=[0.,4.2,7.7,7.8,7.9,8.,10.8,12.28]
    widths=[170,400,400];height=300
    canvas=Image.new('RGB',(sum(widths),len(times)*(height+25)),(20,26,35));draw=ImageDraw.Draw(canvas)
    for col,(path,width,label) in enumerate(zip((source,old,new),widths,('SOURCE','PREVIOUS','CONTACT'))):
        cap=cv2.VideoCapture(str(path));x=sum(widths[:col])
        try:
            for row,t in enumerate(times):
                cap.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,frame=cap.read()
                if not ok:raise RuntimeError(f'Cannot decode {path} at {t}s')
                im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));im.thumbnail((width,height))
                y=row*(height+25);draw.text((x+5,y+5),f'{label}  {t:.2f}s',fill='white')
                canvas.paste(im,(x+(width-im.width)//2,y+25+(height-im.height)//2))
        finally:cap.release()
    canvas.save(out/'contact_review.jpg',quality=92)
    # Keep a compact floor-contact view for visual inspection.
    canvas.crop((0,2*(height+25),sum(widths),6*(height+25))).save(out/'floor_contact_review.jpg',quality=92)
    print(out/'source_vs_g1.mp4');print(out/'repaired_vs_contact.mp4');print(out/'contact_review.jpg')


if __name__=='__main__':main()
