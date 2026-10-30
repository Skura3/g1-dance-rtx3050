"""Inspect video and prepare a loss-minimizing H.264 input without resizing."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from fractions import Fraction
from common import ROOT

def prepare(source, dest, start=0.0, duration=None):
    source, dest = Path(source).resolve(), Path(dest).resolve()
    if source == dest:
        raise ValueError('Input must not be overwritten')
    if start < 0 or (duration is not None and duration <= 0):
        raise ValueError('Invalid trim interval')
    info = json.loads(subprocess.check_output(['ffprobe','-v','error','-show_format','-show_streams','-of','json',str(source)]))
    stream = next(s for s in info['streams'] if s['codec_type'] == 'video')
    total = float(info['format']['duration'])
    if start >= total:
        raise ValueError('Start is past end of video')
    dest.parent.mkdir(parents=True, exist_ok=True)
    meta = {'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'source_frames':stream.get('nb_frames'),'source_fps':float(Fraction(stream['avg_frame_rate'])),
            'width':stream['width'],'height':stream['height'],'start_seconds':start,
            'requested_duration_seconds':duration,'fps':30,'output':str(dest)}
    manifest = dest.with_suffix('.json')
    if dest.exists():
        if manifest.exists() and json.loads(manifest.read_text()) == meta:
            return meta
        raise FileExistsError(f'{dest} exists without matching metadata; choose a new output')
    cmd=['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-n','-ss',str(start),'-i',str(source)]
    if duration is not None: cmd += ['-t',str(duration)]
    cmd += ['-map','0:v:0','-an','-vf','fps=30','-c:v','libx264','-crf','18','-preset','fast','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)]
    subprocess.run(cmd,check=True)
    manifest.write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    return meta

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=ROOT/'reference_video.mp4')
    p.add_argument('--output',type=Path,default=ROOT/'data/video/reference_full.mp4')
    p.add_argument('--start',type=float,default=0)
    p.add_argument('--duration',type=float)
    a=p.parse_args(); print(json.dumps(prepare(a.input,a.output,a.start,a.duration),ensure_ascii=False,indent=2))

