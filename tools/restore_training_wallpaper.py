"""Resume only the wallpaper processes paused for this training session."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--record',type=Path,required=True);a=p.parse_args()
    if not a.record.exists():return
    record=json.loads(a.record.read_text());results=[]
    if record.get('restored_at'):return
    if record.get('restore_deferred'):
        print('Wallpaper restoration deferred during authorized training transition.',flush=True);return
    if record.get('restore_mode')=='restart':
        command=record.get('restart_command')
        if command!=['/System/Applications/komorebi']:raise ValueError('Unexpected wallpaper restart command')
        running=[]
        for candidate in Path('/proc').iterdir():
            if not candidate.name.isdigit():continue
            try:
                if os.readlink(candidate/'exe')==command[0]:running.append(int(candidate.name))
            except OSError:pass
        if running:result={'status':'already_running','pids':running}
        else:
            subprocess.run(['systemd-run','--user','--collect','--unit=supermimic-wallpaper-restored',
                '--property=Type=exec',*command],check=True)
            result={'status':'restart_service_started','unit':'supermimic-wallpaper-restored.service'}
        record.update(restored_at=datetime.datetime.now().astimezone().isoformat(),restore_results=[result])
        a.record.write_text(json.dumps(record,indent=2)+chr(10));print(json.dumps(result),flush=True);return
    for entry in [*record.get('children',[]),record]:
        result={'pid':entry['pid']}
        try:
            proc=Path('/proc')/str(entry['pid']);fields=(proc/'stat').read_text().split(') ',1)[1].split()
            if fields[19]!=entry['start_ticks']:result['status']='pid_reused_no_signal'
            elif entry.get('was_stopped'):result['status']='originally_stopped_no_signal'
            elif fields[0] not in ('T','t'):result['status']='already_running'
            else:os.kill(entry['pid'],signal.SIGCONT);result['status']='resumed'
        except ProcessLookupError:result['status']='already_exited'
        except FileNotFoundError:result['status']='already_exited'
        results.append(result)
    record.update(restored_at=datetime.datetime.now().astimezone().isoformat(),restore_results=results)
    a.record.write_text(json.dumps(record,indent=2)+chr(10));print(json.dumps(results),flush=True)


if __name__=='__main__':main()
