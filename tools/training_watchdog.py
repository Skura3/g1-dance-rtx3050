"""Independent, CPU-only overnight guard for one exact systemd training invocation."""
import argparse
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time


def now():
    return datetime.datetime.now().astimezone().isoformat()


def clock():
    return time.clock_gettime(time.CLOCK_BOOTTIME)


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
        stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def pin(source,destination):
    """Trainer replaces checkpoints atomically, so a hard link pins immutable bytes."""
    source=Path(source);destination=Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_name(destination.name+'.tmp')
    temporary.unlink(missing_ok=True);os.link(source,temporary);os.replace(temporary,destination)


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(1024*1024),b''):digest.update(data)
    return digest.hexdigest()


def checkpoint_snapshot(run,destination):
    """Reject mismatched checkpoint/report pairs during atomic best updates."""
    run=Path(run);destination=Path(destination)
    report=read_json(run/'best_evaluation.json')
    expected=report['nominal']['checkpoint_sha256']
    score=report['nominal']['selection_score_lower_is_better']
    if score is None or not math.isfinite(score):raise ValueError('Invalid best score')
    staging=destination.with_name(destination.name+'.candidate')
    try:
        pin(run/'ckpt_best_evaluated.pt',staging)
        actual=sha256(staging)
        if actual!=expected:raise ValueError('Best report/checkpoint mismatch; retry next poll')
        os.replace(staging,destination)
    finally:
        staging.unlink(missing_ok=True)
    return {'path':str(destination),'source_run':str(run),'iteration':report['iteration'],
        'sha256':actual,'selection_score_lower_is_better':score,'preserved_at':now(),
        'definition':'Best evaluated checkpoint within this source run under its existing ranking; not guaranteed visually approved or strict-target qualified.'}


def latest_iteration(run):
    run=Path(run);values=[]
    try:values.append(int(read_json(run/'status.json')['iteration']))
    except (OSError,ValueError,KeyError,TypeError):pass
    try:
        with (run/'updates.jsonl').open('rb') as stream:
            stream.seek(0,2);stream.seek(max(0,stream.tell()-16384))
            for line in stream.read().splitlines()[-12:]:
                try:values.append(int(json.loads(line)['iteration']))
                except (ValueError,KeyError,TypeError):pass
    except OSError:pass
    return max(values) if values else None


def update_progress(state,iteration,at):
    if iteration is not None and (state.get('highest_iteration') is None or iteration>state['highest_iteration']):
        state.update(highest_iteration=iteration,last_progress_clock=at,last_progress_at=now())
    state['no_progress_seconds']=max(0.0,at-state['last_progress_clock'])


def trigger_reason(temperature,no_progress_seconds,config):
    if temperature is not None and temperature>config['temperature_limit_c']:
        return 'gpu_temperature_exceeded'
    if no_progress_seconds>config['stall_limit_seconds']:
        return 'training_iteration_stalled'
    return None


class Watchdog:
    def __init__(self,config):
        self.config=config;self.out=Path(config['output_dir']);self.run=Path(config['run_dir'])
        self.out.mkdir(parents=True,exist_ok=True);self.best_signatures={}
        self.boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        try:self.state=read_json(self.out/'status.json')
        except (OSError,ValueError):self.state={}
        if self.state.get('boot_id')!=self.boot or self.state.get('invocation_id')!=config['invocation_id']:
            self.state={'boot_id':self.boot,'invocation_id':config['invocation_id'],
                'started_at':now(),'last_progress_at':now(),'last_progress_clock':clock(),
                'highest_iteration':None,'snapshots':{}}
        self.state['configuration']=config

    def event(self,kind,**fields):
        row={'time':now(),'event':kind,**fields}
        with (self.out/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
        print(json.dumps(row),flush=True)

    def publish(self,**fields):
        self.state.update(fields,updated_at=now(),watchdog_pid=os.getpid())
        atomic_json(self.out/'status.json',self.state)

    def service(self):
        raw=subprocess.check_output(['systemctl','--user','show',self.config['training_service'],
            '-p','ActiveState','-p','SubState','-p','InvocationID'],text=True,timeout=5)
        return dict(line.split('=',1) for line in raw.splitlines() if '=' in line)

    def same_active(self):
        service=self.service()
        return service.get('InvocationID')==self.config['invocation_id'] and service.get('ActiveState') in ('active','activating','deactivating')

    def command(self,*arguments):
        if not self.same_active():return False
        result=subprocess.run(['systemctl','--user',*arguments,self.config['training_service']],
            text=True,capture_output=True,timeout=10)
        self.event('service_command',arguments=list(arguments),returncode=result.returncode,stderr=result.stderr[-2000:])
        return result.returncode==0

    def temperature(self):
        raw=subprocess.check_output(['nvidia-smi','-i',self.config['gpu_uuid'],
            '--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],text=True,timeout=5)
        value=float(raw.strip())
        if not math.isfinite(value) or value<0 or value>150:raise ValueError('Invalid temperature reading')
        return value

    def refresh_best(self):
        for name,run in self.config['best_sources'].items():
            path=Path(run);checkpoint=path/'ckpt_best_evaluated.pt';report=path/'best_evaluation.json'
            if not checkpoint.exists() or not report.exists():continue
            signature=(checkpoint.stat().st_ino,report.stat().st_mtime_ns)
            if self.best_signatures.get(name)==signature:continue
            try:
                result=checkpoint_snapshot(path,self.out/'snapshots'/f'{name}.pt')
                self.state['snapshots'][name]=result;self.best_signatures[name]=signature
                self.event('best_checkpoint_preserved',name=name,**result)
            except (OSError,ValueError,KeyError,TypeError) as exc:
                self.event('best_snapshot_retry',name=name,error=repr(exc))

    def wait_for_exit(self,seconds):
        deadline=clock()+seconds
        while clock()<deadline:
            if not self.same_active():return True
            time.sleep(min(2.0,max(0.0,deadline-clock())))
        return not self.same_active()

    def stop_training(self,reason):
        incident_path=self.out/'incident.json'
        if incident_path.exists():incident=read_json(incident_path)
        else:
            self.refresh_best()
            incident={'reason':reason,'triggered_at':now(),'trigger_iteration':self.state.get('highest_iteration'),
                'temperature_c':self.state.get('temperature_c'),'no_progress_seconds':self.state.get('no_progress_seconds'),
                'configuration':self.config,'preserved_best':{},'status':'stop_requested'}
            for name,entry in self.state.get('snapshots',{}).items():
                dst=self.out/'incident_checkpoints'/f'{name}.pt'
                pin(entry['path'],dst);incident['preserved_best'][name]=dict(entry,path=str(dst))
            incident['preferred_best']=min(incident['preserved_best'],key=lambda name:(
                incident['preserved_best'][name]['selection_score_lower_is_better'],
                -incident['preserved_best'][name]['iteration']),default=None)
            if incident['preferred_best'] is not None:
                pin(incident['preserved_best'][incident['preferred_best']]['path'],
                    self.out/'incident_checkpoints/ckpt_best_evaluated.pt')
            latest=self.run/'ckpt_latest.pt'
            if latest.exists():pin(latest,self.out/'incident_checkpoints/ckpt_latest_before_stop.pt')
            atomic_json(incident_path,incident)
        self.publish(status='stopping',stop_reason=reason)
        if not self.same_active():
            incident['status']='target_already_exited_or_replaced'
        else:
            if reason=='gpu_temperature_exceeded':
                try:
                    p=Path(self.config['wallpaper_record']);record=read_json(p)
                    record.update(restore_deferred=True,watchdog_deferral_reason=reason,watchdog_incident=str(incident_path))
                    atomic_json(p,record)
                except (OSError,ValueError) as exc:incident['wallpaper_deferral_error']=repr(exc)
            atomic_json(self.run/'watchdog_stop.json',incident)
            atomic_json(self.run/'STOP',{'requested_by':'training_watchdog','reason':reason,'incident':str(incident_path),'time':now()})
            self.event('stop_requested',reason=reason,incident=str(incident_path))
            grace=self.config['overheat_grace_seconds'] if reason=='gpu_temperature_exceeded' else self.config['stall_grace_seconds']
            exited=self.wait_for_exit(grace)
            if not exited:
                self.command('stop','--no-block')
                exited=self.wait_for_exit(self.config['termination_grace_seconds'])
            if not exited:
                incident['forced_kill_requested']=True
                self.command('kill','--kill-who=all','--signal=SIGKILL')
                exited=self.wait_for_exit(15)
            incident['status']='stopped' if exited else 'termination_unconfirmed'
        latest=self.run/'ckpt_latest.pt';after=self.out/'incident_checkpoints/ckpt_latest_after_stop.pt'
        if latest.exists():
            pin(latest,after);incident['latest_checkpoint']={'path':str(after),'sha256':sha256(after)}
        try:
            report=read_json(self.run/'run_report.json');incident['training_run_report']=report
            incident['latest_state_save_confirmed']=bool(report.get('ok') and report.get('stop_reason')=='stopped_by_request')
        except (OSError,ValueError):incident['latest_state_save_confirmed']=False
        incident['finished_at']=now();atomic_json(incident_path,incident)
        atomic_json(self.run/'watchdog_stop.json',incident)
        self.event('shutdown_finished',status=incident['status'],latest_state_save_confirmed=incident['latest_state_save_confirmed'])
        self.publish(status=incident['status'],stop_reason=reason,incident=str(incident_path))
        if incident['status']=='termination_unconfirmed':raise RuntimeError('Could not confirm target service exit')

    def run_forever(self):
        incident=self.out/'incident.json'
        if incident.exists():
            self.stop_training(read_json(incident)['reason']);return
        self.event('watchdog_started',service=self.config['training_service'])
        self.refresh_best()
        while True:
            service=self.service()
            if service.get('InvocationID')!=self.config['invocation_id']:
                self.publish(status='target_invocation_changed');self.event('target_invocation_changed');return
            if service.get('ActiveState') not in ('active','activating','deactivating'):
                self.publish(status='target_exited',service=service);self.event('target_exited',service=service);return
            update_progress(self.state,latest_iteration(self.run),clock())
            previous_error=self.state.get('temperature_read_error')
            try:
                temperature=self.temperature();error=None
            except (OSError,ValueError,subprocess.SubprocessError) as exc:
                temperature=None;error=repr(exc)
            if error!=previous_error:self.event('temperature_sensor_status',error=error)
            self.state.update(temperature_c=temperature,temperature_read_error=error)
            reason=trigger_reason(temperature,self.state['no_progress_seconds'],self.config)
            if reason:
                self.stop_training(reason);return
            self.refresh_best()
            self.publish(status='watching',service=service)
            time.sleep(self.config['poll_seconds'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();config=read_json(args.config)
    if config['temperature_limit_c']!=100 or config['stall_limit_seconds']!=14400:
        raise ValueError('This deployment must match user-authorized thresholds: >100 C or >4 hours')
    out=Path(config['output_dir']);out.mkdir(parents=True,exist_ok=True)
    with (out/'watchdog.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        Watchdog(config).run_forever()


if __name__=='__main__':main()
