"""Low-frequency full-clip evaluation for one uninterrupted training process."""
import json
import hashlib
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from common import ROOT
from compare_finetuning import assess


class TrainingStopped(Exception):
    pass


def atomic_json(path, data):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w') as f:
        json.dump(data,f,indent=2,default=str,allow_nan=False);f.write('\n')
        f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def link_checkpoint(source, destination):
    """Immutable snapshots share disk space until latest is atomically replaced."""
    source=Path(source);destination=Path(destination)
    if source.resolve()==destination.resolve():return
    tmp=destination.with_name(destination.name+'.tmp')
    tmp.unlink(missing_ok=True)
    os.link(source,tmp);os.replace(tmp,destination)


class SuccessGate:
    def __init__(self, consecutive=2, required_seeds=(11,23,37)):
        self.consecutive=consecutive;self.required_seeds=tuple(required_seeds)
        self.streak=0;self.last_iteration=None

    def nominal(self, iteration, passed):
        if self.last_iteration is not None and iteration<=self.last_iteration:
            raise ValueError('Success requires distinct increasing checkpoints')
        self.last_iteration=iteration
        self.streak=self.streak+1 if passed else 0
        return self.streak>=self.consecutive

    def robustness(self, passes):
        ok=self.streak>=self.consecutive and all(passes.get(s) is True for s in self.required_seeds)
        if not ok:self.streak=0
        return ok


class TrainingSupervisor:
    def __init__(self, trainer, settings, motion, initial_iteration):
        self.trainer=trainer;self.settings=settings;self.motion=Path(motion)
        self.out=Path(trainer.log_dir);self.initial=initial_iteration
        self.interval=int(settings['evaluation_interval_steps'])
        if self.interval<1:raise ValueError('Evaluation interval must be positive')
        self.next_eval=int(settings.get('first_evaluation_iteration',self.initial+self.interval))
        if self.next_eval<=self.initial:raise ValueError('First evaluation must be after the continuation checkpoint')
        self.last_heartbeat=-1;self.started=time.monotonic()
        self.gate=SuccessGate(int(settings['consecutive_nominal_passes']),settings['robustness_seeds'])
        self.expected_steps=int(trainer.lib.clips[0].length)
        self.criteria=json.loads((ROOT/settings['criteria_file']).read_text())
        self.criteria_sha256=hashlib.sha256(json.dumps(self.criteria,sort_keys=True).encode()).hexdigest()
        atomic_json(self.out/'criteria_snapshot.json',self.criteria)
        self.best_score=math.inf;self.history=[];self.stop_requested=False
        self.review_evaluations=int(settings.get('review_after_evaluations',0))
        if self.review_evaluations<0:raise ValueError('Review count must be nonnegative')
        self.state={'status':'initializing','pid':os.getpid(),'initial_iteration':self.initial,
            'reference':str(motion),'evaluation_interval_steps':self.interval,
            'expected_rollout_steps':self.expected_steps,'checkpoint_retention':settings['retained_evaluation_checkpoints'],
            'stop_rule':'Two consecutive nominal full-clip passes and all three initial-state perturbation seeds pass; visual review still required.',
            'mathematical_convergence_proven':False,'visual_review_complete':False,
            'criteria_revision':self.criteria.get('revision','legacy'),
            'criteria_sha256':self.criteria_sha256,
            'criteria_snapshot':str(self.out/'criteria_snapshot.json')}
        atomic_json(self.out/'supervision.json',settings)
        self.publish('training')

    def publish(self,status=None,**updates):
        if status is not None:self.state['status']=status
        self.state.update(updates)
        self.state.update(iteration=int(self.trainer.iteration),
            additional_steps=int(self.trainer.iteration)-self.initial,
            transitions_collected=(int(self.trainer.iteration)-self.initial)*self.trainer.cfg.env.num_envs,
            num_updates=int(self.trainer.num_updates),
            next_evaluation_iteration=self.next_eval,nominal_pass_streak=self.gate.streak,
            elapsed_seconds=round(time.monotonic()-self.started,1),updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
        self.state.update(completed_evaluations=len(self.history),review_after_evaluations=self.review_evaluations,
            review_window_complete=bool(self.review_evaluations and len(self.history)>=self.review_evaluations))
        resource_snapshot=getattr(self.trainer,'resource_snapshot',None)
        if resource_snapshot is not None:self.state['resources']=resource_snapshot()
        atomic_json(self.out/'status.json',self.state)

    def before_step(self):
        it=int(self.trainer.iteration)
        if self.stop_requested or (self.out/'STOP').exists():
            self.trainer.save();self.publish('stopped_by_request')
            raise TrainingStopped('stopped_by_request')
        if it-self.last_heartbeat>=25:
            free=shutil.disk_usage(self.out).free/2**30
            if free<float(self.settings['minimum_free_disk_gib']):
                self.publish('stopped_disk_space',free_disk_gib=round(free,3))
                raise TrainingStopped('stopped_disk_space; last atomic checkpoint remains available')
            self.publish('training',free_disk_gib=round(free,3));self.last_heartbeat=it
        if it>=self.next_eval:
            self.evaluate()

    def run_rollout(self,checkpoint,folder,seed,reset_noise):
        folder.mkdir(parents=True,exist_ok=True)
        cmd=[sys.executable,str(ROOT/'tools/record_nju_rollout.py'),'--checkpoint',str(checkpoint),
            '--motions',str(self.motion),'--output',str(folder),'--no-render','--seed',str(seed)]
        if reset_noise:cmd.append('--reset-noise')
        with (folder/'execution.log').open('w') as f:
            subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True,
                timeout=float(self.settings['evaluation_timeout_seconds']))
        result=json.loads((folder/'report.json').read_text())
        json.dumps(result,allow_nan=False)  # Invalid metrics must never count as success.
        if int(result['checkpoint_iteration'])!=int(self.trainer.iteration):
            raise ValueError('Evaluation loaded a different checkpoint iteration')
        if len(result['clips'])!=1 or result['clips'][0]['name']!=self.motion.stem:
            raise ValueError('Unexpected evaluated motion')
        return dict(result['clips'][0],**assess(result['clips'][0],self.criteria,self.expected_steps),
            checkpoint_sha256=result['checkpoint_sha256'],report=str(folder/'report.json'))

    def evaluate(self):
        it=int(self.trainer.iteration);self.trainer.save()
        snapshot=self.out/f'eval_{it:09d}.pt'
        link_checkpoint(self.out/'ckpt_latest.pt',snapshot)
        folder=self.out/'evaluations'/f'{it:09d}'
        self.publish('evaluating',evaluation_checkpoint=str(snapshot))
        print(f'[SUPERVISOR] Full-clip evaluation at {it}, additional {it-self.initial}; replay retained.',flush=True)
        entry={'iteration':it,'additional_steps':it-self.initial,'checkpoint':str(snapshot),
            'criteria_revision':self.criteria.get('revision','legacy'),
            'criteria_sha256':self.criteria_sha256,'criteria':self.criteria}
        try:
            nominal=self.run_rollout(snapshot,folder/'nominal',0,False);entry['nominal']=nominal
            score=nominal['selection_score_lower_is_better']
            if score is not None and score<self.best_score:
                self.best_score=score;link_checkpoint(snapshot,self.out/'ckpt_best_evaluated.pt')
                atomic_json(self.out/'best_evaluation.json',entry)
            if self.gate.nominal(it,nominal['provisional_targets_met']):
                self.publish('confirming_robustness');passes={};entry['robustness']=[]
                for seed in self.gate.required_seeds:
                    row=self.run_rollout(snapshot,folder/f'reset_seed_{seed}',seed,True)
                    if row['checkpoint_sha256']!=nominal['checkpoint_sha256']:raise ValueError('Snapshot changed during evaluation')
                    passes[seed]=row['provisional_targets_met'];entry['robustness'].append(dict(row,seed=seed))
                if self.gate.robustness(passes):
                    link_checkpoint(snapshot,self.out/'ckpt_accepted.pt')
                    entry['operational_targets_met']=True
            entry.setdefault('operational_targets_met',False)
        except (OSError,ValueError,KeyError,subprocess.SubprocessError) as exc:
            entry.update(evaluation_error=repr(exc),operational_targets_met=False)
            self.gate.streak=0
            print(f'[SUPERVISOR] Evaluation infrastructure error; training will continue: {exc!r}',flush=True)
        atomic_json(folder/'assessment.json',entry)
        self.history.append(entry)
        if entry.get('nominal',{}).get('provisional_targets_met'):
            link_checkpoint(snapshot,self.out/'ckpt_latest_quality_pass.pt')
        if self.review_evaluations:
            review={'planned_evaluations':self.review_evaluations,'completed_evaluations':len(self.history),
                'review_window_complete':len(self.history)>=self.review_evaluations,
                'automatic_configuration_changes':False,
                'note':'Observation only: reaching the review count neither stops learning nor changes configuration.',
                'results':[{'iteration':x['iteration'],'additional_steps':x['additional_steps'],
                    'nominal':x.get('nominal'),'evaluation_error':x.get('evaluation_error')} for x in self.history]}
            atomic_json(self.out/'review_window.json',review)
        with (self.out/'evaluations.jsonl').open('a') as f:f.write(json.dumps(entry,allow_nan=False)+'\n')
        self.next_eval=it+self.interval
        for old in sorted(self.out.glob('eval_*.pt'))[:-int(self.settings['retained_evaluation_checkpoints'])]:old.unlink()
        print('[SUPERVISOR] '+json.dumps(entry,allow_nan=False),flush=True)
        if entry['operational_targets_met']:
            self.publish('targets_met_video_review_pending',accepted_checkpoint=str(self.out/'ckpt_accepted.pt'),
                last_evaluation=entry)
            raise TrainingStopped('targets_met_video_review_pending')
        self.publish('training',last_evaluation=entry)
