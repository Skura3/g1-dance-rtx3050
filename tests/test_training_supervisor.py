import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from training_supervisor import SuccessGate,TrainingSupervisor,TrainingStopped,link_checkpoint
from compare_finetuning import assess

ROOT=Path(__file__).resolve().parents[1]

class SupervisionTests(unittest.TestCase):
    def test_nominal_failure_resets_streak(self):
        gate=SuccessGate()
        self.assertFalse(gate.nominal(4000,True))
        self.assertFalse(gate.nominal(8000,False))
        self.assertFalse(gate.nominal(12000,True))
        self.assertTrue(gate.nominal(16000,True))
        with self.assertRaises(ValueError):gate.nominal(16000,True)
        self.assertFalse(gate.robustness({11:True,23:False,37:True}))
        self.assertEqual(gate.streak,0)

    def test_nonfinite_or_missing_metrics_cannot_pass(self):
        c=json.loads((ROOT/'configs/dance_acceptance.json').read_text())
        good=dict(failed=False,steps=616,tilt_mean_deg=8.,tilt_max_deg=42.,
            yaw_error_mean_abs_deg=25.,yaw_error_max_abs_deg=70.,joint_rmse_rad=.28,
            root_error_mean_m=.4,root_error_max_m=.8)
        for value in (float('nan'),float('inf'),None):
            self.assertFalse(assess(dict(good,joint_rmse_rad=value),c)['provisional_targets_met'])
        self.assertFalse(assess(dict(good,steps=float('nan')),c)['provisional_targets_met'])

    def test_atomic_replacement_does_not_change_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);(p/'latest').write_bytes(b'old')
            link_checkpoint(p/'latest',p/'snapshot')
            (p/'replacement').write_bytes(b'new');(p/'replacement').replace(p/'latest')
            self.assertEqual((p/'snapshot').read_bytes(),b'old')

    def test_continuous_controller_waits_then_checks_robustness(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)
            trainer=SimpleNamespace(iteration=291800,num_updates=100,log_dir=out,
                cfg=SimpleNamespace(env=SimpleNamespace(num_envs=64)),
                lib=SimpleNamespace(clips=[SimpleNamespace(length=616)]))
            def save():
                tmp=out/'tmp';tmp.write_bytes(str(trainer.iteration).encode());tmp.replace(out/'ckpt_latest.pt')
            trainer.save=save
            settings=json.loads((ROOT/'configs/g1_3050_long_training.json').read_text())
            s=TrainingSupervisor(trainer,settings,ROOT/'data/preview/reference_full_contact.npz',291800)
            self.assertEqual(json.loads((out/'criteria_snapshot.json').read_text()),s.criteria)
            self.assertEqual(json.loads((out/'status.json').read_text())['criteria_sha256'],s.criteria_sha256)
            calls=[]
            def rollout(checkpoint,folder,seed,noise):
                folder.mkdir(parents=True,exist_ok=True);calls.append((trainer.iteration,seed,noise))
                return {'provisional_targets_met':True,'selection_score_lower_is_better':1.,'checkpoint_sha256':'test'}
            s.run_rollout=rollout
            with patch('builtins.print'):
                trainer.iteration=295799;s.before_step();self.assertFalse(calls)
                trainer.iteration=295800;s.before_step()
                self.assertEqual(len(calls),1);self.assertEqual(s.next_eval,299800)
                trainer.iteration=299800
                with self.assertRaises(TrainingStopped):s.before_step()
            self.assertEqual([c[1] for c in calls],[0,0,11,23,37])
            self.assertTrue((out/'ckpt_accepted.pt').exists())
            self.assertEqual(json.loads((out/'status.json').read_text())['status'],'targets_met_video_review_pending')

    def test_failed_nominal_rollout_continues(self):
        gate=SuccessGate()
        for it in range(4000,40001,4000):self.assertFalse(gate.nominal(it,False))
        self.assertEqual(gate.streak,0)

    def test_resume_preserves_first_scheduled_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            trainer=SimpleNamespace(iteration=328252,num_updates=100,log_dir=Path(temp),
                cfg=SimpleNamespace(env=SimpleNamespace(num_envs=64)),
                lib=SimpleNamespace(clips=[SimpleNamespace(length=616)]))
            settings=json.loads((ROOT/'configs/g1_3050_long_training.json').read_text())
            settings['first_evaluation_iteration']=329526
            supervisor=TrainingSupervisor(trainer,settings,ROOT/'data/preview/reference_full_contact.npz',328252)
            self.assertEqual(supervisor.next_eval,329526)
            settings['first_evaluation_iteration']=328252
            with self.assertRaises(ValueError):TrainingSupervisor(trainer,settings,ROOT/'data/preview/reference_full_contact.npz',328252)

    def test_three_evaluation_review_does_not_stop_or_change_training(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)
            trainer=SimpleNamespace(iteration=10000,num_updates=100,log_dir=out,
                cfg=SimpleNamespace(env=SimpleNamespace(num_envs=64)),
                lib=SimpleNamespace(clips=[SimpleNamespace(length=616)]))
            def save():
                tmp=out/'tmp';tmp.write_bytes(str(trainer.iteration).encode());tmp.replace(out/'ckpt_latest.pt')
            trainer.save=save
            settings=json.loads((ROOT/'configs/g1_3050_long_training.json').read_text())
            settings['review_after_evaluations']=3;settings['retained_evaluation_checkpoints']=3
            supervisor=TrainingSupervisor(trainer,settings,ROOT/'data/preview/reference_full_contact.npz',10000)
            def rollout(checkpoint,folder,seed,noise):
                folder.mkdir(parents=True,exist_ok=True)
                return {'provisional_targets_met':False,'selection_score_lower_is_better':2.,'checkpoint_sha256':'test'}
            supervisor.run_rollout=rollout
            with patch('builtins.print'):
                for iteration in [14000,18000,22000]:
                    trainer.iteration=iteration;supervisor.before_step()
            status=json.loads((out/'status.json').read_text())
            self.assertTrue(status['review_window_complete']);self.assertEqual(status['status'],'training')
            self.assertEqual(supervisor.next_eval,26000);self.assertFalse(supervisor.stop_requested)
            self.assertEqual(len(list(out.glob('eval_*.pt'))),3)

if __name__=='__main__':unittest.main()
