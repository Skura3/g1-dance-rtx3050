import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from training_watchdog import (Watchdog,atomic_json,checkpoint_snapshot,
    latest_iteration,trigger_reason,update_progress)


class WatchdogTests(unittest.TestCase):
    def config(self,root):
        return {'output_dir':str(root/'guard'),'run_dir':str(root/'train'),
            'invocation_id':'test-invocation','training_service':'fake.service',
            'temperature_limit_c':100,'stall_limit_seconds':14400,
            'overheat_grace_seconds':15,'stall_grace_seconds':60,
            'termination_grace_seconds':15,'poll_seconds':10,'gpu_uuid':'fake',
            'wallpaper_record':str(root/'wallpaper.json'),'best_sources':{}}

    def test_strictly_greater_threshold_boundaries(self):
        cfg={'temperature_limit_c':100,'stall_limit_seconds':14400}
        self.assertIsNone(trigger_reason(100,14400,cfg))
        self.assertEqual(trigger_reason(101,0,cfg),'gpu_temperature_exceeded')
        self.assertEqual(trigger_reason(80,14400.01,cfg),'training_iteration_stalled')
        self.assertEqual(trigger_reason(None,14401,cfg),'training_iteration_stalled')
        self.assertIsNone(trigger_reason(None,100,cfg))

    def test_heartbeat_and_regressing_iteration_do_not_reset_stall_clock(self):
        state={'last_progress_clock':0,'highest_iteration':100}
        update_progress(state,100,7200);self.assertEqual(state['last_progress_clock'],0)
        update_progress(state,99,14401);self.assertEqual(state['no_progress_seconds'],14401)
        update_progress(state,101,14402);self.assertEqual(state['no_progress_seconds'],0)

    def test_reads_updates_even_if_status_writer_is_delayed(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);atomic_json(run/'status.json',{'iteration':100})
            (run/'updates.jsonl').write_text('{"iteration": 103}\n{unfinished')
            self.assertEqual(latest_iteration(run),103)

    def test_snapshot_survives_replacement_and_rejects_mismatched_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'run';run.mkdir();dst=root/'saved.pt'
            source=run/'ckpt_best_evaluated.pt';source.write_bytes(b'best')
            report={'iteration':10,'nominal':{'checkpoint_sha256':hashlib.sha256(b'best').hexdigest(),
                'selection_score_lower_is_better':1.2}}
            atomic_json(run/'best_evaluation.json',report)
            result=checkpoint_snapshot(run,dst);self.assertEqual(result['iteration'],10)
            replacement=run/'new';replacement.write_bytes(b'next');replacement.replace(source)
            self.assertEqual(dst.read_bytes(),b'best')
            with self.assertRaises(ValueError):checkpoint_snapshot(run,dst)
            self.assertEqual(dst.read_bytes(),b'best')

    def test_watchdog_restart_preserves_four_hour_timer(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=self.config(Path(tmp));first=Watchdog(cfg)
            first.state.update(highest_iteration=300,last_progress_clock=123.0)
            first.publish(status='watching')
            restarted=Watchdog(cfg)
            self.assertEqual(restarted.state['last_progress_clock'],123.0)
            self.assertEqual(restarted.state['highest_iteration'],300)

    def setup_stop(self,root):
        cfg=self.config(root);run=Path(cfg['run_dir']);run.mkdir()
        (run/'ckpt_latest.pt').write_bytes(b'latest')
        best=root/'known_best.pt';best.write_bytes(b'best')
        atomic_json(root/'wallpaper.json',{'restore_deferred':False})
        guard=Watchdog(cfg)
        guard.state.update(highest_iteration=200,temperature_c=101,no_progress_seconds=0,
            snapshots={'inherited_recent_best':{'path':str(best),'iteration':190,
                'selection_score_lower_is_better':1.1}})
        return guard,run

    def test_graceful_stop_preserves_best_and_records_confirmed_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);guard,run=self.setup_stop(root)
            def finish(seconds):
                self.assertTrue((run/'STOP').exists())
                atomic_json(run/'run_report.json',{'ok':True,'stop_reason':'stopped_by_request','final_iteration':201})
                return True
            with patch.object(guard,'same_active',return_value=True),patch.object(guard,'wait_for_exit',side_effect=finish),patch.object(guard,'command') as command,patch('builtins.print'):
                guard.stop_training('gpu_temperature_exceeded')
                command.assert_not_called()
            incident=json.loads((guard.out/'incident.json').read_text())
            self.assertTrue(incident['latest_state_save_confirmed'])
            self.assertEqual(incident['reason'],'gpu_temperature_exceeded')
            self.assertEqual((guard.out/'incident_checkpoints/ckpt_best_evaluated.pt').read_bytes(),b'best')
            self.assertTrue(json.loads((root/'wallpaper.json').read_text())['restore_deferred'])

    def test_stuck_process_escalates_and_does_not_claim_a_new_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);guard,run=self.setup_stop(root)
            with patch.object(guard,'same_active',return_value=True),patch.object(guard,'wait_for_exit',side_effect=[False,False,True]),patch.object(guard,'command',return_value=True) as command,patch('builtins.print'):
                guard.stop_training('training_iteration_stalled')
            incident=json.loads((guard.out/'incident.json').read_text())
            self.assertFalse(incident['latest_state_save_confirmed'])
            self.assertTrue(incident['forced_kill_requested'])
            self.assertEqual(command.call_count,2)
            self.assertEqual(command.call_args.args,('kill','--kill-who=all','--signal=SIGKILL'))
            self.assertFalse(json.loads((root/'wallpaper.json').read_text())['restore_deferred'])

    def test_different_invocation_is_not_signaled(self):
        with tempfile.TemporaryDirectory() as tmp:
            guard=Watchdog(self.config(Path(tmp)))
            with patch.object(guard,'service',return_value={'InvocationID':'replacement','ActiveState':'active'}),patch('training_watchdog.subprocess.run') as run:
                self.assertFalse(guard.command('stop','--no-block'));run.assert_not_called()


if __name__=='__main__':unittest.main()
