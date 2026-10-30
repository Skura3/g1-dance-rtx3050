import sys
import unittest
from pathlib import Path
import json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from compare_finetuning import assess

class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.criteria=json.loads((Path(__file__).resolve().parents[1]/'configs/dance_acceptance.json').read_text())
        self.good={'failed':False,'steps':616,'tilt_mean_deg':4.,'tilt_max_deg':12.,
            'yaw_error_mean_abs_deg':4.,'yaw_error_max_abs_deg':18.,'joint_rmse_rad':.18,
            'root_error_mean_m':.08,'root_error_max_m':.2,
            'arms_joint_rmse_rad':.23,'worst_2s_joint_rmse_rad':.28}
    def test_accepts_approximate_complete_dance(self):
        self.assertTrue(assess(self.good,self.criteria)['provisional_targets_met'])
    def test_survival_does_not_hide_half_turn_error(self):
        d=dict(self.good,yaw_error_max_abs_deg=179.)
        self.assertFalse(assess(d,self.criteria)['provisional_targets_met'])
    def test_partial_or_failed_clip_is_rejected(self):
        for d in [dict(self.good,steps=400),dict(self.good,failed=True)]:
            self.assertIn('incomplete_physical_rollout',assess(d,self.criteria)['exceeded_targets'])
    def test_global_pass_cannot_hide_local_errors(self):
        for metric in ('arms_joint_rmse_rad','worst_2s_joint_rmse_rad'):
            for value in (1.0,float('nan'),None):
                self.assertFalse(assess(dict(self.good,**{metric:value}),self.criteria)['provisional_targets_met'])
            missing=dict(self.good);missing.pop(metric)
            self.assertFalse(assess(missing,self.criteria)['provisional_targets_met'])
    def test_strict_joint_boundary(self):
        limit=self.criteria['joint_rmse_rad_max']
        self.assertTrue(assess(dict(self.good,joint_rmse_rad=limit),self.criteria)['provisional_targets_met'])
        self.assertFalse(assess(dict(self.good,joint_rmse_rad=limit+1e-6),self.criteria)['provisional_targets_met'])
    def test_old_loose_pass_is_no_longer_accepted(self):
        row=dict(self.good,tilt_mean_deg=7.06,yaw_error_mean_abs_deg=7.48,
            yaw_error_max_abs_deg=36.19,joint_rmse_rad=.342,root_error_mean_m=.133,root_error_max_m=.281)
        self.assertFalse(assess(row,self.criteria)['provisional_targets_met'])

if __name__=='__main__':unittest.main()
