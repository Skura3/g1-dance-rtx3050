import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from record_nju_rollout import orientation_quality,local_joint_quality


def poses(angles):
    angles=np.deg2rad(angles)
    q=np.zeros((len(angles),36));q[:,3]=np.cos(angles/2);q[:,6]=np.sin(angles/2)
    return q


class OrientationMetricTests(unittest.TestCase):
    def test_quaternion_sign_does_not_change_rotation(self):
        a=poses([20.,179.]);b=a.copy();b[:,3:7]*=-1
        r=orientation_quality(a,b,.02)
        self.assertLess(r['orientation_error_max_deg'],1e-5)
        self.assertLess(r['yaw_error_max_abs_deg'],1e-10)

    def test_yaw_error_wraps_at_180(self):
        r=orientation_quality(poses([-179.]),poses([179.]),.02)
        self.assertAlmostEqual(r['yaw_error_max_abs_deg'],2.,places=8)
        self.assertAlmostEqual(r['orientation_error_max_deg'],2.,places=8)

    def test_half_turn_detected_despite_upright_orientation(self):
        r=orientation_quality(poses([0.,180.]),poses([0.,0.]),.02)
        self.assertAlmostEqual(r['yaw_error_max_abs_deg'],180.)
        self.assertAlmostEqual(r['orientation_error_max_deg'],180.)
        self.assertAlmostEqual(r['yaw_error_peak_time_s'],.02)


class LocalJointMetricTests(unittest.TestCase):
    def test_zero_error_and_localized_arm_error(self):
        names=['left_hip_pitch_joint','left_shoulder_yaw_joint']
        reference=np.zeros((300,9));actual=reference.copy()
        zero=local_joint_quality(actual,reference,names,.02)
        self.assertEqual(zero['arms_joint_rmse_rad'],0.)
        self.assertEqual(zero['worst_2s_joint_rmse_rad'],0.)
        actual[100:200,8]=1.0
        result=local_joint_quality(actual,reference,names,.02)
        self.assertAlmostEqual(result['arms_joint_rmse_rad'],np.sqrt(1/3))
        self.assertAlmostEqual(result['worst_2s_joint_rmse_rad'],np.sqrt(.5))
        self.assertEqual(result['worst_2s_window_start_s'],2.)
        self.assertEqual(result['joint_quality_window_samples'],100)
    def test_invalid_or_short_trajectory_is_rejected(self):
        ref=np.zeros((100,8));actual=ref.copy();actual[0,7]=np.nan
        with self.assertRaises(ValueError):local_joint_quality(actual,ref,['left_elbow_joint'],.02)
        with self.assertRaises(ValueError):local_joint_quality(ref[:50],ref[:50],['left_elbow_joint'],.02)

if __name__=='__main__':unittest.main()
