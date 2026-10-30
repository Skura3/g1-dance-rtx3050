"""Regression checks for the opt-in differentiable G1 FK optimization."""
import sys
import unittest
import importlib.util
from pathlib import Path

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from common import add_sources
add_sources()
from supermimic.data.motion_convert import load_g1_model
HAS_MJLAB=importlib.util.find_spec('mjlab') is not None
if HAS_MJLAB:
    from supermimic.kinematics.g1_fk import G1Kinematics
    from fast_g1_fk import LayeredG1Kinematics


@unittest.skipUnless(HAS_MJLAB,'Run FK tests with envs/training/.venv/bin/python (mjlab/Python 3.12)')
class LayeredFKTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=load_g1_model()
        cls.original=G1Kinematics(cls.model,dtype=torch.float64)
        cls.layered=LayeredG1Kinematics(cls.model,dtype=torch.float64)

    def check_shape(self,shape,velocities):
        rng=torch.Generator().manual_seed(42)
        q=torch.randn((*shape,self.model.nq),generator=rng,dtype=torch.float64,requires_grad=True)
        v=torch.randn((*shape,self.model.nv),generator=rng,dtype=torch.float64,requires_grad=True)
        lhs=self.original(q,v if velocities else None)
        rhs=self.layered(q,v if velocities else None)
        self.assertEqual(set(lhs),set(rhs))
        for key in lhs:torch.testing.assert_close(lhs[key],rhs[key],rtol=1e-10,atol=1e-11)
        weights={k:torch.randn(x.shape,generator=rng,dtype=x.dtype) for k,x in lhs.items()}
        inputs=(q,v) if velocities else (q,)
        a=torch.autograd.grad(sum((x*weights[k]).sum() for k,x in lhs.items()),inputs)
        b=torch.autograd.grad(sum((x*weights[k]).sum() for k,x in rhs.items()),inputs)
        for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=1e-9,atol=1e-10)

    def test_scalar_pose(self):self.check_shape((),False)
    def test_scalar_velocity(self):self.check_shape((),True)
    def test_batch_pose(self):self.check_shape((4,),False)
    def test_nested_batch_velocity(self):self.check_shape((2,4),True)


if __name__=='__main__':unittest.main()
