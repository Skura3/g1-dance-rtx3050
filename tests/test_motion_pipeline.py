"""Numerical tests use temporary synthetic fixtures, never the user's training data."""
import sys
import tempfile
import unittest
import pickle
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from common import add_sources
add_sources()
from convert_motion import convert,align_height,capsule_bottoms,DEFAULT_XML
from supermimic.data.motion_convert import G1_JOINT_NAMES,load_g1_model
from repair_motion import height_correction
from audit_motion import intervals,bottom
from refine_contacts import geometry_bottom,refine
from audit_dynamics import contact_force_jacobian

class MotionConversionTests(unittest.TestCase):
    def test_contact_surface_force_matches_mujoco_virtual_work(self):
        import mujoco
        data=mujoco.MjData(self.model)
        data.qpos[:]=self.model.qpos0
        mujoco.mj_forward(self.model,data)
        force=np.array([13.,-9.,85.])
        checked=0
        for geom in range(self.model.ngeom):
            if not (self.model.geom_contype[geom] or self.model.geom_conaffinity[geom]):continue
            point,jac=contact_force_jacobian(self.model,data,geom)
            expected=np.zeros(self.model.nv)
            mujoco.mj_applyFT(self.model,data,force,np.zeros(3),point,int(self.model.geom_bodyid[geom]),expected)
            np.testing.assert_allclose(jac.T@force,expected,rtol=1e-12,atol=1e-12)
            checked+=1
        self.assertGreater(checked,10)

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.dir=Path(self.tmp.name)
        self.model=load_g1_model(DEFAULT_XML)
        t=np.arange(31)/30
        pos=np.column_stack([0.2*t,np.zeros(31),0.85+0.1*np.sin(np.pi*t)])
        q=np.zeros((31,4));q[:,3]=1
        self.motion={'fps':30,'root_pos':pos,'root_rot':q,'dof_pos':np.zeros((31,29)),
                     'joint_names':list(G1_JOINT_NAMES),'root_quaternion_order':'xyzw'}
    def tearDown(self):self.tmp.cleanup()
    def write(self):
        path=self.dir/'fixture.pkl'
        with path.open('wb') as f:pickle.dump(self.motion,f)
        return path
    def test_height_offset_preserves_jump(self):
        z=self.motion['root_pos'];aligned=align_height(z,-0.13)
        np.testing.assert_allclose(np.diff(aligned,axis=0),np.diff(z,axis=0),atol=1e-14)
        np.testing.assert_allclose(aligned[:,:2],z[:,:2])
    def test_resampling_geometry_velocity_and_names(self):
        self.motion['dof_pos'][:,0]=0.1
        self.motion['joint_names']=list(reversed(G1_JOINT_NAMES))
        self.motion['dof_pos']=self.motion['dof_pos'][:,::-1]
        out=self.dir/'result.npz';report=convert(self.write(),out,ground_offset=-0.1)
        self.assertEqual(report['frames_out'],51)
        with np.load(out,allow_pickle=False) as d:
            self.assertEqual(d['body_pos_w'].shape,(51,30,3))
            self.assertEqual(list(d['joint_names']),list(G1_JOINT_NAMES))
            np.testing.assert_allclose(d['joint_pos'][:,0],0.1)
            np.testing.assert_allclose(d['body_lin_vel_w'][:,0,0],0.2,atol=1e-6)
            np.testing.assert_allclose(np.linalg.norm(d['body_quat_w'],axis=-1),1,atol=1e-6)
            np.testing.assert_allclose(d['body_pos_w'][[0,-1],0,2],0.75,atol=1e-6)
            self.assertGreater(np.ptp(d['body_pos_w'][:,0,2]),0.099)
    def test_capsule_heights_translate_exactly(self):
        q=np.repeat(self.model.qpos0[None],3,axis=0);q[:,2]=[0.8,0.9,1.]
        z=capsule_bottoms(self.model,q)
        np.testing.assert_allclose(np.diff(z),[0.1,0.1],atol=1e-12)
    def test_rejects_unknown_joint_order(self):
        del self.motion['joint_names']
        with self.assertRaisesRegex(ValueError,'joint_names'):convert(self.write(),self.dir/'x.npz',ground_offset=0)
    def test_rejects_zero_quaternion(self):
        self.motion['root_rot'][2]=0
        with self.assertRaisesRegex(ValueError,'Zero quaternion'):convert(self.write(),self.dir/'x.npz',ground_offset=0)
    def test_support_range_uses_one_constant(self):
        out=self.dir/'x.npz';r=convert(self.write(),out,support_range=(0,0.2))
        with np.load(out) as d:
            self.assertGreater(np.ptp(d['body_pos_w'][:,0,2]),0.099)
        self.assertEqual(r['support_range_s'],(0,0.2))
    def test_sparse_drift_repair_keeps_airborne_arc(self):
        fps=50.;t=np.arange(101)/fps
        drift=.1+.05*t
        jump=np.where((t>.5)&(t<1.5),.2*np.sin(np.pi*np.clip(t-.5,0,1)),0.)
        z=drift+jump;anchors=np.array([0,20,80,100])
        shift,report=height_correction(z,z,anchors,fps)
        fixed=z+shift
        self.assertLess(np.max(abs(fixed[anchors]-.002)),.003)
        self.assertGreater(fixed[50],.18)
        self.assertGreaterEqual(fixed.min(),-.00101)
        self.assertLess(np.max(abs(np.diff(shift))),.002)
    def test_interval_report_preserves_frame_endpoints(self):
        self.assertEqual(intervals([False,True,True,False,True],50),[
            {'start_frame':1,'end_frame':2,'start_s':.02,'end_s':.04,'frames':2},
            {'start_frame':4,'end_frame':4,'start_s':.08,'end_s':.08,'frames':1}])
    def test_audit_foot_geometry_matches_conversion(self):
        import mujoco
        data=mujoco.MjData(self.model);q=self.model.qpos0.copy();q[2]=.85
        data.qpos[:]=q;mujoco.mj_forward(self.model,data)
        ids=[i for i in range(self.model.ngeom) if self.model.geom(i).name.startswith(('left_foot','right_foot'))]
        self.assertAlmostEqual(min(bottom(self.model,data,i) for i in ids),capsule_bottoms(self.model,q[None])[0],places=12)
    def test_contact_height_jacobian_matches_finite_difference(self):
        import mujoco
        data=mujoco.MjData(self.model);q=self.model.qpos0.astype(float).copy()
        q[2]=.85;q[7:]=np.clip(np.linspace(-.25,.35,29),self.model.jnt_range[1:,0],self.model.jnt_range[1:,1])
        for name in ['left_hand_collision','left_foot3_collision','pelvis_collision']:
            data.qpos[:]=q;mujoco.mj_forward(self.model,data);g=self.model.geom(name).id
            point,jac=geometry_bottom(self.model,data,g)
            self.assertAlmostEqual(point[2],bottom(self.model,data,g),places=12)
            for col in [0,2,6,18,21,25]:
                dv=np.zeros(self.model.nv);dv[col]=1.;perturbed=q.copy()
                mujoco.mj_integratePos(self.model,perturbed,dv,1e-6);data.qpos[:]=perturbed;mujoco.mj_forward(self.model,data)
                shifted,_=geometry_bottom(self.model,data,g)
                self.assertLess(abs((shifted[2]-point[2])/1e-6-jac[2,col]),2e-6)
    def test_contact_refinement_cannot_overwrite_source(self):
        path=self.dir/'same.npz'
        with self.assertRaisesRegex(ValueError,'Keep original'):refine(path,path,self.dir/'x.json',self.dir/'report')

if __name__=='__main__':unittest.main(verbosity=2)
