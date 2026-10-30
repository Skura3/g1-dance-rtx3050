"""Depth-batched G1 FK. Same tree/equations as upstream, fewer small GPU launches."""
from common import add_sources
add_sources()
import torch
import mujoco
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.kinematics.rotations import quat_apply,quat_from_angle_axis,quat_mul,quat_normalize

class LayeredG1Kinematics(G1Kinematics):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if any(link.joint_type!=mujoco.mjtJoint.mjJNT_HINGE for link in self.links[1:]):
            raise ValueError('Layered optimization requires the validated G1 tree: one hinge per nonroot body')
        depths=[0]*self.nbody
        for i in range(1,self.nbody):depths[i]=depths[self.links[i].parent_index]+1
        previous=[0];self.layers=[];order=[0]
        for depth in range(1,max(depths)+1):
            ids=[i for i,d in enumerate(depths) if d==depth]
            ix=torch.tensor(ids,device=self.device);parent=torch.tensor([previous.index(self.links[i].parent_index) for i in ids],device=self.device)
            self.layers.append({'parent':parent,'pos':self.body_pos[ix],'quat':self.body_quat[ix],
                'jpos':self.jnt_pos[ix],'axis':self.jnt_axis[ix],'q0':self.qpos0[ix],
                'qadr':torch.tensor([self.links[i].qpos_adr for i in ids],device=self.device),
                'vadr':torch.tensor([self.links[i].dof_adr for i in ids],device=self.device)})
            order.extend(ids);previous=ids
        self.restore_order=torch.tensor([order.index(i) for i in range(self.nbody)],device=self.device)

    def forward(self,qpos,qvel=None):
        root_quat=quat_normalize(qpos[...,3:7]);pos=qpos[...,None,:3];quat=root_quat.unsqueeze(-2)
        positions=[pos];quaternions=[quat];velocities=[];angular=[]
        if qvel is not None:
            lin=qvel[...,None,:3];ang=quat_apply(root_quat,qvel[...,3:6]).unsqueeze(-2);velocities=[lin];angular=[ang]
        for layer in self.layers:
            idx=layer['parent'];pp=pos[...,idx,:];pq=quat[...,idx,:]
            bpos=pp+quat_apply(pq,layer['pos'].expand_as(pp));bquat=quat_mul(pq,layer['quat'].expand_as(pq))
            anchor=bpos+quat_apply(bquat,layer['jpos'].expand_as(bpos));axis_w=quat_apply(bquat,layer['axis'].expand_as(bpos))
            angle=qpos[...,layer['qadr']]-layer['q0'];local=quat_from_angle_axis(angle,layer['axis'].expand_as(bpos))
            quat=quat_mul(bquat,local);pos=anchor-quat_apply(quat,layer['jpos'].expand_as(bpos))
            if qvel is not None:
                wjoint=axis_w*qvel[...,layer['vadr'],None];parent_ang=ang[...,idx,:];parent_lin=lin[...,idx,:]
                ang=parent_ang+wjoint;lin=parent_lin+torch.cross(parent_ang,pos-pp,dim=-1)+torch.cross(wjoint,pos-anchor,dim=-1)
                velocities.append(lin);angular.append(ang)
            positions.append(pos);quaternions.append(quat)
        ix=self.restore_order;out={'pos':torch.cat(positions,dim=-2)[...,ix,:],'quat':torch.cat(quaternions,dim=-2)[...,ix,:]}
        if qvel is not None:out.update(lin_vel=torch.cat(velocities,dim=-2)[...,ix,:],ang_vel=torch.cat(angular,dim=-2)[...,ix,:])
        return out

    __call__=forward


def enable_layered_fk():
    """Opt-in substitution at the Trainer's FK factory; preserve upstream files."""
    import supermimic.training.loop as training_loop
    training_loop.G1Kinematics=LayeredG1Kinematics
