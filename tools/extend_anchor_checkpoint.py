"""Extend 680-d pretrained goals with nine simulation anchor features, initially ignored.

Preserve original policy/critic outputs and Adam moments, inserting zero new columns.
Uses upstream anchor observation and matching imagined features; no physics alteration.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from common import ROOT,add_sources


def insert_columns(x,at,count):
    import torch
    return torch.cat((x[:,:at],x.new_zeros((x.shape[0],count)),x[:,at:]),dim=1)


def adapt(state):
    import torch
    state=copy.deepcopy(state)
    cfg=state['config'];old=int(cfg['policy'].get('goal_dim',680));extra=9
    if cfg['env'].get('anchor_obs',False) or old!=680:raise ValueError('Expected unextended 680-d motion goals')
    for part,key,cut,scale,opt in [
        ('policy','encoders.motion.mlp.net.0.weight',old,'encoders.motion.scale','policy_opt'),
        ('critic','mlp.net.0.weight',71+old,'goal_scale','critic_opt')]:
        weights=state[part];weights[key]=insert_columns(weights[key],cut,extra)
        weights[scale]=torch.cat((weights[scale],weights[scale].new_ones(extra)))
        # Resolve the first-layer parameter by its position, not an assumed state id.
        params=[k for k in weights if k.endswith(('.weight','.bias'))]
        index=params.index(key)
        ids=[v for group in state[opt]['param_groups'] for v in group['params']]
        moments=state[opt]['state'][ids[index]]
        for name,value in list(moments.items()):
            if isinstance(value,torch.Tensor) and value.ndim==2:moments[name]=insert_columns(value,cut,extra)
    cfg['env']['anchor_obs']=True;cfg['policy']['goal_dim']=old+extra;cfg['policy_trainer']['anchor_obs']=True
    cfg['resume_reset_sampler']=True
    return state


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    os.environ['G1_SUPERMIMIC_SOURCE']='nju';add_sources()
    import torch
    from supermimic.policy.nets import Policy,PolicyCfg,Critic,CriticCfg
    torch.set_num_threads(1);torch.manual_seed(7)
    before=torch.load(a.input,map_location='cpu',weights_only=False);after=adapt(before)
    def models(s):
        pc=PolicyCfg(**s['config']['policy']);p=Policy(pc);v=Critic(CriticCfg(**s['config']['critic']),pcfg=pc)
        p.load_state_dict(s['policy'],strict=True);v.load_state_dict(s['critic'],strict=True)
        return p.eval(),v.eval()
    p0,v0=models(before);p1,v1=models(after)
    with torch.no_grad():
        obs=torch.randn(32,930);g=torch.randn(32,680);anchor=torch.randn(32,9);g1=torch.cat((g,anchor),dim=-1)
        feat=torch.randn(32,71);xi=torch.randn(32,33);error=torch.randn(32,4)
        ap=(p0(obs,g)-p1(obs,g1)).abs().max().item();av=(v0(feat,g,xi,error)-v1(feat,g1,xi,error)).abs().max().item()
        assert ap<2e-5 and av<2e-5,(ap,av)
    # Confirm migrated Adam tensors have the exact shapes of trainable parameters.
    for model,key in [(p1,'policy_opt'),(v1,'critic_opt')]:
        opt=torch.optim.Adam(model.parameters());opt.load_state_dict(after[key])
        for group in opt.param_groups:
            for param in group['params']:
                for name,tensor in opt.state[param].items():
                    if name!='step' and isinstance(tensor,torch.Tensor):assert tensor.shape==param.shape
    a.output.parent.mkdir(parents=True,exist_ok=True);torch.save(after,a.output)
    report={'input':str(a.input.resolve()),'output':str(a.output.resolve()),'goal_dim_before':680,'goal_dim_after':689,
        'policy_max_abs_difference':ap,'critic_max_abs_difference':av,'optimizer_shapes_verified':True,
        'policy_function_preserved_at_initialization':True,'source_sha256':hashlib.sha256(a.input.read_bytes()).hexdigest(),
        'scope':'Simulation-only behavior-preserving initialization; new observation must be learned and evaluated.'}
    a.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
