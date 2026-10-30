"""Evaluate a published NJU policy in the existing CUDA12 mjlab environment."""
import argparse,json,os
from pathlib import Path
from common import ROOT,add_sources

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--motions',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/nju_validation/policy_evaluation.json')
    a=p.parse_args()
    if not a.checkpoint.is_file():p.error('Checkpoint missing; see outputs/nju_validation/download_status.json')
    if any(not f.is_file() for f in a.motions):p.error('A reference motion is missing')
    os.environ['G1_SUPERMIMIC_SOURCE']='nju';add_sources()
    import torch,supermimic
    from supermimic.policy.nets import Policy,PolicyCfg
    from supermimic.eval.benchmark import run_benchmark,save_results,summarize
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    # Keep optimizer/world-model checkpoint state off the GPU during policy evaluation.
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    cfg=ck['config'];policy=Policy(PolicyCfg(**{k:tuple(v) if isinstance(v,list) else v for k,v in cfg['policy'].items()}))
    policy.load_state_dict(ck['policy']);del ck
    policy.to('cuda:0').eval();env=cfg['env']
    a.output.parent.mkdir(parents=True,exist_ok=True)
    wc=(env.get('waist_coupling_k',1.4),env.get('waist_coupling_c',.95)) if env.get('waist_coupling',False) else None
    results=run_benchmark(policy,[str(f.resolve()) for f in a.motions],num_envs=min(8,len(a.motions)),future_frames=env['future_frames'],anchor_obs=env.get('anchor_obs',False),lock_waist=env.get('lock_waist',False),lock_waist_refs=env.get('lock_waist_refs','feasible'),waist_coupling=wc,waist_roll_range=env.get('waist_roll_range'))
    save_results(results,a.output)
    report=json.loads(a.output.read_text());report.update(checkpoint=str(a.checkpoint.resolve()),source_module=str(supermimic.__file__),torch_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,scope='Deterministic learned-policy physics evaluation from frame zero; no training or real robot')
    a.output.write_text(json.dumps(report,indent=2));print(json.dumps(summarize(results),indent=2))

if __name__=='__main__':main()
