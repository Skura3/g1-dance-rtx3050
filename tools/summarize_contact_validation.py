"""Assemble measured results without promoting the candidate to training data."""
import json
from pathlib import Path
import numpy as np
from common import ROOT


def read(path):
    return json.loads((ROOT/path).read_text())


def main():
    contact=read('outputs/contact_validation/contact_report.json')
    result={'training_approved':False,'frames':contact['frames'],'reference':'data/preview/reference_full_contact.npz'}
    before=np.load(ROOT/contact['source'],allow_pickle=False)
    after=np.load(ROOT/contact['output'],allow_pickle=False)
    fps=float(after['fps'][0]);speeds={key:[] for key in ('before','after')}
    for body in ('left_foot','right_foot'):
        pair_mask=np.zeros(len(after['joint_pos'])-1,dtype=bool)
        for c in contact['contacts']:
            if c['body']!=body:continue
            frames=np.asarray(c['frames'])
            pair_mask[frames[:-1]]=True
        name=body.replace('_foot','_ankle_roll_link')
        index=list(after['body_names']).index(name)
        for label,arr in [('before',before),('after',after)]:
            velocity=np.linalg.norm(np.diff(arr['body_pos_w'][:,index,:2],axis=0)*fps,axis=1)
            speeds[label].extend(velocity[pair_mask].tolist())
    result['matched_foot_contact_motion']={key:{'samples':len(v),'mean_xy_speed_m_s':float(np.mean(v)),'p95_xy_speed_m_s':float(np.percentile(v,95))} for key,v in speeds.items()}
    result['max_joint_change_rad']=contact['max_joint_change_rad']
    result['max_root_translation_change_m']=contact['max_root_translation_change_m']
    result['gpu_playback']=read('outputs/contact_validation/play_report.json')
    result['dynamics']=read('outputs/contact_validation/dynamics/dynamics_report.json')
    result['physics_probes']=read('outputs/contact_validation/physics_probes/probe_report.json')
    result['geometry']=read('outputs/contact_validation/audit/quality.json')
    result['performance']={}
    for p in sorted((ROOT/'outputs/performance').glob('training_*.json')):
        d=json.loads(p.read_text())
        if not d.get('ok'):continue
        excluded=d.get('timing_warmup_updates_excluded',1);steady=d['updates'][excluded:]
        keys=['num_envs','batch','cpu_threads','matmul_precision','kinematics','environment_step_median_s','aggregate_environment_steps_per_s','estimated_iteration_s_at_utd_0_4','gpu_total_memory_used_peak_mib','torch_peak_allocated_mib','torch_peak_reserved_mib','gpu_temperature_max_c']
        row={k:d.get(k) for k in keys}
        row.update(optimizer_updates=len(d['updates']),world_model_update_median_s=float(np.median([u['world_model_s'] for u in steady])),policy_critic_update_median_s=float(np.median([u['policy_critic_s'] for u in steady])))
        row['estimated_transitions_per_wall_s_at_utd_0_4']=d['num_envs']/d['estimated_iteration_s_at_utd_0_4']
        result['performance'][p.name]=row
    a=result['performance']['training_512_b128_t1_original_8.json']
    b=result['performance']['training_512_b128_t1_layered_8.json']
    result['matched_training_speedup']=a['estimated_iteration_s_at_utd_0_4']/b['estimated_iteration_s_at_utd_0_4']
    result['performance_scope']='Short measured optimizer/step costs with warmup; iteration time is UTD-weighted estimate, excluding periodic diagnostics/checkpoint IO. No long training or convergence claim.'
    result['next_work']='Refine support schedule and ballistic/centroidal motion, then torque-aware trajectory optimization and staged physical tracking; do not start a long full-dance job with this candidate.'
    destination=ROOT/'outputs/contact_validation/summary.json'
    destination.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ['reference','training_approved','matched_foot_contact_motion','matched_training_speedup']},indent=2))
    before.close();after.close()


if __name__=='__main__':main()
