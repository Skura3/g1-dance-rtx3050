"""Rank independently recorded physics rollouts without treating survival as fidelity."""
import argparse
import json
import math
from pathlib import Path
from common import ROOT


def assess(row, criteria, expected_steps=616):
    mapping={'mean_tilt_error_deg_max':'tilt_mean_deg','peak_tilt_error_deg_max':'tilt_max_deg',
        'mean_absolute_yaw_error_deg_max':'yaw_error_mean_abs_deg','peak_absolute_yaw_error_deg_max':'yaw_error_max_abs_deg',
        'joint_rmse_rad_max':'joint_rmse_rad','mean_root_position_error_m_max':'root_error_mean_m',
        'peak_root_position_error_m_max':'root_error_max_m'}
    for criterion,metric in {
        'arms_joint_rmse_rad_max':'arms_joint_rmse_rad',
        'worst_2s_joint_rmse_rad_max':'worst_2s_joint_rmse_rad',
    }.items():
        if criterion in criteria:mapping[criterion]=metric
    failures=[]
    steps=row.get('steps',0)
    if row.get('failed') is not False or not isinstance(steps,(int,float)) or not math.isfinite(steps) or steps<expected_steps:
        failures.append('incomplete_physical_rollout')
    invalid=[]
    for criterion,metric in mapping.items():
        value=row.get(metric)
        if not isinstance(value,(int,float)) or not math.isfinite(value):
            invalid.append(metric);failures.append('invalid_'+metric)
        elif value>criteria[criterion]:failures.append(metric)
    # A ranking convenience, not a training reward or externally validated score.
    score=None if invalid else (row['yaw_error_mean_abs_deg']/45 + row['joint_rmse_rad']/.35
        +row['tilt_mean_deg']/15 +row['root_error_mean_m']/.6)
    if score is not None and 'incomplete_physical_rollout' in failures:score+=100
    return {'provisional_targets_met':not failures,'exceeded_targets':failures,'selection_score_lower_is_better':score}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reports',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();criteria=json.loads((ROOT/'configs/dance_acceptance.json').read_text());rows=[]
    for path in a.reports:
        r=json.loads(path.read_text())
        if r.get('provenance_valid') is False:continue
        for c in r['clips']:
            if c['name']!='reference_full_contact':continue
            rows.append(dict(c,report=str(path),checkpoint=r['checkpoint'],seed=r.get('seed',0),reset_noise=r.get('reset_noise',False),**assess(c,criteria)))
    nominal=[x for x in rows if not x['reset_noise']]
    if not nominal:raise ValueError('Need a nominal reference_full_contact rollout')
    best=min(nominal,key=lambda x:float('inf') if x['selection_score_lower_is_better'] is None else x['selection_score_lower_is_better'])
    result={'criteria':criteria,'results':rows,'preferred_checkpoint':best['checkpoint'],
        'preferred_report':best['report'],'provisional_targets_met':best['provisional_targets_met'],
        'scope':'Same-reference whole-clip physics comparisons; require visual review and robustness checks before final acceptance.'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
