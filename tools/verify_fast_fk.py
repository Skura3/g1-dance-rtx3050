"""Compare local layered FK with upstream values/gradients and matched timings."""
import argparse
import json
import time
from pathlib import Path

from common import ROOT, add_sources


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT/'outputs/performance/fk_validation.json')
    a = p.parse_args()
    add_sources()
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation
    from supermimic.data.motion_convert import load_g1_model
    from supermimic.kinematics.g1_fk import G1Kinematics
    from fast_g1_fk import LayeredG1Kinematics

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261029)
    model = load_g1_model()
    with np.load(ROOT/'data/preview/reference_full_contact.npz', allow_pickle=False) as d:
        root = list(d['body_names']).index('pelvis')
        q = np.column_stack([d['body_pos_w'][:,root], d['body_quat_w'][:,root], d['joint_pos']])
        rotation = Rotation.from_quat(q[:,[4,5,6,3]])
        v = np.column_stack([d['body_lin_vel_w'][:,root], rotation.inv().apply(d['body_ang_vel_w'][:,root]), d['joint_vel']])
    report = {'ok':False, 'reference_frames':len(q), 'cases':[], 'timings':[]}
    for device, dtype in [('cpu', torch.float64), ('cuda:0', torch.float32)]:
        original = G1Kinematics(model, device=device, dtype=dtype)
        layered = LayeredG1Kinematics(model, device=device, dtype=dtype)
        # All 616 real frames, plus scalar and two-leading-dimension batches.
        for shape in [(len(q),), (), (2,8)]:
            n = int(np.prod(shape)) if shape else 1
            ids = np.linspace(0,len(q)-1,n).astype(int)
            qp = torch.tensor(q[ids].reshape(*shape,model.nq), device=device, dtype=dtype, requires_grad=True)
            qv = torch.tensor(v[ids].reshape(*shape,model.nv), device=device, dtype=dtype, requires_grad=True)
            for with_velocity in (False, True):
                lhs = original(qp, qv if with_velocity else None)
                rhs = layered(qp, qv if with_velocity else None)
                errors = {}
                weights = {k:torch.randn_like(x) for k,x in lhs.items()}
                for key in lhs:
                    torch.testing.assert_close(rhs[key], lhs[key], rtol=2e-5, atol=2e-6)
                    errors[key] = float((rhs[key]-lhs[key]).abs().max())
                loss1 = sum((x*weights[k]).sum() for k,x in lhs.items())/n
                loss2 = sum((x*weights[k]).sum() for k,x in rhs.items())/n
                inputs = (qp,qv) if with_velocity else (qp,)
                grad1 = torch.autograd.grad(loss1, inputs)
                grad2 = torch.autograd.grad(loss2, inputs)
                for i,(g1,g2) in enumerate(zip(grad1,grad2)):
                    torch.testing.assert_close(g2,g1,rtol=3e-5,atol=3e-5)
                    errors[f'gradient_{i}'] = float((g2-g1).abs().max())
                report['cases'].append({'device':device,'dtype':str(dtype),'shape':shape,'velocities':with_velocity,'max_abs_errors':errors})
        if device.startswith('cuda'):
            for batch in (64,128):
                qp = torch.tensor(q[:batch], device=device, dtype=dtype, requires_grad=True)
                qv = torch.tensor(v[:batch], device=device, dtype=dtype, requires_grad=True)
                for name,fk in [('upstream',original),('layered',layered)]:
                    times=[]
                    for trial in range(15):
                        torch.cuda.synchronize(); start=time.perf_counter()
                        out=fk(qp,qv)
                        loss=sum(x.square().mean() for x in out.values())
                        torch.autograd.grad(loss,(qp,qv))
                        torch.cuda.synchronize()
                        if trial>=5:times.append(time.perf_counter()-start)
                    report['timings'].append({'implementation':name,'batch':batch,'forward_backward_median_s':float(np.median(times))})
    report['ok']=True
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
