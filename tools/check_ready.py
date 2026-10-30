"""Report actual assets and environment readiness without pretending missing stages ran."""
import importlib
import json
import platform
import subprocess
from pathlib import Path
from common import ROOT,CHECKPOINTS,add_sources

def main():
    add_sources();report={'python':platform.python_version(),'root':str(ROOT),'modules':{},'files':{}}
    for name in ['torch','torchvision','mujoco','mink','smplx','hydra','pytorch_lightning','pytorch3d','pycolmap','general_motion_retargeting']:
        try:
            m=importlib.import_module(name);report['modules'][name]={'ok':True,'version':getattr(m,'__version__',None)}
        except Exception as e:report['modules'][name]={'ok':False,'error':str(e)}
    required=['yolo/yolov8x.pt','vitpose/vitpose-h-multi-coco.pth','hmr2/epoch=10-step=25000.ckpt',
              'gvhmr/gvhmr_siga24_release.ckpt','body_models/smplx/SMPLX_NEUTRAL.npz']
    for name in required:
        p=CHECKPOINTS/name;p.parent.mkdir(parents=True,exist_ok=True)
        report['files'][name]={'path':str(p),'present':p.is_file() and p.stat().st_size>0}
    try:
        import torch
        report['cuda_available']=torch.cuda.is_available()
        if report['cuda_available']:
            report['gpu']=torch.cuda.get_device_name();free,total=torch.cuda.mem_get_info()
            report['cuda_free_mib']=free/1024**2;report['cuda_total_mib']=total/1024**2
    except Exception as e:report['cuda_error']=str(e)
    report['motion_assets_ready']=all(x['present'] for x in report['files'].values())
    audit_path=ROOT/'outputs/download_audit/assets.json'
    if audit_path.exists():
        audit=json.loads(audit_path.read_text())
        report['download_audit']={'path':str(audit_path),'all_ok_at_audit':audit['all_ok'],
                                  'checked_at':audit['checked_at'],'scope':audit['scope']}
    report['motion_assets_ready_scope']='Presence only; use tools/audit_downloads.py to recheck actual content.'
    train_report=ROOT/'outputs/training_readiness.json'
    report['training_environment']=json.loads(train_report.read_text()) if train_report.exists() else 'Not validated; see docs/TRAINING.md'
    report['inference_result_files']=[str(p) for p in (ROOT/'outputs/gvhmr').glob('*/hmr4d_results.pt')]
    report['video_3d_inference_executed']=bool(report['inference_result_files'])
    validation=ROOT/'outputs/environment_validation/summary.json'
    if validation.exists():
        report['g1_environment_validation']=json.loads(validation.read_text())
    report['pytorch3d_native_extensions']=False
    nju_audit=ROOT/'outputs/download_audit/nju_checkpoint.json'
    if nju_audit.exists():
        audit=json.loads(nju_audit.read_text())
        checkpoint=Path(audit['checkpoint'])
        report['nju_pretrained_checkpoint']={'audit':str(nju_audit),
            'verified_at_audit':audit['ok'],'checkpoint':str(checkpoint),
            'present':checkpoint.is_file(),'iteration':audit['iteration'],
            'scope':'Stored hash/content audit; does not rehash the checkpoint on this readiness check.'}
    for key,relative in [('nju_pretrained_evaluation','outputs/nju_validation/policy_evaluation.json'),
                         ('nju_recorded_rollouts','outputs/nju_validation/rollouts/report.json')]:
        path=ROOT/relative
        if path.exists():report[key]={'path':str(path),'result':json.loads(path.read_text())}
    report['note']='Readiness check only; does not establish training convergence or peak memory.'
    dest=ROOT/'outputs/readiness.json';dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()

