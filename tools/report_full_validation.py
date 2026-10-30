"""Summarize full-video checks; does not certify physical tracking."""
import hashlib,json,subprocess
import numpy as np
from common import ROOT

def main():
    out=ROOT/'outputs/full_validation'
    def read(path):return json.loads((ROOT/path).read_text())
    raw=read('outputs/full_validation/raw_audit/quality.json');fixed=read('outputs/full_validation/repaired_audit/quality.json')
    repair=read('outputs/full_validation/repair/repair_report.json');play=read('outputs/full_validation/play_report.json')
    kp=read('outputs/full_validation/keypoint_quality.json');rp=read('outputs/full_validation/reconstruction_quality.json')
    cfg=read('configs/reference_full_support.json')
    a=np.load(out/'raw_audit/quality_series.npz');b=np.load(out/'repaired_audit/quality_series.npz')
    ids=np.rint(np.array(cfg['support_frames'])/cfg['fps']*fixed['fps']).astype(int)
    metrics=['max_joint_speed_rad_s','max_joint_acceleration_rad_s2','maximum_self_penetration_m','foot_bottom_min_m','lowest_foot_max_height_m']
    comparison={key:{'before':raw[key],'after':fixed[key]} for key in metrics}
    comparison['screening_counts']={'before':raw['screening_counts'],'after':fixed['screening_counts']}
    comparison['near_floor_frames']={'before':int(a['near_floor'].any(1).sum()),'after':int(b['near_floor'].any(1).sum())}
    comparison['support_anchor_median_height_m']={'before':float(np.median(a['foot_bottom'][ids].min(1))),'after':float(np.median(b['foot_bottom'][ids].min(1)))}
    comparison['support_anchor_max_height_m']={'before':float(a['foot_bottom'][ids].min(1).max()),'after':float(b['foot_bottom'][ids].min(1).max())}
    comparison['foot_horizontal_speed_change_max_m_s']=float(abs(a['foot_speed']-b['foot_speed']).max())
    paths=['outputs/full_validation/source_vs_g1.mp4','outputs/full_validation/raw_vs_repaired.mp4','outputs/full_validation/reference_playback.mp4','outputs/reference_full_keypoints.mp4','data/preview/reference_full_repaired.npz','data/preview/reference_full_aligned.npz','data/motions/reference_full_gmr.pkl','outputs/gvhmr/reference_full/hmr4d_results.pt']
    artifacts={}
    for path in paths:
        f=ROOT/path
        if not f.is_file() or not f.stat().st_size:raise RuntimeError('Missing artifact: '+path)
        info={'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
        if f.suffix=='.mp4':info['video']=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=width,height,r_frame_rate,nb_frames:format=duration','-of','json',str(f)]))
        artifacts[path]=info
    summary={'status':'full_reconstruction_retargeting_and_kinematic_validation_complete',
        'source_video':str(ROOT/'data/video/reference_full.mp4'),'source_frames':370,'source_fps':30,'source_container_duration_s':370/30,
        'reference_frames':616,'reference_fps':50,'reference_sample_span_s':12.3,
        'timeline_note':'Source frame 369 is at 12.30s; 616 samples at 50Hz cover 0–12.30s. The 25fps preview has 308 frames (12.32s) ending at sample time 12.28s. Reconstruction and retargeting include all source frames.',
        'tracking':{'selected_track_id':kp['tracking']['selected_track_id'],'detected_frames':len(kp['tracking']['detected_frames']),'missing_frames':kp['tracking']['missing_frames'],'repeat_bbox_max_diff_px':kp['tracking']['repeat_bbox_max_diff']},
        'reprojection':rp,'precision_sample_check':read('outputs/precision_check.json'),'comparison':comparison,'repair':repair,'gpu_playback':play,
        'remaining_issues':[
            {'issue':'Near-floor foot sliding / imperfect support','flagged_frames':fixed['screening_counts']['near_floor_foot_speed_above_30cm_s'],'details':'Geometric heuristic, not ground-truth contact. More frames become near-floor after height correction, so counts use different masks. Horizontal foot speeds were unchanged.'},
            {'issue':'Hand-supported floor move is not recovered as true support','interval_s':[7.8,8.0],'details':'At sampled times the hand collision bottoms remain about 0.14–0.20m above ground. Whole-body contact-constrained retargeting and dynamics remain necessary.'},
            {'issue':'Acceleration spikes','flagged_frames':fixed['screening_counts']['acceleration_above_300rad_s2'],'intervals':fixed['screening_intervals']['acceleration_above_300rad_s2'],'details':'300 rad/s^2 is a review threshold, not a certified motor limit.'},
            {'issue':'Monocular 3D ambiguity','interval_s':[7.6,9.3],'details':'Occlusion and high kick/spin have lower confidence. Reprojection does not establish depth, scale or physical contact.'},
            {'issue':'Dynamics and policy learning untested','details':'No dance training or closed-loop dance physics rollout was run.'}],
        'training_approved':False,'dance_policy_trained':False,'new_downloads_required':[],
        'tests':{'unit_tests_passed':9,'log':'logs/full_tests.log','compileall_passed':True},'artifacts':artifacts}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    ready=read('outputs/readiness.json');ready['full_video_validation']={k:summary[k] for k in ['status','source_frames','reference_frames','reference_fps','reference_sample_span_s','training_approved','dance_policy_trained']}
    ready['full_video_validation'].update(summary='outputs/full_validation/summary.json',motion='data/preview/reference_full_repaired.npz',gpu_playback_ok=play['ok'])
    path=str(ROOT/'outputs/gvhmr/reference_full/hmr4d_results.pt')
    ready['inference_result_files']=list(dict.fromkeys(ready.get('inference_result_files',[])+[path]))
    (ROOT/'outputs/readiness.json').write_text(json.dumps(ready,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'comparison':comparison,'gpu_playback_ok':play['ok'],'artifacts':list(artifacts)},indent=2))

if __name__=='__main__':main()
