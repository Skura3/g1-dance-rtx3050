"""Idempotent, guarded patches for the pinned GVHMR commit."""
import json
from common import GVHMR, ROOT

PATCHES = {
 'hmr4d/utils/preproc/tracker.py': [
  ('track_id = id_sorted[0]', 'track_id = id_sorted[0]\n        self.tracking_diagnostics = {"selected_track_id": int(track_id), "detected_frames": [int(i) for i in id_to_frame_ids[track_id]], "tracks": {int(k): len(v) for k,v in id_to_frame_ids.items()}, "total_frames": len(track_history)}'),
 ],
 'hmr4d/utils/geo_transform.py': [
  ('import pytorch3d.ops.knn as knn', '# KNN is optional; import only in the helper that actually calls it.'),
  ('squared_dist, _, _ = knn.knn_points(points[None], query_verts[None], K=1, return_nn=False)',
   'from pytorch3d.ops import knn_points  # Requires the optional native extension.\n        squared_dist, _, _ = knn_points(points[None], query_verts[None], K=1, return_nn=False)'),
 ],
 'hmr4d/utils/preproc/vitpose_pytorch/src/vitpose_infer/model_builder.py': [
  ('check = torch.load(checkpoint)', 'check = torch.load(checkpoint, map_location="cpu")'),
 ],
 'hmr4d/utils/preproc/vitpose.py': [
  ('import torch\n','import os\nimport torch\n'),
  ('self.pose.cuda().eval()', 'self.pose.to(os.environ.get("GVHMR_PREPROC_DEVICE", "cuda")).eval()'),
  ('self.flip_test = True','self.flip_test = os.environ.get("GVHMR_FLIP_TEST", "1") == "1"'),
  ('batch_size = 16','batch_size = max(1, int(os.environ.get("GVHMR_BATCH_SIZE", "1")))'),
  ('imgs_batch = imgs[j : j + batch_size, :, :, 32:224].cuda()',
   'imgs_batch = imgs[j : j + batch_size, :, :, 32:224].to(next(self.pose.parameters()).device)'),
  ('self.pose.to(os.environ.get("GVHMR_PREPROC_DEVICE", "cuda")).eval()',
   'device = os.environ.get("GVHMR_PREPROC_DEVICE", "cuda")\n        dtype = torch.float16 if os.environ.get("GVHMR_PREPROC_PRECISION", "fp32") == "fp16" else torch.float32\n        self.pose.to(device=device, dtype=dtype).eval()'),
  ('imgs_batch = imgs[j : j + batch_size, :, :, 32:224].to(next(self.pose.parameters()).device)',
   'imgs_batch = imgs[j : j + batch_size, :, :, 32:224].to(next(self.pose.parameters()))'),
  ('heatmap = heatmap.clone().cpu().numpy()', 'heatmap = heatmap.float().cpu().numpy()'),
 ],
 'hmr4d/utils/preproc/vitfeat_extractor.py': [
  ('import torch\n','import os\nimport torch\n'),
  ('load_hmr2().cuda().eval()', 'load_hmr2().to(os.environ.get("GVHMR_PREPROC_DEVICE", "cuda")).eval()'),
  ('imgs = imgs.cuda()','imgs = imgs.cpu()  # Keep the full sequence off GPU; transfer only each minibatch.'),
  ('batch_size = 16  # 5GB GPU memory, occupies all CUDA cores of 3090',
   'batch_size = max(1, int(os.environ.get("GVHMR_BATCH_SIZE", "1")))'),
  ('imgs_batch = imgs[j : j + batch_size]',
   'imgs_batch = imgs[j : j + batch_size].to(next(self.extractor.parameters()).device)'),
  ('load_hmr2().to(os.environ.get("GVHMR_PREPROC_DEVICE", "cuda")).eval()',
   'load_hmr2().to(device=os.environ.get("GVHMR_PREPROC_DEVICE", "cuda"), dtype=torch.float16 if os.environ.get("GVHMR_PREPROC_PRECISION", "fp32") == "fp16" else torch.float32).eval()'),
  ('imgs_batch = imgs[j : j + batch_size].to(next(self.extractor.parameters()).device)',
   'imgs_batch = imgs[j : j + batch_size].to(next(self.extractor.parameters()))'),
  ('with torch.no_grad():\n                feature = self.extractor',
   'with torch.no_grad(), torch.autocast(device_type=imgs_batch.device.type, dtype=torch.float16, enabled=imgs_batch.dtype == torch.float16, cache_enabled=False):\n                feature = self.extractor'),
  ('features.append(feature.detach().cpu())', 'features.append(feature.detach().float().cpu())'),
 ],
 'hmr4d/utils/preproc/vitpose_pytorch/src/vitpose_infer/builder/model_builder.py': [
  ('check = torch.load(checkpoint)', 'check = torch.load(checkpoint, map_location="cpu")'),
 ],
}

def apply():
    changed=[]
    for relative, edits in PATCHES.items():
        path=GVHMR/relative
        text=path.read_text()
        original=text
        for index,(before,after) in enumerate(edits):
            if after in text: continue
            # Later replacements may intentionally consume an earlier patch.
            final_after=after
            for later_before,later_after in edits[index+1:]:
                final_after=final_after.replace(later_before,later_after)
            if final_after in text: continue
            if text.count(before)!=1:
                raise RuntimeError(f'Upstream changed; refusing ambiguous patch: {relative}: {before}')
            text=text.replace(before,after,1)
        if text!=original:
            backup=ROOT/'outputs/upstream_backups/GVHMR'/relative
            backup.parent.mkdir(parents=True,exist_ok=True)
            if not backup.exists(): backup.write_text(original)
            path.write_text(text);changed.append(relative)
    return changed

if __name__=='__main__': print(json.dumps({'changed':apply()},indent=2))
