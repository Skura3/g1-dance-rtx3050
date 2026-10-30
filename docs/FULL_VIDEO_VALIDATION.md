# 完整视频重建、重定向与修复验证

验证日期：2026-10-29。本轮已在本机 RTX 3050 Laptop 4GB 完成完整视频处理，未运行舞蹈策略训练。

## 结果与查看位置

- 原视频：370 帧，30 FPS，容器播放时长 12.333 秒，最后一帧时间 12.300 秒。
- GVHMR 和 GMR 均包含全部 370 帧；50 Hz 参考有 616 帧，覆盖 0–12.300 秒。
- 25 FPS 预览有 308 帧，播放时长 12.320 秒，末次展示采样时刻 12.280 秒；重建/动作文件没有因此丢掉末帧。
- 原视频与修正版 G1 同步对照：source_vs_g1.mp4（历史本地产物，未随Git分发：../outputs/full_validation/source_vs_g1.mp4）。
- 修复前后对照：raw_vs_repaired.mp4（历史本地产物，未随Git分发：../outputs/full_validation/raw_vs_repaired.mp4），左 RAW、右 REPAIRED。
- 全片二维关键点：reference_full_keypoints.mp4（历史本地产物，未随Git分发：../outputs/reference_full_keypoints.mp4）。
- 修正版动作：data/preview/reference_full_repaired.npz；原始固定高度对齐版：data/preview/reference_full_aligned.npz。
- 完整机器可读报告：summary.json（历史本地产物，未随Git分发：../outputs/full_validation/summary.json）。逐帧数据：raw_audit/ 与 repaired_audit/ 下的 frame_audit.csv、quality_series.npz、quality.json。

原始人体预测 outputs/gvhmr/reference_full/hmr4d_results.pt 和原始 GMR data/motions/reference_full_gmr.pkl 保留；预览文件未放进训练目录。

## 已执行的检查

1. YOLO 跟踪选中的同一 ID 覆盖 370/370 帧，无漏检插值帧；另一次带诊断的跟踪得到相同包围框。
2. 关键点、视觉特征、SMPL-X 姿态/形状/位移均有限。低于 0.3 的关键点置信度主要出现在 7.6–8.3 秒，以及踢腿/转身遮挡附近。
3. 相机背景特征相对首帧存在数十像素位移，不支持直接假设完全固定相机。保留 SimpleVO 运动相机路径。
4. SMPL-X 相机坐标关节重投影与高置信度 ViTPose 比较：误差中位数 11.64 像素，95 分位 27.05 像素；分别约为人体裁剪框尺度的 1.48%、3.60%。两者不是独立真值，且关节定义近似，不能由此证明深度/尺度/接触准确。
5. 对 616 帧逐帧检查训练 G1 的关节限位、脚底和非脚部碰撞几何高度、启用的自碰撞对、速度、加速度、根位移/旋转变化。
6. SuperMimic MotionLibrary 在 GPU 读取成功，真实 mjlab / MuJoCo Warp 对全部 616 帧、30 个身体做 FK 比对。修正版最大身体位置误差约 3.73e-7 米；四元数符号等价误差约 2.29e-7。没有运行闭环舞蹈物理控制。
7. 9 项数值测试通过，包括稀疏支撑校正保持合成跳跃、脚底几何一致性、重采样和关节顺序检查；工具编译检查和补丁幂等检查通过。

## 修复方法与实测变化

| 指标 | 原始对齐版 | 修正版 |
|---|---:|---:|
| 关节角超限帧 | 0 | 0 |
| 自穿透超过 5mm 的帧 | 27 | 0 |
| 检测到的最大自穿透 | 66.8mm | 无超过审计 1mm 门限的穿透 |
| 最大关节速度 | 29.39 rad/s | 18.26 rad/s |
| 最大关节加速度 | 696.15 rad/s² | 563.47 rad/s² |
| 加速度超过审查门限 300 rad/s² 的帧 | 16 | 8 |
| 人工核对的支撑时刻：最低脚底高度中位数 | 18.91cm | 1.17cm |
| 人工核对的支撑时刻：最低脚底最高值 | 24.32cm | 3.76cm |
| 最深脚底穿地 | 约 1.01mm | 约 1.00mm |

- 高度：依据 configs/reference_full_support.json 的 12 个视频支撑时刻，解带平滑项和地面不穿透约束的高度修正。最大降低约 22.18cm；没有逐帧强制脚底归零，也没有删除空中帧。支撑标注和单目尺度仍有误差。
- 自碰撞：仅局部调整手臂关节，约束训练 G1 的手–髋、手–手、腕–手几何距离，同时惩罚姿态偏移和时序不连续。避碰优化收敛，避碰本身最大关节调整约 0.101 rad。
- 肩部突变：仅 8.58–8.96 秒右肩、9.58–9.96 秒左肩使用短窗平滑和渐入渐出，保留时间轴。综合最大关节变化 0.329 rad（18.85°）；关节全体 RMS 变化约 0.0105 rad。代价是这两个窗口手腕局部轨迹最大变化约 11.4cm / 14.3cm，不能把修正版视为毫无失真的复原。
- 根部水平轨迹、朝向与腿部关节没有人为修改；脚部水平速度仅存在浮点量级差异。

## 仍未解决的问题

- **滑脚/支撑误差**：近地脚水平速度超过 0.3m/s 的启发式告警有 100 帧。原始版只有 33 帧告警，但原始版大量悬空，参与近地检测的帧只有 79，修正版为 284，不能把计数增加解释为滑脚被修坏。此轮没有修正水平轨迹。
- **手撑地缺失**：7.8–8.0 秒检查点，机器人手部碰撞几何仍约离地 14–20cm。G1 与人体比例差异和遮挡重建误差不能只靠整体降高解决，否则会使支撑脚穿地。需要手/脚联合接触标注与全身约束 IK，随后检查力矩、平衡与支持力。
- **局部加速度仍高**：0.36、7.00、7.50–7.52、7.60、9.10、9.76、9.80 秒被标记；门限是筛查用途，不是厂商电机安全认证。没有为了全部通过数值门限而过度平滑原舞蹈。
- **深度与腾空高度不确定**：约 7.6–9.3 秒的倒身、遮挡、高踢腿最难；修正版最低脚底最高约 34.8cm，发生在 9.12 秒的腾空附近，不能单凭该值判断重建正确。
- **尚无物理跟踪/训练结论**：training_approved=false；未开始舞蹈策略训练，未验证完整训练峰值显存、收敛、抗扰动或实机可执行性。

下一步优先解决 7.6–8.2 秒手撑地的多接触约束、落地/支撑脚水平约束，再做闭环物理跟踪和小规模训练验证。不要把运动学回放当作已学会跳舞。

## 4GB 显存工程改动

ViTPose/HMR2 新增显式 CUDA FP16 预处理，模型和输入使用一致类型，HMR2 的内部 FP32 临时张量通过 autocast 处理；关键点/特征缓存仍为 FP32。batch=1，全段图像仍在 CPU，各阶段用独立进程。

在前 5 秒的 7 个抽样帧上与已有 CPU FP32 结果对比：关键点最大距离偏差 0.0958 像素，HMR2 最低余弦相似度 0.9999982。PyTorch 峰值 reserved 分别 1352 / 1418 MiB，**不是包含桌面、驱动等的显卡总占用，也不是训练峰值**。完整视频随后成功完成，未 OOM。

改变预处理精度/设备/批量且命中有来源记录的缓存时会拒绝混用。所有上游补丁由 tools/patch_gvhmr_lowmem.py 管理，原文件保存在 outputs/upstream_backups/，差异另存 configs/upstream_changes.patch。

本轮新增 tools/audit_motion.py、tools/audit_reconstruction.py、tools/repair_motion.py、tools/report_full_validation.py，增强 tools/run_gvhmr.py 和 tools/validate_g1.py。没有新增待下载文件。

## 复现与交互回放

从 /home/djs/supermimic 运行：

~~~bash
.venv-motion/bin/python tools/patch_gvhmr_lowmem.py
.venv-motion/bin/python tools/run_gvhmr.py --video data/video/reference_full.mp4 --preproc-device cuda --preproc-precision fp16 --batch-size 1 --threads 3
.venv-motion/bin/python tools/retarget_g1.py --input outputs/gvhmr/reference_full/hmr4d_results.pt --output data/motions/reference_full_gmr.pkl
.venv-motion/bin/python tools/convert_motion.py --input data/motions/reference_full_gmr.pkl --output data/preview/reference_full_aligned.npz --support-range 0.9 1.2
.venv-motion/bin/python tools/audit_motion.py --input data/preview/reference_full_aligned.npz --output outputs/full_validation/raw_audit
OPENBLAS_NUM_THREADS=3 OMP_NUM_THREADS=3 .venv-motion/bin/python tools/repair_motion.py --input data/preview/reference_full_aligned.npz --output-motion data/preview/reference_full_repaired.npz --annotations configs/reference_full_support.json --report-dir outputs/full_validation/repair
.venv-motion/bin/python tools/audit_motion.py --input data/preview/reference_full_repaired.npz --output outputs/full_validation/repaired_audit
.venv-motion/bin/python tools/audit_reconstruction.py --reconstruction outputs/gvhmr/reference_full/hmr4d_results.pt --video data/video/reference_full.mp4 --output outputs/full_validation
envs/training/.venv/bin/python tools/validate_g1.py --mode play --input data/preview/reference_full_repaired.npz --output outputs/full_validation
.venv-motion/bin/python tools/view_g1.py --input data/preview/reference_full_repaired.npz
~~~

交互播放器循环显示参考姿态，关闭窗口结束。原始/修正版数据都属于运动学参考预览。
