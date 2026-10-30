# G1 环境验证

本文记录初始机器人加载、真实物理站立和前 5 秒参考动作验证。完整视频已在后续完成，最新结果见 [完整视频验证](FULL_VIDEO_VALIDATION.md)；未训练舞蹈策略。

## 已验证：模型加载与站立

- 使用训练项目同一份 mjlab G1 模型和内置位置执行器：29 个控制关节、36 个 qpos、35 个 qvel、35 个网格。
- 单环境 MuJoCo Warp 在 RTX 3050 上运行，物理步长 0.005 秒，重力 -9.81 m/s²，保留地面接触和原有关节力矩上限。
- 原始默认姿态加固定 PD 目标约 1.2 秒失稳，记录保留在 `outputs/environment_validation/failed_baseline/`。
- 专用站立控制器先用逆动力学求接触静力平衡，再通过原有 PD 执行器执行有限时域 LQR 反馈。机器人基座未固定；仿真期间不重写根状态、不重置、不施加外部扶持力。
- GPU 实测站立 10 秒通过，最大倾角约 0.336°，骨盆高度约 0.757 米。只是无扰动站立验证，并非训练出的策略或抗扰动能力证明。
- 当前驱动使 mjlab 自动禁用 CUDA Graphs；GPU 仿真可以运行，但性能不能按开启图加速时估计。没有修改系统驱动。

视频：`outputs/environment_validation/standing.mp4`。详细状态：`stand_report.json`。

重新执行物理站立：

```bash
cd "$G1_PROJECT_ROOT"
OPENBLAS_NUM_THREADS=1 envs/training/.venv/bin/python tools/validate_g1.py --mode stand --seconds 10
```

本机交互窗口回放已记录的站立状态（鼠标旋转/缩放，关闭窗口结束）：

```bash
.venv-motion/bin/python tools/view_g1.py
```

交互回放只显示保存的状态；要重新验证受力与控制，请运行上面的物理站立命令。
交互窗口使用 motion 环境的 MuJoCo 3.3.7；训练环境的 MuJoCo 3.16 原生窗口在退出时曾发生段错误，已改用隔离的播放器并验证正常关闭。GPU 物理仿真仍使用训练环境。

## 视频参考动作

此处使用原始舞蹈前 5 秒、150 帧进行验证。完整 12.33 秒现已另行处理，见完整视频验证报告。
大视觉网络使用 CPU 小批量，避免与 4GB 显存的物理仿真抢占空间；GVHMR 时序预测单独在 GPU 执行。
参考动作播放是运动学回放，不能据此宣称机器人已经通过策略学会跳舞。

### 已完成的真实数据链路

1. 原视频前 5 秒 → 150 帧 YOLO 跟踪和 ViTPose 关键点，HMR2 特征 `[150,1024]`。
2. GVHMR GPU 推理 → `outputs/gvhmr/reference_pilot/hmr4d_results.pt`，全部人体参数为有限值。
3. CPU SMPL-X + GMR → `data/motions/reference_pilot_gmr.pkl`。修复了分批 SMPL-X expression 与 betas 的 batch 维度不一致，包含全部 150 帧。
4. 根据原视频 0.9–1.2 秒双脚着地区间做固定高度平移 -0.014079 米；未逐帧压脚、未裁剪关节角。
5. 以 50 Hz 导出 `data/preview/reference_pilot_aligned.npz`，249 帧、4.96 秒采样跨度。原视频 150 帧/30 FPS 的播放时长是 5 秒，末帧时刻 4.9667 秒，50 Hz 网格末时刻为 4.96 秒。
6. SuperMimic MotionLibrary 在 GPU 上成功读取；在 mjlab 中逐帧写入并前向计算，所有参考帧验证后以 25 FPS 渲染。

### 观看结果

- `outputs/environment_validation/reference_playback.mp4`：G1 参考动作播放。
- `outputs/environment_validation/source_vs_g1.mp4`：原视频与机器人同步对照。
- `outputs/reference_pilot_keypoints.mp4`：二维关键点叠加。
- `outputs/environment_validation/play_report.json`：GPU 动作库与播放检查。
- `outputs/environment_validation/reference_quality.json`：参考数据质量检查。

交互回放舞蹈参考（循环播放，关闭窗口结束）：

```bash
.venv-motion/bin/python tools/view_g1.py --input data/preview/reference_pilot_aligned.npz
```

重新运行 GPU 参考播放验证并录制：

```bash
envs/training/.venv/bin/python tools/validate_g1.py --mode play --input data/preview/reference_pilot_aligned.npz
```

### 剩余质量问题与范围

全部 249 帧数值有限、关节角没有越界，最大四元数归一化误差约 6e-8。最小脚底高度约 -1.0 毫米，没有超过 2 厘米的穿地；最下方脚底最高约 6.8 厘米，既包含跳跃，也可能有重建悬空/漂移，尚不能作为接触已正确的证明。最大关节速度约 15.22 rad/s，仍需结合动力学检查。

文件保持在 `data/preview/`，未混入训练集。环境验证完成的范围是：模型/网格加载、无扰动站立、真实视频参考动作读取和播放；没有运行舞蹈策略训练或物理舞蹈跟踪。完整视频后半段的后续处理与剩余问题见完整视频验证报告。
