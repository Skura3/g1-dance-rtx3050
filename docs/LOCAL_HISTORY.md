# 视频到 G1 舞蹈动作：本地实现

输入：`reference_video.mp4`，720×1280、30 FPS、370 帧、12.33 秒，HEVC。
原视频保留；另生成同分辨率、30 FPS 的 H.264 完整版及前 5 秒验证片段。
流程：GVHMR（ViTPose、HMR2）→ GMR → 固定地面高度对齐 → mjlab/SuperMimic。

## 当前状态（2026-10-30）

- **本轮训练已结束。** 人工认可的展示版为429526步，最终续训状态为452376步；训练和看护已正常退出。不是全部严格阈值或扰动测试均已通过。
- **零基础复刻教程：** 从舞蹈视频到G1仿真的完整教程（历史本地产物，未随Git分发：docs/tutorial/README.md），包括环境、视频重建、重定向、论文原理、NJU借鉴、训练、评估、播放与续训。
- **成果与恢复：** 最终归档说明（历史本地产物，未随Git分发：outputs/finalized_dance/20261030_130751/README.md）。不要直接启动旧训练服务，它指向历史恢复点。

## 历史过程记录（以下为当时状态）

旧内容中“当前训练中”“尚未训练”“training_approved=false”等描述对应各自日期的阶段，不代表最终状态。独立复刻以新教程为入口。

**2026-10-29 21:52 更新：** 用户进一步授权释放壁纸显存。壁纸已正常退出，训练完整保存至 328259 后以 batch1024 继续，当前服务为 `supermimic-g1-batch1024.service`，状态位于 `logs/train/reference_contact_anchor_batch1024/status.json`。原定评测点 329526、333526、337526 保持不变；其余学习参数不变，详见 显存释放与接续记录（历史本地产物，未随Git分发：docs/BATCH1024_REALLOCATION.md）。

**2026-10-29 21:32 更新：** 按用户授权，动态壁纸已暂停，训练从完整保存的 325526 步接续到 `reference_contact_anchor_batch512`，训练 batch 32→512，其他学习参数不变。当前状态位于 `logs/train/reference_contact_anchor_batch512/status.json`；三个观察评测点为 329526、333526、337526。详见 批量优化与观察窗口（历史本地产物，未随Git分发：docs/BATCH512_TRIAL.md）。下文的 `reference_contact_anchor_long` 为此次切换前的历史运行。

**完整视频已生成 616 帧 G1 参考，现已进入本舞蹈的连续微调。** 2026-10-29 16:11 启动长期训练，从 anchor 分支的 291800 步恢复，保留网络、优化器及采样统计；每新增 4000 步才做完整物理评测，连续达标后复核初态扰动并自动停止。实时状态见 `logs/train/reference_contact_anchor_long/status.json`，运行说明见 长期训练（历史本地产物，未随Git分发：docs/LONG_TRAINING.md）。目前尚无收敛或舞蹈质量已达标的结论。源码、完整网络和权重均已核验，基线评测见 策略评测（历史本地产物，未随Git分发：docs/PRETRAINED_EVALUATION.md）。

接触修正见 接触与性能报告（历史本地产物，未随Git分发：docs/CONTACT_AND_PERFORMANCE.md）。data/preview/reference_full_contact.npz 保留为首选策略评测参考；瞬时筛查 23/616 或 PD 失败不等于学习策略无法完成，也不作为试训硬性门槛。新 NJU 几何候选有自碰撞，未覆盖该参考；候选状态见 configs/motion_candidate_status.json。

- 已获取 GVHMR、GMR、第三方 SuperMimic 复现、指定 mjlab 源码。
- `.venv-motion` 是本项目 Python 3.10 环境，复用本机可工作的 torch 2.5.1+cu121；没有更换系统驱动和系统 PyTorch。
- 两个视频的 SimpleVO 相机运动估计已执行并缓存。
- 小批量推理补丁、分进程续跑、无窗口 GMR 转换、训练格式转换、参考渲染工具已实现。
- 运动处理 12 项数值回归、训练环境 4 项 FK 回归和 GPU 全片 FK 输出/梯度检查通过，测试数据未混入真实动作。
- 人体模型和四个权重均已核验；前 5 秒真实视频的跟踪、二维关键点、HMR2 特征、GVHMR 三维人体与 GMR/G1 重定向已完成。参考 NPZ 位于 `data/preview/reference_pilot_aligned.npz`，尚未批准进入训练。
- 自由基座真实重力下专用控制站立 10 秒和参考回放通过，见 环境验证（历史本地产物，未随Git分发：docs/ENVIRONMENT_VALIDATION.md）。真实训练组件已短测至 2048 环境 / batch 128，总显存峰值 2270 MiB；没有长期收敛结论或已学会舞蹈的策略。
- 直接观看：`outputs/environment_validation/standing.mp4`、`reference_playback.mp4`、`source_vs_g1.mp4`。本地交互播放器使用 `.venv-motion/bin/python tools/view_g1.py`。

## 执行顺序

以下命令均从项目根目录执行。无需激活环境，直接调用指定解释器，避免混入 ROS 等环境。

```bash
cd "$G1_PROJECT_ROOT"
# 已完成，可重复检查；同一输入不重复转码
.venv-motion/bin/python tools/prepare_video.py
.venv-motion/bin/python tools/prepare_video.py --output data/video/reference_pilot.mp4 --duration 5
.venv-motion/bin/python tools/check_ready.py

# 模型已核验：后续先处理前 5 秒，每个模型进程结束后释放 GPU
.venv-motion/bin/python tools/run_gvhmr.py
.venv-motion/bin/python tools/render_keypoints.py
# 显存不足时，把大型视觉模型放 CPU（慢）；不改变已缓存结果
.venv-motion/bin/python tools/run_gvhmr.py --preproc-device cpu

# 三维人体 → 29 自由度 G1；完整保留首帧和时间
.venv-motion/bin/python tools/retarget_g1.py

# 首次用零偏移输出到预览目录，先检查着地阶段
.venv-motion/bin/python tools/convert_motion.py --ground-offset 0 --output data/preview/reference_pilot_unaligned.npz
.venv-motion/bin/python tools/render_reference.py --input data/preview/reference_pilot_unaligned.npz --output outputs/reference_pilot_unaligned.mp4
```

确认一段确实着地的时间区间后，用 `--support-range START END` 导出训练文件；或者使用经检查的 `--ground-offset METRES`。**不预设第一帧必然着地**：本视频有跳跃。示例参数 START、END、METRES 需要替换成检查后的数值：

```text
.venv-motion/bin/python tools/convert_motion.py --support-range START END
```

地面对齐对整段只加一个 z 偏移，不逐帧把脚压到地面。转换报告会列出穿地、关节限位和速度；不能用固定偏移解决高度漂移，也不会自动把异常动作裁剪成假正常数据。

## 处理完整 12.33 秒

```bash
.venv-motion/bin/python tools/run_gvhmr.py --video data/video/reference_full.mp4 --preproc-device cuda --preproc-precision fp16 --batch-size 1 --threads 3
.venv-motion/bin/python tools/retarget_g1.py --input outputs/gvhmr/reference_full/hmr4d_results.pt --output data/motions/reference_full_gmr.pkl
```

完整视频后半段含贴地旋转和腾空，现已全片检查并保留原始/修正版对照；修复步骤和未解决问题见完整视频验证报告。

## 小显存改动

- ViTPose/HMR2 默认 batch=1；全段图像保留 CPU，按批传 GPU；权重先加载 CPU。
- 视觉预处理可显式指定 CPU。默认关闭 ViTPose 翻转增强，可 `--flip-test` 启用；改变增强/相机设定需使用新输出目录，避免混用缓存。
- 每个阶段单独进程；结果文件和输入 SHA256/参数记录在输出目录。
- 仅注册 GVHMR 推理配置，不加载训练数据集、评估回调或原生人体渲染器。
- PyTorch3D 只安装官方纯 PyTorch 变换函数；不支持其原生 KNN/网格渲染。本项目 G1 预览用 MuJoCo EGL，已验证可用。
- GMR 包装脚本使用 CPU、小批量 SMPL-X，按帧执行 IK，不启动交互窗口；保存关节名称和 xyzw 四元数元数据。
- 转换时按训练 G1 模型重新做 FK，以 50Hz 导出 wxyz 身体姿态、关节速度和身体速度；显式重排关节。

上游修改由 `tools/patch_gvhmr_lowmem.py` 管理，原文件保存在 `outputs/upstream_backups/`。训练依赖改为本地 mjlab 和 CUDA 12.6，详见训练说明。

## 验证

```bash
.venv-motion/bin/python -m unittest discover -s tests -v
python -m compileall -q tools tests
```

`outputs/reference_contact_sheet.jpg` 是真实视频取样预览。`outputs/camera_validation.json` 是相机结果的形状/有限值检查。
前 5 秒及完整视频流程已执行。分层 FK 同配置预计迭代加速约 1.56 倍；`configs/g1_3050_tuned.json` 和 `configs/g1_3050_throughput.json` 属于历史小网络试验，不能直接加载完整尺寸预训练权重。当前训练使用 `configs/g1_3050_nju_anchor.json` 与 `configs/g1_3050_long_training.json`，64 环境 / batch 32、完整尺寸网络。已保存两组短程微调检查点，当前接续 anchor 分支持续学习；不会根据早期一次失败切换方案。运动学视频不代表策略完成舞蹈。
