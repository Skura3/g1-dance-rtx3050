# NJU SuperMimic 仓库评估与本机验证

更新：2026-10-29。目标仅为本机 mjlab 仿真，无需真机部署。

## 结论

后续实测更新：e13i 权重现已核验、两版动作均完成无中途重置物理回放；朝向误差最高接近 180°，尚不符合复现质量要求。当前结论见 [全片策略评测](PRETRAINED_EVALUATION.md)。

该仓库与当前使用的 CuboiLeo/SuperMimic-Reproduced 同源，能补充预训练模型、自定义舞蹈处理、困难片段采样和策略评测。下一步应先测试已发布策略对本视频的跟踪，再根据实际失败片段修正参考并微调。简单 PD 失败或瞬时逆动力学筛查不通过，不能证明学习策略无法完成，也不应成为禁止试训的硬性门槛。

已克隆到 third_party/SuperMimic-Repro-Sim2Real，固定提交 f8f60d7bbdba68468c050ef100af02d752ca238e；该副本保持未修改。来源：https://github.com/NJU-R-L-Group-Embodied-Lab/SuperMimic-Repro-Sim2Real 。

作者报告约 46 秒真机组合舞蹈；v1.1 报告困难舞蹈同一模型连续成功两次、使用松弛保护绳，并明确没有消融实验。这些是作者报告，未在本机独立复现，也不能证明我们的参考动作已经可执行。

## 采用与取舍

| 部分 | 本项目处理 |
|---|---|
| 已发布跟踪策略 | 优先评测 e13i_0291000（BONES-SEED 加 4 段舞蹈训练）；e11c_0220000 是可选通用基线，无需现在都下载 |
| e24_0355000 | 作者说明仅适配特定舞蹈，父模型 e23_0342000 未发布；不能照抄脚本复现，不作为本视频首选起点 |
| GMR 初始化及 IK 分支恢复 | 已作为可选参数加入本地脚本并跑全片；分支变化必须比较自碰撞和保真度 |
| 滑动最低足高去漂移 | 已生成独立候选，结果未优于原接触候选，不自动替换 |
| 起止站立与平滑过渡 | 后续可生成候选；插值过渡仍需物理回放验证 |
| 自适应采样 | 新源码含时长归一化与访问次数预热；换动作库时重置 sampler，避免旧片段统计污染 |
| 困难落地/转身片段 | 根据本机仿真失败轨迹选窗口，不照搬作者另一段舞蹈的帧号或真机状态 |
| 倾斜奖励 | 如试验启用，保持真实环境与世界模型想象奖励一致，保留基线对照 |
| anchor_obs | 目标维度由 680 变为 689；首轮保留发布权重原设定，不能直接开启后强行加载 |

不需要 DDS、Unitree SDK、真机通讯、标定、IMU 饱和或真机状态日志。保留已验证的 CUDA 12 栈与驱动，不照搬 CUDA 13 安装示例、不放大电机力矩限制。

依据：固定提交内 README.md、docs/VIDEO_TO_MOTION.md、docs/CUSTOM_MOTIONS.md、docs/METHOD.md、docs/REAL_ROBOT_LEAN35KFS.md、scripts/train/train_lean35kfs_e24.sh 及源码。

## 已完成的本机改动

- tools/common.py 支持 G1_SUPERMIMIC_SOURCE=original|nju，默认 original；每次新进程选择源码，无需重装环境。
- tools/benchmark_training.py 增加来源、网络结构记录和 --full-size。
- tools/retarget_g1.py 增加可选 --init-from-human / --restart-tol。完整 370 帧已完成；源帧 261、273、283、287 发生 IK 分支恢复，最大混合修正约 2.825 rad，不能视为无损修复。
- tools/make_nju_candidate.py 生成 616 帧、50 Hz 独立候选。MIT 声明见 docs/NJU_THIRD_PARTY_NOTICE.txt。
- tools/evaluate_nju.py 准备按 checkpoint 配置构造策略，在自由基座和真实重力下从首帧评测；完整权重已核验，两版参考均完成 616 步基线评测；结果与质量限制见 PRETRAINED_EVALUATION.md。

## RTX 3050 4GB 实测

现有 Python 3.12 / CUDA 12 环境、1 个 CPU 线程、分层 FK、64 并行环境、batch 32。先采样 44 步，再执行 3 轮世界模型、策略与 critic 优化更新。

| 网络 | GPU 总显存采样峰值 | 结果 |
|---|---:|---|
| 小网络 | 1261 MiB | 3 轮更新通过 |
| 完整尺寸网络 | 1595 MiB（约 1.56 GiB） | 3 轮更新通过，损失/梯度有限 |

完整尺寸：世界模型 2048×3，策略 encoder/decoder 均为 2048、1024、512、256。最高温度 72°C；PyTorch 峰值已分配 385 MiB、预留 450 MiB，其余含仿真和桌面等显存。见 outputs/nju_validation/full_size_benchmark.json。

这是随机初始化的训练组件短测，不是已发布策略跟踪验证；没有保存 checkpoint 或证明收敛。估算 0.792 秒/迭代来自采样与更新耗时组合，不是连续长训吞吐。之前 2048 环境测试是小网络，不能直接外推到完整尺寸。

本机可从完整尺寸、小并行度评测/微调起步。预训练权重要求保持网络和观测维度；通过降低环境数、batch、回放容量和想象时域控制显存。连续温度、存盘/恢复和学习质量仍需验证，不能承诺全规模从零训练或本舞蹈必然收敛。

## 动作候选比较

| 候选 | 已知结果 | 当前用途 |
|---|---|---|
| reference_full_contact.npz | 瞬时筛查 23/616；实际接触 WBC 的六个一秒片段中三个通过 | 首选已检查的学习策略评测参考，未证明全片可跟踪 |
| reference_full_support.npz | 瞬时筛查 69/616，但优化到达最大迭代，WBC 未整体改善 | 保留对照，不作为默认 |
| reference_full_dynamic.npz | 原生接触逆动力学修复失败，绝对残差和所需力矩仍严重超标 | 标记失败，排除正式训练选择 |
| reference_full_nju_drift.npz | 27 帧自碰撞超过 5 mm，最深约 6.68 cm；瞬时筛查 8/616；WBC 3/6 | 方法对照，不覆盖 contact |

WBC 判失败使用完整根位置偏差 0.25 m 或朝向偏差 0.8 rad，与 SuperMimic 的高度/投影重力终止条件不同，不可直接比较成功率。瞬时筛查反映参考精确执行的疑点，不是学习策略准入门槛。状态记录在 configs/motion_candidate_status.json。

## 手动下载项已补齐

此前自动下载两次超时，用户已手动补齐。归档 SHA256 与 4 个有效载荷全部核验通过，.partial 不参与使用。

下载：https://github.com/NJU-R-L-Group-Embodied-Lab/SuperMimic-Repro-Sim2Real/releases/download/v1.0/e13i_0291000.tar.gz

保存到：/home/djs/supermimic/data/downloads/nju/e13i_0291000.tar.gz

- 大小：266591326 字节，约 254.24 MiB。
- Release asset SHA256：2aa3f4ca81d7a2e4cdf357d9a3b60d6809fd7d87e0db3005cad55d3ef22709fa。
- 实际加载路径：data/downloads/nju/e13i_0291000/ckpt_e13i_0291000.pt；无需搬动。核验见 outputs/download_audit/nju_checkpoint.json。
- 状态见 outputs/nju_validation/download_status.json，Release 元数据见 outputs/physics_repair/repo_releases.json。

## 后续顺序

1. 已完成：归档、策略配置和 680 维目标/历史观测兼容性核验。
2. 已完成：tools/evaluate_nju.py 全片评测 contact 与 nju_drift，两者均未触发失败条件，但最大倾斜误差约 39.4° / 33.0°。详细物理回放见 PRETRAINED_EVALUATION.md。
3. 根据真实失败窗口修正接触、穿插、速度，必要时补起止过渡，保留原片作为保真度对照。
4. 如需训练，加载预训练模型、重置 sampler，从 64 环境 / batch 32 短程微调开始；验证存盘/恢复、连续显存和吞吐再扩大。
5. 以全片不重置的物理跟踪、多个种子与扰动测试评估效果，同时报告动作偏差。当前没有长训或学会舞蹈的策略。

CPU 动力学辅助实验额外依赖记录在 envs/training/requirements-physics.txt，不要求替换 CUDA 包。
