# 训练环境与 3050 初始配置

训练使用 `third_party/SuperMimic-Reproduced` 的独立复现，底层是其指定的 mjlab 提交。
源码默认 Python 3.12、CUDA 13；本机驱动 535.309.01，因此改为 CUDA 12.6 依赖路径。
mjlab 自带 `docs/source/installation.rst` 说明 cu126 路径支持 driver 525+；实际可用性仍需 Torch、Warp 编译和仿真测试。
不因 nvidia-smi 显示 CUDA 12.2 就直接判定所有 CUDA 12.x wheel 不兼容，也不假设全部运行时特性可用。

**2026-10-29 状态：CUDA 12 Warp wheel 已核验并离线安装；Torch GPU 运算和 Warp GPU 内核编译、执行全部通过。**
已将不兼容的普通 Warp 替换为官方 `warp-lang==1.18.0+cu12`，其运行日志显示 CUDA Toolkit 12.9；在当前 535.309.01 驱动和 RTX 3050 4GB 上通过实际内核测试。PyTorch 仍为 2.14.1+cu126。
两个 wheel 均固定为本地来源，SHA256 已校验；未修改系统驱动。当前不缺安装文件。基础检查之外，G1 单环境真实物理站立和参考动作播放也已通过，详见 [环境验证](ENVIRONMENT_VALIDATION.md)；尚未运行完整 SuperMimic 训练或验证收敛。当前驱动禁用 CUDA Graphs，训练性能仍需实测。

## 独立环境

```bash
cd "$G1_PROJECT_ROOT"
bash tools/setup_training.sh --offline
envs/training/.venv/bin/python tools/check_training.py
```

下载每项最多尝试两次；本轮全程离线，安装日志 `logs/setup_training_warp_cu12.log`，运行验证日志 `logs/check_training_warp_cu12.log`。
训练使用 `envs/training/.venv`，与 `.venv-motion` 的 PyTorch 2.5.1/CUDA12.1 隔离。
Python 安装位置 `.runtime/python`，缓存 `.runtime/uv-cache`。没有修改系统 GPU 驱动。

`envs/training/pyproject.toml` 及第三方复现的 `pyproject.toml` 改为本地 mjlab 路径和 pytorch-cu126，避免重复 git 下载和 CUDA13 依赖；原始文件保存在 `outputs/upstream_backups/SuperMimic-Reproduced/`。

## 最新性能实测

2026-10-29：已完成一次性短程采样、世界模型、策略和 critic 更新；无新 checkpoint。分层 FK 在同配置下预计迭代耗时从 1.063 秒降至 0.682 秒。1024 环境 / batch 256 显存峰值 1866 MiB；2048 环境 / batch 128 为 2270 MiB。尚未验证长期收敛、完整 Trainer.run 周期诊断和存盘。

新增 configs/g1_3050_tuned.json 和 configs/g1_3050_throughput.json，参数 --threads 1 --fast-fk 显式启用优化，精度保留 highest。测量方法与限制见 [接触与性能报告](CONTACT_AND_PERFORMANCE.md)。当前参考动力学未通过，正式动作目录仍为空，没有启动长训。

## 初版小显存配置

见 `configs/g1_3050.json`：

- 16 个仿真环境；每环境回放 512 步，共 8192 行，初版仍用上游 GPU buffer。先缩小容量避免未经测试地改写回放逻辑。
- 世界模型 512×512×512，8 步监督窗口，每批 32，2 批/更新。
- 策略编码器/解码器 512×256×128；10 帧历史与 10 帧未来参考。
- 想象 16 步、32 条，保留 critic；关闭 torch.compile，先降低首次编译和图缓存开销。
- 上游日志中的世界模型诊断批量原本固定为 1024，已改成不超过配置批量，避免训练小批量却在日志诊断时显存激增。
- 调试初版关闭物理参数随机化；稳定后启用并重新验证。
- episode 上限 15 秒，避免完整 12.33 秒舞蹈被默认 10 秒强制截断。
- 初次尝试上限 1 小时/20000 迭代，仅为作业边界，不代表一小时能收敛。

这些是工程试验值，不是论文默认值或已证明可在 4GB 收敛的参数。与官方大网络 checkpoint 的形状不同，不能直接把大网络 checkpoint 当小网络 resume。

## 动作准备完成后的验证与训练

先检查真实动作的 G1 参考视频和转换报告，把合格 NPZ 放在 `data/npz/pilot/`。不要将测试/预览数据放到训练目录。

```bash
# 80 次环境步，包括世界模型/策略更新；先检查是否 OOM
envs/training/.venv/bin/python tools/train_3050.py --smoke
# 验证通过后进行有时长上限的训练
envs/training/.venv/bin/python tools/train_3050.py
```

训练器保留上游的终止前状态 recorder、8 步连续回放监督、冻结世界模型权重但保留输入梯度、critic bootstrap。未把 PPO 冒充 SuperMimic。

记录 GPU 总占用要使用 nvidia-smi，不能只看 torch.cuda：MuJoCo Warp 与桌面显示显存不在 PyTorch 的峰值统计内。
首轮观察数值稳定性、真实仿真回报与世界模型误差；不能只用想象回报判定舞蹈完成。

最终要从第一帧持续执行完整参考动作，报告完成时长、是否触发失败、跟踪误差；随机起点训练中短暂不摔倒不等于整段成功。

## 真实状态

完整视频已经历两轮修复；最新 data/preview/reference_full_contact.npz 为 616 帧 / 50 Hz，全片 GPU FK 通过。接触段滑动减小，但动力学仍待修正，training_approved=false。已完成一次性短程性能更新，无长期训练或 checkpoint。见 [最新报告](CONTACT_AND_PERFORMANCE.md)。下文为前 5 秒阶段历史记录。

如存在 `outputs/training_readiness.json`，查看其 `ready` 和 `note`；该文件只证明所注明的验证范围。
前 5 秒的真实 SMPL-X/G1 参考已生成；`data/preview/reference_pilot_aligned.npz` 为预览文件，已通过 MotionLibrary 加载和运动学播放。关节角无超限，但脚底有悬空变化，需要质量复查。`data/npz/pilot/` 仍保持不含未批准的预览动作，尚未开始舞蹈策略训练。
