# 从 GitHub 副本复刻

本仓库公开的是独立整理的源码副本；作者原电脑的训练目录与成果没有被迁移或删除。历史教程中的 `/home/djs/supermimic` 是原机路径，不是必须使用的用户名。

## 1. 设置项目根目录

克隆后进入仓库，在每个新终端执行：

```bash
export G1_PROJECT_ROOT="$(pwd)"
```

此后教程中的 `cd "$G1_PROJECT_ROOT"` 就回到仓库。若新终端不在仓库，先手动cd到你克隆的实际目录再设置。

## 2. 恢复固定版本源码

```bash
python3 tools/bootstrap_sources.py --list
python3 tools/bootstrap_sources.py
```

按固定commit下载GVHMR、GMR、CuboiLeo复现、NJU复现和mjlab；从内附BSD许可的源码包恢复本机PyTorch3D纯变换所需源文件；为GVHMR和CuboiLeo应用已验证补丁并核对文件SHA256。不会安装Python包、下载SMPL-X、覆盖已有checkout或启动训练。每项下载最多两次，失败打印网址与应存位置。

已有但未验证的third_party不会被覆盖；若补丁检查失败，先检查是否错误版本，勿强行应用。只有指定版本有补丁兼容保证。

## 3. 环境和外部文件

按[教程第02章](tutorial/02_环境与下载.md)安装两个环境。SMPL-X需自行注册许可；视觉权重、Torch/Warp wheel按该章网址自行获取。无需真机SDK。新电脑的洁净安装路线没有在第二台机器端到端验证，保留这个边界。

完整论文PDF和原舞蹈视频不随仓库发布。论文：SuperMimic https://arxiv.org/abs/2610.28480 ，mjlab https://arxiv.org/abs/2601.22074 。视频流程可使用自己有权使用的素材。换新视频必须重做接触标注和缓存，不能照搬原片时间窗口。

## 4. 快速看现有成果

无需环境就可先看 [参考与策略对照MP4](assets/reference_vs_policy_429526.mp4)。它是已记录的物理轨迹与运动学参考对比，不是再次在线训练。

下载仓库Release v1.0.0中的 `checkpoint_429526.pt`、`g1_scene.mjb`，放到 `data/releases/v1.0.0/`。校验和见 `configs/release_assets.json`。

```bash
python3 tools/prepare_public_demo.py --mode display
MUJOCO_GL=glfw OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
envs/training/.venv/bin/python -B docs/tutorial/scripts/view_comparison.py \
  --report outputs/public_demo/replay_report.json \
  --checkpoint data/releases/v1.0.0/checkpoint_429526.pt \
  --model data/releases/v1.0.0/g1_scene.mjb \
  --status outputs/public_demo/view_status.json
```

二进制场景要求对应MuJoCo3.16环境。左参考、右策略，一轮12.32秒、墙钟1x；空格暂停、回车重播。它需要桌面图形会话。

## 5. 从已准备参考重新微调

`examples/reference_dance/reference_full_contact.npz` 是整理好的G1参考，`policy_429526_states.npz` 是已验证策略轨迹，不含SMPL-X人体模型。

```bash
python3 docs/tutorial/scripts/prepare_run.py --name my_dance_01 \
  --motion examples/reference_dance/reference_full_contact.npz --batch 32
```

然后按[第04章](tutorial/04_开始训练.md)下载e13i、扩展anchor、短测、持续训练。把该章的实验名替换为my_dance_01，评测动作路径也同步替换。首次此路径默认关闭历史重点窗口；要完全沿用原片重点采样，先准备data/preview原参考，再传 `--use-original-focus`。

## 6. 从最终452376继续学习

另下载Release中的 `checkpoint_final_452376.pt` 至相同目录，执行：

```bash
python3 tools/prepare_public_demo.py --mode resume
MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
envs/training/.venv/bin/python -B tools/finetune_nju.py \
  --config configs/public_resume.json \
  --checkpoint data/releases/v1.0.0/checkpoint_final_452376.pt \
  --run-name resumed_public_01 --keep-sampler --until-targets \
  --supervision outputs/public_demo/resume_supervision.json
```

这条会真正训练。换新run-name，先保证没有其他训练程序。保留权重、三个优化器和采样统计；回放缓存/现场仿真/RNG未保存，启动后64步预热。首评是起始步数+4000。它不自动设置看护；无人值守参见教程第07章。

## 7. 发布范围

Git中不放虚拟环境、CUDA wheel、视觉权重、SMPL-X、原视频、完整论文、训练缓存或所有历史检查点。两份成果权重与场景通过Release单独分发；源码和原创文档MIT，第三方与权重/数据许可见[第三方声明](../THIRD_PARTY_NOTICES.md)。
