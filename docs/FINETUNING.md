# 单舞蹈微调执行记录

日期：2026-10-29。用户允许适度动作误差，目标为稳定完成且动作可辨认；不是逐关节精确复现。

## 运行设置

只使用明确指定的 data/preview/reference_full_contact.npz，不扫描整个预览目录；失败的逆动力学候选不参与训练。预训练来源为已核验 e13i_0291000。

保留完整网络、状态归一化、世界模型、策略、critic 及 Adam 状态。64 个环境、batch 32、512 步回放缓存、16 步想象时域、UTD 0.4、世界模型每次更新 2 个 batch。新库重置旧 sampler 统计，取消作者大动作库路径及真机相关随机化。每次新进程填充 64 步回放后开始更新。

以 4.2 秒踢腿、7.8 秒手撑、9.32 秒倾斜峰值、11.86 秒朝向偏差为参考，25% 的 reset 从这些时刻之前两秒窗口采样；这属于质量问题采样，不伪称它们均发生了终止失败。其余采样沿用上游方法。

## 第一轮：680 维兼容微调

命令：

    envs/training/.venv/bin/python tools/finetune_nju.py --steps 600 --run-name reference_contact_e13i_ft1

600 次环境迭代、214 次世界模型/策略/critic 更新已完成，约 397 秒（含初始化）。保存 checkpoint 为 logs/train/reference_contact_e13i_ft1/ckpt_0291200.pt、ckpt_0291400.pt、ckpt_0291600.pt。与独立评测短暂并行时总显存峰值 1813 MiB、温度 76°C，因此这不是单独训练的纯峰值。独立训练观察值约 1600 MiB。

200 步模型完整运行，但平均根位置误差增至 0.715 m、最大 2.486 m，暂不选用。600 步模型在 9.64 秒触发失败，排除。保留全部记录，没有覆盖下载模型。

## 第二轮：仿真位置/朝向反馈

原策略的目标不含机器人相对参考的平面位置和偏航误差，难以纠正累积漂移。采用上游已有 anchor_obs，为目标增加 9 维参考锚点相对位置/旋转表示，实际仿真与世界模型想象使用同一定义。

tools/extend_anchor_checkpoint.py 对策略第一层追加零权重列；对 critic 在目标块后插入零列，并同步扩展 Adam 一阶/二阶矩。原有列和后续 xi/error 输入列顺序保留。初始策略和 critic 最大输出差为 3.81e-6 / 7.63e-6，优化器形状检查通过。迁移后是 689 维策略，仅用于仿真，不声明可直接部署到真实机器人。

命令：

    envs/training/.venv/bin/python tools/extend_anchor_checkpoint.py --input data/downloads/nju/e13i_0291000/ckpt_e13i_0291000.pt --output outputs/finetuning/e13i_anchor_initialized.pt
    envs/training/.venv/bin/python tools/finetune_nju.py --config configs/g1_3050_nju_anchor.json --checkpoint outputs/finetuning/e13i_anchor_initialized.pt --steps 800 --run-name reference_contact_anchor_ft2

第二轮学习率 2e-5，第一轮 1e-5。不同轮次不仅是单变量消融，不将改善归因于单一设置。以上为历史短程试验；用户现已授权连续长期训练，新的运行规则见 [LONG_TRAINING.md](LONG_TRAINING.md)。

## 验收与选择

configs/dance_acceptance.json 为暂定工程目标，非用户指定的数值：完整无重置运行；平均/峰值倾斜误差不超过 15°/50°；平均/峰值偏航误差不超过 45°/100°；关节 RMSE 不超过 0.35 rad；平均/峰值根位置误差不超过 0.6/1.2 m。

这些目标容许明显但有限的偏差，仍排除摔倒和半圈朝向错误。tools/compare_finetuning.py 比较同一个完整参考；评分只是排序辅助，不能替代视觉检查。先做完整轨迹，再做初态扰动回放；不通过缩短视频、重置、对齐消除偏航或改变计分时轴掩盖问题。

## 工程修复

- 修复配置覆盖器无法把可选 tuple 参数置为 None 的错误，并通过回归测试。
- checkpoint 在 CPU 加载，仅迁移所需模型与优化器张量；不复制旧大库的采样状态到 GPU。
- 降低在线诊断 batch 到 64，训练 batch 不受影响；每次更新记录有限值检查和 JSONL 指标。
- 分层 FK 沿用已验证优化，不修改上游源码、驱动或电机限制。
- 每轮保存 run_report.json、config.json、updates.jsonl 和独立评测报告。最新结果汇总在 outputs/finetuning/ 下。

## 解释更正：短测不是收敛判断

此前“排除”仅能表示该检查点暂不用于演示，不能否定继续训练或整个方法；早期摔倒与误差应被允许。600/800 步是预设短程预算，不是达到姿态误差门槛后停止。两个阶段均从已发布 291000 步模型分别起步，第二阶段没有接着第一阶段训练，不能合称连续 1400 步学习。

论文第 4–5 页描述采样、拟合可微世界模型、冻结世界模型权重并通过 BPTT 更新策略/critic 的循环；第 15 页 Algorithm A.1、第 16 页 Table A.5 给出 4096 环境、320000 iterations、32 步想象；第 18 页 Fig A.1 每 4000 iterations 评测。当前 64 环境、batch 32、16 步想象和数百步训练只足以作工程与短期诊断。不能用原论文 H100 的时间承诺本机的收敛时间。

原版与 anchor 版同时改变了观测和学习率，且训练量很短，因此不能声称已经证明哪种算法更好。后续应固定主要配置持续训练，低频评测趋势、保留检查点；不因单次早期回放失败反复换方法。最终误差阈值是暂定工程指标，不是用户要求，也不限制训练中的错误。真实计数见 outputs/finetuning/methodology_audit.json。
