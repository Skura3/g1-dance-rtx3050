# 2026-10-29：暂停动态壁纸与 batch 512 续训

用户授权范围：本次训练期间暂停动态壁纸，提高显存利用率，保持配置观察 2–3 个评测点后再考虑下一步改动。此次未采用前一轮讨论的小学习率方案，未更改奖励、网络、参考动作、环境数、想象时域或停止阈值。

## 已执行

1. 21:27 暂停 komorebi（PID 2885）及其子进程 WebKitNetworkPr（PID 3160），用 SIGSTOP 保留进程现场。原 PID 与 /proc 启动标识已记录，恢复时验证身份避免误操作重用 PID。暂停保留部分内存/显存分配，主要释放计算竞争，不卸载程序、不更改自启动。
2. 原训练主动保存并正常结束于 **325526**，num_updates **129983**。没有回滚到更早版本，也没有丢弃停训前的更新。
3. 该检查点独立保存在 outputs/training_resource_trial/resume_checkpoint.pt；SHA256 a6344117816e7c2919be644bcfc61667e03408b8afb720f0f65d5bbdfa9cb216。原运行及之前归档的最佳 319800 均保留。
4. 使用副本做 batch 32/64/128/256/512 的工程短测，每档 5 次更新，前 2 次不计稳态时间。短测仅检查显存、耗时与有限值，未用于评价动作或决定是否收敛，短测权重均丢弃。
5. 新运行 reference_contact_anchor_batch512 从 325526 继续，世界模型和策略 batch 都从 32 增至 512，其余学习配置与原运行完全一致。加载后逐项核对三个网络、三个优化器、三个采样统计张量一致。
6. 新进程回放缓存只需重建一次，预热 64 步；网络、Adam 状态和 sampler 未重置。

## 工程短测结果

| Batch | WM+policy/critic 稳态每次更新 | 总显存采样峰值 | Torch 峰值 allocated |
|---:|---:|---:|---:|
| 32 | 0.830 s | 1608 MiB | 389 MiB |
| 64 | 0.846 s | 1661 MiB | 440 MiB |
| 128 | 0.856 s | 1777 MiB | 548 MiB |
| 256 | 0.871 s | 1972 MiB | 779 MiB |
| 512 | 0.892 s | 2489 MiB | 1257 MiB |

所有档位数值有限且未发生 OOM。512 的样本数为原来的 16 倍，更新耗时较同条件 batch 32 增加约 7.4%；不代表收敛速度提升 16 倍。512 末尾 CUDA 报告空闲约 1404 MiB，预留独立全片评测和桌面所需空间，不以占满 4 GiB 为目标。

这些是小规模工程测量，包含启动阶段的回放分布；实际长跑速度以 console.log 为准。同期关闭壁纸和增加 batch，无法仅从长跑前后速度分离两项变化的因果贡献。

## 固定观察窗口

仍每新增 4000 步评测，按本次切换点计算：

- 第 1 次：329526（新增 4000 步）。
- 第 2 次：333526（新增 8000 步）。
- 第 3 次：337526（新增 12000 步）。

原计划 327800 已被新阶段完整 4000 步观察周期替代，这是明确记录的评测排程变化。每次均评测完整 616 步，仍按连续两次常规达标加三组扰动的既有逻辑决定成功停训。未达标时不调整配置、不因单次失败重启。三个评测完成后 review_window.json 标记可复盘；该标记本身不停止学习，也不自动换算法或调参。如果提前满足既有成功条件，则按原规则停止。

额外一次 transition_baseline 仅对切换前 325526 的固定权重做完整回放，建立前后对照并检验评测进程显存余量；不作为 batch 512 的质量评测点、不改训练状态。

新增保留最近三份评测权重，并保留最近通过质量门槛的检查点，避免观察窗口内有价值的候选被轮转。资源统计仅用于观察，不触发自动超参变更。

## 查看与恢复

当前服务：supermimic-g1-batch512.service；原 supermimic-g1-long.service 已安全退出。

```bash
cat logs/train/reference_contact_anchor_batch512/status.json
cat logs/train/reference_contact_anchor_batch512/resume_validation.json
tail -n 10 logs/train/reference_contact_anchor_batch512/console.log
systemctl --user status supermimic-g1-batch512.service --no-pager
```

本次服务成功结束、人工停止或故障结束后，由 ExecStopPost 自动恢复此前暂停的壁纸进程。必要时手动恢复：

```bash
python3 -B tools/restore_training_wallpaper.py --record outputs/training_resource_trial/wallpaper_pause.json
```

如需安全停止训练并保存：

```bash
touch logs/train/reference_contact_anchor_batch512/STOP
```

聊天助手不会在会话结束后自行观察或自动决定下一步修改；保存、评测、成功停止、壁纸恢复及三点汇总由本地程序执行。
