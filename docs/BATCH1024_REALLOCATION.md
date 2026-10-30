# 释放动态壁纸显存并继续训练（2026-10-29）

用户授权：退出动态壁纸释放其显存，并让训练程序使用更多显存。未改变此前观察 2–3 个评测点再考虑下一步调参的安排。

## 实际操作

- 验证 PID、启动标识和可执行路径后，对原已暂停的 komorebi 及其子进程发送 TERM 和 CONT，使其正常退出。没有卸载程序或更改开机自启动。
- 壁纸进程原占 479 MiB。退出后整卡占用由 2498 降至 1886 MiB；额外下降包括桌面图形资源释放，不能全部计为壁纸私有显存。
- 原 batch512 运行安全保存到 **328259**，更新计数 **131051**，并正常退出。保存的完整状态位于 outputs/training_resource_reallocation/resume_checkpoint.pt，SHA256 146e84680a5d7819445bd1d56684a33cb637fd1d29acedb7de2a3313e5453fbb。
- 对该副本做一次性 batch768/1024 工程短测，结果权重全部丢弃。它们不用于宣称动作改善或收敛。
- 采用 batch1024，从 **328259** 原样恢复网络、三个优化器和 sampler；逐项一致性校验通过。新进程回放缓存重建并预热 64 步，已有学习成果未重置。
- 与上一运行相比，学习配置差异仅为 wm_trainer.batch_size 和 policy_trainer.batch_size 从 512 改为 1024。学习率、网络、环境数、时域、噪声、参考动作、奖励、停止阈值均保持不变。

## 工程测量

| batch | 单次 WM+policy/critic 更新稳态中位耗时 | 总显存采样峰值 | CUDA 空闲 |
|---:|---:|---:|---:|
| 768 | 0.926 s | 2424 MiB | 1469 MiB |
| 1024 | 1.005 s | 2898 MiB | 995 MiB |

1024 有限值检查通过、未 OOM。并非将释放的显存静态绑定给训练，而是由更大的实际训练批量增加显存需求。没有分配无用途的占位张量，也没有更改驱动或 GPU 时钟。

实际恢复后单次检查：训练进程约 **2532 MiB**，整卡约 **2942 MiB**。相比原训练进程 1476 MiB，训练自身占用提高约 1056 MiB，利用了壁纸释放和此前剩余的显存。另执行固定切换检查点的全片回放以验证并存显存余量；结果在 transition_baseline/report.json，此回放不算新批量的质量评测点。

## 观察安排

评测时间点继续使用 **329526、333526、337526**，不因重启重新推迟。第一次评测相对本次切换只有 1267 步，属于 batch512/batch1024 混合区间，不能称为完整 4000 步纯 batch1024 结果；随后两次各包含完整 4000 步。除既有成功停止或真实运行故障外，保持该配置。

现有最高质量历史检查点和两次转换前检查点仍保留。原训练目录与参考文件均未覆盖。

## 当前入口

服务：supermimic-g1-batch1024.service

```bash
cat logs/train/reference_contact_anchor_batch1024/status.json
cat logs/train/reference_contact_anchor_batch1024/resume_validation.json
tail -n 10 logs/train/reference_contact_anchor_batch1024/console.log
```

训练结束后，通过 ExecStopPost 启动独立的 supermimic-wallpaper-restored.service，重新打开原壁纸程序。旧暂停进程已退出，不能再用 SIGCONT 恢复；恢复工具会检测是否已有同程序运行，避免重复启动。

若需人工安全停训：

```bash
touch logs/train/reference_contact_anchor_batch1024/STOP
```

当前训练未使用学习率降低或奖励修改方案。几个工程短测只验证能够执行，最终动作效果仍由原定完整评测与扰动复核判断。
