# G1 连续训练（2026-10-29）

用户授权：固定合理配置持续学习，允许早期失败及可承受误差，低频检查；达到期望的近似舞蹈质量后停止。此前几百步试验不能用于判定方法失败或收敛。

## 当前运行

- 服务：`supermimic-g1-long.service`，2026-10-29 16:11 启动；无需保持当前终端或聊天窗口。未设置开机自启或异常无限重启。
- 目录：`logs/train/reference_contact_anchor_long/`。
- 起点：`logs/train/reference_contact_anchor_ft2/ckpt_latest.pt`，iteration 291800。其中 291000 步来自作者；本地此前 anchor 分支只训练了 800 步。
- 输入：完整的 `data/preview/reference_full_contact.npz`，616 帧；未缩短、对齐消除朝向误差或混入其他失败候选。
- 保留世界模型、策略、critic、三个 Adam 状态与 13 个采样 bin 的统计。新进程只在启动时重建回放，之后评测不重启 Trainer、不清空回放。
- 固定 64 环境、batch 32、世界模型 2 次更新、k=8、想象 H=16、UTD 0.4、学习率 2e-5；anchor_obs 为 NJU 的仿真扩展。低显存配置与论文规模不同，不能承诺论文中的收敛速度。
- 去掉旧配置 0.4 小时上限；`--until-targets` 不使用默认 600 步短程预算。

## 检查与停止

每新增 **4000 次向量环境 step** 才评测整段，即每轮约采集 256000 条环境转移。第一次 295800，第二次 299800。每 100 步日志/25 步状态心跳只检查运行状况，不播放或评价动作，不中断学习。每 1000 绝对 step 原子保存恢复点，不触发质量评测。

完整评测禁止中途 reset。最终暂定误差容忍见 `configs/dance_acceptance.json`：平均/峰值倾斜 15°/50°、平均/峰值偏航 45°/100°、关节 RMSE 0.35 rad、平均/峰值根位置误差 0.6/1.2 m。它们只作为最终演示质量参考，不限制训练中的探索和失败。

**两次不同检查点连续达标** 后，在同一不可变检查点上用 seed 11、23、37 加入初态扰动，三次均达标才停训。任一次动作评测不达标都继续原配置学习。NaN、缺失指标、未完整播放、评测进程异常均不得计入成功；评测基础设施错误记录后继续训练。

达标状态为 `targets_met_video_review_pending`，保存 `ckpt_accepted.pt`，释放训练 GPU 后自动渲染 `accepted_preview/policy.mp4` 和 `reference.mp4`。这是重复物理回放达到暂定指标，不是数学收敛证明；视频仍需检查动作是否符合期望。系统不会把运行一段时间、平台期或单次高奖励当作成功。可能长期无法达标，目前不承诺必然收敛或固定完成时间。

真正的 NaN/Inf 训练故障、资源故障或人为停止允许中断；旧有效检查点保留。磁盘不足 1.5 GiB 时停止并保留最近恢复点，不冒险覆盖文件。进程崩溃会标记失败，不会误称达标。

## 保留与运行方式

新增运行目录最多保留最新、上一份、最优评测、最近两份评测检查点和达标检查点；相同版本以硬链接共用磁盘。原有下载和历史试验不会删除。写入临时文件完成并 fsync 后再原子替换，不允许边训练覆盖边评测。更新日志按 32 MiB 轮转。

查看状态：

```bash
cat logs/train/reference_contact_anchor_long/status.json
tail -n 15 logs/train/reference_contact_anchor_long/console.log
systemctl --user status supermimic-g1-long.service --no-pager
```

如需手动停止并保存当前状态：

```bash
touch logs/train/reference_contact_anchor_long/STOP
```

状态由脚本自动更新，聊天助手本身并不在会话结束后持续观察。服务阻止空闲休眠以完成训练；关机、强制休眠或完全注销仍可能中断。保持供电。

重启/续训使用新的 run-name，输入本轮最新有效 checkpoint，并加 `--keep-sampler --until-targets`；不要向已有检查点的运行目录覆盖写入。重启后回放缓存需重新预热，优化器和采样统计仍可保留。

## 验证

新增测试覆盖：两次独立检查点及三组扰动的停止逻辑、失败继续学习、非有限指标拒绝、原子替换不修改评测快照。此前配置覆盖、旋转指标和选择测试均通过。实际运行加载及性能数值以 status.json、console.log 为准；最终训练结果不能由这些工程测试代替。
