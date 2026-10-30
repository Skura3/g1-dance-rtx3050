# 夜间独立看护（2026-10-29）

用户授权的两个保护条件为 GPU 温度严格超过 100°C，或训练步数连续严格超过 4 小时没有增长。不是奖励或动作误差 4 小时未改善，也不是累计训练达到 4 小时。100°C 是用户指定触发值，不是工作温度目标。

## 部署

- 配置：configs/g1_3050_watchdog.json
- 程序：tools/training_watchdog.py（仅标准库，不加载 CUDA 或模型）
- 看护服务：supermimic-g1-watchdog.service
- 训练服务：supermimic-g1-strict.service，绑定其本次 InvocationID，避免误操作后续替换的进程。
- 每 10 秒通过 nvidia-smi 读取指定 GPU 温度，并读取 status.json 与 updates.jsonl 中最新步数。只更新时间戳不会重置停滞计时；步数真正增长才会。
- CLOCK_BOOTTIME 计时独立于系统墙钟调整；同次开机看护重启后保留已累计停滞时间。看护自身异常退出后 10 秒重试，训练不会因此被重启。
- 安装看护无需重启训练。本次训练 Python PID 保持 199780，网络、优化器、回放缓存不变。

## 保护触发后的行为

1. 保留当前运行最近发布的最佳评测检查点，同时保留 batch1024 前段与历史长训最佳；根据既有 selection_score_lower_is_better 选择主要最佳，分数相同取较新者。此排名不等于严格阈值通过或人工视觉认可。历史文件不会删除。
2. 检查点使用同文件系统硬链接固定已原子保存的文件，再核对 SHA256 与评测报告一致；训练后续替换源路径不会覆盖固定快照，也不额外复制整份权重占用磁盘。
3. 写 incident.json 和训练目录 watchdog_stop.json，记录温度、停滞时长、步数、配置、最佳检查点来源与哈希，再写 STOP 请求训练在安全边界保存最新状态并退出。
4. 过温给予 15 秒安全退出机会；停滞给予 60 秒。仍未退出则对该服务发出 systemctl stop --no-block，再等待 15 秒；依然无响应则终止该服务控制组。强制退出时不声称内存中最新进度已保存，保留已落盘最佳与最近检查点，并记录 latest_state_save_confirmed=false。
5. 过温事件暂停动态壁纸恢复，避免训练刚退出就重新增加 GPU 负载；正常训练完成仍沿用原恢复规则。
6. 保护触发后不自动重启训练；正常完成或训练已退出时，看护记录状态并退出。

## 文件位置

- 实时看护状态：outputs/training_watchdog/strict/status.json
- 事件日志：outputs/training_watchdog/strict/events.jsonl；同时进入 systemd 用户日志。
- 已固定最佳权重：outputs/training_watchdog/strict/snapshots/
- 触发后事故报告：outputs/training_watchdog/strict/incident.json
- 触发后保留权重：outputs/training_watchdog/strict/incident_checkpoints/
- 训练目录摘要：logs/train/reference_contact_anchor_strict/watchdog_stop.json

## 验证和边界

八项隔离测试覆盖 >100°C、>14400 秒的严格边界、真实步数进展、缓存时间戳不误判、检查点原子替换与哈希不一致、看护重启计时、保存后正常退出、卡死升级终止，以及不向更换后的服务发信号。测试使用临时目录和模拟服务，不故意使实际 GPU 过热，也不打断真实训练。

若温度传感器查询失败，记录读取错误，不把缺失值伪装成低温；停滞监控仍继续，但这段时间无法判断温度是否超限。断电、操作系统整体崩溃或不可中断的内核卡死无法由用户态看护保证现场保存。现有周期性检查点仍是恢复依据。

这是本次训练的看护服务，没有设置开机自动恢复训练，也没有更改奖励、学习率、batch 或评测频率。睡前保持插电、散热畅通，锁屏即可。
