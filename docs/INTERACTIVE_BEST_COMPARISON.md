# 最佳策略与无策略参考：交互仿真对照

服务：supermimic-g1-live-comparison.service。程序：tools/view_best_comparison.py。固定检查点：429526，哈希与最佳评测报告一致。

左侧为无策略的运动学参考，右侧为该检查点实时推理驱动的 mjlab/Warp 物理仿真。显示平移仅作用于 CPU 可视化数据，不修改策略观测或仿真状态。

空格暂停/继续；Enter 从头播放；- 和 = 调整目标速度；鼠标拖动和滚轮调整视角。每次完整动作结束停留 0.7 秒后循环；失败会暂停并显示，不通过中途重置掩盖。

关闭窗口只结束预览，不影响训练。未录制新视频。训练与实时仿真共享显卡，已观察实际约 0.3 倍速度，界面显示 actual 实际倍率。提高目标速度不会保证加速。

查看状态：outputs/interactive_comparison/iteration_429526/status.json；日志：同目录 console.log。此服务不会自动重启或随开机启动。

## 原速交互回放（2026-10-30）

用户要求录屏时原速播放。原实时推理预览已关闭，现启用 supermimic-g1-replay-comparison.service，程序 tools/view_best_replay.py。
仍是原生 MuJoCo 可交互窗口：左侧无策略参考，右侧最佳429526步已验证物理轨迹。这里不重新进行策略推理或积分，不是 MP4 播放器；按原记录50Hz时间轴进行1x播放，无额外对齐或动作重定时。
空格暂停/继续，Enter从头播放，鼠标旋转/缩放，关闭窗口仅结束预览。原始12.32秒动作实测循环耗时约12.320秒、显示约50fps。
训练原进程持续运行。状态及验证：outputs/interactive_comparison/iteration_429526/replay_status.json、replay_verification.json。
