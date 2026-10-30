# 接触修正与 RTX 3050 性能实测

更新（2026-10-29）：后续顺序改为预训练策略全片评测 → 按失败窗口修正参考 → 单舞蹈微调。PD 失败及瞬时动力学筛查不是学习策略评测/试训的硬性门槛。NJU 源码已核验，e13i 权重已补齐并完成全片策略基线评测；当前结论见 [NJU 仓库评估](NJU_REPO_ASSESSMENT.md) 和 [下载清单](DOWNLOADS.md)。


日期：2026-10-29。原视频 370 帧、30 FPS；完整动作保留 616 帧、50 Hz，首末采样跨度 12.30 秒，视频容器时长约 12.33 秒。

## 当前结论

4GB 显存已实际完成 SuperMimic 复现的采样、世界模型、策略和 critic 优化更新。局部代码优化在相同工作量下约加速 1.56 倍；较大的并行环境和批量也已短测通过。仍没有训练完成的舞蹈策略、收敛结果或新训练 checkpoint。

新的接触候选为 data/preview/reference_full_contact.npz。全部 616 帧通过 GPU 身体运动学比对；触地和接触段滑动明显改善。但大部分帧未通过瞬时动力学筛查，简单 PD 无法持续跟踪，因此仍为 training_approved=false，没有复制到正式训练目录。

## 看结果

- outputs/contact_validation/source_vs_g1.mp4：完整原视频与新 G1 参考对照。
- outputs/contact_validation/repaired_vs_contact.mp4：上一版修复与本次接触修正对照。
- outputs/contact_validation/contact_review.jpg：关键时间截图；floor_contact_review.jpg 放大检查 7.7–8.0 秒支撑。
- outputs/contact_validation/summary.json：质量、物理检查和性能汇总。

视频是运动学回放，不能代表重力下的策略跟踪。已目视检查支撑关键帧；相机视角与源视频不同，腿部和躯干仍有重定向变形。交互播放命令：

    .venv-motion/bin/python tools/view_g1.py --input data/preview/reference_full_contact.npz

## 接触修复

configs/reference_full_contacts.json 保存人工检查的手撑地及左脚支撑时间。其他足部支撑段由 GVHMR 静止置信度和二维脚踝置信度联合推断；标签有单目遮挡、时序和草地高度的不确定性，并非测量真值。

tools/refine_contacts.py 使用稀疏最小二乘，联合优化根节点平移及 29 个关节角，保留根节点旋转、全片时间和未选中区段。接触位置、碰撞体底部高度和时间平滑共同约束。求解 78 次评估、约 34.5 秒，没有覆盖旧动作。

| 指标 | 上一版 | 接触修正版 |
|---|---:|---:|
| 左手支撑段最大离地高度绝对值 | 19.89 cm | 0.21 cm |
| 右手支撑段最大离地高度绝对值 | 19.94 cm | 0.36 cm |
| 扫腿时左脚平均水平速度 | 1.073 m/s | 0.078 m/s |
| 相同足部接触区间平均水平速度 | 0.1880 m/s | 0.0238 m/s |
| 相同区间水平速度 P95 | 0.8143 m/s | 0.0782 m/s |

最后两项使用相同接触区间内部相邻帧，共 195 个样本，不包含接触边界外的移动。其他“近地面”启发式统计的样本集合会随修正改变，不能直接用来宣称全片滑脚都改善。

代价：单关节最大改动 0.669 rad（约 38.3°），根节点最大平移 18.9 cm。新动作更贴地，但不等于完全忠实原舞蹈。几何检查无关节超限、无启用碰撞对的自穿透、无超过 2 cm 的穿地；最低脚底约 -1.03 mm。最大关节速度 18.31 rad/s、最大加速度 563.47 rad/s²，仍有 8 帧超过 300 rad/s² 的复查阈值，该阈值不是电机额定值。

实际 mjlab GPU 校验全部 616 帧、30 个身体：最大位置偏差 3.83e-7 m，最大四元数范数差 2.35e-7。记录见 play_report.json。

## 动力学检查及限制

tools/audit_dynamics.py 使用真实 mjlab 模型质量、惯量、执行器力矩限值及被动项，拟合参考加速度所需力矩和单向摩擦接触力。已修正执行器顺序与关节顺序差异；另修正接触力臂：高度包络 Jacobian 不能代替实际球/胶囊表面接触点的完整 Jacobian，已用 MuJoCo 力映射回归检查验证。

- 站立平衡校准：5/5 帧通过，归一化基座力残差约 1.8e-6。
- 上一版：9/616 帧通过；接触修正版：23/616 帧通过。
- 修正版仍有 333 帧找不到 1 cm 内的候选碰撞体接触；腾空帧需要满足弹道动力学，不能一律视作错误或自动通过。
- 六个起点的真实自由基座 PD 探测均在 0.20–0.64 秒达到跟踪失败阈值；没有固定根节点、外力或运行中重置。结果见 physics_probes/probe_report.json。

筛查假设摩擦系数 0.6、使用四棱锥，没有接触互补性或完整粘着约束；参考微分噪声也影响结果。因此不通过是继续修正的证据，不能严格证明学习控制器不可能完成。手撑地区段需要同时调整质心、姿态、速度及支撑受力。

## 性能优化及测量

硬件：RTX 3050 Laptop 4096 MiB、i5-11400H、约 15 GiB RAM。训练使用独立 Python 3.12 环境，保持现有驱动；未超频、未改动功耗或温度保护。当前 mjlab 仍禁用 CUDA Graphs。

tools/fast_g1_fk.py 将机器人树同一层身体批量计算，减少小 CUDA 操作。tools/verify_fast_fk.py 已对照原实现验证 CPU float64、GPU float32、全部 616 帧、标量和多维批次的姿态、速度及输入梯度。GPU 输出本次完全一致，梯度误差处于容差内。64/128 批量 FK 前向加反向独立测试约快 2.4 倍。通过 --fast-fk 显式启用，不修改上游 FK 源文件或网络结构。

以下测试使用实际 Trainer、回放缓存和优化器，无渲染；每个作业先收集 44 步，再执行 5 或 8 组世界模型、策略和 critic 更新。预热前两次更新不计稳定耗时。训练步耗时为“实测采样步 + 0.4 × 实测更新中位数”的估算，不是连续长训吞吐，未包括周期诊断、存盘和长期热降频。

| 配置 | FK | 预计秒/训练迭代 | 总显存采样峰值 | 最高温度 |
|---|---|---:|---:|---:|
| 512 环境 / batch 128 | 原实现 | 1.063 | 1546 MiB | 73°C |
| 512 环境 / batch 128 | 分层批量 | 0.682 | 1546 MiB | 73°C |
| 1024 环境 / batch 128 | 分层批量 | 0.669 | 1752 MiB | 73°C |
| 1024 环境 / batch 256 | 分层批量 | 0.675 | 1866 MiB | 73°C |
| 2048 环境 / batch 128 | 分层批量 | 0.676 | 2270 MiB | 74°C |

同配置 512/128 比较约加速 1.56 倍，预计迭代耗时减少约 36%。2048 环境是本轮采样吞吐最高的配置，不是全局最优，也没有证明学习最快。扩大环境数但维持更新数会改变每条样本的训练复用率。1024/256 在耗时接近时每次更新使用更大批次，作为后续质量通过后首轮尝试的折中配置。

总显存包含桌面、Warp、PyTorch 等，nvidia-smi 每约 0.25 秒采样可能漏掉瞬时峰值，不能将 PyTorch 统计等同显卡总占用。短测损失均为有限值，不证明策略学会动作。high 精度仅探索测试，推荐仍用 highest、PyTorch CPU 1 线程、单个 GPU 作业串行运行。

## 配置与复现

旧 configs/g1_3050.json 保留。新增：

- configs/g1_3050_tuned.json：1024 环境 / batch 256，首轮折中候选。
- configs/g1_3050_throughput.json：2048 环境 / batch 128，采样吞吐候选。

两者仍指向 data/npz/pilot/，未混入预览动作。小网络、512 步回放容量、16 步想象、UTD 0.4、关闭随机化等是本机工程设置，不代表论文原始大规模训练设置。

可立即复现短程性能验证，显式读取预览动作且不保存 checkpoint：

    envs/training/.venv/bin/python tools/benchmark_training.py --envs 1024 --batch 256 --threads 1 --fast-fk --updates 5 --output outputs/performance/recheck_1024_b256.json

重跑接触修正和逐帧 GPU 回放：

    .venv-motion/bin/python tools/refine_contacts.py --input data/preview/reference_full_repaired.npz --output-motion data/preview/reference_full_contact.npz --report-dir outputs/contact_validation
    envs/training/.venv/bin/python tools/validate_g1.py --mode play --input data/preview/reference_full_contact.npz --output outputs/contact_validation

以下仅在动作质量通过、放入正式目录后使用，当前会因无数据退出：

    envs/training/.venv/bin/python tools/train_3050.py --config configs/g1_3050_tuned.json --threads 1 --fast-fk --smoke

train_3050.py 会保存线程、精度和 FK 模式到日志目录 runtime.json；恢复时使用相同参数。正式训练前仍需验证周期日志、存盘与连续运行。

## 验证及下一阶段

运动处理环境 12 项数值回归通过；另 4 项 FK 回归在 Python 3.12 训练环境通过。GPU 真实轨迹与梯度测试通过，对照视频完整解码无错误。

    .venv-motion/bin/python -m unittest discover -s tests -v
    envs/training/.venv/bin/python -m unittest discover -s tests -p test_fast_fk.py -v
    envs/training/.venv/bin/python tools/verify_fast_fk.py

下一阶段完善全片支撑时序，修正腾空段质心弹道及角动量，再加入力矩和接触力约束做轨迹调整；先验证可跟踪的小段，再扩展全片。增加显存使用率或训练时长不能确认当前参考可执行。原七项依赖完整；新增 e13i 权重已补齐，两版动作均完成上游基线评测，姿态跟踪仍有偏差。没有后台长训作业，也没有实现 G1 自主跳完整舞蹈。
