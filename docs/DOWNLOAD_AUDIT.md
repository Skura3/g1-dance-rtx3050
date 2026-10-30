# 下载资产核验记录

日期：2026-10-29。原清单六项与新增 CUDA 12 Warp wheel 全部核验通过，当前无需手动补下载，见 [下载清单](DOWNLOADS.md)。

## 已修正

1. ViTPose 原名多了 `-001`，内容符合 ViTPose-H COCO17；建立源码要求文件名的相对软链接，没有改动权重。
2. HMR2 原先只有解包目录。在 `/home/djs/下载/epoch%3D10-step%3D25000.ckpt` 找到完整原件，硬链接到正确位置。
3. SMPL-X 原先只有解出的 NPY。在 `/home/djs/下载/SMPLX_NEUTRAL.npz` 找到完整原件，硬链接到正确位置。硬链接没有再占一份磁盘空间，解出的旧内容保留。
4. GVHMR 加载被系统旧版 transforms3d 的 `np.float` 引用阻塞；仅在 `.venv-motion` 安装 transforms3d 0.4.2，保留 NumPy 1.26.4。修复后加载通过。
5. Torch wheel 通过锁文件 SHA256 核验并完成离线安装，训练 GPU 算术通过。没有更换 motion 环境的 torch 2.5.1+cu121。
6. 增加 `tools/use_local_warp.py`，setup 会校验并固定 CUDA 12 Warp wheel，防止再次选择 CUDA 13 构建。
7. 用户补齐 Warp wheel 后，确认 164625100 字节、官方 SHA256、归档 CRC、版本与平台标签均正确；离线替换为 `warp-lang==1.18.0+cu12`。RTX 3050 上实际编译并运行 Warp kernel 通过。

## 结果与范围

| 项目 | 实际检查结果 |
|---|---|
| YOLO | YOLOv8x，COCO80，CPU 合成单帧前向通过 |
| ViTPose | ViTPose-H COCO17，403 项权重完全匹配，单帧输出 `[1,17,64,48]` |
| HMR2 | HMR2.0a，epoch 10 / step 25000；上游使用的 503 项权重匹配，特征输出 `[1,1024]` |
| GVHMR | epoch 419 / step 158340，247 项权重匹配，缺失/多余键均为零，骨架层通过 |
| SMPL-X | 10475 顶点、55 骨架关节；加载与零姿态前向通过 |
| Torch | 2.14.1+cu126，cp312 Linux x86_64 标签、锁文件 SHA256 匹配，GPU 算术通过 |
| Warp | 1.18.0+cu12，Linux x86_64，官方 SHA256 匹配，GPU kernel 编译和执行通过 |

七个文件均通过完整归档 CRC 检查；逐项大小、SHA256、结构与日志在 `outputs/download_audit/`，汇总为 `assets.json`。
SMPL-X 没有可验证的内嵌发行版/性别证明；核验的是 neutral 原始文件名、标准维度和实际加载，不声称认证了官方来源。

资产核验阶段未执行视频推理；后续环境验证现已完成前 5 秒视频重建/GMR、G1 物理站立及参考回放，见 [环境验证](ENVIRONMENT_VALIDATION.md)。策略训练尚未执行，不保证 4GB 显存训练收敛。
基础训练运行时检查见 `outputs/training_readiness.json`；机器人实际验证见 `outputs/environment_validation/summary.json`。
