# 下载文件核验与剩余清单

更新：2026-10-29。项目根目录 `/home/djs/supermimic`。

## 已补齐并验证：NJU 预训练权重

2026-10-29 已核验用户下载的 data/downloads/SuperMimic-Repro-Sim2Real-main.zip：ZIP CRC 完整；解压后的 145 个文件与运行目录 third_party/SuperMimic-Repro-Sim2Real 完全一致。ZIP 提交为 f8f60d7bbdba68468c050ef100af02d752ca238e。下载目录可保留，不需要重复安装或搬动源码。报告见 outputs/download_audit/nju_source.json。

源码包本身不含权重；用户已另行下载并解压下述 Release 包。完整归档 SHA256、4 个有效文件与 CPU 权重加载全部通过，无需重下。

下载：<https://github.com/NJU-R-L-Group-Embodied-Lab/SuperMimic-Repro-Sim2Real/releases/download/v1.0/e13i_0291000.tar.gz>

保存到：/home/djs/supermimic/data/downloads/nju/e13i_0291000.tar.gz

- 完整大小：266591326 字节（约 254.24 MiB）。
- Release SHA256：2aa3f4ca81d7a2e4cdf357d9a3b60d6809fd7d87e0db3005cad55d3ef22709fa。
- 先前自动下载两次超时；现在用户手动补齐。旧 .partial 保留作历史残片，不参与加载。
- 已确认实际权重：data/downloads/nju/e13i_0291000/ckpt_e13i_0291000.pt；解压位置可直接使用。模型 iteration=291000，策略目标维度 680，全部策略参数有限。核验见 outputs/download_audit/nju_checkpoint.json。
- 包中的 ._ 前缀文件为归档附带元数据，加载时忽略；不属于缺失/损坏模型。
- 两版全片参考均已执行预训练策略评测，616/616 步未触发上游失败条件；这不代表动作误差达标，见 PRETRAINED_EVALUATION.md。
- 不需要下载作者特定舞蹈的 e24 包、整个 BONES-SEED 动作库或真机 SDK。

详细兼容性及本机短测结果见 [NJU 仓库评估](NJU_REPO_ASSESSMENT.md)。

## 原七项文件均已通过核验

完整结果见 [核验记录](DOWNLOAD_AUDIT.md) 和 `outputs/download_audit/assets.json`。人体模型与四个权重无需重下，Torch 和 CUDA 12 Warp wheel 已核验并离线安装到独立训练环境。

```text
third_party/GVHMR/inputs/checkpoints/
├── yolo/yolov8x.pt
├── vitpose/vitpose-h-multi-coco.pth -> vitpose-h-multi-coco-001.pth
├── hmr2/epoch=10-step=25000.ckpt
├── gvhmr/gvhmr_siga24_release.ckpt
└── body_models/smplx/SMPLX_NEUTRAL.npz
data/downloads/wheels/
├── torch-2.14.1+cu126-cp312-cp312-manylinux_2_28_x86_64.whl
└── warp_lang-1.18.0+cu12-py3-none-manylinux_2_28_x86_64.whl
```

HMR2 与 SMPL-X 已从“下载”目录的完整原件建立硬链接，未改动文件内容，也未重复占用磁盘。此前解出的文件夹保留。
`.npz`、`.ckpt` 和 `.whl` 均应保留为完整文件，**不要解开成为同名目录**。SMPL-X 外层下载包可以解压，以取得其中的 `.npz`；`.npz` 本身不要再解开。

## 已补齐：CUDA 12 Warp wheel

此前普通 `warp-lang 1.18.0` wheel 是 CUDA 13.4 构建，本机驱动 535.309.01 无法初始化其 GPU。现已替换为用户下载的官方 CUDA 12 兼容构建，Torch GPU 运算和 Warp 内核编译、执行通过；没有修改系统驱动。以下链接保留作来源记录，无需重下。

直接下载：<https://github.com/NVIDIA/warp/releases/download/v1.18.0/warp_lang-1.18.0%2Bcu12-py3-none-manylinux_2_28_x86_64.whl>

官方发布页：<https://github.com/NVIDIA/warp/releases/tag/v1.18.0>

保存为：

```text
/home/djs/supermimic/data/downloads/wheels/warp_lang-1.18.0+cu12-py3-none-manylinux_2_28_x86_64.whl
```

- 完整大小：164625100 字节，约 157 MiB。
- 官方 Release asset SHA256：`6dda3c163dde4edaa7d46ade9fff3ddb1126ee7b147d9d890cd38cc70dafd656`。
- 文件名必须含 `+cu12` 和 `x86_64`，保留 wheel，不解压。
- 前轮自动下载两次均超时；本轮用户手动补齐，文件已校验并离线安装，记录见 `outputs/download_audit/warp_wheel.json`。
- `.partial` 是未完成文件，不能改名冒充完整 wheel。

本轮已执行；如需复查：

```bash
cd "$G1_PROJECT_ROOT"
bash tools/setup_training.sh --offline
envs/training/.venv/bin/python tools/check_training.py
```

安装脚本核验两个 wheel 的 SHA256，再固定为本地来源和 `warp-lang==1.18.0+cu12`，避免再次选中 CUDA 13 包。缺少兼容包时明确停止。
当前 **基础 GPU 检查、G1 单环境物理站立与参考播放均通过**，见 [环境验证](ENVIRONMENT_VALIDATION.md)。训练组件短测及已发布策略的全片评测已完成；所有当前所需下载项已齐。没有针对本舞蹈微调完成的策略，不能据此保证最终跟踪质量。

## 已有文件原始入口（无需重复下载）

- SMPL-X：<https://smpl-x.is.tue.mpg.de/>，注册、接受许可后下载 NPZ；当前只需 neutral。
- 四个权重：<https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD?usp=drive_link>。
- GVHMR 安装说明：<https://github.com/zju3dv/GVHMR/blob/main/docs/INSTALL.md>。
- Torch wheel：<https://download-r2.pytorch.org/whl/cu126/torch-2.14.1%2Bcu126-cp312-cp312-manylinux_2_28_x86_64.whl>。
- Torch SHA256：`75727245346d8f92224a860bf40c56bd681edcfa0456d01d7dfcb0bf6f0cf86c`。

当前 SimpleVO / 无人体网格渲染路线不需要 DPVO、AMASS/BEDLAM、SMPL 或男性/女性 SMPL-X，不必额外下载。

## 源码状态（均已获取）

以下源码均已下载，无需你再次下载。精确提交见 `configs/upstream_sources.json`：

| 项目 | 本地目录 | 官方/上游链接 |
|---|---|---|
| GVHMR | `third_party/GVHMR` | <https://github.com/zju3dv/GVHMR> |
| GMR | `third_party/GMR` | <https://github.com/YanjieZe/GMR> |
| SuperMimic 第三方复现 | `third_party/SuperMimic-Reproduced` | <https://github.com/CuboiLeo/SuperMimic-Reproduced> |
| mjlab，使用复现指定提交 | `third_party/mjlab` | <https://github.com/mujocolab/mjlab> |
| PyTorch3D，仅安装纯 PyTorch 变换功能 | `third_party/pytorch3d` | <https://github.com/facebookresearch/pytorch3d> |

SuperMimic 这里是独立第三方复现，不是论文作者官方源码。

## 重新核验

```bash
cd "$G1_PROJECT_ROOT"
.venv-motion/bin/python tools/audit_downloads.py
.venv-motion/bin/python tools/check_ready.py
```

`audit_downloads.py` 在 CPU 上逐个加载，检查完整归档、权重结构和合成输入前向，不下载文件，不执行舞蹈推理或训练。
`check_ready.py` 汇总基础环境与已有验证报告；文件存在性不等于内容有效，报告不能证明视频预测质量或训练收敛。
