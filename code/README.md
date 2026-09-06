# code/

算法实现代码目录。

**实现路线已确定：以 [HuggingFace LeRobot](https://github.com/huggingface/lerobot) 的 ACT 实现为准**（它封装了数据集格式、训练/评估脚本与多种策略，比官方 `tonyzhaozh/act` 更便于复现和后续扩展对比）。

```
code/
├── README.md                   ← 本文件
├── reproduction_act_aloha.md   ← ACT 复现实验记录（LeRobot + aloha_mobile_cabinet，5k 训练/评测闭环，✅ 已闭环）
├── reproduction_smolvla.md     ← SmolVLA 复现实验记录（🔄 进行中，实验数据待回填）
├── data/                       ← LeRobot 数据集（gitignored，含本地复现实验数据）
└── outputs/                    ← 实验产物（act_aloha_5k、act_aloha_68ep：曲线图/日志/评测结果）
```

**复现状态：**
- ✅ **ACT 已复现**——完整实验记录见 [reproduction_act_aloha.md](reproduction_act_aloha.md)
- 🔄 **SmolVLA 复现进行中**——记录骨架见 [reproduction_smolvla.md](reproduction_smolvla.md)（实验数据待回填）

实验数据（数据集、checkpoint、日志）不提交到 git（见根目录 `.gitignore` 的 `data/`、`notebooks/`）。

学习路径：

1. `pip install lerobot`，clone [lerobot](https://github.com/huggingface/lerobot) 参考其教程
2. 用 ACT policy 配置跑通训练与评估，对照 `papers/01-act/` 笔记
3. 笔记中"动手实现 / 复现实验"章节以 LeRobot 的实现细节为准
4. 用 smolvla policy（LeRobot 内置）复现 SmolVLA，对照 `papers/02-smolvla/` 笔记与 `reproduction_smolvla.md`，重点与已复现的 ACT 在同一任务口径下对比
