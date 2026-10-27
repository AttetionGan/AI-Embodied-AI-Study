# 具身智能算法学习

具身智能（Embodied AI）算法的阅读笔记与公式推导。从 ACT（Action Chunking with Transformers）出发，按学习路线逐篇推进。

> **English intro**: Personal deep-study notes & reproductions on Embodied AI / imitation learning, from ACT to VLA (π0, SmolVLA). Each paper note comes with math derivations, architecture diagrams, **paper-vs-code discrepancy tables**（论文与官方实现差异核对表）, ablations, and honest reproduction records with clear evidence boundaries（证据边界）.

## 这个仓库有什么不一样？（能帮你省什么时间）

这是一个"边读边写、边复现边挑错"的个人学习仓库。如果你也在学具身智能/模仿学习，这里的笔记可能比论文解读博客多一层价值：

- **论文 vs 官方代码核对表**：不只复述论文，还逐条核对论文说法与官方实现的实际差异（如 ACT 论文 Algorithm 1 写 MSE、官方代码实为 L1；论文写 decoder 7 层、实际仅 1 层生效）。这类"踩坑点"是读论文时最容易卡住、又没人写的地方。
- **公式推导与符号表**：关键公式给到直觉解释与推导过程，而不是贴公式了事。
- **诚实的复现记录**：每个实验都写明证据边界——什么能证明、什么不能证明；无实机不虚构成功率；结论与局限分开列。
- **成体系的学习路线**：从 ACT 出发 → VLA，每篇笔记标好前置依赖，方便按序跟进。

> **欢迎纠错与交流**：笔记中凡涉及论文数字、消融结论、代码行为的断言，都欢迎指正——我的目标是准确性优先。发现错误或想讨论某篇论文，欢迎提 [Issue](https://github.com/AttetionGan/AI-Embodied-AI-Study/issues) 或 [Discussion](https://github.com/AttetionGan/AI-Embodied-AI-Study/discussions)；每一处纠错都会记入 [更新日志](#更新日志)。

## 仓库结构

```
.
├── README.md                ← 本页（论文清单 + 学习路线 + 结构说明）
├── code/                    ← 算法实现代码（随阅读进度逐步添加）
├── papers/
│   ├── _template/           ← 新建论文时复制此目录
│   │   └── README.md        ← 模板
│   │
│   └── 01-act/              ← ACT: Action Chunking with Transformers (Zhao et al., RSS 2023)
│       ├── Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware.md
│       └── Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware.pdf
│
│   ├── 02-smolvla/          ← SmolVLA (Shukor et al., arXiv 2025)
│   │   ├── SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics.md
│   │   └── SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics.pdf
│   │
│   └── 03-pi0/              ← π0: A Vision-Language-Action Flow Model for General Robot Control (Physical Intelligence, arXiv 2024)
│       ├── π0: A Vision-Language-Action Flow Model for General Robot Control.md
│       └── π0: A Vision-Language-Action Flow Model for General Robot Control.pdf
```

每篇论文一个子目录，笔记和 PDF 均以论文标题命名。

---

## 论文清单

| # | 论文 | 状态 |
|---|------|------|
| 1 | [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](papers/01-act/Learning%20Fine-Grained%20Bimanual%20Manipulation%20with%20Low-Cost%20Hardware.md) — Zhao et al., RSS 2023 | ✅ 已完成 |
| 2 | [SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics](papers/02-smolvla/SmolVLA:%20A%20Vision-Language-Action%20Model%20for%20Affordable%20and%20Efficient%20Robotics.md) — Shukor et al., arXiv 2025 | ✅ 已完成 |
| 3 | [π0: A Vision-Language-Action Flow Model for General Robot Control](papers/03-pi0/%CF%800:%20A%20Vision-Language-Action%20Flow%20Model%20for%20General%20Robot%20Control.md) — Black et al. (Physical Intelligence), arXiv 2024 | ✅ 已完成 |

> 状态: 📝 待读 / 🔍 阅读中 / ✅ 已完成

---

## 学习路线

ACT 是一条主线，建议按以下顺序推进（读完在对应项打勾，并把新论文加入上方清单）：

- [ ] **预备知识**：Transformer、CVAE（条件变分自编码器）、行为克隆（BC）/ 模仿学习基础
- [x] **核心论文**：ACT (RSS 2023) — 论文阅读 + 公式推导（笔记见 [papers/01-act/](papers/01-act/)）
- [x] **动手实现**：用 [LeRobot](https://github.com/huggingface/lerobot) 复现 ACT（✅ 已复现，记录见 [code/reproduction_act_aloha.md](code/reproduction_act_aloha.md)）
- [ ] **对比算法**：Diffusion Policy (Chi et al., RSS 2023)
- [x] **延伸方向：VLA**：SmolVLA (arXiv 2025) — 紧凑 VLA 架构 + 社区数据预训练 + 异步推理（笔记见 [papers/02-smolvla/](papers/02-smolvla/)，复现 🔄 进行中，记录见 [code/reproduction_smolvla.md](code/reproduction_smolvla.md)）
- [x] **延伸对比：π0**：精读 π0 (arXiv 2024) — 大 VLM 骨干 + 流匹配动作专家 + 跨形态预训练（笔记见 [papers/03-pi0/](papers/03-pi0/)）
- [ ] **延伸对比：OpenVLA**：精读 OpenVLA，对照 π0 的"VLM + 流匹配专家"与 SmolVLA 的"小模型 + 冻结骨干"两条路线差异
- [ ] **VLA 复现**：用 LeRobot smolvla policy 复现 SmolVLA，与已复现 ACT 在同一任务口径下对比（🔄 进行中）

---

## 更新日志

| 日期 | 内容 |
|------|------|
| 2026-08-23 | 初始化仓库结构，创建笔记模板 |
| 2026-08-23 | 添加第一篇论文：Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware（Zhao et al., RSS 2023），创建骨架 |
| 2026-08-30 | 完成 ACT 论文精读笔记（含公式推导、框架图、论文/LeRobot 实现核对表、消融分析）；实现路线定为 LeRobot 并标注已复现 |
| 2026-08-30 | 新增 ACT 复现实验记录 [code/reproduction_act_aloha.md](code/reproduction_act_aloha.md)（LeRobot + aloha_mobile_cabinet，5k 训练/评测闭环） |
| 2026-09-03 | 添加第二篇论文：SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics（Shukor et al., arXiv 2025），完成精读笔记（紧凑 VLA 架构、社区数据预训练、异步推理、LIBERO/MetaWorld/SO100 结果与 8 组消融）；学习路线勾选 VLA 延伸方向 |
| 2026-09-03 | 新增 SmolVLA 复现记录骨架 [code/reproduction_smolvla.md](code/reproduction_smolvla.md)（🔄 实验进行中，数据待回填） |
| 2026-09-03 | 图规范升级：ACT 与 SmolVLA 笔记的架构/数据流/流程/历史脉络图由 ASCII 改为 Mermaid；同步更新 [笔记模板](papers/_template/README.md) 与 [paper-study-workflow skill](.opencode/skills/paper-study-workflow/SKILL.md)（Mermaid 惯例 + 防坑说明，算法伪代码框保留 ASCII） |
| 2026-09-10 | 添加第三篇论文：π0: A Vision-Language-Action Flow Model for General Robot Control（Black et al. / Physical Intelligence, arXiv 2024），完成精读笔记（流匹配 VLA 架构、块状因果注意力、10,000h 跨形态预训练/后训练 recipe、openpi 论文-代码核对表 13 条）；学习路线勾选"延伸对比：π0"并新增 OpenVLA 待办 |
| 2026-10-27 | 补齐 ACT 与 SmolVLA 笔记缺失的 §12 Discussion Points（ACT 4 条、SmolVLA 5 条），全部锚定论文数字与本工程复现证据（per-horizon 平坦曲线、prior/posterior 差值、async 成功率下降、481 数据集不可复现等），三篇笔记结构与[模板](papers/_template/README.md)对齐；`.gitignore` 忽略 macOS 的 `.DS_Store` |
