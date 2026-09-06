# 具身智能算法学习

具身智能（Embodied AI）算法的阅读笔记与公式推导。从 ACT（Action Chunking with Transformers）出发，按学习路线逐篇推进。

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
│   └── 02-smolvla/          ← SmolVLA (Shukor et al., arXiv 2025)
│       ├── SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics.md
│       └── SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics.pdf
```

每篇论文一个子目录，笔记和 PDF 均以论文标题命名。

---

## 论文清单

| # | 论文 | 状态 |
|---|------|------|
| 1 | [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](papers/01-act/Learning%20Fine-Grained%20Bimanual%20Manipulation%20with%20Low-Cost%20Hardware.md) — Zhao et al., RSS 2023 | ✅ 已完成 |
| 2 | [SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics](papers/02-smolvla/SmolVLA:%20A%20Vision-Language-Action%20Model%20for%20Affordable%20and%20Efficient%20Robotics.md) — Shukor et al., arXiv 2025 | ✅ 已完成 |

> 状态: 📝 待读 / 🔍 阅读中 / ✅ 已完成

---

## 学习路线

ACT 是一条主线，建议按以下顺序推进（读完在对应项打勾，并把新论文加入上方清单）：

- [ ] **预备知识**：Transformer、CVAE（条件变分自编码器）、行为克隆（BC）/ 模仿学习基础
- [x] **核心论文**：ACT (RSS 2023) — 论文阅读 + 公式推导（笔记见 [papers/01-act/](papers/01-act/)）
- [x] **动手实现**：用 [LeRobot](https://github.com/huggingface/lerobot) 复现 ACT（✅ 已复现，记录见 [code/reproduction_act_aloha.md](code/reproduction_act_aloha.md)）
- [ ] **对比算法**：Diffusion Policy (Chi et al., RSS 2023)
- [x] **延伸方向：VLA**：SmolVLA (arXiv 2025) — 紧凑 VLA 架构 + 社区数据预训练 + 异步推理（笔记见 [papers/02-smolvla/](papers/02-smolvla/)，复现 🔄 进行中，记录见 [code/reproduction_smolvla.md](code/reproduction_smolvla.md)）
- [ ] **延伸对比**：精读 π0 / OpenVLA，对照 SmolVLA 的"小模型+流匹配专家"与 π0 的"大 VLM+扩散"路线差异
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
