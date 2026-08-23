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
```

每篇论文一个子目录，笔记和 PDF 均以论文标题命名。

---

## 论文清单

| # | 论文 | 状态 |
|---|------|------|
| 1 | [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](papers/01-act/Learning%20Fine-Grained%20Bimanual%20Manipulation%20with%20Low-Cost%20Hardware.md) — Zhao et al., RSS 2023 | 🔍 阅读中 |

> 状态: 📝 待读 / 🔍 阅读中 / ✅ 已完成

---

## 学习路线

ACT 是一条主线，建议按以下顺序推进（读完在对应项打勾，并把新论文加入上方清单）：

- [ ] **预备知识**：Transformer、CVAE（条件变分自编码器）、行为克隆（BC）/ 模仿学习基础
- [x] **核心论文**：ACT (RSS 2023) — 论文阅读 + 公式推导
- [ ] **动手实现**：跑通官方代码 [tonyzhaozh/act](https://github.com/tonyzhaozh/act)，在仿真中复现
- [ ] **对比算法**：Diffusion Policy (Chi et al., RSS 2023)
- [ ] **延伸方向**：VLA（如 RT-2、OpenVLA）等视觉语言动作模型

---

## 更新日志

| 日期 | 内容 |
|------|------|
| 2026-08-23 | 初始化仓库结构，创建笔记模板 |
| 2026-08-23 | 添加第一篇论文：Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware（Zhao et al., RSS 2023），创建骨架 |
