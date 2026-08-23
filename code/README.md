# code/

算法实现代码目录。

参考仓库（AI-World-Models-Study）为纯笔记仓库，不包含代码；但 ACT 是可运行的算法，因此预留此目录，随学习进度逐步添加：

```
code/
├── act/              ← ACT 实现（可先 clone 官方仓库 tonyzhaozh/act，再逐行阅读/复现）
├── requirements.txt  ← 依赖清单（随实现添加）
└── README.md         ← 本文件
```

建议学习顺序：

1. `git clone https://github.com/tonyzhaozh/act.git` 到本目录
2. 通读 `policy.py`（ACT 模型定义）与 `train.py`（训练循环），对照 `papers/01-act/` 笔记
3. 在仿真环境（如 ALOHA 提供的 sim）中跑通训练与评估
