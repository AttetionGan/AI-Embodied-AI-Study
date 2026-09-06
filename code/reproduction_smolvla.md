# SmolVLA 复现实验记录（LeRobot + SO-100/101）

> **核心实验**：SmolVLA（LeRobot smolvla policy）复现——数据集/评测任务以实际跑通为准
> **日期**：2026-09-03 建档
> **状态**：🔄 **实验进行中**——本文档为骨架，所有实验数据留白待回填；回填时删除"待回填"标记并更新状态
> **固定边界**：无实机则写"无实机成功率"；仿真/离线条目不得写成实机性能；无匹配仿真就说清证据止于离线指标

> **🔁 论文对照**：论文为 [SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics](../papers/02-smolvla/SmolVLA:%20A%20Vision-Language-Action%20Model%20for%20Affordable%20and%20Efficient%20Robotics.md)，实验口径（成功率的细粒度评分、sync/async 定义、chunk=50、flow matching 10 步）以论文 §4 为准。

---

## 1. 实验概述与证据边界

### 1.1 Smoke test

<!-- 待回填：smoke run 步数与验证内容（只验证工程链路，不进入性能评价） -->

> 待回填：smoke 链路验证结论（forward/loss/backward/optimizer 能跑）。

### 1.2 三类证据（模板，跑完按实际增删）

| 证据 | 对应实验 | 能证明什么 | 不能证明什么 |
|---|---|---|---|
| 训练稳定性 | <!-- 待回填：数据集 + steps run --> | 真实/仿真数据上的训练链路、优化稳定性 | 未见任务成功率、泛化 |
| 离线/仿真策略评估 | <!-- 待回填 --> | 策略在评测集/仿真环境的任务成功率 | 实机成功率（若为仿真） |
| 实机闭环 | <!-- 待回填：SO100/SO101 任务 --> | 真机成功率、async 收益 | <!-- 待回填 --> |

> 三类证据必须分栏展示，不能把仿真结果写成实机性能。

---

## 2. 数据集（Dataset Card）

<!-- 待回填：以实际使用的数据集 meta/info.json 为准。
官方评测数据集（论文 §4.1，SO100）：lerobot/svla_so100_pickplace、lerobot/svla_so100_stacking、lerobot/svla_so100_sorting；
SO101：lerobot/svla_so101_pickplace（Pick-Place-Lego，OOD 设置：Lego 放在训练未见位置）。
仿真：LIBERO physical-intelligence/libero（1,693 eps）、Meta-World lerobot/metaworld_mt50（2,500 eps）。
若复现走社区数据预训练 + 微调，预训练集为 481 个 lerobot-tag 数据集（论文附录 A.1 全清单）。 -->

| 字段 | 数值 |
|---|---|
| repo | <!-- 待回填 --> |
| robot type | <!-- SO100 / SO101 / 仿真 --> |
| episodes | <!-- 待回填 --> |
| frames | <!-- 待回填 --> |
| tasks | <!-- 待回填 --> |
| fps | <!-- 待回填 --> |
| cameras | <!-- 待回填（论文归一化为 OBS_IMAGE_1/2/3） --> |
| image shape | <!-- 待回填（论文 resize 512×512） --> |
| state/action 维度 | <!-- 待回填 --> |
| 任务标注 | <!-- 待回填：原始 or Qwen2.5-VL-3B 清洗 --> |

---

## 3. 训练配置（resolved）

<!-- 待回填：以 checkpoint 内保存的 resolved train config 为准（LeRobot commit/version、dataset revision、
policy 类型 smolvla、VLM backbone/是否冻结/截层 N、视觉 token 数、chunk_size、n_action_steps、
flow matching 推理步数、batch、lr/scheduler/optimizer、precision、参数量、wall time、GPU 型号）。
论文参考值（仅核对用，非最终证据）：SmolVLM-2 冻结 + 前 16 层、chunk=50、FM 10 步、
预训练 200k/batch256/cosine 1e-4→2.5e-6、微调 100k(仿真)/200k(真实)、bf16+torch.compile。 -->

| 字段 | 记录 |
|---|---|
| status | 🔄 running |
| policy | smolvla（LeRobot） |
| VLM 骨干 | <!-- 待回填 --> |
| VLM 截层 N / 冻结 | <!-- 待回填 --> |
| chunk size / action steps | <!-- 待回填（论文 50） --> |
| flow matching 采样步数 | <!-- 待回填（论文 10） --> |
| optimizer / lr / scheduler | <!-- 待回填 --> |
| steps / batch | <!-- 待回填 --> |
| precision / compile | <!-- 待回填 --> |
| 评测模式 | <!-- sync / async（g 与相似度阈值） --> |
| wall time / GPU / 参数量 | <!-- 待回填 --> |

---

## 4. 评测结果

### 4.1 主结果（对照论文 Table 2/3/4/5）

<!-- 待回填：成功率表。仿真/真实分栏，评分口径与论文一致（细粒度评分注明）。
论文对照值（SO100 多任务+社区预训练 0.45B）：Pick-Place 75 / Stacking 90 / Sorting 70 / Avg 78.3；
SO101 Lego：ID 90 / OOD 50。LIBERO Avg 87.3、MetaWorld Avg 57.3。 -->

| 任务 | 论文值（0.45B，对照） | 本工程复现值 | 差异说明 |
|---|---|---|---|
| <!-- 待回填：任务名 --> | <!-- 待回填 --> | <!-- 待回填 --> | <!-- 待回填 --> |

### 4.2 预训练 vs 无预训练 / 多任务 vs 单任务（对照论文 Table 5）

<!-- 待回填：若复现含消融，按 单任务无pt / 多任务无pt / 多任务有pt 三行对照。论文：40 / 51.7 / 78.3。 -->

### 4.3 Sync vs Async（对照论文 Figure 5）

| 指标 | Sync（论文 13.75s / 9 次） | Async（论文 9.70s / 19 次） |
|---|---|---|
| 成功率 | <!-- 待回填 --> | <!-- 待回填 --> |
| 任务完成时间 | <!-- 待回填 --> | <!-- 待回填 --> |
| 固定时间完成数 | <!-- 待回填 --> | <!-- 待回填 --> |

### 4.4 与 ACT 的对比（本工程 01-act 的延续，学习主线重点）

<!-- 待回填：同一数据集/实机上 SmolVLA vs ACT（本工程已复现 ACT，aloha_mobile_cabinet）的对照；
若任务不同则注明口径不可比，不强行并表。 -->

---

## 5. 已知限制与冻结口径

<!-- 待回填：实验完成后，参照 ACT 复现文档 §5 记录：评测口径差异、未能闭环的证据、元数据缺项（LeRobot commit/dataset revision 等）、冻结决策。 -->

---

## 6. 复现边界声明

<!-- 待回填：本次复现不做什么（如：不重跑多 seed、不做全 481 数据集预训练、不做 LIBERO/Meta-World 全量仿真、不调 async 的 g/ε 等），避免无限扩张。 -->

---

## 7. 闭环清单

以下各项构成此次 SmolVLA 复现闭环，随实验推进逐项勾选：

- [ ] 训练 run 完成（resolved config + 日志 + checkpoint 归档）
- [ ] 评测结果回填（§4 各表，含仿真/实机分栏）
- [ ] 与论文数值差异分析 + 与 ACT 基线对照
- [ ] 局限/证据边界/复现边界声明定稿
- [ ] 产物清单登记（§8）
- [ ] 笔记 🔁 引用点与本文档交叉核对

---

## 8. 实验产物

```text
code/reproduction_smolvla.md   ← 本文档
code/outputs/<run>/            ← 待回填：训练日志/曲线/评测结果/视频 清单
```
