# π*₀.₆ RECAP 价值模型调研与本项目落地

## 结论

π*₀.₆ 的 RECAP 价值模型不是二分类器，而是一个语言条件、分布式的状态价值模型。论文把 Monte-Carlo return 离散为 201 个 bin；非终止步奖励为 −1，成功终止奖励为 0，失败终止为负惩罚，并按任务时长把价值归一化到 `[-1, 0]`。因此，越接近成功，期望价值越接近 0；失败轨迹保持低价值。

本项目保留了这条主线，并做了一个明确扩展：用 Qwen3‑VL‑2B 读取三帧历史与共享的粗粒度任务语言，同时使用仅在训练期可见未来帧的 LAPO teacher 提供 32D transition latent 辅助监督。部署时只保留 Qwen3‑VL LoRA 和 201-bin Value Head，因此不存在未来信息泄漏。

## 从 RECAP 复现的设计

- 输入必须有任务语言条件；相同画面在不同任务下的价值可以不同。
- 输出是 201-bin categorical distribution，标量价值由 bin 概率的期望得到。
- 成功轨迹按剩余步数形成从 −1 向 0 上升的监督；失败轨迹给低价值。
- 划分必须在 episode 级完成，避免同一轨迹的相邻帧跨 train/test。
- 验收不能只看 MAE，还要看轨迹内 Spearman、时间单调性、成功/失败分离和 transition advantage 的符号质量。

## 相比论文的工程差异

| 项目 | π*₀.₆ RECAP | 本实现 |
|---|---|---|
| 视觉语言骨干 | 约 670M Gemma 3 价值网络 | Qwen3‑VL‑2B + LoRA |
| 价值输出 | 201 bins | 201 bins，`[-1,0]` |
| 视觉历史 | 论文价值观测路径 | `t−10,t−5,t` 三帧，15 Hz |
| 语言 | 任务指令 | success/failure 共享的 17 类 coarse family |
| 动力学辅助 | 无此 LAPO 扩展 | 训练期 32D LAPO latent，Huber 对齐 |
| 部署依赖未来帧 | 否 | 否；teacher 和 latent head 均不导出 |

LAPO 辅助监督必须通过 ablation 证明有效：baseline（无 latent）、proposed（真实 latent）、random latent 和 shuffled latent 使用相同数据预算。只有 proposed 优于 baseline，且 random/shuffled 不能复现提升时，才能把收益归因于 transition-aware supervision。

## 数据审计摘要

- 原始 success：53,282 episodes / 14,153,535 frames。
- 原始 failure：13,747 episodes / 3,476,037 frames。
- 去除跨 outcome 重复 UUID、failure sidecar 中非明确失败、过短轨迹、缺失相机和不共享 station 后：52,475 success + 13,422 failure。
- 最终 526,718 anchors：train 419,344 / val 53,772 / test 53,602。
- task-family `other`：success 约 1.8%，failure 约 2.8%，低于 15% gate。
- 唯一部署相机：`observation.images.left_external`。

## 参考资料

1. Physical Intelligence, *π*₀.₆: a VLA That Learns From Experience*, arXiv:2511.14759, <https://arxiv.org/abs/2511.14759>。
2. 论文 HTML 版本（便于检索公式与附录），<https://ar5iv.labs.arxiv.org/html/2511.14759>。
3. DROID: A Large-Scale In-the-Wild Robot Manipulation Dataset, <https://droid-dataset.github.io/>。
4. Qwen3‑VL‑2B‑Instruct 本地快照：`models/Qwen3-VL-2B-Instruct`。
