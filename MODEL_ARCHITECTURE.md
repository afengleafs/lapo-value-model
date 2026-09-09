# LAPO + Qwen3-VL 价值模型架构与输入输出

## 1. 总体设计

当前模型采用“两阶段训练、单模型部署”的设计：

1. **LAPO Teacher** 在训练期同时观察当前帧与未来帧，学习描述状态变化的 32D latent action。
2. **Qwen3-VL Student** 只观察历史图像与任务语言，在学习价值函数的同时对齐 Teacher latent。
3. 部署时删除 LAPO Teacher 和未来帧路径。旧 `direct` 模型可删除 Latent Head；新
   `latent_hidden` 模型保留因果的 Latent Trunk，因为 Value Head 读取其 128D 隐层。

```text
训练期 LAPO Teacher
───────────────────
当前帧 o_t + 未来帧 o_{t+5}
          │
          ▼
    ResNet-18 IDM
          │
     μ, logσ² ∈ R³²
          │
          ▼
   latent action z_t ∈ R³²
          │
          ▼
 FiLM-conditioned U-Net FDM
          │
          ▼
重建未来帧 ô_{t+5}


Qwen3-VL latent_hidden Value Student
──────────────────────
[o_{t-10}, o_{t-5}, o_t] + 任务语言
                  │
                  ▼
            Qwen3-VL-2B
                  │
         shared hidden h ∈ R²⁰⁴⁸
                  │
                  ▼
     Latent Trunk 2048→512→128
              ┌───┴────────┐
              ▼            ▼
       128→32 projection   Value Head
        latent supervision 128→512→256→201
                           │
                           ▼
                     标量价值 V̂
```

核心约束是：**未来帧只允许进入训练期 Teacher，不能进入 Student 或最终部署模型。**

## 2. 数据与模型输入

### 2.1 Student/部署模型输入

Student 使用以下输入：

- 相机字段：`observation.images.left_external`
- 三帧 RGB 历史图像：`o_{t-10}`、`o_{t-5}`、`o_t`
- 数据帧率：15 Hz
- 三帧对应时间约为：当前前 0.67 秒、当前前 0.33 秒、当前时刻
- 图像预处理：保持宽高比 letterbox 到 `336×336`
- 语言输入：归一化后的粗粒度任务族提示

任务族示例包括：

- `container_transfer`
- `reposition`
- `lid`
- `slidable_open_close`
- `clean`
- `fold_spread`

同一任务族的成功和失败轨迹使用相同提示，语言中不包含当前样本的 outcome 标签。

### 2.2 Teacher 输入

Teacher 使用：

- 当前图像 `o_t`
- 未来图像 `o_{t+5}`
- Teacher horizon：5 帧，即约 0.33 秒
- Teacher 图像尺寸：`224×224`

未来图像仅用于学习 transition latent 和检验未来帧重建效果。

## 3. LAPO Teacher

Teacher 由 Inverse Dynamics Model 和 Forward Dynamics Model 组成。

### 3.1 Inverse Dynamics Model（IDM）

- Backbone：ResNet-18
- 分别编码 `o_t` 和 `o_{t+5}`，各得到一个 512D 特征
- 拼接当前特征、未来特征以及两者差值
- MLP 输出 32D 高斯分布参数：
  - `mu`：`[B, 32]`
  - `logvar`：`[B, 32]`
- 通过 VAE 重参数化得到：
  - `latent`：`[B, 32]`

离线生成 Student 监督数据时使用确定性的 `mu`，避免采样噪声导致 target 漂移。

### 3.2 Forward Dynamics Model（FDM）

- 架构：FiLM-conditioned U-Net
- 输入：当前图像 `o_t` 和 32D latent action
- latent 通过 FiLM 调制 U-Net 的多尺度解码特征
- 输出未来帧重建：
  - `reconstruction`：`[B, 3, 224, 224]`

Teacher 完整输出为：

| 输出 | 形状 | 含义 |
|---|---:|---|
| `reconstruction` | `[B, 3, 224, 224]` | 预测的未来图像 `ô_{t+5}` |
| `mu` | `[B, 32]` | 确定性 transition latent |
| `logvar` | `[B, 32]` | latent 对数方差 |
| `latent` | `[B, 32]` | 重参数化采样得到的 latent |

Teacher 损失为：

```text
L_teacher = MSE(ô_{t+5}, o_{t+5}) + β · KL
β = 1e-4
```

正式生成 latent 前要求 validation reconstruction gain 大于 0，即模型重建未来帧必须优于直接复制当前帧。

## 4. Qwen3-VL Value Student

### 4.1 Backbone

- 基础模型：Qwen3-VL-2B-Instruct
- 总参数量约 2.14B
- 隐藏维度：2048
- 精度：bf16
- 基础权重冻结，使用 LoRA 微调
- LoRA rank：16
- LoRA 注入位置：
  - 语言 attention：`q_proj`、`k_proj`、`v_proj`、`o_proj`
  - 视觉 attention：`qkv`、`proj`
- LoRA 与两个 Head 合计可训练参数约 11.1M
- 两卡训练 micro-batch：160/卡，梯度累积 1，全局 batch 320
- validation micro-batch：80/卡（只控制验证显存，不改变训练预算）
- Adapter 学习率：`5e-5`；Value/Latent Head 学习率：`5e-4`

在当前 ROCm 环境中，语言 attention 使用高效实现；视觉 fused attention 的 bf16 backward 会产生非有限 LoRA 梯度，因此视觉塔固定使用数值稳定的 eager attention。

三帧图像和语言经过 Qwen3-VL 后，模型取最后一个有效 token 的最终隐藏状态：

```text
h ∈ R²⁰⁴⁸
```

实现支持两条显式路径：历史 `direct` Baseline 让该隐藏状态直接进入 Value Head；
新 `latent_hidden` Proposed 先经过 Latent Head 的 128D 隐层，再进入 Value Head。

### 4.2 Value Head

旧 `direct` Baseline：

```text
LayerNorm(2048)
→ Linear(2048, 512)
→ GELU + Dropout
→ Linear(512, 256)
→ GELU + Dropout
→ Linear(256, 201)
```

新 `latent_hidden` Proposed：

```text
Latent Head 的 128D penultimate feature
→ LayerNorm(128)
→ Linear(128, 512)
→ GELU + Dropout
→ Linear(512, 256)
→ GELU + Dropout
→ Linear(256, 201)
```

Value Head 输出：

- `value_logits`：`[B, 201]`
- 201 个 bin 均匀覆盖 `[-1, 0]`
- `softmax(value_logits)` 得到价值概率分布
- 对 bin center 求概率期望得到标量价值 `value`：`[B]`

```text
p(V | history, language) = softmax(value_logits)
V̂ = Σ_i p_i · bin_i
```

价值含义：

- 越接近 `0`：越接近任务成功完成
- 越接近 `-1`：离成功更远，或更符合失败状态

### 4.3 Latent Head

```text
LayerNorm(2048)
→ Linear(2048, 512)
→ GELU + Dropout
→ Linear(512, 128)
→ GELU + Dropout
→ Linear(128, 32)
```

输出：

- `latent`：`[B, 32]`

32D 输出对齐冻结的 LAPO Teacher `mu`。新 Proposed 的价值路径读取最终
`128→32` 投影之前的 128D 隐层：价值 CE 更新 Qwen LoRA 和 Latent Trunk，
但不更新 32D projection；Huber latent loss 更新完整 Latent Head。

### 4.4 Student 损失

```text
L_student = L_value + λ_z · L_latent

L_value  = CrossEntropy(value_logits, target_bin)
L_latent = Huber(ẑ_t, stopgrad(z_t))
λ_z      = 0.1
```

## 5. 价值监督目标

### 5.1 成功轨迹

成功轨迹的价值由剩余步数构造：

```text
V_t = clip(-(T - 1 - t) / T_max, -1, 0)
```

轨迹越接近成功终点，价值越接近 0。

### 5.2 失败轨迹

当前 V1 中失败轨迹使用低价值目标：

```text
V_t = -1
```

### 5.3 201-bin 离散化

连续价值被映射到 201 个 bin：

```text
target_bin = round((V_t + 1) × 200)
```

因此 `-1` 对应 bin 0，`0` 对应 bin 200。

## 6. 训练输出与部署输出

OpenPI OOF 微调按 checkpoint 中的 `value_path` 固定模型结构。30 Hz OpenPI 的 Student 三帧输入为
`[o_{t-20}, o_{t-10}, o_t]`，与 DROID 15 Hz 的 `[-10,-5,0]` 覆盖相同物理时间；
Teacher 使用 `o_t, o_{t+10}`。未来帧只离线生成 32D latent target，held-out forward
仍严格只接收三帧历史和任务文本。每折输出的 201-bin value logits、期望价值和训练期
latent 与 DROID 阶段完全同构。

### 6.1 Student 训练期输出

| 输出 | 形状 | 用途 |
|---|---:|---|
| `value_logits` | `[B, 201]` | Distributional value 的分类监督 |
| `value` | `[B]` | 201-bin 分布的期望价值 |
| `latent` | `[B, 32]` | 对齐 LAPO Teacher；价值预测不直接读取该最终投影 |

### 6.2 最终部署结构

两种部署包都删除以下组件：

- LAPO Teacher
- Inverse Dynamics Model
- Forward Dynamics Model
- 未来帧输入路径

`direct` 模型还会删除 Student Latent Head；`latent_hidden` 模型必须保留其
2048→512→128 trunk。两者都只使用历史图像和任务语言，均无未来信息泄漏。

最终只保留：

```text
3 张历史图像 + 任务族语言
            │
            ▼
Qwen3-VL-2B + LoRA +（可选 Latent Trunk）+ Value Head
            │
            ├── value_logits: [B, 201]
            ├── value_distribution: softmax(value_logits)
            └── value: [B]，范围 [-1, 0]
```

其中 `value_distribution` 可由 `value_logits` 直接计算，`value` 可用于：

- 对状态或轨迹进行价值排序
- 识别任务进展
- 区分成功/失败趋势
- 计算相邻状态的 value difference 或 RECAP-style advantage

最终模型的输入和输出均不依赖未来观测，满足因果部署要求。
