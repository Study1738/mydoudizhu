# Action-Level Fusion Strategy

动作级融合方案实现，在每次决策时动态融合两个 DouZero 模型的输出。

## 核心思路

- 每次出牌决策时，两套 DouZero 模型各自计算动作价值
- 融合 MLP 网络读取两份输出及游戏状态，决定如何组合或选择
- 通过强化学习训练融合网络，学习在不同局面更信任哪个模型
- 支持多种融合策略：门控、注意力、混合专家等

## 模块说明

### 1. `dual_model.py` - 双模型推理
- `DualModelInference`: 管理两个模型的并行推理
  - 加载两套 DouZero 模型
  - 并行前向计算动作价值
  - 返回两份模型输出

- `DualModelAgent`: 使用融合策略的智能体
  - 集成双模型推理和融合网络
  - 执行动作选择

- `EnsemblePredictor`: 简单集成方法（baseline）
  - 平均集成 (average)
  - 最大值集成 (max)
  - 投票集成 (vote)

### 2. `fusion_network.py` - 融合网络架构
提供多种融合网络实现：

#### `ActionFusionNetwork`
- 直接输出动作概率分布
- 输入：[values_a, values_b, state_features]
- 输出：动作概率 (softmax)

#### `GatingFusionNetwork`
- 门控网络，输出融合权重 β
- 支持全局门控或逐动作门控
- 输出：`β * values_a + (1-β) * values_b`

#### `AttentionFusionNetwork`
- 基于注意力机制的融合
- 使用查询-键-值架构
- 动态分配两个模型的权重

#### `MixtureOfExpertsNetwork`
- 混合专家 (MoE) 风格
- 将两个模型视为专家
- 学习专家选择策略

#### `AdaptiveFusionNetwork`
- 自适应融合，可在多种策略间切换
- 元学习选择最优融合方式

### 3. `feature_extractor.py` - 特征提取
- `ActionLevelFeatureExtractor`: 提取丰富的特征
  - 手牌分布特征
  - 游戏阶段特征（细粒度划分）
  - 动作历史特征
  - 对手建模特征
  - 战略特征（控制权、炸弹、获胜紧迫度）
  - **模型分歧特征**（两模型的不一致性）

- `BatchActionFeatureExtractor`: 批量特征提取

### 4. `trainer.py` - PPO 训练器
- `ActionFusionTrainer`: 动作级融合训练器
  - 管理双模型推理（冻结）
  - 训练融合网络（可训练）
  - PPO 策略梯度更新
  - 支持多种融合网络类型

### 5. `evaluator.py` - 评估器
- `ActionFusionEvaluator`: 主评估器
  - 对战评估
  - 轨迹收集
  - 统计分析

- `SimpleActionEvaluator`: 简化评估
  - 比较不同融合方法
  - 分析模型分歧

### 6. `train_fusion.py` - 训练脚本
主训练入口。

## 使用方法

### 训练融合网络

```bash
cd DouZero/fusion_action_level

# 训练门控融合网络
python train_fusion.py \
    --model_a ../baselines/douzero_ADP/landlord.ckpt \
    --model_b ../baselines/douzero_WP/landlord.ckpt \
    --position landlord \
    --fusion_type gating \
    --eval_data ../eval_data.pkl \
    --num_iterations 1000 \
    --device cuda

# 训练注意力融合网络
python train_fusion.py \
    --model_a ../baselines/douzero_ADP/landlord.ckpt \
    --model_b ../baselines/douzero_WP/landlord.ckpt \
    --position landlord \
    --fusion_type attention \
    --device cuda
```

### 参数说明

**必需参数:**
- `--model_a`: 第一个模型路径
- `--model_b`: 第二个模型路径
- `--position`: 训练位置 [landlord, landlord_up, landlord_down]

**融合网络类型:**
- `--fusion_type`: 融合类型 [gating, action, attention, moe]

**训练参数:**
- `--num_iterations`: number of training iterations
- `--games_per_iteration`: games collected per iteration
- `--eval_interval`: evaluation interval
- `--eval_games`: number of games for evaluation
- `--objective`: reward objective (`adp` / `wp` / `logadp`)
- `--gate_lr_scale`: gating network learning-rate multiplier (gating, default 0.05)
- `--gate_warmup_iters`: gating warm-up iterations (gating, default 3)
- `--gate_reg_coeff`: gating regularisation coefficient (gating, default 1e-3)

- **对手配置:**
  - `--opponent`: 默认对手类型（perfectdou / douzero / random / rlcard）
  - `--opponent_landlord` / `--opponent_landlord_up` / `--opponent_landlord_down`: 针对每个位置单独指定对手，可填写关键字或直接给出 `.ckpt/.onnx` 权重路径（未指定则使用 `--opponent` 的默认值）

- 使用 perfectdou 对手时，可将 `landlord.pt/landlord.onnx` 等权重放在 `perfectdou/model/`（或设置 `PERFECTDOU_MODEL_DIR`），脚本会自动加载；如果提供 ONNX 权重则需安装 onnxruntime。

**网络参数:**
- `--num_actions`: 动作数量（默认 309）
- `--feature_dim`: 特征维度（默认 512）
- `--lr`: 学习率
- `--gamma`: 折扣因子
- `--value_loss_coef`: 价值损失系数
- `--entropy_coef`: 熵正则系数

### 测试简单集成方法

```python
from dual_model import EnsemblePredictor

# 平均集成
predictor = EnsemblePredictor(
    model_path_a='path/to/model_a.ckpt',
    model_path_b='path/to/model_b.ckpt',
    position='landlord',
    ensemble_method='average'
)

action = predictor.predict(infoset)
```

### 分析模型分歧

```python
from evaluator import SimpleActionEvaluator

evaluator = SimpleActionEvaluator()

# 分析两模型的不一致性
disagreement_stats = evaluator.analyze_model_disagreement(
    model_path_a='path/to/model_a.ckpt',
    model_path_b='path/to/model_b.ckpt',
    position='landlord',
    eval_data_path='eval_data.pkl'
)

print(f"Agreement rate: {disagreement_stats['agreement_rate']:.2%}")
print(f"Avg value difference: {disagreement_stats['avg_value_diff']:.4f}")
```

### 使用训练好的融合网络

```python
from trainer import ActionFusionTrainer

# 加载训练好的融合网络
trainer = ActionFusionTrainer(
    model_path_a='path/to/model_a.ckpt',
    model_path_b='path/to/model_b.ckpt',
    position='landlord',
    fusion_type='gating'
)

trainer.load_checkpoint('checkpoints/best_fusion.pt')

# 使用融合网络选择动作
action_info = trainer.select_action(infoset, deterministic=True)
action = action_info['action']
```

## 融合网络类型对比

| 类型 | 输出方式 | 特点 | 适用场景 |
|------|----------|------|----------|
| **Gating** | 权重系数 β | 简单高效，全局或逐动作门控 | 快速验证，基础融合 |
| **Action** | 动作概率分布 | 直接学习最优分布 | 需要探索性策略 |
| **Attention** | 注意力加权 | 动态关注不同模型 | 模型特征差异大 |
| **MoE** | 专家选择 | 专家混合，可扩展 | 多模型融合 (>2) |
| **Adaptive** | 自适应策略 | 元学习最优融合方式 | 复杂场景 |

## 训练流程

1. **双模型推理** (冻结)
   - 加载两套 DouZero 模型
   - 对每个状态并行前向计算
   - 获得两份动作价值

2. **特征提取**
   - 提取游戏状态特征
   - 计算模型分歧特征
   - 拼接为融合网络输入

3. **融合网络前向**
   - 输入两份模型输出 + 状态特征
   - 输出融合后的动作策略
   - 选择动作执行

4. **PPO 训练**
   - 收集对战轨迹
   - 计算 GAE 优势
   - 更新融合网络参数

5. **定期评估**
   - 对战 PerfectDou 测试性能
   - 分析融合行为
   - 保存最优模型

## 输出文件

训练产生的文件：
- `checkpoints/action_fusion/checkpoint_iter_*.pt`: 定期 checkpoint
- `checkpoints/action_fusion/best_fusion.pt`: 最优模型
- `checkpoints/action_fusion/final_fusion.pt`: 最终模型

Checkpoint 内容：
- 融合网络参数
- 价值网络参数
- 优化器状态
- 训练统计

## 优势与限制

**优势:**
- 细粒度动作级控制
- 可学习复杂融合策略
- 易扩展到多模型 (>2)
- 提供模型分歧分析

**限制:**
- 推理成本高（双模型 + 融合网络）
- 训练样本需求大
- 需要更多超参数调优

## 性能优化建议

1. **推理加速**
   - 使用 ONNX 或 TorchScript 优化推理
   - 批量处理多个决策
   - GPU 并行计算

2. **训练效率**
   - 使用经验回放缓冲区
   - 优先经验回放 (PER)
   - 并行环境采样

3. **网络优化**
   - 使用轻量级融合网络
   - 剪枝不重要的连接
   - 知识蒸馏到单一模型

## 扩展方向

1. **多模型融合**: 扩展到 >2 个模型的融合
2. **动态模型选择**: 根据置信度动态选择模型
3. **层级融合**: 结合权重级和动作级融合
4. **在线适应**: 对战中实时调整融合策略
5. **迁移学习**: 将学到的融合策略迁移到其他游戏

## 依赖

- PyTorch >= 1.8.0
- NumPy
- DouZero 环境
- PerfectDou（用于评估）

## 与权重级融合的对比

| 维度 | 权重级融合 | 动作级融合 |
|------|------------|------------|
| 融合粒度 | 模型权重 | 动作输出 |
| 推理成本 | 低（单模型） | 高（双模型） |
| 表达能力 | 权重空间插值 | 动作空间任意组合 |
| 训练难度 | 较简单 | 较复杂 |
| 适用场景 | 模型相似，需要效率 | 模型差异大，追求性能 |
