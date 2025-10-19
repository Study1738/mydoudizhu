# Readme

这是一个为斗地主（Doudizhu）AI设计的**动态动作级模型融合**项目。

本项目的核心思想不是从零开始训练一个全新的模型，而是借鉴 [DouZero (Kwai)](https://github.com/kwai/DouZero) 和 [PerfectDou (Netease)](https://github.com/Netease-Games-AI-Lab-Guangzhou/PerfectDou) 的思路，将两个或多个预训练的强大模型（例如 DouZero 的 ADP 和 WP 模型）进行**动态融合**。

融合在“动作层面”(Action-Level）进行：在游戏的每一步决策时，一个可训练的**门控融合网络 (Gating Fusion Network)** 会实时分析当前的游戏状态（包括手牌、出牌历史、甚至两个基础模型的决策差异），然后动态地决定两个基础模型（如 ADP 和 WP）的输出权重，最后加权计算出最终的动作价值。

这种方法使得AI可以根据局势动态切换策略：在优势局，它可能更相信 WP（胜率）模型的决策以求稳；在劣势局或关键时刻，它可能更相信 ADP（得分）模型的决策来博取高分。

主要代码在 `DouZero/fusion_action_level` 下。

## 目录

[TOC]

## 网络架构

本项目的核心是训练一个**门控融合网络 (Gating Network)**，而两个基础模型（Base Model A, Base Model B）的权重是**冻结不变**的。

### 整体数据流图 (Text-based Diagram)

为了帮助您理解，这里是整个决策和训练的数据流：

```mermaid
graph TD
    subgraph "游戏状态输入 (Infoset)"
        Infoset["Infoset (游戏状态)"];
    end

    subgraph "基础动作价值预测 (Base Action-Value Prediction)"
        Infoset --> BaseModelA["Base Model A (Frozen)"];
        Infoset --> BaseModelB["Base Model B (Frozen)"];
        BaseModelA --> Values_A["Values_A (Q值/动作价值)"];
        BaseModelB --> Values_B["Values_B (Q值/动作价值)"];
    end
    
    subgraph "特征提取 (Action-Level Feature Extraction)"
        Infoset --> FeatureExtractor["ActionLevelFeatureExtractor"];
        
        Values_A --> FeatureExtractor;
        Values_B --> FeatureExtractor;
        

        FeatureExtractor --> State_Features["State_Features (游戏状态特征)"];
        FeatureExtractor --> Disagreement_Features["Disagreement_Features (模型分歧特征)"];

        State_Features --> Combined_Features["Combined_Features (最终特征输入)"];
        Disagreement_Features --> Combined_Features;
    end

    subgraph "融合网络输入准备 (Fusion Network Input Prep)"
        Values_A --> Fusion_Input_Prep["Fusion Network Input Prep"];
        Values_B --> Fusion_Input_Prep;
        Combined_Features --> Fusion_Input_Prep;
        
        Fusion_Input_Prep --> Fused_Actor_Input["Actor Input (Values_A, Values_B, Features)"];
        Fusion_Input_Prep --> Critic_Input["Critic Input (Features)"];
    end

    subgraph "训练部分 (Trainable Components)"
        %% Actor
        Fused_Actor_Input --> GatingFusionNetwork["GatingFusionNetwork (Actor, 策略网络)"];
        
        %% Critic
        Critic_Input --> ValueNetwork["ValueNetwork (Critic, 价值网络)"];
        ValueNetwork --> Predicted_Reward["Predicted_Reward (V(s) 状态价值)"];

        %% 训练目标
        Trajectory_Data["轨迹数据 (R_t)"] --> Training_Target["Training_Target (Target Return 目标回报)"];

        %% 1. 计算优势函数 (Advantage)
        Predicted_Reward --> Advantage_Calc("Advantage A = Target - V(s)");
        Training_Target --> Advantage_Calc;

        %% 2. Actor 更新 (Policy Gradient Update)
        Advantage_Calc -.-> PPO_Update_Actor["PPO 策略梯度更新"];
        PPO_Update_Actor --> GatingFusionNetwork;
		%% 更新融合网络参数

        %% 3. Critic 更新 (Value Loss Update)
        Predicted_Reward --> Value_Loss_Calc("Value Loss = (Target - V(s))^2");
        Training_Target --> Value_Loss_Calc;

        Value_Loss_Calc --> PPO_Update_Critic["PPO 价值损失更新"];
        PPO_Update_Critic --> ValueNetwork;
		%% 更新价值网络参数
    end

    subgraph "输出与融合 (Output and Fusion)"
        GatingFusionNetwork --> Gate_Weight["Gate_Weight (β)"];

        %% 融合公式: Fused_Values = Beta*A + (1-Beta)*B
        Gate_Weight ==> Fused_Values["Fused_Values (β*A + (1-β)*B)"];
        Values_A ==> Fused_Values;
        Values_B ==> Fused_Values;

        Fused_Values --> Final_Action["最终动作"];
    end
    
    style Predicted_Reward fill:#E0FFFF,stroke:#000
    style Training_Target fill:#FFC0CB,stroke:#000
    style Advantage_Calc fill:#ADD8E6,stroke:#000
    style Value_Loss_Calc fill:#FFFFE0,stroke:#000
```

我们真正训练的神经网络架构分为两个独立的部分：

1.  **`fusion_net` (门控融合网络)**：扮演 **Actor (演员)** 角色，负责**决策**，即动态决定如何融合基础模型的输出。
2.  **`value_net` (价值网络)**：扮演 **Critic (评论家)** 角色，负责**评估**，即判断当前状态最终能带来多少回报。

这两个网络都依赖于一个关键的**非神经网络**组件：

3.   **`feature_extractor` (特征提取器)**：负责为 `fusion_net` 和 `value_net` 准备高维度的输入向量。

### 特征提取器 (ActionLevelFeatureExtractor)

这是架构的输入预处理单元，它不是一个神经网络。

-   **文件**: `feature_extractor.py`
-   **作用**: 它的核心方法 `extract_features` 从游戏状态 `infoset` 和两个基础模型的输出 `model_outputs` 中提取一个高维特征向量。
-   **关键特征**:
    -   手牌特征（单牌、对子、炸弹数量等）。
    -   游戏阶段特征（开局、中局、残局）。
    -   对手信息（剩余手牌数）。
    -   **模型差异特征**：分析 `Values_A` 和 `Values_B` 之间的统计差异，例如Q值均值差、最大值差、Top-K动作的重合度等。
-   **输出**: `State_Features`，一个被填充或截断到 `feature_dim` (默认为 **512**) 维的 `torch.tensor`。这个向量是后续两个神经网络的主要输入。

### 门控融合网络 (GatingFusionNetwork) - (Actor)

这是**第一块可训练**的网络，是**决策者 (Actor)**。

-   **文件**: `fusion_network.py`
-   **作用**: 接收状态特征和模型输出摘要，然后输出一个**标量门控权重 $\beta$**，用于决定 `Values_A` 和 `Values_B` 各占多少比重。
-   **输入**: 它的输入由两部分在 `forward` 方法中动态拼接而成：
    1.  **`state_features`**: 来自特征提取器的 **512** 维向量。
    2.  **`summary` (摘要统计)**: 一个实时的 **6** 维向量，包含 `Values_A` 和 `Values_B` 的均值、最大值和差异等统计。
-   **总输入维度**: 512 + 6 = **518** 维。
-   **网络层级** (基于 `hidden_dim=256`)：

| **层 (Layer)** | **类型 (Type)** | **输入神经元 (Input)** | **输出神经元 (Output)** | **激活函数 (Activation)** | **作用 (Function)**                                        |
| -------------- | --------------- | ---------------------- | ----------------------- | ------------------------- | ---------------------------------------------------------- |
| `fc1`          | `nn.Linear`     | 518                    | 256                     | `ReLU`                    | 第一次特征变换                                             |
| `fc2`          | `nn.Linear`     | 256                    | 128                     | `ReLU`                    | 第二次特征变换                                             |
| `gate`         | `nn.Linear`     | 128                    | 1                       | `0.1 + 0.8 * Sigmoid(x)`  | 输出最终的门控权重 $\beta$，确保其范围在 `[0.1, 0.9]` 之间 |

-   **输出**: `Gate_Weight (Beta)`，一个标量。
-   **最终计算**: 融合后的Q值 `Fused_Values` 通过 `Beta * Values_A + (1 - Beta) * Values_B` 计算得出。

### 价值网络 (ValueNetwork) - (Critic)

这是**第二块可训练**的网络，是**评估者 (Critic)**。

-   **文件**: `trainer.py` (在 `ActionFusionTrainer` 的 `__init__` 中定义)
-   **作用**: 接收状态特征，并预测当前状态的**预期最终回报**（例如，预测这局游戏最终得分是 +2 还是 -4）。这个预测值在训练中用于计算 `value_loss` (价值损失)。
-   **输入**: **仅** `state_features` 向量。
-   **总输入维度**: **512** 维。
-   **网络层级** (基于 `feature_dim=512`)：

| **层 (Layer)** | **类型 (Type)** | **输入神经元 (Input)** | **输出神经元 (Output)** | **激活函数 (Activation)** | **作用 (Function)**    |
| -------------- | --------------- | ---------------------- | ----------------------- | ------------------------- | ---------------------- |
| Layer 1        | `nn.Linear`     | 512                    | 256                     | `ReLU`                    | 第一次状态评估         |
| Layer 2        | `nn.Linear`     | 256                    | 128                     | `ReLU`                    | 第二次状态评估         |
| Layer 3        | `nn.Linear`     | 128                    | 1                       | `None`                    | 输出最终的标量价值预测 |

-   **输出**: `Predicted_Reward` (标量)，代表对最终回报的预测。

## 训练方法

本项目的训练方法采用了 **Actor-Critic (演员-评论家)** 框架，其更新方式类似于 [DouZero](https://github.com/kwai/DouZero) 中使用的简化版蒙特卡洛策略回归（在 `trainer.py` 的文档字符串中提到，这是一种简化的、不含 PPO 复杂度的 MSE 优化方法）。

训练的核心目标**不是**基础模型（例如 Model A：ADP，Model B：WP），它们的权重在整个过程中保持**冻结**。我们的目标是训练两个**新**的网络：

1.  **`fusion_net` (Actor - 演员)**：即 `GatingFusionNetwork`。它的职责是根据当前状态，动态生成一个融合权重 $\beta$，用于决策。
2.  **`value_net` (Critic - 评论家)**：一个标准的多层感知机 (MLP)。它的职责是评估当前状态的预期最终回报（即预测这局游戏的最终得分）。

整个训练流程在 `train_fusion.py` 脚本的 `train` 函数中定义，并依赖 `ActionFusionEvaluator` 和 `ActionFusionTrainer` 两个核心类。

### 训练步骤详解

训练是一个迭代循环，每次迭代 (`iteration`) 包含三个主要阶段：轨迹收集、网络更新和评估。

#### 阶段一：轨迹收集 (Trajectory Collection)

此阶段的目标是使用**当前**的融合网络（Actor）玩一定数量的游戏，并记录下决策过程中的关键数据。

1.  **执行者**：`ActionFusionEvaluator` 类的 `collect_trajectories` 方法。
2.  **设置**：
    -   训练器 `trainer` 中的 `fusion_net` (融合网络) 被设置为评估模式 (`.eval()`)。
    -   根据配置（`--opponent` 和 `--opponent_map`）加载对手智能体。
3.  **游戏循环**：
    -   循环 `num_games`（例如 `games_per_iteration=1000`）次。
    -   为了在游戏中记录数据，`evaluator` 会实例化一个临时的内部智能体 `TrajectoryRecordingAgent`。这个智能体被放置在我们要训练的 `position` 上（例如 `landlord`）。
4.  **`TrajectoryRecordingAgent` 的决策过程**：
    -   在轮到它出牌时（且非唯一合法动作时）：
    -   **a. 基础模型推理**：调用 `trainer.dual_inference.get_dual_action_values(infoset)` 获取两个冻结模型的 `values_a` 和 `values_b`。
    -   **b. 状态特征提取**：调用 `trainer.feature_extractor.extract_features(...)` 提取高维状态特征 `state_features`。这个特征包含了手牌、游戏阶段以及**模型差异**等信息。
    -   **c. 融合网络决策 (Actor)**：调用 `trainer.fusion_net(values_a, values_b, state_features)` 得到 `fusion_output`（包含融合后的Q值 `fused_values` 和门控权重 `gate_weights`）。
    -   **d. 动作选择**：根据 `fusion_output` 中的Q值或概率，（通常是贪婪地 `argmax`）选择一个动作 `action_idx`。
5.  **数据记录**：
    -   将这一步的关键信息存储在 `trajectory` 缓冲区中，包括：
        -   `obs_z`, `obs_x`：用于基础模型推理的原始输入。
        -   `action_indices`：实际选择的动作索引。
        -   `state_features`：**(关键)** 提取出来并**缓存**的高维状态特征，用于后续训练。
        -   `legal_actions`：合法的动作列表。
6.  **游戏结束**：
    -   游戏结束后，调用 `_compute_reward` 方法，根据训练目标 (`--objective`) 计算一个**最终回报 `R_target`**。例如，如果 `objective='wp'`，`R_target` 是 `+1.0` (赢) 或 `-1.0` (输)。
    -   这个**单一的** `R_target` 值被赋给整个轨迹的 `target` 字段。

#### 阶段二：网络更新 (Network Update)

此阶段使用收集到的轨迹数据，通过反向传播来更新 `fusion_net` (Actor) 和 `value_net` (Critic) 的参数。

1.  **执行者**：`ActionFusionTrainer` 类的 `update` 方法。
2.  **设置**：
    -   `fusion_net` 和 `value_net` 被设置为训练模式 (`.train()`)（由 `trainer` 在 `collect_trajectories` 结束后设置回 `train` 模式）。
3.  **更新循环**：
    -   外循环 `num_epochs` (例如 4) 次，重复使用所有收集到的轨迹数据。
    -   中循环遍历 `trajectories` 列表中的**每一局游戏 (traj)**。
    -   内循环遍历该局游戏 (traj) 中的**每一步 (step_idx)**。
4.  **每一步的更新逻辑**：
    -   **a. 加载数据**：获取该步骤的 `obs_z`, `obs_x`, `action_idx` 和**缓存的 `cached_features`**。
    -   **b. 重计算基础Q值**：在 `torch.no_grad()` 上下文中，**重新**运行两个冻结的基础模型 (`model_a` 和 `model_b`)，得到 `values_a` 和 `values_b`。
    -   **c. Actor 前向传播 (fusion_net)**：
        -   使用**当前**（正在训练的）`fusion_net`，输入 `values_a`, `values_b` 和 `cached_features`，得到 `fusion_output`。
        -   从 `fusion_output['fused_values']` 中，取出**当时轨迹中实际选择的动作** `action_idx` 所对应的 Q 值，记为 `predicted_values`。
    -   **d. Critic 前向传播 (value_net)**：
        -   使用**当前**（正在训练的）`value_net`，输入 `cached_features`，得到状态价值预测 `value_pred`。
    -   **e. 计算损失 (Loss)**：
        -   获取该局游戏的最终回报 `R_target` (即 `traj['target']`)。
        -   **策略损失 (Actor Loss)**：`policy_loss = MSE(predicted_values, R_target)`。
            -   *含义*：迫使 `fusion_net` (Actor) 对其在轨迹中选择的动作 `action_idx`，给出一个与该局游戏**最终结果** `R_target` 相符的 Q 值评估。
        -   **价值损失 (Critic Loss)**：`value_loss = MSE(value_pred, R_target)`。
            -   *含义*：迫使 `value_net` (Critic) 仅根据状态特征 `cached_features` 就能准确预测该局的**最终结果** `R_target`。
        -   **门控正则化损失 (Gating Reg Loss)**：如果启用了 `gate_reg_coeff`，会额外计算一个正则化损失 `gate_reg = coeff * (gate_weights - initial_gate)^2`。这用于防止门控权重 $\beta$ 过早地收敛到 0 或 1，保持模型的探索性。
    -   **f. 反向传播与优化**：
        -   `total_loss = policy_loss + value_loss_coef * value_loss + gate_reg`。
        -   清空两个优化器 (`fusion_optimizer` 和 `value_optimizer`) 的梯度。
        -   执行 `total_loss.backward()`。
        -   对 `fusion_net` 和 `value_net` 的参数进行梯度裁剪 (`clip_grad_norm_`)。
    -   **g. 优化器步进 (Step)**：
        -   **门控预热 (Gate Warmup)**：`trainer` 会检查当前是否处于预热步骤中（`_update_calls < gate_warmup_iters`）。在预热期内，`fusion_optimizer.step()` **不会被调用**。
        -   `self.value_optimizer.step()` (Critic 始终更新)。
        -   `self.fusion_optimizer.step()` (Actor 在预热期结束后更新)。

#### 阶段三：评估与保存 (Evaluation & Checkpointing)

在每次迭代（`iteration`）的 `update` 步骤完成后，立即对更新后的模型进行性能评估。

1.  **执行者**：主循环 (`train_fusion.py`) 调用 `ActionFusionEvaluator` 的 `evaluate` 方法。
2.  **评估**：
    -   `evaluate` 方法会使用**更新后**的 `fusion_net`（通过 `DualModelAgent` 包装）在固定的评估数据集（`--eval_data`）上与对手（`--opponent`）进行 `eval_games` 局游戏。
    -   它不记录轨迹，只计算最终的胜率 (`win_rate`)、平均回报 (`avg_reward`)、WP指标 (`avg_reward_wp`) 和 ADP 指标 (`avg_reward_adp`)。
3.  **保存**：
    -   主循环根据 `--objective` 确定关键指标（例如 `wp` 对应 `win_rate`）。
    -   如果 `current_metric`（当前评估指标）大于 `best_metric`（历史最佳指标）：
    -   `best_metric` 被更新。
    -   调用 `trainer.save_checkpoint`，将当前模型（`fusion_net` 和 `value_net` 的状态）保存为 `best_fusion_{objective}.pt`。
    -   此外，还会按固定的 `save_interval` 保存常规的 `checkpoint_iter_X.pt`。

## 模块介绍

-   `train_fusion.py`: **训练入口脚本**。解析命令行参数，初始化 `ActionFusionTrainer`，并执行训练循环（收集轨迹 -> 更新网络 -> 评估）。
-   `play_with_fusion.py`: **评估和运行脚本**。加载训练好的融合网络 (`--checkpoint`) 和基础模型 (`--model_a`, `--model_b`)，在指定数量的游戏中测试其表现。
-   `trainer.py`: **核心训练器**。`ActionFusionTrainer` 类包含了网络（融合网络、价值网络）、优化器、双模型推理实例以及核心的 `update` 逻辑。
-   `fusion_network.py`: **融合网络架构**。定义了 `GatingFusionNetwork` (MLP + Sigmoid 门控)。
-   `dual_model.py`: **双模型推理**。`DualModelInference` 类负责加载两个冻结的基础模型，并提供 `get_dual_action_values` 接口。`DualModelAgent` 是一个将融合网络包装好的智能体类，用于实际游戏。
-   `feature_extractor.py`: **特征提取器**。`ActionLevelFeatureExtractor` 从游戏状态中提取高维特征向量，包括手牌、历史和模型差异，作为门控网络的输入。
-   `evaluator.py`: **评估器与轨迹收集器**。`ActionFusionEvaluator` 实现了 `evaluate` (评估) 和 `collect_trajectories` (为训练收集数据) 两个关键功能。
-   `opponent_loader.py`: **对手加载器**。用于加载训练和评估时对手座位上的智能体，支持 `douzero`, `perfectdou`, `random` 或自定义模型路径。

## 如何运行

### 环境准备

1.  **PyTorch**: `pip install torch`
2.  **NumPy**: `pip install numpy`
3.  **DouZero 环境**:
    -   你需要能够访问 DouZero 的项目代码（本项目依赖其环境 `douzero.env` 和模型 `douzero.dmc.models`）。
    -   通常，这意味着你需要将 DouZero 的根目录添加到你的 `PYTHONPATH` 中，或者将本`fusion_action_level`文件夹放置在 DouZero 项目的根目录下。
4.  **PerfectDou (可选, 推荐作为对手)**:
    -   为了获得强大的对手（`--opponent perfectdou`），你需要安装 PerfectDou 及其依赖。
    -   如果使用 `onnx` 模型，需要 `pip install onnxruntime`。
5.  **预训练模型**:
    -   你必须拥有 DouZero 的预训练模型（`.ckpt` 文件），例如 `douzero_ADP` 和 `douzero_WP` 文件夹，包含 `landlord.ckpt`, `landlord_up.ckpt`, `landlord_down.ckpt`。
6.  **评估数据**:
    -   你需要一个评估数据集（例如 `eval_data.pkl` 或 `eval_data_copy.pkl`），这是 DouZero 项目中用于评估的标准牌局数据。
    -   评估数据可以通过运行脚本 `generate_eval_data.py` 生成。

### 训练融合模型

使用 `train_fusion.py` 脚本开始训练。你必须指定要融合的两个基础模型（A 和 B）、要训练的位置以及训练目标。

**示例：训练农民上家的WP（胜率）融合模型**

```
python -m fusion_action_level.train_fusion \
	--model_a model/douzero_ADP/landlord_up.ckpt \
	--model_b model/douzero_WP/landlord_up.ckpt \
	--eval_data eval_data_copy.pkl \
	--num_iterations 20 \
	--games_per_iteration 1000 \
	--eval_interval 1 \
	--eval_games 1000  \
	--opponent_landlord model/douzero_ADP/landlord.ckpt  \
	--opponent_landlord_down model/douzero_WP/landlord_down.ckpt \
	--device cuda  \
	--objective wp  \
	--position landlord_up \
	--opponent perfectdou
```

**关键参数说明：**

-   `--model_a`: 基础模型A (例如 ADP)。
-   `--model_b`: 基础模型B (例如 WP)。
-   `--position`: 要训练的位置 (`landlord`, `landlord_up`, `landlord_down`)。三个位置的融合网络需要**分别训练**。
-   `--objective`: 训练目标 (`wp` 或 `adp`)。这决定了游戏结束时 `R_target` 的计算方式（`wp` 为 +1/-1，`adp` 为 2^bombs）。
-   `--opponent`: 训练时其他两个座位的对手类型。推荐使用 `perfectdou` 以获得高质量的训练数据。
-   `--save_dir`: 训练好的模型检查点保存目录。
-   `--num_iterations`: 训练迭代次数（(收集N局游戏 -> 更新M次) * 迭代次数）。
-   `--games_per_iteration`: 每次迭代收集多少局游戏轨迹。

### 评估与运行

使用 `play_with_fusion.py` 脚本来评估训练好的融合模型。

**示例：评估地主的WP融合模型**

```
python play_with_fusion.py \
	--position landlord \
	--num_games 10000 \
	--eval_data eval_data_copy.pkl \
	--opponent_landlord_up perfectdou \
	--opponent_landlord_down perfectdou \
	--checkpoint checkpoints/action_fusion_landlord_wp/best_fusion_wp.pt \
	--model_a model/douzero_ADP/landlord.ckpt \
	--model_b model/douzero_WP/landlord.ckpt \
	--opponent perfectdou
```

**关键参数说明：**

-   `--checkpoint`: 指向你训练好的**融合网络** (`.pt` 文件)。
-   `--model_a`, `--model_b`: **必须**与训练时使用的基础模型一致。
-   `--position`: 指定融合智能体要扮演的位置。
-   `--opponent`: 评估时其他座位的对手。
-   `--num_games`: 运行多少局游戏进行评估。

**多智能体评估**

`play_with_fusion.py` 脚本还支持为**所有三个位置**加载不同的融合智能体：

```
python -m fusion_action_level.play_with_fusion \
    --position landlord \
    --checkpoint /path/to/landlord_fusion.pt \
    --model_a /path/to/landlord_A.ckpt \
    --model_b /path/to/landlord_B.ckpt \
    \
    --landlord_up_checkpoint /path/to/farmer_up_fusion.pt \
    --landlord_up_model_a /path/to/farmer_up_A.ckpt \
    --landlord_up_model_b /path/to/farmer_up_B.ckpt \
    \
    --landlord_down_checkpoint /path/to/farmer_down_fusion.pt \
    --landlord_down_model_a /path/to/farmer_down_A.ckpt \
    --landlord_down_model_b /path/to/farmer_down_B.ckpt \
    \
    --num_games 10000
```

**预期输出**

评估脚本会打印出详细的统计数据，包括地主和农民各自的平均得分（ADP）：

```
============================================================
GAMEPLAY RESULTS
============================================================
Total Games Played: 10000
Primary Position (landlord) Wins: 5210 (52.10%)
------------------------------------------------------------
Avg Landlord ADP: 0.2345
Avg Farmer (Total) ADP: -0.2345
============================================================

Sample Game Results (last 5):
  Game 9996: [WIN] Winner=landlord, LL Reward=2.00, Farmer Reward=-2.00, Bombs=0
  Game 9997: [LOSS] Winner=farmer, LL Reward=-2.00, Farmer Reward=2.00, Bombs=0
  Game 9998: [LOSS] Winner=farmer, LL Reward=-4.00, Farmer Reward=4.00, Bombs=1
  Game 9999: [WIN] Winner=landlord, LL Reward=2.00, Farmer Reward=-2.00, Bombs=0
  Game 10000: [WIN] Winner=landlord, LL Reward=4.00, Farmer Reward=-4.00, Bombs=1
```

## 结果展示

1k 副牌

| 地主             | 农民上家         | 农民下家         | 地主wp | 农民wp | 地主adp | 农民adp |
| ---------------- | ---------------- | ---------------- | ------ | ------ | ------- | ------- |
| douzero_ADP      | perfectdou       | perfectdou       | 0.355  | 0.645  | -0.666  | 0.666   |
| perfectdou       | douzero_ADP      | douzero_ADP      | 0.449  | 0.551  | -0.416  | 0.416   |
| douzero_WP       | perfectdou       | perfectdou       | 0.403  | 0.597  | -0.744  | 0.744   |
| perfectdou       | douzero_WP       | douzero_WP       | 0.387  | 0.613  | -0.32   | 0.32    |
| douzero_ADP      | douzero_WP       | douzero_WP       | 0.375  | 0.625  | -0.276  | 0.276   |
| douzero_WP       | douzero_ADP      | douzero_ADP      | 0.477  | 0.523  | -0.444  | 0.444   |
| douzero_ADP      | douzero_ADP      | douzero_ADP      | 0.418  | 0.582  | -0.438  | 0.438   |
| douzero_WP       | douzero_WP       | douzero_WP       | 0.421  | 0.579  | -0.348  | 0.348   |
| perfectdou       | perfectdou       | perfectdou       | 0.405  | 0.595  | -0.522  | 0.522   |
| ADP+WP+gating_wp | perfectdou       | perfectdou       | 0.407  | 0.593  | -0.47   | 0.47    |
| perfectdou       | ADP+WP+gating_wp | ADP+WP+gating_wp | 0.386  | 0.614  | -0.548  | 0.548   |
| ADP+WP+gating_wp | douzero_ADP      | douzero_ADP      | 0.473  | 0.527  | -0.294  | 0.294   |
| douzero_ADP      | ADP+WP+gating_wp | ADP+WP+gating_wp | 0.354  | 0.646  | -0.604  | 0.604   |
| ADP+WP+gating_wp | douzero_WP       | douzero_WP       | 0.417  | 0.583  | -0.2    | 0.2     |
| douzero_WP       | ADP+WP+gating_wp | ADP+WP+gating_wp | 0.419  | 0.581  | -0.584  | 0.584   |
| perfectdou       | ADP+WP+gating_wp | perfectdou       | 0.389  | 0.611  | -0.524  | 0.524   |
| perfectdou       | perfectdou       | ADP+WP+gating_wp | 0.389  | 0.611  | -0.528  | 0.528   |

1k 副牌对阵表

| WP/ADP           | ADP+WP+gating_wp | perfectdou   | douzero_WP  | douzero_ADP  |
| ---------------- | ---------------- | ------------ | ----------- | ------------ |
| ADP+WP+gating_wp | /                | 0.5105/0.039 | 0.499/0.192 | 0.5595/0.155 |
| perfectdou       | 0.4895/-0.039    | /            | 0.492/0.212 | 0.547/0.125  |
| douzero_WP       | 0.501/-0.192     | 0.508/-0.212 | /           | 0.551/-0.084 |
| douzero_ADP      | 0.4405/-0.155    | 0.453/-0.125 | 0.449/0.084 | /            |

1w 副牌

| 地主             | 农民上家         | 农民下家         | 地主wp | 农民wp | 地主adp | 农民adp |
| ---------------- | ---------------- | ---------------- | ------ | ------ | ------- | ------- |
| douzero_ADP      | perfectdou       | perfectdou       | 0.359  | 0.641  | -0.67   | 0.67    |
| perfectdou       | douzero_ADP      | douzero_ADP      | 0.4532 | 0.5468 | -0.3636 | 0.3636  |
| douzero_WP       | perfectdou       | perfectdou       | 0.4041 | 0.5959 | -0.7018 | 0.7018  |
| perfectdou       | douzero_WP       | douzero_WP       | 0.3867 | 0.6133 | -0.303  | 0.303   |
| douzero_ADP      | douzero_WP       | douzero_WP       | 0.3608 | 0.6392 | -0.3738 | 0.3738  |
| douzero_WP       | douzero_ADP      | douzero_ADP      | 0.4636 | 0.5364 | -0.5322 | 0.5322  |
| douzero_ADP      | douzero_ADP      | douzero_ADP      | 0.4283 | 0.5717 | -0.4182 | 0.4182  |
| douzero_WP       | douzero_WP       | douzero_WP       | 0.4054 | 0.5946 | -0.4448 | 0.4448  |
| perfectdou       | perfectdou       | perfectdou       | 0.3838 | 0.6162 | -0.6052 | 0.6052  |
| ADP+WP+gating_wp | perfectdou       | perfectdou       | 0.4029 | 0.5971 | -0.5334 | 0.5334  |
| perfectdou       | ADP+WP+gating_wp | ADP+WP+gating_wp | 0.3777 | 0.6223 | -0.5756 | 0.5756  |
| ADP+WP+gating_wp | douzero_ADP      | douzero_ADP      | 0.4016 | 0.5984 | -0.2908 | 0.2908  |
| douzero_ADP      | ADP+WP+gating_wp | ADP+WP+gating_wp | 0.3555 | 0.6445 | -0.6318 | 0.6318  |
| ADP+WP+gating_wp | douzero_WP       | douzero_WP       | 0.4651 | 0.5349 | -0.3466 | 0.3466  |
| douzero_WP       | ADP+WP+gating_wp | ADP+WP+gating_wp | 0.4015 | 0.5985 | -0.6918 | 0.6918  |

1w 副牌对阵表

| WP/ADP           | ADP+WP+gating_wp | perfectdou     | douzero_WP    | douzero_ADP    |
| ---------------- | ---------------- | -------------- | ------------- | -------------- |
| ADP+WP+gating_wp | /                | 0.5126/0.0211  | 0.5318/0.1705 | 0.52305/0.1726 |
| perfectdou       | 0.4874/-0.0211   | /              | 0.4913/0.1994 | 0.5471/0.1532  |
| douzero_WP       | 0.4682/-0.1705   | 0.5087/-0.1994 | /             | 0.5514/-0.0792 |
| douzero_ADP      | 0.47695/-0.1726  | 0.4529/-0.1532 | 0.4486/0.0792 | /              |

在 1w 副牌的实验中，本文提出的 **ADP+WP+gating_wp** 模型在总体表现上优于所有基线模型。具体而言，该模型对 **PerfectDou**、**DouZero_WP** 和 **DouZero_ADP** 的胜率分别为 **51.26%**、**53.18%** 和 **52.31%**，平均得分差均为正值，表明其在不同类型对手面前均保持稳定优势。


该结果表明，gating 模型能够在胜率最大化与收益期望之间取得更优平衡，从而实现更强的博弈适应性与泛化性。性能提升的关键在于其动态权重机制，可根据局面自适应地融合 ADP 与 WP 的策略优势，使模型更加灵活且稳健。总体而言，**ADP+WP+gating_wp** 模型在大规模对局中展现出最优的综合实力与策略稳定性。
