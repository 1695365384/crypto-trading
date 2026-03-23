# 系统架构文档

## 1. 系统概述

本系统是一个基于深度强化学习的加密货币量化交易决策系统，采用 PPO 算法进行策略学习。

## 2. 架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                    加密货币量化交易决策系统                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   数据层      │    │   环境层      │    │   智能体层    │      │
│  │  Data Layer  │───▶│  Env Layer   │───▶│  Agent Layer │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│        │                    │                    │              │
│        ▼                    ▼                    ▼              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ - 数据获取    │    │ - 状态空间    │    │ - PPO 算法   │      │
│  │ - 特征工程    │    │ - 动作空间    │    │ - 网络结构   │      │
│  │ - 数据分割    │    │ - 奖励函数    │    │ - 训练循环   │      │
│  │ - 标准化      │    │ - 环境逻辑    │    │ - 模型保存   │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                     回测与评估模块                        │   │
│  │  - 策略回测  - 风险指标  - 收益分析  - 可视化报告         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 3. 数据流

```
原始数据 (OHLCV)
     │
     ▼
┌─────────────────┐
│  特征工程        │
│  - 技术指标      │
│  - 时间特征      │
│  - 标准化        │
└─────────────────┘
     │
     ▼
┌─────────────────┐
│  数据分割        │
│  - 训练集 70%    │
│  - 验证集 15%    │
│  - 测试集 15%    │
└─────────────────┘
     │
     ▼
┌─────────────────┐
│  经验回放缓冲    │
│  (离线数据集)    │
└─────────────────┘
     │
     ▼
┌─────────────────┐
│  RL 智能体训练   │
│  - 策略学习      │
│  - 价值估计      │
└─────────────────┘
     │
     ▼
┌─────────────────┐
│  模型评估        │
│  - 回测         │
│  - 风险分析      │
└─────────────────┘
```

## 4. 核心模块详解

### 4.1 配置模块 (config/)

**config.py** - 全局配置管理

```python
@dataclass
class DataConfig:
    tickers: List[str]        # 交易对列表
    start_date: str           # 开始日期
    end_date: str             # 结束日期
    timeframe: int            # K线周期 (分钟)
    indicators: List[str]     # 技术指标列表

@dataclass
class EnvConfig:
    initial_amount: float     # 初始资金
    max_position_pct: float   # 最大仓位比例
    transaction_cost_pct: float  # 手续费率
    lookback_window: int      # 观察窗口

@dataclass
class ModelConfig:
    algorithm: str            # 算法名称
    learning_rate: float      # 学习率
    gamma: float              # 折扣因子
    clip_ratio: float         # PPO 裁剪比率
    entropy_coef: float       # 熵系数
```

### 4.2 数据模块 (data/)

**data_loader.py** - 数据加载器

| 方法 | 说明 |
|------|------|
| `load(path)` | 从文件或 API 加载数据 |
| `get_price_array()` | 获取价格数组 |
| `get_combined_dataframe()` | 获取合并数据框 |
| `save(path)` | 保存数据到文件 |

**feature_engineer.py** - 特征工程

| 指标 | 说明 |
|------|------|
| MACD | 趋势指标 |
| RSI | 相对强弱指标 |
| EMA | 指数移动平均 |
| Bollinger Bands | 布林带 |
| ATR | 平均真实波幅 |
| OBV | 能量潮指标 |
| CCI | 商品通道指标 |
| ADX | 平均趋向指标 |

### 4.3 环境模块 (envs/)

**crypto_env.py** - 交易环境

```python
class CryptoTradingEnv:
    """
    状态空间: Box(-inf, inf, (obs_dim,))
        - 历史价格窗口
        - 技术指标
        - 账户状态

    动作空间: Box(-1, 1, (n_assets,))
        - 正值: 买入比例
        - 负值: 卖出比例

    奖励函数:
        - 基础: 组合收益率
        - 可选: 风险调整 (夏普比率惩罚)
    """
```

### 4.4 智能体模块 (agents/)

**ppo_agent.py** - PPO 智能体

```python
class PPOAgent:
    """
    Actor 网络:
        - 输入: 状态
        - 输出: 动作均值和标准差

    Critic 网络:
        - 输入: 状态
        - 输出: 状态价值

    核心方法:
        - get_action(): 获取动作
        - get_value(): 获取价值估计
        - compute_gae(): 计算 GAE
        - update(): PPO 更新
    """
```

### 4.5 神经网络架构详解

#### 4.5.1 架构总览

本系统采用 **LSTM + 3层MLP + Dropout** 的混合架构，结合了时序建模能力和强大的特征提取能力。

```
                           输入状态
                              │
                              ▼
              ┌───────────────────────────────┐
              │  时间序列窗口 (60步 × 特征维度)  │
              │  shape: [batch, 60, ~50]       │
              └───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        LSTM 层                                  │
│                    (捕捉时间序列依赖)                             │
│                    Dropout = 0.1 (输入Dropout)                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   t=1      t=2      t=3      ...      t=60                     │
│    │        │        │                  │                       │
│    ▼        ▼        ▼                  ▼                       │
│  ┌───┐    ┌───┐    ┌───┐              ┌───┐                    │
│  │LSTM│───▶│LSTM│───▶│LSTM│─── ... ───▶│LSTM│                   │
│  └───┘    └───┘    └───┘              └───┘                    │
│   128      128      128                128                      │
│                                             │                   │
│                                             ▼                   │
│                                         [128维]                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │      Dropout(p=0.2)           │
              │    (随机丢弃20%神经元)          │
              └───────────────────────────────┘
                              │
            ┌─────────────────┴─────────────────┐
            │                                   │
            ▼                                   ▼
┌───────────────────────┐           ┌───────────────────────┐
│     Actor 网络        │           │     Critic 网络       │
│   (3层 MLP + Dropout) │           │   (3层 MLP + Dropout) │
├───────────────────────┤           ├───────────────────────┤
│                       │           │                       │
│  Linear(128, 256)     │           │  Linear(128, 256)     │
│  LayerNorm(256)       │           │  LayerNorm(256)       │
│  ReLU                 │           │  ReLU                 │
│  Dropout(0.2)         │           │  Dropout(0.2)         │
│         │             │           │         │             │
│         ▼             │           │         ▼             │
│  Linear(256, 128)     │           │  Linear(256, 128)     │
│  LayerNorm(128)       │           │  LayerNorm(128)       │
│  ReLU                 │           │  ReLU                 │
│  Dropout(0.2)         │           │  Dropout(0.2)         │
│         │             │           │         │             │
│         ▼             │           │         ▼             │
│  Linear(128, 64)      │           │  Linear(128, 64)      │
│  LayerNorm(64)        │           │  LayerNorm(64)        │
│  ReLU                 │           │  ReLU                 │
│  Dropout(0.2)         │           │  Dropout(0.2)         │
│         │             │           │         │             │
│         ▼             │           │         ▼             │
│  Linear(64, action_dim)│          │  Linear(64, 1)        │
│  Tanh → 动作均值       │           │  → 状态价值            │
│  + 可学习 std          │           │                       │
│                       │           │                       │
└───────────────────────┘           └───────────────────────┘
            │                                   │
            ▼                                   ▼
      动作分布 (π(a|s))                    价值估计 V(s)
      mean ∈ [-1, 1]                    标量值
      std > 0
```

#### 4.5.2 数据流详解

```
┌──────────────────────────────────────────────────────────────────┐
│  输入: 60分钟窗口数据                                              │
│                                                                  │
│  时间步:  t-59   t-58   t-57  ...  t-2    t-1    t(当前)          │
│           │      │      │          │      │      │               │
│           ▼      ▼      ▼          ▼      ▼      ▼               │
│  特征:  [特征]  [特征]  [特征]     [特征] [特征] [特征]            │
│         向量    向量    向量       向量   向量   向量              │
│         (50维)  (50维)  (50维)     (50维) (50维) (50维)           │
│                                                                  │
│  shape: [batch_size, 60, ~50]                                    │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼ LSTM 处理
┌──────────────────────────────────────────────────────────────────┐
│  LSTM 逐时间步处理，记住历史信息                                    │
│                                                                  │
│  隐藏状态 h_t = LSTM(x_t, h_{t-1})                               │
│                                                                  │
│  最后输出: h_{t=60} → 128维向量 (压缩了60分钟的信息)               │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼ MLP 处理
┌──────────────────────────────────────────────────────────────────┐
│  3层 MLP 逐步抽象特征                                              │
│                                                                  │
│  128 → 256 → 128 → 64 → 输出                                      │
│                                                                  │
│  每一层: Linear → LayerNorm → ReLU → Dropout                     │
└──────────────────────────────────────────────────────────────────┘
```

#### 4.5.3 组件详解

**LSTM 层**

```
作用: 捕捉时间序列中的长期依赖关系

特性:
- 输入维度: ~50 (特征数)
- 隐藏维度: 128
- 层数: 1
- Dropout: 0.1 (输入dropout)

LSTM 内部结构:
┌─────────────────────────────────────────┐
│                                         │
│   x_t ──▶ [遗忘门] ──┐                  │
│          [输入门] ───┼──▶ h_t (输出)    │
│          [输出门] ───┘                  │
│          [单元状态 C_t]                  │
│                                         │
│   记忆机制:                              │
│   - 遗忘门: 决定丢弃哪些历史信息          │
│   - 输入门: 决定记住哪些新信息            │
│   - 输出门: 决定输出哪些信息              │
└─────────────────────────────────────────┘

优势:
- 能记住长期趋势 (如持续上涨/下跌)
- 能识别时序模式 (如周期性波动)
- 处理变长序列
```

**MLP 层**

```
作用: 特征提取和非线性变换

每层结构:
┌─────────────────────────────────────────┐
│  Linear(in, out)                        │
│       │                                 │
│       ▼                                 │
│  LayerNorm(out)  ─ 稳定训练              │
│       │                                 │
│       ▼                                 │
│  ReLU()          ─ 引入非线性            │
│       │                                 │
│       ▼                                 │
│  Dropout(p)      ─ 防止过拟合            │
└─────────────────────────────────────────┘

逐层维度变化:
  输入: 128  (LSTM 输出)
    ↓
  Layer 1: 256  (扩展特征空间)
    ↓
  Layer 2: 128  (压缩提取)
    ↓
  Layer 3: 64   (进一步抽象)
    ↓
  输出: action_dim / 1
```

**Dropout 正则化**

```
训练时 (Dropout 开启):              推理时 (Dropout 关闭):

  全连接层                           全连接层
  ┌─────────┐                       ┌─────────┐
  │ ● ● ○ ● │  随机丢弃             │ ● ● ● ● │  全部激活
  │ ● ○ ● ● │  ────────►            │ ● ● ● ● │  (输出×0.8)
  │ ○ ● ● ○ │  20%神经元             │ ● ● ● ● │
  └─────────┘                       └─────────┘

作用: 强迫网络不依赖任何单个神经元，学习更鲁棒的特征
```

#### 4.5.4 Dropout 配置策略

| 位置 | Dropout率 | 原因 |
|------|-----------|------|
| LSTM 输入 | 0.1 | 轻微即可，保护时序信息 |
| LSTM 输出 | 0.2 | 主要防过拟合点 |
| MLP 每层 | 0.2 | 标准配置，每层都保护 |

```
Dropout 率选择原则:

0.0 ──────── 0.1 ──────── 0.2 ──────── 0.3 ──────── 0.5
 │            │            │            │            │
无保护       轻度         标准         较强         过强
                          ✓推荐                      (欠拟合风险)
```

#### 4.5.5 参数量统计

| 组件 | 参数量 | 计算公式 |
|------|--------|----------|
| LSTM (128维) | ~92K | 4 × (input_dim + hidden + 1) × hidden |
| Actor MLP (3层) | ~115K | 见下表 |
| Critic MLP (3层) | ~115K | 见下表 |
| **总计** | **~322K** | |

MLP 参数量详解:

| 层 | 形状 | 参数量 |
|----|------|--------|
| Layer 1 | 128 → 256 | 128×256 + 256 = 33,024 |
| Layer 2 | 256 → 128 | 256×128 + 128 = 32,896 |
| Layer 3 | 128 → 64 | 128×64 + 64 = 8,256 |
| 输出 (Actor) | 64 → action_dim | ~128 |
| 输出 (Critic) | 64 → 1 | 65 |

> 注: LayerNorm 参数未计入，每层额外增加 2×hidden

#### 4.5.6 训练 vs 推理模式

```python
# 训练时
model.train()   # Dropout 生效，随机丢弃

# 推理/评估时
model.eval()    # Dropout 关闭，全部激活
```

#### 4.5.7 与传统架构对比

| 架构类型 | 结构 | 优势 | 劣势 |
|----------|------|------|------|
| **纯 MLP** | 全连接 | 简单快速 | 无法捕捉时序依赖 |
| **CNN** | 卷积层 | 局部模式识别 | 需要空间结构 |
| **纯 LSTM** | 循环层 | 时序建模 | 特征提取能力有限 |
| **本架构 (LSTM+MLP)** | 混合 | 时序+特征 | 计算量稍大 |

#### 4.5.8 配置参数

```python
@dataclass
class NetworkConfig:
    # LSTM 配置
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 1
    lstm_dropout: float = 0.1

    # MLP 配置
    actor_hidden_sizes: List[int] = field(
        default_factory=lambda: [256, 128, 64]
    )
    critic_hidden_sizes: List[int] = field(
        default_factory=lambda: [256, 128, 64]
    )
    mlp_dropout: float = 0.2

    # 输出配置
    action_dim: int = 2  # 交易对数量
    use_layer_norm: bool = True
    activation: str = "relu"
```

### 4.5 训练模块 (training/)

**trainer.py** - 训练器

```python
class Trainer:
    """
    训练流程:
        1. 收集经验到缓冲区
        2. 计算 GAE 和回报
        3. 执行 PPO 更新
        4. 验证和保存
    """
```

**callbacks.py** - 训练回调

| 回调类 | 说明 |
|--------|------|
| `EarlyStoppingCallback` | 早停机制 |
| `CheckpointCallback` | 检查点保存 |
| `TensorBoardCallback` | TensorBoard 日志 |
| `MetricsLoggerCallback` | 指标记录 |

### 4.6 评估模块 (evaluation/)

**backtest.py** - 回测引擎

```python
@dataclass
class BacktestResult:
    # 收益指标
    total_return: float       # 总收益率
    annual_return: float      # 年化收益率
    sharpe_ratio: float       # 夏普比率
    sortino_ratio: float      # Sortino 比率

    # 风险指标
    max_drawdown: float       # 最大回撤
    volatility: float         # 波动率
    var_95: float            # 95% VaR
    cvar_95: float           # 95% CVaR

    # 交易统计
    total_trades: int         # 总交易次数
    win_rate: float           # 胜率
    profit_factor: float      # 盈亏比
```

**metrics.py** - 评估指标

| 函数 | 说明 |
|------|------|
| `calculate_metrics()` | 计算常用指标 |
| `calculate_drawdown()` | 计算回撤 |
| `calculate_trade_metrics()` | 计算交易指标 |
| `calculate_benchmark_comparison()` | 基准比较 |

### 4.7 推理模块 (inference/)

**predictor.py** - 交易预测器

```python
class TradingPredictor:
    """
    主要方法:
        - predict(): 预测交易动作
        - should_trade(): 判断是否交易
        - get_position_adjustment(): 获取仓位调整
    """

class EnsemblePredictor:
    """
    多模型集成预测:
        - mean: 均值集成
        - median: 中位数集成
        - vote: 加权投票
    """
```

**risk_manager.py** - 风险管理

```python
class RiskManager:
    """
    风控措施:
        - 单币种仓位限制
        - 日损失限制
        - 最大回撤限制
        - 杠杆限制
    """

class PositionSizer:
    """
    仓位计算方法:
        - kelly: Kelly Criterion
        - fixed: 固定比例
        - volatility: 波动率倒数
    """
```

## 5. 算法详解

### 5.1 PPO 算法

```
PPO 目标函数:
L^CLIP(θ) = E[min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)]

其中:
- r(θ) = π_θ(a|s) / π_θ_old(a|s)  # 重要性采样比率
- A = GAE  # 广义优势估计
- ε = 0.2  # 裁剪比率
```

### 5.2 GAE 计算

```
GAE: A_t = Σ (γλ)^l * δ_{t+l}

其中:
- δ_t = r_t + γV(s_{t+1}) - V(s_t)  # TD 残差
- γ = 0.99  # 折扣因子
- λ = 0.95  # GAE 参数
```

### 5.3 奖励函数

```python
def calculate_reward(new_value, old_value, returns_history):
    # 收益率
    returns = (new_value - old_value) / old_value

    # 风险惩罚 (可选)
    if len(returns_history) > 10:
        volatility = std(returns_history[-10:])
        risk_penalty = 0.1 * volatility
    else:
        risk_penalty = 0

    # 缩放
    reward = (returns - risk_penalty) * reward_scaling

    return reward
```

## 6. 使用指南

### 6.1 训练流程

```bash
# 1. 准备数据
# 将数据放入 ./data/ 目录

# 2. 配置参数
# 编辑 config/prod.yaml

# 3. 开始训练
python scripts/train.py --config config/prod.yaml

# 4. 监控训练
tensorboard --logdir logs/
```

### 6.2 回测流程

```bash
# 运行回测
python scripts/backtest.py \
    --model models/best_model.pt \
    --data data/test.csv \
    --output results/
```

### 6.3 推理使用

```python
from inference.predictor import TradingPredictor

# 加载模型
predictor = TradingPredictor(
    model_path='models/best_model.pt',
    obs_dim=100,
    action_dim=2
)

# 获取预测
action, confidence = predictor.predict(observation)

# 风险检查
from inference.risk_manager import RiskManager
risk_mgr = RiskManager(initial_capital=10000)

is_valid, reason = risk_mgr.check_trade(
    action, positions, prices, balance
)
```

## 7. 性能优化

### 7.1 训练优化

- 使用 GPU 加速
- 调整 batch_size 和 buffer_size
- 使用混合精度训练

### 7.2 推理优化

- 模型量化
- ONNX 导出
- 批量推理

## 8. 扩展开发

### 8.1 添加新指标

```python
# 在 feature_engineer.py 中添加
def _add_custom_indicator(self, df):
    # 实现自定义指标
    df['custom'] = ...
    return df
```

### 8.2 自定义奖励函数

```python
# 在 crypto_env.py 中修改
def _calculate_reward(self, new_value):
    # 实现自定义奖励
    return custom_reward
```

### 8.3 添加新算法

```python
# 创建新的智能体类
class SACAgent:
    def __init__(self, config):
        # 初始化 SAC 网络
        pass

    def update(self, batch):
        # 实现 SAC 更新
        pass
```

## 9. 故障排除

| 问题 | 解决方案 |
|------|----------|
| 训练不收敛 | 降低学习率，检查奖励函数 |
| 内存不足 | 减小 buffer_size，使用梯度累积 |
| GPU 内存不足 | 减小 batch_size |
| 回测结果差 | 检查数据泄露，调整超参数 |

## 10. 参考资料

- [PPO 论文](https://arxiv.org/abs/1707.06347)
- [GAE 论文](https://arxiv.org/abs/1506.02438)
- [FinRL 框架](https://github.com/AI4Finance-Foundation/FinRL)
- [Gymnasium 文档](https://gymnasium.farama.org/)
