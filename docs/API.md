# API 参考

## 配置模块

### Config

主配置类，整合所有子配置。

```python
from config.config import Config

# 从 YAML 加载
config = Config.from_yaml('config/prod.yaml')

# 保存配置
config.to_yaml('config/custom.yaml')

# 属性访问
config.data.tickers      # 交易对列表
config.env.initial_amount  # 初始资金
config.model.learning_rate  # 学习率
```

### DataConfig

数据配置。

| 属性 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `tickers` | List[str] | ["BTCUSDT", "ETHUSDT"] | 交易对 |
| `start_date` | str | "2023-01-01" | 开始日期 |
| `end_date` | str | "2025-12-31" | 结束日期 |
| `timeframe` | int | 1 | K线周期 (分钟) |
| `indicators` | List[str] | [...] | 技术指标列表 |

### EnvConfig

环境配置。

| 属性 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `initial_amount` | float | 10000.0 | 初始资金 |
| `max_position_pct` | float | 0.5 | 最大仓位比例 |
| `transaction_cost_pct` | float | 0.001 | 手续费率 |
| `slippage_pct` | float | 0.0005 | 滑点 |
| `reward_scaling` | float | 1e-4 | 奖励缩放 |
| `lookback_window` | int | 60 | 观察窗口 |

### ModelConfig

模型配置。

| 属性 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `algorithm` | str | "PPO" | 算法名称 |
| `learning_rate` | float | 3e-5 | 学习率 |
| `gamma` | float | 0.99 | 折扣因子 |
| `gae_lambda` | float | 0.95 | GAE 参数 |
| `clip_ratio` | float | 0.2 | PPO 裁剪比率 |
| `entropy_coef` | float | 0.01 | 熵系数 |
| `batch_size` | int | 64 | 批次大小 |
| `total_timesteps` | int | 1000000 | 总训练步数 |

---

## 数据模块

### DataLoader

数据加载器。

```python
from data.data_loader import DataLoader
from config.config import DataConfig

config = DataConfig()
loader = DataLoader(config)

# 加载数据
data = loader.load('./data/')  # 从目录加载
data = loader.load('./data/btc.csv')  # 从文件加载

# 获取价格数组
prices = loader.get_price_array()  # shape: (timesteps, assets)

# 保存数据
loader.save('./processed_data/')
```

### FeatureEngineer

特征工程。

```python
from data.feature_engineer import FeatureEngineer

engineer = FeatureEngineer(indicators=['macd', 'rsi_14', 'ema_20'])

# 添加特征
featured_df = engineer.add_features(df)

# 获取特征列名
cols = engineer.get_feature_columns()
```

### Preprocessor

数据预处理器。

```python
from data.preprocessor import Preprocessor

preprocessor = Preprocessor(scaler_type='robust')

# 分割数据
train, val, test = preprocessor.split(data, 0.7, 0.15, 0.15)

# 标准化 (拟合训练数据)
train = preprocessor.fit_transform(train)

# 转换验证和测试数据
val = preprocessor.transform(val)
test = preprocessor.transform(test)

# 创建环境数据
features, prices, cols = preprocessor.create_env_data(train)

# 保存/加载预处理器
preprocessor.save('preprocessor.pkl')
preprocessor.load('preprocessor.pkl')
```

---

## 环境模块

### CryptoTradingEnv

交易环境。

```python
from envs.crypto_env import CryptoTradingEnv
from config.config import EnvConfig

config = EnvConfig()
env = CryptoTradingEnv(
    config=config,
    price_data=prices,      # shape: (timesteps, assets)
    feature_data=features,  # shape: (timesteps, assets, features)
    feature_columns=cols
)

# 重置环境
obs, info = env.reset()

# 执行步骤
action = np.array([0.5, -0.3])  # 买入 BTC 50%, 卖出 ETH 30%
obs, reward, terminated, truncated, info = env.step(action)

# 获取统计
stats = env.get_portfolio_stats()
# 返回: {'total_return', 'sharpe_ratio', 'max_drawdown', ...}
```

#### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `observation_space` | Box | 观察空间 |
| `action_space` | Box | 动作空间 [-1, 1] |
| `portfolio_value` | float | 当前组合价值 |
| `balance` | float | 当前余额 |
| `positions` | ndarray | 各币种持仓数量 |

---

## 智能体模块

### PPOAgent

PPO 智能体。

```python
from agents.ppo_agent import PPOAgent
from config.config import ModelConfig

config = ModelConfig()
agent = PPOAgent(config, device='cuda:0')

# 初始化网络
agent.init_networks(obs_dim=100, action_dim=2)

# 获取动作
action, log_prob = agent.get_action(obs, deterministic=False)

# 获取价值
value = agent.get_value(obs)

# 计算 GAE
advantages = agent.compute_gae(rewards, values, dones, next_value)

# 更新策略
update_info = agent.update(batch)

# 保存/加载模型
agent.save('model.pt')
agent.load('model.pt')
```

#### 方法

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `init_networks` | obs_dim, action_dim | None | 初始化网络 |
| `get_action` | obs, deterministic | (action, log_prob) | 获取动作 |
| `get_value` | obs | float | 获取价值估计 |
| `compute_gae` | rewards, values, dones, next_value | Tensor | 计算 GAE |
| `update` | batch | Dict | 执行更新 |
| `save` | path | None | 保存模型 |
| `load` | path | None | 加载模型 |

### ReplayBuffer

经验回放缓冲区。

```python
from agents.replay_buffer import ReplayBuffer

buffer = ReplayBuffer(
    buffer_size=2048,
    obs_dim=100,
    action_dim=2
)

# 添加经验
buffer.add(obs, action, reward, value, log_prob, done)

# 获取所有数据
obs, actions, rewards, values, log_probs, dones = buffer.get()

# 清空
buffer.clear()

# 检查
buffer.is_full()
len(buffer)
```

---

## 训练模块

### Trainer

训练器。

```python
from training.trainer import Trainer
from training.callbacks import CheckpointCallback, EarlyStoppingCallback

# 创建回调
callbacks = [
    CheckpointCallback(save_freq=50000, save_path='./models/'),
    EarlyStoppingCallback(patience=10, min_delta=0.001)
]

# 创建训练器
trainer = Trainer(
    config=config,
    agent=agent,
    train_env=train_env,
    val_env=val_env,
    callbacks=callbacks
)

# 训练
final_stats = trainer.train()

# 保存日志
trainer.save_training_logs('logs/training.json')
```

---

## 评估模块

### Backtester

回测引擎。

```python
from evaluation.backtest import Backtester, BacktestResult

backtester = Backtester(config)

# 运行回测
result = backtester.run(agent, env, deterministic=True)

# 访问结果
print(result.total_return)    # 总收益
print(result.sharpe_ratio)    # 夏普比率
print(result.max_drawdown)    # 最大回撤
print(result.win_rate)        # 胜率

# 生成摘要
print(result.summary())

# 保存结果
result.save('backtest_results.json')

# 转换为字典
result_dict = result.to_dict()
```

### Visualizer

可视化工具。

```python
from evaluation.visualizer import Visualizer

viz = Visualizer(figsize=(14, 10))

# 绘制组合价值
viz.plot_portfolio_value(result, save_path='portfolio.png')

# 绘制收益分布
viz.plot_returns_distribution(result, save_path='returns.png')

# 绘制回撤
viz.plot_drawdown(result, save_path='drawdown.png')

# 策略对比
viz.plot_comparison({'PPO': result1, 'Buy&Hold': result2})

# 生成完整报告
viz.generate_report(result, output_dir='./reports/', prefix='backtest')
```

### 指标函数

```python
from evaluation.metrics import (
    calculate_metrics,
    calculate_drawdown,
    calculate_trade_metrics
)

# 计算指标
metrics = calculate_metrics(returns)
# 返回: {'sharpe_ratio', 'sortino_ratio', 'var_95', ...}

# 计算回撤
dd = calculate_drawdown(portfolio_values)
# 返回: {'max_drawdown', 'max_dd_duration', ...}

# 计算交易指标
trade_metrics = calculate_trade_metrics(trades)
# 返回: {'win_rate', 'profit_factor', ...}
```

---

## 推理模块

### TradingPredictor

交易预测器。

```python
from inference.predictor import TradingPredictor

predictor = TradingPredictor(
    model_path='models/best_model.pt',
    obs_dim=100,
    action_dim=2,
    device='cuda:0'
)

# 预测
action, confidence = predictor.predict(observation, deterministic=True)

# 批量预测
actions, confidences = predictor.predict_batch(observations)

# 判断是否交易
should_trade = predictor.should_trade(
    action,
    confidence,
    action_threshold=0.1,
    confidence_threshold=0.5
)

# 获取仓位调整
target_positions = predictor.get_position_adjustment(
    current_positions,
    action,
    max_position=0.5
)

# 获取模型信息
info = predictor.get_model_info()
```

### RiskManager

风险管理器。

```python
from inference.risk_manager import RiskManager

risk_mgr = RiskManager(
    initial_capital=10000,
    max_position_pct=0.5,
    max_daily_loss_pct=0.05,
    max_drawdown_pct=0.2
)

# 检查交易
is_valid, reason = risk_mgr.check_trade(
    action=action,
    current_positions=positions,
    current_prices=prices,
    current_balance=balance
)

# 调整动作
adjusted_action = risk_mgr.adjust_action(
    action, positions, prices, balance
)

# 更新状态
risk_mgr.update(portfolio_value, positions, trade)

# 获取风险指标
metrics = risk_mgr.get_metrics(portfolio_value, positions, prices)

# 检查是否停止
should_stop, reason = risk_mgr.should_stop_trading(portfolio_value)
```

### PositionSizer

仓位计算器。

```python
from inference.risk_manager import PositionSizer

sizer = PositionSizer(
    method='kelly',      # 'kelly', 'fixed', 'volatility'
    max_position=0.5,
    risk_per_trade=0.02
)

# 计算仓位
position = sizer.calculate(
    confidence=0.8,
    volatility=0.02,
    win_rate=0.55
)
```

---

## 脚本使用

### train.py

```bash
python scripts/train.py \
    --config config/prod.yaml \
    --timesteps 1000000 \
    --device cuda:0 \
    --seed 42
```

### backtest.py

```bash
python scripts/backtest.py \
    --model models/best_model.pt \
    --data data/test.csv \
    --config config/prod.yaml \
    --output results/ \
    --device cuda:0
```

### evaluate.py

```bash
python scripts/evaluate.py \
    --model models/best_model.pt \
    --data data/test.csv \
    --verbose
```

### quickstart.py

```bash
python scripts/quickstart.py
```
