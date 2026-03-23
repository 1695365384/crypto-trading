<div align="center">
  <img src="assets/banner.svg" alt="Crypto Trading Agent Banner" width="100%">
</div>

<p align="center">
  <a href="docs/ARCHITECTURE.md"><img src="https://img.shields.io/badge/Docs-Architecture-10B981?style=flat-square" alt="Architecture"></a>
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Gymnasium-0081A7?style=flat-square&logo=openaigym&logoColor=white" alt="Gymnasium">
  <img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=flat-square" alt="Matplotlib">
  <img src="https://img.shields.io/badge/Exchange-OKX-black?style=flat-square" alt="OKX">
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License">
</p>

---

基于 PPO + LSTM 的加密货币分钟级交易决策系统，支持多币种策略学习与回测。

## 项目背景

加密货币市场具有高波动性、7x24 小时连续交易的特点，传统技术分析方法难以应对复杂的非线性价格行为。本项目采用深度强化学习技术，让智能体通过与环境交互自动学习交易策略。

**为什么选择 PPO + LSTM：**

| 方案 | 优势 |
|------|------|
| PPO (Proximal Policy Optimization) | 策略梯度方法中稳定性最好，适合连续动作空间，不会出现策略更新过大导致的崩溃 |
| LSTM (Long Short-Term Memory) | 能够记忆长期时序依赖，捕捉价格趋势和周期性模式 |
| Actor-Critic 架构 | Actor 负责决策，Critic 负责评估，两者协同提升学习效率 |

**适用场景：** 分钟级至小时级的中频量化交易，支持多币种组合策略。

## 核心特性

| 特性 | 说明 |
|------|------|
| PPO 算法 | 稳定的策略梯度方法，适合连续动作空间 |
| LSTM + MLP | 捕捉时序依赖 + 特征提取 |
| 分钟级决策 | 适用于中频交易场景 |
| 多币种支持 | 可同时交易多个加密货币 |
| 完整回测 | 内置回测引擎和评估指标 |

## 项目结构

```
crypto_trading_agent/
├── config/           # 配置模块
├── data/             # 数据模块 (OKX 数据源)
├── envs/             # 交易环境 (Gymnasium)
├── agents/           # PPO 智能体 + 网络结构
├── training/         # 训练器 + 回调
├── evaluation/       # 回测 + 指标 + 可视化
├── inference/        # 预测器 + 风险管理
├── scripts/          # 命令行脚本
├── tests/            # 测试
└── docs/             # 文档
```

## 快速开始

### 安装

```bash
pip install -r requirements.txt
```

### 下载数据

```bash
python scripts/download_okx_data.py \
    --symbols BTC-USDT ETH-USDT \
    --start 2024-01-01 \
    --end 2024-03-31 \
    --bar 1m \
    --output ./data/okx
```

### 训练

```bash
python scripts/train.py --config config/okx.yaml
```

### 回测

```bash
python scripts/backtest.py --model models/best_model.pt --data data/test.csv
```

### 评估

```bash
python scripts/evaluate.py --model models/best_model.pt
```

## 网络架构

```mermaid
flowchart TB
    subgraph INPUT["输入层"]
        I["时间窗口: 60步 x 50特征<br/>[batch, 60, 50]"]
    end

    subgraph LSTM["LSTM 层 (时序建模)"]
        L1["t=1 -> 128维"]
        L2["t=2 -> 128维"]
        L3["..."]
        L4["t=60 -> 128维"]
        L1 --> L2 --> L3 --> L4
        LO["输出: 128维<br/>压缩60分钟时序信息"]
        L4 --> LO
    end

    subgraph DROP["Dropout"]
        D["p=0.2"]
    end

    subgraph ACTOR["Actor 网络 (策略)"]
        A1["Layer 1: 128->256<br/>LayerNorm + ReLU + Dropout"]
        A2["Layer 2: 256->128<br/>LayerNorm + ReLU + Dropout"]
        A3["Layer 3: 128->64<br/>LayerNorm + ReLU + Dropout"]
        A4["Output: 动作分布<br/>mean [-1,1], std>0"]
        A1 --> A2 --> A3 --> A4
    end

    subgraph CRITIC["Critic 网络 (价值)"]
        C1["Layer 1: 128->256<br/>LayerNorm + ReLU + Dropout"]
        C2["Layer 2: 256->128<br/>LayerNorm + ReLU + Dropout"]
        C3["Layer 3: 128->64<br/>LayerNorm + ReLU + Dropout"]
        C4["Output: V(s)<br/>状态价值评估"]
        C1 --> C2 --> C3 --> C4
    end

    I --> L1
    LO --> D
    D --> A1
    D --> C1
```

### 层级参数表

| 模块 | 层 | 输入维度 | 输出维度 | 激活函数 | Dropout |
|------|-----|---------|---------|---------|---------|
| **LSTM** | - | 50 | 128 | Tanh/Sigmoid | 0.1 |
| **Actor** | Layer 1 | 128 | 256 | ReLU | 0.2 |
| | Layer 2 | 256 | 128 | ReLU | 0.2 |
| | Layer 3 | 128 | 64 | ReLU | 0.2 |
| | Output | 64 | action_dim | Tanh | - |
| **Critic** | Layer 1 | 128 | 256 | ReLU | 0.2 |
| | Layer 2 | 256 | 128 | ReLU | 0.2 |
| | Layer 3 | 128 | 64 | ReLU | 0.2 |
| | Output | 64 | 1 | Linear | - |

### 组件功能

| 组件 | 功能 | 说明 |
|------|------|------|
| **LSTM** | 时序建模 | 记忆60分钟内的价格趋势变化 |
| **Actor** | 策略决策 | 输出买卖动作 (-1=卖, 0=持有, 1=买) |
| **Critic** | 价值评估 | 评估当前状态好坏，指导策略优化 |
| **Dropout** | 正则化 | 防止过拟合，提升泛化能力 |
| **LayerNorm** | 归一化 | 稳定训练，加速收敛 |

**总参数量**: ~322K

## 配置参数

| 类别 | 参数 | 默认值 |
|------|------|--------|
| **数据** | 交易对 | BTC-USDT, ETH-USDT |
| | K线周期 | 1分钟 |
| | 技术指标 | MACD, RSI, EMA, ATR |
| **环境** | 初始资金 | 10,000 USDT |
| | 最大仓位 | 50% |
| | 手续费 | 0.1% |
| | 观察窗口 | 60分钟 |
| **模型** | 学习率 | 3e-5 |
| | 折扣因子 | 0.99 |
| | 训练步数 | 1,000,000 |

## 数据源

| 数据类型 | 说明 |
|----------|------|
| K线数据 | OHLCV + 成交量 |
| 技术指标 | MACD, RSI, EMA, 布林带, ATR, OBV, CCI, ADX |
| 时间特征 | 周期性编码 (sin/cos) |
| 收益率 | 多周期收益率 + 波动率 |

支持的 K 线周期: `1m, 5m, 15m, 30m, 1H, 2H, 4H, 6H, 12H, 1D, 1W, 1M`

## 评估指标

| 类别 | 指标 |
|------|------|
| 收益 | 总收益、年化收益、夏普比率、Sortino 比率 |
| 风险 | 最大回撤、波动率、VaR、CVaR |
| 交易 | 胜率、盈亏比、交易次数 |

## 风险提示

> **免责声明**
> - 本项目仅供学习和研究使用
> - 历史回测结果不代表未来收益
> - 加密货币市场波动剧烈，请谨慎投资

## 许可证

[MIT License](LICENSE)

---

<div align="center">

如果这个项目对你有帮助，请给一个 Star 支持一下！

</div>
