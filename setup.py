"""Setup script for crypto trading agent"""

from setuptools import find_packages, setup

setup(
    name="crypto_trading_agent",
    version="0.1.0",
    description="Deep Reinforcement Learning based Crypto Trading Agent",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "gymnasium>=0.29.0",
        "scikit-learn>=1.2.0",
        "matplotlib>=3.7.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "tensorboard>=2.13.0",
        ],
        "viz": [
            "seaborn>=0.12.0",
            "plotly>=5.14.0",
        ],
    },
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "crypto-train=crypto_trading_agent.scripts.train:main",
            "crypto-backtest=crypto_trading_agent.scripts.backtest:main",
            "crypto-evaluate=crypto_trading_agent.scripts.evaluate:main",
        ],
    },
)
