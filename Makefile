.PHONY: help install clean format lint test all

# 默认目标
help:
	@echo "可用命令:"
	@echo "  make install     - 安装依赖"
	@echo "  make format      - 格式化代码 (black + isort)"
	@echo "  make lint        - 代码检查 (flake8 + mypy)"
	@echo "  make lint-all    - 完整检查 (flake8 + mypy + pylint)"
	@echo "  make test        - 运行测试"
	@echo "  make test-cov    - 运行测试并生成覆盖率报告"
	@echo "  make clean       - 清理缓存文件"
	@echo "  make all         - 格式化 + 检查 + 测试"
	@echo "  make pre-commit  - 安装 pre-commit hooks"

# 安装依赖
install:
	pip install -r requirements.txt

# 格式化代码
format:
	black .
	isort .

# 代码检查
lint:
	flake8 .
	mypy .

# 完整检查
lint-all:
	flake8 .
	mypy .
	pylint --rcfile=pyproject.toml agents config data envs evaluation inference scripts training

# 运行测试
test:
	pytest tests/ -v

# 测试覆盖率
test-cov:
	pytest tests/ -v --cov=. --cov-report=term-missing --cov-report=html

# 清理
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf htmlcov/ .coverage

# 安装 pre-commit
pre-commit:
	pre-commit install

# 全部执行
all: format lint test
