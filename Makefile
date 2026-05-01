.PHONY: help install install-dev test test-cov lint format typecheck clean run-cwru run-phm run-nasa run-comparison

# Default target
help:
	@echo "BSS-Test: Bearing Fault Diagnosis with BSS"
	@echo ""
	@echo "Available targets:"
	@echo "  install       - Install core dependencies"
	@echo "  install-dev   - Install development dependencies"
	@echo "  install-all   - Install all dependencies (extended + dev)"
	@echo "  test          - Run tests"
	@echo "  test-cov      - Run tests with coverage"
	@echo "  lint          - Run linter (flake8)"
	@echo "  format        - Format code (black + isort)"
	@echo "  typecheck     - Run type checking (mypy)"
	@echo "  clean         - Remove build artifacts and caches"
	@echo "  run-cwru      - Run CWRU experiment"
	@echo "  run-phm       - Run PHM 2010 experiment"
	@echo "  run-nasa      - Run NASA milling experiment"
	@echo "  run-comparison - Run comparison report"
	@echo "  run-all       - Run all experiments"
	@echo "  help          - Show this help message"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements_dev.txt

install-all:
	pip install -r requirements_extended.txt
	pip install -r requirements_dev.txt

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-parallel:
	pytest tests/ -v -n auto

test-fast:
	pytest tests/ -v -m "not slow"

# Code Quality
lint:
	flake8 src/ tests/ --max-line-length=88 --extend-ignore=E203,W503

format:
	black src/ tests/
	isort src/ tests/

typecheck:
	mypy src/ --ignore-missing-imports

# Cleaning
clean:
	rm -rf __pycache__/
	rm -rf src/__pycache__/
	rm -rf tests/__pycache__/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Running experiments
run-cwru:
	python experiments/single/cwru.py

run-phm:
	python experiments/single/phm_milling.py

run-nasa:
	python experiments/single/nasa_milling.py

run-comparison:
	python experiments/comparison/bss_methods.py

run-summary:
	python experiments/reports/summary.py

run-all: run-cwru run-phm run-nasa run-comparison

# Development
check: lint typecheck test

pre-commit:
	pre-commit run --all-files

# Documentation
docs:
	cd docs && make html

# Docker (if needed)
docker-build:
	docker build -t bss-test .

docker-run:
	docker run -v $(PWD)/data:/app/data -v $(PWD)/outputs:/app/outputs bss-test
