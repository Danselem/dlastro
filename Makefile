.DEFAULT_GOAL := help
SRC := src

## ──────────────────────────────────────────────
## Setup
## ──────────────────────────────────────────────

.PHONY: init
init: ## Create venv, pyproject.toml, and install all deps
	uv venv --python 3.12
	uv init && rm main.py

.PHONY: bootstrap
bootstrap: ## Install deps into existing venv
	uv add --requirements requirements.txt --link-mode=copy

.PHONY: install
install: ## Install deps into existing venv
	uv sync

.PHONY: cuda
cuda: ## Start interactive CUDA session on ilifu
	srun --x11 \
		--time=1:00:00 \
		--nodes=1 \
		--partition=GPU \
		--mem=64GB \
		--nodelist=gpu-[001-007] \
		--gres=gpu:1 \
		--pty bash