.PHONY: help install run run-mcp-bank dataset dataset-upload

# Default target
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "📋 Available commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install project dependencies
	uv sync

run: ## Start the Telegram bot
	uv run python src/bot.py

run-mcp-bank: ## Start MCP bank agent server (port 8000)
	cd mcp/mcp-bank-agent && make run

dataset: ## Create QA dataset from documents
	uv run python src/dataset_synthesizer.py --create

dataset-upload: ## Upload dataset to LangSmith
	uv run python src/dataset_synthesizer.py --upload

