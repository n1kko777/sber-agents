.PHONY: help install run test clean regenerate-data

# Default target
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "📋 Available commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies via uv
	@echo "📦 Installing dependencies..."
	uv sync
	@echo "✅ Dependencies installed successfully!"

run: ## Start HTTP MCP server on port 8000
	@echo "🚀 Starting HTTP MCP server..."
	@echo "📍 Server will be available at: http://localhost:8000/mcp"
	@echo "⏹️  Press Ctrl+C to stop"
	@echo ""
	uv run server.py

test: ## Test database connection and search functionality
	@echo "🧪 Testing database and search..."
	@uv run python -c "from server import ticket_db; tickets = ticket_db.search_tickets(priority='critical', status='open'); print(f'✅ Found {len(tickets)} critical open tickets'); [print(f\"  - [{t['ticket_id']}] {t['title']}\") for t in tickets[:3]]"

regenerate-data: ## Regenerate sample ticket database
	@echo "🔄 Regenerating sample data..."
	@rm -f data/requests.xls
	@uv run python -c "from sample_data import get_sample_data; import pandas as pd; pd.DataFrame(get_sample_data()).to_excel('data/requests.xls', index=False)"
	@echo "✅ Sample data regenerated with 50 tickets!"

clean: ## Clean cache and temporary files
	@echo "🧹 Cleaning cache and temporary files..."
	@rm -rf __pycache__
	@rm -rf .pytest_cache
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ Cleanup complete!"

info: ## Show server information
	@echo "📊 HTTP MCP Server Information"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  Name:          tickets-http"
	@echo "  Transport:     streamable-http"
	@echo "  Port:          8000"
	@echo "  URL:           http://localhost:8000/mcp"
	@echo "  Database:      data/requests.xls"
	@echo ""
	@if [ -f data/requests.xls ]; then \
		echo "  📊 Database status: ✅ Ready"; \
		uv run python -c "import pandas as pd; df = pd.read_excel('data/requests.xls'); print(f'  📈 Total tickets:   {len(df)}')"; \
	else \
		echo "  📊 Database status: ❌ Not found (run 'make regenerate-data')"; \
	fi
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

