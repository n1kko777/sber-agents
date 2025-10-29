.PHONY: install run

install:
	cd bot && uv sync

run:
	@set -a; \
	if [ -f .env ]; then . .env; fi; \
	set +a; \
	test -n "$${TELEGRAM_BOT_TOKEN:-}" || { echo "TELEGRAM_BOT_TOKEN is required"; exit 1; }; \
	test -n "$${OPENROUTER_API_KEY:-}" || { echo "OPENROUTER_API_KEY is required"; exit 1; }; \
	cd bot && uv run python -m bot.main
