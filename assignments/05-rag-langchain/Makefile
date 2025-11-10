.PHONY: install run

install:
	uv sync

run:
	uv run python src/bot.py

test:
	uv run python src/local_test.py --log-level INFO

