.PHONY: install run dataset dataset-upload

install:
	uv sync

run:
	uv run python src/bot.py

dataset:
	uv run python src/dataset_synthesizer.py --create

dataset-upload:
	uv run python src/dataset_synthesizer.py --upload

