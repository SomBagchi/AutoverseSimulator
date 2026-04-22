.PHONY: help install test lint format validate whatif clean

help:
	@echo "Autoverse Simulator — commands:"
	@echo "  install   uv sync + install package"
	@echo "  test      run pytest"
	@echo "  lint      ruff check + mypy"
	@echo "  format    ruff format"
	@echo "  validate  Tier 1+: calibrate against real measurements"
	@echo "  whatif    Tier 3+: run counterfactual experiments"
	@echo "  clean     remove build / cache artifacts"

install:
	uv sync

test:
	uv run pytest -q

lint:
	uv run ruff check src/ tests/
	uv run mypy src/

format:
	uv run ruff format src/ tests/

validate:
	@echo "TODO: implemented at Tier 1 (Day 2)"

whatif:
	@echo "TODO: implemented at Tier 3 (Day 4)"

clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .mypy_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
