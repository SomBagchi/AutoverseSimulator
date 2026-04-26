# --- Tier-1 inputs ---
# Update MEASUREMENTS when a new sweep lands.
MEASUREMENTS ?= measurements/h100_sxm/run_20260426_233235.json
FIT_OUT      ?= reports/calibration_fit.json
PLOT_OUT     ?= reports/figures/measured_vs_predicted.png

.PHONY: help install test lint format measure calibrate validate whatif clean

help:
	@echo "Autoverse Simulator — commands:"
	@echo "  install     uv sync + install package"
	@echo "  test        run pytest"
	@echo "  lint        ruff check + mypy"
	@echo "  format      ruff format"
	@echo "  measure     Tier 1: collect H100 measurements (CUDA + --extra measure required)"
	@echo "  calibrate   Tier 1: fit (F, B, O) on the committed measurement JSON"
	@echo "  validate    Tier 1: calibrate + regenerate residual plot"
	@echo "  whatif      Tier 3+: run counterfactual experiments"
	@echo "  clean       remove build / cache artifacts"

install:
	uv sync

test:
	uv run pytest -q

lint:
	uv run ruff check src/ tests/ scripts/
	uv run mypy src/

format:
	uv run ruff format src/ tests/ scripts/

measure:
	@if [ -z "$$MEASUREMENT_OUT" ]; then \
		echo "Set MEASUREMENT_OUT=measurements/h100_sxm/run_<tag>.json"; exit 2; \
	fi
	uv run python scripts/collect_measurements.py --device cuda --out $$MEASUREMENT_OUT

calibrate:
	uv run python scripts/calibrate.py $(MEASUREMENTS) --out $(FIT_OUT)

validate: calibrate
	uv run python scripts/make_validation_plot.py \
		--measurements $(MEASUREMENTS) --fit $(FIT_OUT) --out $(PLOT_OUT)
	@echo ""
	@echo "Validation artifacts:"
	@echo "  fit summary : $(FIT_OUT)"
	@echo "  scatter plot: $(PLOT_OUT)"
	@echo "  report      : reports/01_validation.md"

whatif:
	@echo "TODO: implemented at Tier 3 (Day 4)"

clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .mypy_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
