PY ?= python3
PYTHON ?= $(PY)

.PHONY: gradable-check gradable-dry-run

## gradable-check: claim-number gate + gradable unit tests
gradable-check:
	$(PYTHON) scripts/check_gradable_claim_numbers.py
	$(PYTHON) -m pytest -q tests/test_gradable_behavior_metrics.py tests/test_gradable_size_steering.py

## gradable-dry-run: the four documented model-free dry-runs (read-only)
gradable-dry-run:
	PYTHON=$(PYTHON) DRY_RUN=1 DEVICE=cpu TORCH_DTYPE=float32 bash scripts/run_gradable_cross_domain_low_rank_control.sh
	PYTHON=$(PYTHON) DRY_RUN=1 DEVICE=cpu TORCH_DTYPE=float32 bash scripts/run_gradable_cross_domain_subspace_transfer.sh
	PYTHON=$(PYTHON) DRY_RUN=1 DEVICE=cpu TORCH_DTYPE=float32 bash scripts/run_gradable_cross_domain_delta_basis_transfer.sh
	PYTHON=$(PYTHON) DRY_RUN=1 DEVICE=cpu TORCH_DTYPE=float32 bash scripts/run_gradable_size_steering.sh
