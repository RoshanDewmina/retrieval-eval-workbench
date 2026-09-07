PYTHON := uv run python3

.PHONY: setup test demo benchmark

setup:
	uv sync --frozen

test:
	$(PYTHON) -m pytest

demo:
	$(PYTHON) -m uvicorn retrieval_eval_workbench.app:app --host $${HOST:-127.0.0.1} --port $${PORT:-8113}

benchmark:
	$(PYTHON) -m retrieval_eval_workbench.cli benchmark --receipt $${BENCHMARK_RECEIPT:-evidence/benchmarks/frozen-run.json}
