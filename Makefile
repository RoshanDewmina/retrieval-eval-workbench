PYTHON := uv run python

.PHONY: setup test demo benchmark

setup:
	uv sync --frozen

test:
	uv run pytest

demo:
	uv run uvicorn retrieval_eval_workbench.app:app --host $${HOST:-127.0.0.1} --port $${PORT:-8113}

benchmark:
	$(PYTHON) -m retrieval_eval_workbench.cli benchmark
