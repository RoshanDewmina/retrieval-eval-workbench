FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOST=0.0.0.0 PORT=8113 UV_TORCH_BACKEND=cpu
COPY --from=ghcr.io/astral-sh/uv:0.6.5 /uv /uvx /bin/
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN uv sync --frozen --no-dev
COPY data ./data
COPY evidence ./evidence
EXPOSE 8113
CMD ["sh", "-c", "/bin/uv run uvicorn retrieval_eval_workbench.app:app --host \"${HOST}\" --port \"${PORT}\""]
