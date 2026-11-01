SYNC := .synced
MYPY_FLAGS := "--warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs"

install: $(SYNC)

$(SYNC): pyproject.toml
	git config core.hooksPath .githooks
	@if ! command -v uv >/dev/null;then \
		pip install uv; \
	fi
	uv sync
	@if ! [ -d data/raw/vllm-0.10.1 ];then \
		git clone --depth 1 --branch v0.10.1 https://github.com/vllm-project/vllm.git data/raw/vllm-0.10.1; \
	fi
	@touch $(SYNC)

index: install
	uv run python -m src index


run: index
	
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache .ruff_cache
	rm -rf dist/
	rm -rf $(SYNC)

fclean: clean
	rm -rf .venv $(SYNC)
	uv clean

lint: $(SYNC)
	uv run flake8 .
	uv run mypy --strict .

lint-strict: $(SYNC)
	ruff check .
	ty check .

format:
	ruff format .
	ruff check --fix .

debug: install
	uv run python -m pdb -m src

test: $(SYNC)
	uv run pytest -q

re: clean run install

	
.PHONY: run install clean fclean lint lint-strict test format re
