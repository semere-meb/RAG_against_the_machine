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

search: install
	uv run python -m src search_dataset ./data/datasets/UnansweredQuestions/dataset_docs_public.json 10 _searched
	uv run python -m src search_dataset ./data/datasets/UnansweredQuestions/dataset_code_public.json 10 _searched

search-priv: install
	uv run python -m src search_dataset ./_private/UnansweredQuestions/dataset_docs_private.json 10 _searched
	uv run python -m src search_dataset ./_private/UnansweredQuestions/dataset_code_private.json 10 _searched

evaluate: install
	uv run python -m src evaluate ./_searched/dataset_docs_public.json data/datasets/AnsweredQuestions/dataset_docs_public.json
	uv run python -m src evaluate ./_searched/dataset_code_public.json data/datasets/AnsweredQuestions/dataset_code_public.json

evaluate-priv: install
	uv run python -m src evaluate ./_searched/dataset_docs_private.json ./_private/AnsweredQuestions/dataset_docs_private.json
	uv run python -m src evaluate ./_searched/dataset_code_private.json ./_private/AnsweredQuestions/dataset_code_private.json
	

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
