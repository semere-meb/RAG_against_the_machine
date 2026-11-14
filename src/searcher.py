import json
from pathlib import Path

from bm25q import BM25
from pydantic import ValidationError

from src.index.tokenizer import Tokenizer
from src.models import (
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)


class SearchError(Exception):
    """"""


class Searcher:
    def __init__(self) -> None:
        self.retriever: BM25 | None = None
        self.tokenizer = Tokenizer()

    def load(self) -> BM25:
        return BM25().load("data/processed", load_corpus=True)

    def search(self, query: str, k: int) -> MinimalSearchResults:
        self.retriever = self.load()

        query_tokens = self.tokenizer.tokenize(query)
        docs, scores = self.retriever.retrieve([query_tokens], k=k)
        sources = [MinimalSource.model_validate(d) for d in docs[0]]

        return MinimalSearchResults(question=query, retrieved_sources=sources)

    def search_dataset(
        self, dataset_path: str, k: int, save_direcotry: str
    ) -> StudentSearchResults:
        search_results = []

        try:
            with open(dataset_path) as f:
                raw = json.load(f)
            dataset = RagDataset.model_validate(raw)
        except (OSError, ValidationError) as e:
            raise SearchError("Error reading the input dataset.") from e

        questions = dataset.rag_questions

        for question in questions:
            result = self.search(question.question, k=k)
            search_results.append(result)

        ret = StudentSearchResults(
            search_results=search_results, k=len(dataset.rag_questions)
        )

        try:
            out_dir = Path(save_direcotry)
            out_dir.mkdir(parents=True, exist_ok=True)

            with open(out_dir / "search_dataset.json", "w") as f:
                f.write(ret.model_dump_json(indent=2))
        except OSError as e:
            raise SearchError("Couldn't save the output dataset.") from e

        return ret
