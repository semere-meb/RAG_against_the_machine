import json
import pathlib

from bm25q import BM25

from src.index.tokenizer import Tokenizer
from src.models import (
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)


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

        with open(dataset_path) as f:
            raw = json.load(f)

            dataset = RagDataset.model_validate(raw)

        for question in dataset.rag_questions:
            result = self.search(question.question, k=5)
            search_results.append(result)

        results = StudentSearchResults(
            search_results=search_results, k=len(dataset.rag_questions)
        )

        save_dir = pathlib.Path(save_direcotry)
        save_dir.mkdir(parents=True, exist_ok=True)

        with open(save_dir / "search_results.json", "w") as f:
            f.write(results.model_dump_json(indent=2))

        return results
