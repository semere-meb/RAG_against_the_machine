from pathlib import Path

from bm25q import BM25
from pydantic import ValidationError
from pydantic_core import PydanticSerializationError
from tqdm import tqdm

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
        if not query_tokens:
            return MinimalSearchResults(question=query, retrieved_sources=[])
        k = min(k, len(self.retriever.corpus))
        docs, _scores = self.retriever.retrieve([query_tokens], k=k)
        sources = [MinimalSource.model_validate(d) for d in docs[0]]

        return MinimalSearchResults(question=query, retrieved_sources=sources)

    def search_dataset(
        self, dataset_path: str, k: int, save_directory: str
    ) -> StudentSearchResults:
        input_path = Path(dataset_path)
        try:
            with open(input_path) as f:
                raw = f.read()
            dataset = RagDataset.model_validate_json(raw)
        except (OSError, ValidationError) as e:
            raise SearchError("Couldn't read the input dataset.") from e

        questions = dataset.rag_questions

        search_results = []
        for question in tqdm(questions, "Searching"):
            result = self.search(question.question, k=k)
            search_result = MinimalSearchResults(
                question_id=question.question_id,
                question=question.question,
                retrieved_sources=result.retrieved_sources,
            )
            search_results.append(search_result)

        ret = StudentSearchResults(search_results=search_results, k=k)

        try:
            out_dir = Path(save_directory)
            out_dir.mkdir(parents=True, exist_ok=True)

            with open(out_dir / input_path.name, "w") as f:
                f.write(ret.model_dump_json(indent=2))
        except (OSError, PydanticSerializationError) as e:
            raise SearchError("Couldn't save the output dataset.") from e

        return ret
