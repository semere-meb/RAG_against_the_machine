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
    """Raised when a dataset cannot be read or results cannot be saved."""


class Searcher:
    """Retrieves the most relevant chunks for a question with BM25."""

    def __init__(self) -> None:
        """Create the searcher; the index is loaded on search."""
        self.retriever: BM25 | None = None
        self.tokenizer = Tokenizer()

    def load(self) -> BM25:
        """Load the index and chunk metadata from data/processed.

        Returns:
            The BM25 retriever.
        """
        return BM25().load("data/processed", load_corpus=True)

    def search(self, query: str, k: int) -> MinimalSearchResults:
        """Find the top-k chunks for a query.

        Args:
            query: The question to search for.
            k: Number of sources to return, capped at the corpus size.

        Returns:
            The ranked sources; empty if the query has no usable tokens.
        """
        self.retriever = self.load()

        query_tokens = self.tokenizer.tokenize(query)
        if not query_tokens:
            return MinimalSearchResults(question=query, retrieved_sources=[])
        k = min(k, len(self.retriever.corpus))
        docs, scores = self.retriever.retrieve([query_tokens], k=k)
        sources = [
            MinimalSource.model_validate(d)
            for d, score in zip(docs[0], scores[0])
            if score > 0
        ]

        return MinimalSearchResults(question=query, retrieved_sources=sources)

    def search_dataset(
        self, dataset_path: str, k: int, save_directory: str
    ) -> StudentSearchResults:
        """Search every question of a dataset and save the results.

        The output file has the same name as the input dataset.

        Args:
            dataset_path: Path to the questions JSON file.
            k: Number of sources per question.
            save_directory: Directory to write the results to.

        Returns:
            The search results.

        Raises:
            SearchError: If the dataset cannot be read or saved.
        """
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
