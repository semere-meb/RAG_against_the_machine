from pathlib import Path

from bm25q import BM25

from src.index.chunker import Chunker
from src.index.tokenizer import Tokenizer
from src.loader import RAW_DIR, Loader

PATH_WEIGHT: int = 2


class Indexer:
    """Builds the BM25 index of the corpus."""

    def __init__(self) -> None:
        """Create the indexer with its tokenizer."""
        self.tokenizer = Tokenizer()

    def save(self, max_chunk_size: int) -> None:
        """Load, chunk and index the corpus, saving it to data/processed.

        Each chunk is prefixed with its file path, repeated PATH_WEIGHT
        times, so path words weigh more in the ranking.

        Args:
            max_chunk_size: Maximum chunk length in characters.
        """
        files = Loader().load()
        chunks = Chunker(max_chunk_size).chunk(files)

        corpus_tokens = [
            self.tokenizer.tokenize(
                f"{Path(chunk.file_path).relative_to(RAW_DIR)} " * PATH_WEIGHT
                + chunk.text
            )
            for chunk in chunks
        ]

        retriever = BM25(backend="numba")
        retriever.index(corpus_tokens)
        retriever.save(
            save_dir="data/processed",
            corpus=[chunk.model_dump() for chunk in chunks],
        )
