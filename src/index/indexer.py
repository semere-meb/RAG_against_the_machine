from bm25q import BM25

from src.index.chunker import Chunker
from src.index.tokenizer import Tokenizer
from src.loader import Loader


class Indexer:
    def __init__(self) -> None:
        self.tokenizer = Tokenizer()

    def save(self, max_chunk_size: int) -> None:
        files = Loader().load()

        chunks = Chunker().chunk(files, max_chunk_size)

        corpus_tokens = [
            self.tokenizer.tokenize(chunk.text) for chunk in chunks
        ]

        retriever = BM25(backend="numba", corpus=chunks)

        retriever.index(corpus_tokens)

        retriever.save(
            save_dir="data/processed", corpus=chunks, allow_pickle=True
        )
