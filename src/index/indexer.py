from bm25q import BM25

from src.index.tokenizer import Tokenizer
from src.models import MinimalSource


class Indexer:
    def __init__(self) -> None:
        self.tokenizer = Tokenizer()
        self.retriever = BM25(backend="numba")

    def save(self, chunks: list[MinimalSource]) -> None:

        corpus = [chunk.text for chunk in chunks]
        corpus_tokens = [self.tokenizer.tokenize(text) for text in corpus]

        self.retriever.corpus = corpus

        self.retriever.index(corpus_tokens)

        self.retriever.save(
            save_dir="data/processed", corpus=corpus, allow_pickle=True
        )
