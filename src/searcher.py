from bm25q import BM25

from src.index.indexer import Indexer


class Searcher:
    def __init__(self) -> None:
        self.retriever = BM25()

    def search(self, query: str, k: int) -> None:
        try:
            self.retriever.load("data/processed", load_corpus=True)
        except Exception:
            indexer = Indexer()
            indexer.save(2000)  # TODO: check
