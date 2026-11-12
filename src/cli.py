from src.index.indexer import Indexer
from src.searcher import Searcher


class CLI:
    def index(self, max_chunk_size: int = 2000) -> None:
        indexer = Indexer()
        indexer.save(max_chunk_size)

    def search(self, query: str, k: int) -> None:
        query = str(query)
        searcher = Searcher()

        searcher.search(query, k)

    def search_dataset(
        self,
        dataset_path: str,
        k: int,
        save_direcotry: str,
    ) -> None:
        print("search_dataset: not implemented.")

    def answer(
        self,
        query: str,
        k: int,
    ) -> None:
        print("answer: not implemented.")

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_direcotry: str,
    ) -> None:
        print("answer_dataset: not implemented.")

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
    ) -> None:
        print("evaluate: not implemented.")
