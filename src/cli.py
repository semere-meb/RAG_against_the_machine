from src.answer import Answerer
from src.index.indexer import Indexer
from src.searcher import Searcher


class CLI:
    def index(self, max_chunk_size: int = 2000) -> None:
        print(">>> Indexing")
        indexer = Indexer()
        indexer.save(max_chunk_size)

    def search(self, query: str, k: int) -> None:
        print(">>> Searching")
        query = str(query)
        searcher = Searcher()

        result = searcher.search(query, k)

        for source in result.retrieved_sources:
            print(f"{source.file_path}", end=" ")
            print(
                f"[{source.first_character_index}:{source.last_character_index}]"
            )

    def search_dataset(
        self, dataset_path: str, k: int, save_directory: str
    ) -> None:
        print(">>> Searching using the dataset")
        searcher = Searcher()
        searcher.search_dataset(dataset_path, k, save_directory)

    def answer(self, query: str, k: int) -> None:
        print(">>> Answering")
        answerer = Answerer()
        result = answerer.answer(query, k)
        print(result.answer)

    def answer_dataset(
        self, student_search_results_path: str, save_directory: str
    ) -> None:
        print(">>> answering using the dataset")
        answerer = Answerer()
        answerer.answer_dataset(student_search_results_path, save_directory)

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
    ) -> None:
        print("evaluate: not implemented.")
