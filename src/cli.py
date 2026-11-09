from src.chunker import Chunker
from src.loader import Loader


class CLI:
    def index(
        self,
        max_chunk_size: int = 2000,
    ) -> None:
        indexer = Loader()
        paths = indexer.get_all_paths()
        filter_paths = indexer.filter_sort_paths(paths)
        files = indexer.build_file_list(filter_paths)

        file = [file for file in files if file.suffix == ".md"][10]

        chunker = Chunker(max_chunk_size=2000)
        for c in chunker.chunk_file(file):
            print(c.first_character_index, c.last_character_index)

    def search(
        self,
        query: str,
        k: int,
    ) -> None:
        query = str(query)
        print("search: not implemented.")

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
