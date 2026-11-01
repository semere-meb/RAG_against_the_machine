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
        print(len(files))

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
