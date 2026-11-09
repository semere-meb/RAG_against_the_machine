from src.chunker import Chunker
from src.loader import Loader
from src.tokenizer import Tokenizer


class CLI:
    def index(
        self,
        max_chunk_size: int = 2000,
    ) -> None:
        files = Loader().load()
        chunks = Chunker().chunk(files, max_chunk_size)
        Tokenizer().index(chunks)

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
