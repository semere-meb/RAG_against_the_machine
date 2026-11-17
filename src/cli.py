from src.answer import Answerer
from src.evaluator import Evaluator
from src.index.indexer import Indexer
from src.searcher import Searcher


class CLI:
    """Command-line interface of the RAG pipeline, exposed through Fire."""

    def index(self, max_chunk_size: int = 2000) -> None:
        """Chunk the corpus in data/raw/ and save the index.

        Args:
            max_chunk_size: Maximum chunk length in characters (100-2000).
        """
        print(">>> Indexing")
        if (
            type(max_chunk_size) is not int
            or max_chunk_size < 100
            or max_chunk_size > 2000
        ):
            print("--max-chunk-size must be an integer between 100 - 2000")
            return
        indexer = Indexer()
        indexer.save(max_chunk_size)

    def search(self, query: str, k: int) -> None:
        """Print the top-k source locations for a single query.

        Args:
            query: The question to search for.
            k: Number of sources to return.
        """
        if type(k) is not int or k <= 0:
            print("k has to be a valid int type and at least 1.")
            exit(1)
        print(">>> Searching")
        query = str(query)
        searcher = Searcher()

        result = searcher.search(query, k)

        for source in result.retrieved_sources:
            print(f"{source.file_path}", end=" ")
            print(source.first_character_index, end="")
            print(":", end="")
            print(source.last_character_index)

    def search_dataset(
        self, dataset_path: str, k: int, save_directory: str
    ) -> None:
        """Search every question of a dataset and save the results.

        Args:
            dataset_path: Path to the questions JSON file.
            k: Number of sources per question.
            save_directory: Directory to write the search results to.
        """
        if type(k) is not int or k <= 0:
            print("k has to be a valid int type and at least 1.")
            exit(1)
        print(">>> Searching using the dataset")
        searcher = Searcher()
        searcher.search_dataset(dataset_path, k, save_directory)

    def answer(self, query: str, k: int) -> None:
        """Print an answer to a single query, generated from its sources.

        Args:
            query: The question to answer.
            k: Number of sources to retrieve.
        """
        if type(k) is not int or k <= 0:
            print("k has to be a valid int type and at least 1.")
            exit(1)
        print(">>> Answering")
        query = str(query)
        answerer = Answerer()
        result = answerer.answer(query, k)
        print(result.answer)

    def answer_dataset(
        self, student_search_results_path: str, save_directory: str
    ) -> None:
        """Generate answers for saved search results and save them.

        Args:
            student_search_results_path: Path to the search results JSON.
            save_directory: Directory to write the answers to.
        """
        print(">>> answering using the dataset")
        answerer = Answerer()
        answerer.answer_dataset(student_search_results_path, save_directory)

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
    ) -> None:
        """Print recall@k of search results against a ground truth.

        Args:
            student_search_results_path: Path to the search results JSON.
            dataset_path: Path to the answered questions JSON.
        """
        print(">>> Evaluating")
        evaluator = Evaluator()
        res = evaluator.evaluate(student_search_results_path, dataset_path)
        for k in res:
            print(f"recall@{k}: {res[k]:.2f}")
