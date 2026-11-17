from typing import TypeVar

from pydantic import BaseModel, ValidationError

from src.models import (
    AnsweredQuestion,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)

Model = TypeVar("Model", bound=BaseModel)

IOU_TRESHOLD = 0.05


class EvaluationError(Exception):
    """Raised when an input file cannot be read."""


def load(path: str, model: type[Model]) -> Model:
    """Read a JSON file and validate it against a pydantic model.

    Args:
        path: Path to the JSON file.
        model: The model class to validate against.

    Returns:
        The validated model instance.

    Raises:
        EvaluationError: If the file cannot be read or is invalid.
    """
    try:
        with open(path) as f:
            raw = f.read()
            return model.model_validate_json(raw)
    except (OSError, ValidationError) as e:
        raise EvaluationError(f"Couldn't read {path}") from e


class Evaluator:
    """Computes recall@k of search results against a ground truth."""

    def IoU(self, a: MinimalSource, b: MinimalSource) -> float:
        """Intersection over union of two character ranges.

        Args:
            a: First source.
            b: Second source.

        Returns:
            The overlap ratio, from 0 to 1; file paths are not compared.
        """
        i_start = max(a.first_character_index, b.first_character_index)
        i_end = min(a.last_character_index, b.last_character_index)
        intersection = max(0, i_end - i_start)

        u_start = min(a.first_character_index, b.first_character_index)
        u_end = max(a.last_character_index, b.last_character_index)

        if u_end - u_start <= 0:
            return 0
        return intersection / (u_end - u_start)

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
    ) -> dict[int, float]:
        """Compute recall at k = 1, 3, 5 and 10, up to the results' k.

        A reference source counts as found when a result in the same file
        overlaps it with an IoU of at least IOU_TRESHOLD.

        Args:
            student_search_results_path: Path to the search results JSON.
            dataset_path: Path to the answered questions JSON.

        Returns:
            Recall for each k.

        Raises:
            EvaluationError: If either file cannot be read.
        """
        s_dataset = load(student_search_results_path, StudentSearchResults)
        k = s_dataset.k
        s_results = {
            q.question_id: q.retrieved_sources
            for q in s_dataset.search_results
        }

        dataset = load(dataset_path, RagDataset)

        d_results = {
            q.question_id: q.sources
            for q in dataset.rag_questions
            if isinstance(q, AnsweredQuestion) and q.sources
        }

        if not d_results:
            raise EvaluationError(
                f"{dataset_path} has no questions with reference sources"
            )

        ks = [_k for _k in (1, 3, 5, 10) if _k <= k]
        totals = {_k: 0.0 for _k in ks}

        for q_id, d_sources in d_results.items():
            s_sources = s_results.get(q_id, None)
            if not s_sources:
                continue

            for _k in ks:
                top_k = s_sources[:_k]
                found = sum(
                    any(
                        s.file_path == d.file_path
                        and self.IoU(s, d) >= IOU_TRESHOLD
                        for s in top_k
                    )
                    for d in d_sources
                )
                totals[_k] += found / len(d_sources)

        return {_k: totals[_k] / len(d_results) for _k in ks}
