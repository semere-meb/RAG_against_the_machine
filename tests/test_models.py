import json
from pathlib import Path

from src.models import AnsweredQuestion, RagDataset, UnansweredQuestion


def test_load_public_datasets() -> None:
    """
    Test loading both answered and unanswered public datasets (docs and code)
    via RagDataset to verify Pydantic validation and serialization
    round-trips.
    """
    dataset_paths = [
        "data/datasets/AnsweredQuestions/dataset_docs_public.json",
        "data/datasets/AnsweredQuestions/dataset_code_public.json",
        "data/datasets/UnansweredQuestions/dataset_docs_public.json",
        "data/datasets/UnansweredQuestions/dataset_code_public.json",
    ]

    for path_str in dataset_paths:
        path = Path(path_str)
        assert path.exists(), f"Dataset file missing: {path}"

        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        # Validate via RagDataset
        dataset = RagDataset.model_validate(raw_data)
        assert isinstance(dataset, RagDataset)
        assert len(dataset.rag_questions) > 0

        # Check types of questions
        for q in dataset.rag_questions:
            if "sources" in q.model_dump() and "answer" in q.model_dump():
                assert isinstance(q, AnsweredQuestion)
                assert len(q.sources) > 0
                assert isinstance(q.answer, str)
            else:
                assert isinstance(q, UnansweredQuestion)

        # Test JSON round-trip serialization/deserialization
        json_str = dataset.model_dump_json()
        round_trip_dataset = RagDataset.model_validate_json(json_str)
        assert len(round_trip_dataset.rag_questions) == len(
            dataset.rag_questions
        )
