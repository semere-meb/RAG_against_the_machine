import json
from pathlib import Path

from src.loader import Loader


def test_indexer_loader_and_offsets() -> None:
    """
    Test that Loader discovers files, formats paths correctly,
    and that dataset reference slices can be correctly indexed.
    """
    indexer = Loader()
    paths = indexer.get_all_paths()
    filtered_paths = indexer.filter_sort_paths(paths)
    files = indexer.build_file_list(filtered_paths)

    assert len(files) > 0

    for file_obj in files:
        rel_path = file_obj.path.relative_to(Path.cwd()).as_posix()
        assert rel_path.startswith("data/raw/")
        assert "\\" not in rel_path
        assert isinstance(file_obj.content, str)

    dataset_path = Path(
        "data/datasets/AnsweredQuestions/dataset_docs_public.json"
    )
    if dataset_path.exists():
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if data.get("rag_questions"):
            q = data["rag_questions"][0]
            if q.get("sources"):
                src = q["sources"][0]
                f_path = src["file_path"]
                start = src["first_character_index"]
                end = src["last_character_index"]

                if Path(f_path).exists():
                    with open(f_path, "r", encoding="utf-8", newline="") as sf:
                        content = sf.read()
                    slice_text = content[start:end]
                    assert isinstance(slice_text, str)
                    assert len(slice_text) == (end - start)
