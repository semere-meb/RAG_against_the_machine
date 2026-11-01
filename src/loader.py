from enum import Enum
from pathlib import Path

from pydantic import BaseModel

RAW_DIR: str = "data/raw/vllm-0.10.1/"
DIRS = {
    "py": ["vllm"],
    "md": ["", "docs"],
}
EXCLUDED_DIRS: set[str] = {
    ".eggs",
    "eggs",
    ".pytest_cache",
    ".mypy_cache",
    ".venv",
    ".git",
    "__pycache__",
    "build",
    "dist",
    "tests",
    "vllm.egg-info",
    # "third_party",  # TODO: check, filters apply
    # "examples",  # TODO: check, filters apply
    # "benchmarks",  # TODO: check, filters apply
    # "csrc",  # TODO: check, filters apply
}


class LoadError(Exception):
    """
    An Error during intake.
    """


class FileType(Enum):
    PY = ".py"
    MD = ".md"


class File(BaseModel):
    path: Path
    content: str
    type: FileType


class Loader:
    def get_all_paths(self) -> list[Path]:
        """
        Walks through the entire repo and returns paths of files.
        """
        raw_dir: Path = Path(RAW_DIR).absolute()

        if not (raw_dir.exists() and raw_dir.is_dir()):
            raise LoadError("raw directory not found")

        paths = []

        for type in DIRS:
            for sub in DIRS[type]:
                sub_dir = raw_dir / sub
                files = list(sub_dir.rglob(f"*.{type}"))
                paths += files

        paths.sort()
        return paths

    def filter_sort_paths(self, paths: list[Path]) -> list[Path]:
        """
        filters excluded files.
        """
        filtered_paths = []

        for path in paths:
            if not path.is_file():
                continue
            if EXCLUDED_DIRS.intersection(path.parts):
                print(path)
                continue

            filtered_paths.append(path)

        return filtered_paths

    def build_file_list(self, paths: list[Path]) -> list[File]:
        """
        builds a list of read files with their content and type.
        """
        files = []

        for path in paths:
            try:
                with open(path, encoding="utf-8", newline="") as f:
                    content = f.read()
                    type = FileType(path.suffix)

                    file = File(path=path, content=content, type=type)
                    files.append(file)
            except OSError:
                print(f"Warning: Error during reading {path}, skipped.")

        return files
