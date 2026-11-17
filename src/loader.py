from itertools import chain
from pathlib import Path

from pydantic import BaseModel

RAW_DIR: str = "data/raw/vllm-0.10.1/"
DIRS = {
    "code": {
        "extensions": ["py"],
        "dirs": ["vllm"],
    },
    "prose": {
        "extensions": ["md", "txt"],
        "dirs": [""],
    },
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


class File(BaseModel):
    path: Path
    content: str
    suffix: str


class Loader:
    def get_all_paths(self) -> list[Path]:
        """
        Walks through the entire repo and returns paths of files.
        """
        raw_dir: Path = Path(RAW_DIR)

        if not (raw_dir.exists() and raw_dir.is_dir()):
            raise LoadError("raw directory not found")

        paths = []

        for type in DIRS:
            extensions = DIRS[type]["extensions"]
            sub_dirs = DIRS[type]["dirs"]

            for sub in sub_dirs:
                sub_dir = raw_dir / sub
                files = list(
                    chain.from_iterable(
                        sub_dir.rglob(f"*.{extension}")
                        for extension in extensions
                    )
                )
                paths += files

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
                continue

            filtered_paths.append(path)

        paths.sort()
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
                    suffix = path.suffix
                    file = File(path=path, content=content, suffix=suffix)
                    files.append(file)
            except OSError:
                print(f"Warning: Error during reading {path}, skipped.")

        return files

    def load(self) -> list[File]:
        paths = self.get_all_paths()
        filter_paths = self.filter_sort_paths(paths)
        files = self.build_file_list(filter_paths)

        return files
