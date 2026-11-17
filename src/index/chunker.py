from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)

from src.loader import File
from src.models import Chunk


class Chunker:
    def __init__(self, max_chunk_size: int = 2000) -> None:
        self.max_chunk_size = max_chunk_size
        self.text_separators: list[str] = [
            r"\n[ \t]*\n",
            r"\n",
            r"(?<=[.!?])\s+",
            r"\s+",
        ]

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chunk_size,
            chunk_overlap=self.max_chunk_size // 10,
            separators=self.text_separators,
            is_separator_regex=True,
            add_start_index=True,
        )

        self.markdown_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.MARKDOWN,
            chunk_size=self.max_chunk_size,
            chunk_overlap=self.max_chunk_size // 10,
            add_start_index=True,
        )

        self.python_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=self.max_chunk_size,
            chunk_overlap=self.max_chunk_size // 10,
            add_start_index=True,
        )

    def chunk_file(self, file: File) -> list[Chunk]:
        text = file.content
        if not text:
            return []

        chunks: list[Chunk] = []

        if file.suffix in (".py", ".pyi", ".pyd"):
            splitter = self.python_splitter
        elif file.suffix in (".md",):
            splitter = self.markdown_splitter
        else:
            splitter = self.text_splitter

        docs = splitter.create_documents([text])

        for doc in docs:
            content = doc.page_content
            start = doc.metadata["start_index"]
            end = start + len(content)
            # print(len(content))
            assert text[start:end] == doc.page_content
            chunks.append(
                Chunk(
                    text=content,
                    file_path=str(file.path),
                    first_character_index=start,
                    last_character_index=end,
                )
            )

        return chunks

    def chunk(self, files: list[File]) -> list[Chunk]:
        chunks: list[Chunk] = []

        for file in files:
            chunks.extend(self.chunk_file(file))

        return chunks
