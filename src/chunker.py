from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)

from src.loader import File
from src.models import MinimalSource


class Chunker:
    def __init__(self, max_chunk_size: int = 2000) -> None:
        self.text_separators: list[str] = [
            # r"\n(?=#{1,6}[ \t])",
            r"\n[ \t]*\n",
            r"\n",
            r"(?<=[.!?])\s+",
            r"\s+",
        ]

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=max_chunk_size // 10,
            separators=self.text_separators,
            is_separator_regex=True,
            add_start_index=True,
        )

        self.markdown_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.MARKDOWN,
            chunk_overlap=max_chunk_size // 10,
            add_start_index=True,
        )

        self.python_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_overlap=max_chunk_size // 10,
            add_start_index=True,
        )

    def chunk_file(
        self, file: File, max_chunk_size: int = 2000
    ) -> list[MinimalSource]:
        text = file.content
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be a positive integer")
        if not text:
            return []

        chunks: list[MinimalSource] = []

        if file.suffix in (".py", ".pyi", ".pyd"):
            splitter = self.python_splitter
        elif file.suffix in (".md"):
            splitter = self.markdown_splitter
        elif file.suffix in (".txt"):
            splitter = self.text_splitter

        docs = splitter.create_documents([text])

        for doc in docs:
            content = doc.page_content
            start = doc.metadata["start_index"]
            end = start + len(content)
            # print(len(content))
            assert text[start:end] == doc.page_content
            chunks.append(
                MinimalSource(
                    file_path=str(file.path),
                    first_character_index=start,
                    last_character_index=end,
                )
            )

        return chunks
