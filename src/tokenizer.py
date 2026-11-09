import re
from typing import List

import bm25q

from src.models import MinimalSource


class Tokenizer:
    def __init__(self, lowercase: bool = True) -> None:
        self.lowercase = lowercase

    def _split_identifier(self, identifier: str) -> List[str]:
        clean_id = identifier.strip("-_./\\")
        if not clean_id:
            return []

        tokens = set()

        full_token = clean_id.lower() if self.lowercase else clean_id
        tokens.add(full_token)

        sub_parts = re.split(r"[_./\-]+", clean_id)
        for part in sub_parts:
            if not part:
                continue

            camel_split = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", part)
            camel_split = re.sub(
                r"([A-Z]+)([A-Z][a-z])", r"\1 \2", camel_split
            )

            for sub in camel_split.split():
                if len(sub) > 1:
                    token = sub.lower() if self.lowercase else sub
                    tokens.add(token)

        return list(tokens)

    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []

        raw_tokens = re.findall(r"[a-zA-Z0-9_\-]+", text)

        all_tokens = []
        for raw in raw_tokens:
            sub_tokens = self._split_identifier(raw)
            all_tokens.extend(sub_tokens)

        return all_tokens

    def index(self, chunks: list[MinimalSource]) -> None:

        corpus = [chunk.text for chunk in chunks]
        corpus_tokens = [self.tokenize(text) for text in corpus]

        ret = bm25q.BM25(corpus=corpus, backend="numba")
        ret.index(corpus_tokens)

        ret.save(save_dir="data/processed", corpus=corpus, allow_pickle=True)
