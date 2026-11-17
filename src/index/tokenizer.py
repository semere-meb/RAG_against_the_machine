import re

import Stemmer  # ty: ignore[import-not-found]

STOPWORDS: frozenset[str] = frozenset(
    """
    a an the is are was were be been being of to in on at for by with from
    as and or not no do does did how what which who whom whose why when
    where this that these those it its can could should would will shall
    may might must i you he she we they me my your our their there here
    than then so if into about over under using use used vllm s
    """.split()
)


class Tokenizer:
    def __init__(self, lowercase: bool = True, stem: bool = True) -> None:
        self.lowercase = lowercase
        self.stemmer: Stemmer.Stemmer | None = (
            Stemmer.Stemmer("english") if stem else None
        )

    def _split_identifier(self, identifier: str) -> list[str]:
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

    def tokenize(self, text: str) -> list[str]:
        if not text:
            return []

        raw_tokens = re.findall(r"[a-zA-Z0-9_\-]+", text)

        all_tokens: list[str] = []
        for raw in raw_tokens:
            sub_tokens = self._split_identifier(raw)
            all_tokens.extend(t for t in sub_tokens if t not in STOPWORDS)

        if self.stemmer is not None:
            all_tokens = self.stemmer.stemWords(all_tokens)

        return all_tokens
