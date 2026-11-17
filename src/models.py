import uuid
from typing import List

from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """A character range in a corpus file."""

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """A question, identified by a unique id."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A question with its reference sources and answer."""

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """A dataset of answered or unanswered questions."""

    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """The sources retrieved for one question."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """The sources retrieved for one question, with the generated answer."""

    answer: str


class StudentSearchResults(BaseModel):
    """Search results for a whole dataset, with the k used."""

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Search results and answers for a whole dataset, with the k used."""

    search_results: List[MinimalAnswer]
    k: int


class Chunk(BaseModel):
    """A piece of a corpus file, with its text and character range."""

    file_path: str
    text: str
    first_character_index: int
    last_character_index: int
