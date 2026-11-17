from pathlib import Path

import torch
from pydantic import ValidationError
from pydantic_core import PydanticSerializationError
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models import (
    MinimalAnswer,
    MinimalSource,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from src.searcher import Searcher

TOP_SOURCES_COUNT = 3
SYSTEM_PROMPT = (
    "You answer questions about the vLLM codebase using only the "
    "provided sources. Quote names, commands, parameters and values "
    "exactly as they appear. Answer concisely. Include the exact command "
    "or code from the sources when the question asks how to do something. "
    "If the question is not about vLLM or software, reply exactly: "
    '"The provided sources do not contain this information."'
)


class AnswerError(Exception):
    """"""


class Answerer:
    def __init__(self) -> None:
        model_name = "Qwen/Qwen3-0.6B"
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32 if not torch.cuda.is_available() else "auto",
            device_map="auto",
        )

    def build_prompt(self, query: str, sources: list[tuple[str, str]]) -> str:
        blocks = [
            f"--- Source {i}: {path} ---\n{text.strip()}"
            for i, (path, text) in enumerate(sources, 1)
        ]
        return "\n\n".join(blocks) + f"\n\nQuestion: {query}\nAnswer:"

    def generate(self, query: str, sources: list[MinimalSource]) -> str:
        context = []

        for source in sources[:TOP_SOURCES_COUNT]:
            with open(source.file_path) as f:
                text = f.read()[
                    slice(
                        source.first_character_index,
                        source.last_character_index,
                    )
                ]
                context.append((source.file_path, text))

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self.build_prompt(query, context)},
        ]

        if not self.tokenizer:
            raise AnswerError("Error loading tokenizer.")

        chat = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if not isinstance(chat, str):
            raise AnswerError("Chat template did not produce a string.")

        inputs = self.tokenizer([chat], return_tensors="pt").to(
            self.model.device
        )

        output = self.model.generate(  # type: ignore[misc]
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )  # ty: ignore[invalid-argument-type]

        new_tokens = output[0][slice(inputs.input_ids.shape[1])]
        llm_answer = self.tokenizer.decode(
            new_tokens, skip_special_tokens=True
        )
        return str(llm_answer)

    def answer(self, query: str, k: int) -> MinimalAnswer:
        result = Searcher().search(query, k)
        return MinimalAnswer(
            question=result.question,
            question_id=result.question_id,
            retrieved_sources=result.retrieved_sources,
            answer=self.generate(result.question, result.retrieved_sources),
        )

    def answer_dataset(
        self, student_search_results_path: str, save_directory: str
    ) -> StudentSearchResultsAndAnswer:
        input_path = Path(student_search_results_path)
        try:
            with open(input_path) as f:
                raw = f.read()
            dataset = StudentSearchResults.model_validate_json(raw)
        except (OSError, ValidationError) as e:
            raise AnswerError("Couldn't read the input dataset.") from e

        s_results, k = dataset.search_results, dataset.k

        search_results = []
        for s_result in tqdm(s_results, "Answering"):
            search_answer = MinimalAnswer(
                question_id=s_result.question_id,
                question=s_result.question,
                retrieved_sources=s_result.retrieved_sources,
                answer=self.generate(
                    s_result.question, s_result.retrieved_sources
                ),
            )
            search_results.append(search_answer)

        ret = StudentSearchResultsAndAnswer(
            search_results=search_results,
            k=k,
        )

        try:
            out_dir = Path(save_directory)
            out_dir.mkdir(parents=True, exist_ok=True)

            with open(out_dir / input_path.name, "w") as f:
                f.write(ret.model_dump_json(indent=2))
        except (OSError, PydanticSerializationError) as e:
            raise AnswerError("Couldn't save the output dataset.") from e

        return ret
