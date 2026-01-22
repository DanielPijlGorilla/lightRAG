from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import httpx
import numpy as np
import pytest
import torch 
from tenacity import RetryError, retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from qa_metrics.transformerMatcher import TransformerMatcher
from qa_metrics.pedant import PEDANT


@dataclass(frozen=True)
class EvalCase:
    question: str
    reference_answer: str
    label: str | None = None

    def display_label(self) -> str:
        return self.label or self.question[:60]


class AnswerSimilarityScorer:
    def __init__(self, references: Sequence[str] | None = None) -> None:
        self.metric = PEDANT()

    def similarity(self, prediction: str, reference: str, question: str) -> float:
        return self.metric.get_score(reference, prediction, question)

class LightragQueryClient:
    def __init__(self, base_url: str, api_key: str | None, timeout: float) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._client.close()

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception_type(httpx.HTTPError),
    )
    def query(self, question: str) -> str:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["X-API-Key"] = self._api_key
        response = self._client.post(
            f"{self._base_url}/query",
            headers=headers,
            json={
                "query": question, 
                "stream": False,
                "mode": "mix",
                "response_type": "Single concise sentence matching the length of the reference",
            },
        )
        response.raise_for_status()
        payload = response.json()
        answer = payload.get("response")
        if not isinstance(answer, str):
            raise ValueError("LightRAG /query response missing 'response'")
        return _clean_answer(answer)


@dataclass
class CaseResult:
    case: EvalCase
    answer: str
    similarity: float


@dataclass
class EvaluationReport:
    case_results: list[CaseResult]

    @property
    def average_score(self) -> float:
        if not self.case_results:
            return 0.0
        return sum(result.similarity for result in self.case_results) / len(self.case_results)

    def format_table(self) -> str:
        lines = ["label | similarity"]
        for result in self.case_results:
            lines.append(f"{result.case.display_label()} | {result.similarity:.3f}")
        lines.append(f"average | {self.average_score:.3f}")
        return "\n".join(lines)


class AnswerQualityEvaluator:
    def __init__(
        self,
        cases: Sequence[EvalCase],
        min_case_score: float,
        min_average_score: float,
        base_url: str,
        api_key: str | None,
        timeout: float,
    ) -> None:
        self._cases = list(cases)
        self._case_threshold = min_case_score
        self._avg_threshold = min_average_score
        self._client = LightragQueryClient(base_url, api_key, timeout)
        self._scorer = AnswerSimilarityScorer([case.reference_answer for case in self._cases])

    def evaluate(self) -> EvaluationReport:
        results: list[CaseResult] = []
        try:
            for case in self._cases:
                answer = self._client.query(case.question)
                similarity = self._scorer.similarity(answer, case.reference_answer, case.question)
                print(f"\n[{case.display_label()}] question: {case.question}")
                print(f"[{case.display_label()}] golden: {case.reference_answer}")
                print(f"[{case.display_label()}] lightrag: {answer}")
                print(f"[{case.display_label()}] similarity: {similarity:.3f}")
                results.append(CaseResult(case=case, answer=answer, similarity=similarity))
        finally:
            self._client.close()
        return EvaluationReport(results)

    @property
    def case_threshold(self) -> float:
        return self._case_threshold

    @property
    def average_threshold(self) -> float:
        return self._avg_threshold


def _clean_answer(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"\n?\s*#{2,}\s*References\b[\s\S]*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*\*\*?Answer:?\*\*?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*#{1,6}\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*[-*]\s+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\[[0-9]+\]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _load_cases() -> list[EvalCase]:
    override_path = os.getenv("LIGHTRAG_EVAL_SET")
    if override_path:
        dataset_path = Path(override_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Evaluation set not found: {dataset_path}")
        data = json.loads(dataset_path.read_text(encoding="utf-8"))
        return [EvalCase(item["question"], item.get("reference_answer") or item["answer"], item.get("label")) for item in data]
    
    default_data_path = Path(__file__).parent / "questions.json"
    if default_data_path.exists():
        data = json.loads(default_data_path.read_text(encoding="utf-8"))
        return [EvalCase(item["question"], item.get("reference_answer") or item["answer"], item.get("label")) for item in data]

    return None


@pytest.mark.offline
def test_lightrag_answer_similarity() -> None:
    base_url = os.getenv("LIGHTRAG_API_BASE_URL", "http://localhost:9621")
    api_key = os.getenv("LIGHTRAG_API_KEY")
    
    min_case_score = 0.60
    min_average_score = 0.80
    
    timeout = float(os.getenv("LIGHTRAG_QUERY_TIMEOUT", "60"))

    cases = _load_cases()
    evaluator = AnswerQualityEvaluator(
        cases=cases,
        min_case_score=min_case_score,
        min_average_score=min_average_score,
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
    )
    
    try:
        report = evaluator.evaluate()
    except RetryError as exc:
        pytest.fail(f"LightRAG retries exhausted: {exc}")

    print("\nSimilarity summary:")
    print(report.format_table())

    failing = [res for res in report.case_results if res.similarity < evaluator.case_threshold]
    if failing:
        details = ", ".join(f"{res.case.display_label()}={res.similarity:.3f}" for res in failing)
        pytest.fail(
            f"Answers below threshold {evaluator.case_threshold:.2f}: "
            f"{details}\n{report.format_table()}"
        )

    avg_score = report.average_score
    assert avg_score >= evaluator.average_threshold, (
        f"Average similarity {avg_score:.3f} below minimum {evaluator.average_threshold:.2f}\n"
        f"{report.format_table()}"
    )
 