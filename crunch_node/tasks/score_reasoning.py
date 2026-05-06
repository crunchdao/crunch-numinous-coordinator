"""
ScoreReasoning — evaluates reasoning quality using OpenAI LLM.

Reads unscored reasoning from PG (reasoning_scored=false), only when
the corresponding score row already exists. Checks reasoning content:
- empty/missing → worst score (all 1s)
- valid text → OpenAI evaluation

Updates scores.reasoning_scores JSONB and flags reasoning.reasoning_scored=true.
"""

import json
from pathlib import Path

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from neurons.validator.models.reasoning import MISSING_REASONING_PREFIX
from neurons.validator.scheduler.task import AbstractTask
from neurons.validator.utils.logger.logger import NuminousLogger

from crunch_node.clients.pg_client import PgClient

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "reasoning_eval.md"

_SYSTEM_SUFFIX = """

Score each of the 5 criteria above from 1 to 5.
"""

WORST_SCORE = {"sources": 1, "evidence": 1, "weighting": 1, "uncertainties": 1, "mapping": 1}


class ReasoningScores(BaseModel):
    sources: int = Field(ge=1, le=5)
    evidence: int = Field(ge=1, le=5)
    weighting: int = Field(ge=1, le=5)
    uncertainties: int = Field(ge=1, le=5)
    mapping: int = Field(ge=1, le=5)


def _load_system_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8") + _SYSTEM_SUFFIX


_ERROR_MARKERS = ("Runtime Error:", "HTTP Error:", "Error:", "Traceback")


def _is_empty_reasoning(reasoning: str | None) -> bool:
    if not reasoning or reasoning.startswith(MISSING_REASONING_PREFIX):
        return True
    if any(reasoning.startswith(marker) for marker in _ERROR_MARKERS):
        return True
    return False


class ScoreReasoning(AbstractTask):
    def __init__(
        self,
        interval_seconds: float,
        pg_client: PgClient,
        openai_api_key: str,
        openai_model: str,
        logger: NuminousLogger,
        batch_size: int = 50,
    ):
        self.interval = interval_seconds
        self.pg_client = pg_client
        self.openai_client = AsyncOpenAI(api_key=openai_api_key)
        self.openai_model = openai_model
        self.logger = logger
        self.batch_size = batch_size
        self.system_prompt = _load_system_prompt()

    @property
    def name(self) -> str:
        return "score-reasoning"

    @property
    def interval_seconds(self) -> float:
        return self.interval

    async def run(self) -> None:
        # Only fetch reasoning where the score row already exists (avoids timing issues)
        # The reasoning is exported with the unique event ID, despite the fact that the scoring is exported with the event ID. Numinous is aware of this, and for now, we have one unique event for each event.
        total_scored = 0
        total_empty = 0
        total_failed = 0
        batch_num = 0

        while True:
            rows = await self.pg_client.fetch(
                """
                SELECT r.run_id,
                       r.unique_event_id,
                       r.miner_uid,
                       r.track,
                       r.reasoning
                FROM reasoning r
                         JOIN scores s
                              ON s.event_id = substr(r.unique_event_id, 9)
                                  AND s.miner_uid = r.miner_uid
                                  AND s.track = r.track
                WHERE r.reasoning_scored = false
                    LIMIT $1
                """,
                self.batch_size,
            )

            if not rows:
                break

            batch_num += 1
            self.logger.info(
                "Scoring reasoning batch",
                extra={"batch": batch_num, "size": len(rows)},
            )

            for row in rows:
                run_id = row["run_id"]
                event_id = row["unique_event_id"].removeprefix("ifgames-")
                miner_uid = row["miner_uid"]
                track = row["track"]

                try:
                    if _is_empty_reasoning(row["reasoning"]):
                        scores = WORST_SCORE
                        total_empty += 1
                    else:
                        scores = await self._evaluate(row["reasoning"])
                        if scores is None:
                            total_failed += 1
                            continue

                    scores_json = json.dumps(scores)

                    await self.pg_client.execute(
                        """
                        UPDATE scores
                        SET reasoning_scores = $1::jsonb
                        WHERE event_id = $2 AND miner_uid = $3 AND track = $4
                        """,
                        scores_json, event_id, miner_uid, track,
                    )

                    await self.pg_client.execute(
                        "UPDATE reasoning SET reasoning_scored = true WHERE run_id = $1",
                        run_id,
                    )

                    total_scored += 1

                except Exception:
                    total_failed += 1
                    self.logger.exception(
                        "Failed to score reasoning",
                        extra={"run_id": run_id, "miner_uid": miner_uid},
                    )

            self.logger.info(
                "Batch completed",
                extra={"batch": batch_num, "scored": total_scored, "empty": total_empty, "failed": total_failed},
            )

            if len(rows) < self.batch_size:
                break

        if total_scored > 0 or total_empty > 0 or total_failed > 0:
            self.logger.info(
                "Reasoning scoring finished",
                extra={"batches": batch_num, "scored": total_scored, "empty": total_empty, "failed": total_failed},
            )
        else:
            self.logger.debug("No reasoning to score")

    async def _evaluate(self, reasoning: str) -> dict | None:
        """Call OpenAI to evaluate reasoning. Returns dict with 5 integer scores or None."""
        try:
            response = await self.openai_client.beta.chat.completions.parse(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": reasoning},
                ],
                response_format=ReasoningScores,
                temperature=0.0,
            )

            parsed = response.choices[0].message.parsed
            if parsed is None:
                self.logger.warning("OpenAI returned no parsed response")
                return None

            return parsed.model_dump()

        except Exception:
            self.logger.exception("OpenAI evaluation failed")
            return None
