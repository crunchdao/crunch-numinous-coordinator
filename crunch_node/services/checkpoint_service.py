"""
CheckpointService — creates weekly reward checkpoints from leaderboard data.

Uses exponential reward distribution: weight = exp(alpha / rank).
Called after each leaderboard computation in the scoring worker.
"""

import json
import math
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

from crunch_node.clients.pg_client import PgClient

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL_SECONDS = 7 * 24 * 3600  # 1 week


class CheckpointService:

    def __init__(
        self,
        pg_client: PgClient,
        reward_pool: float = 500,
        top_k: int = 10,
        alpha: float = 1.0,
        interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
        start_date: Optional[datetime] = None,
        benchmark_miner_uid: Optional[int] = None,
    ):
        self.pg_client = pg_client
        self.reward_pool = reward_pool
        self.top_k = top_k
        self.alpha = alpha
        self.interval_seconds = interval_seconds
        self.start_date = start_date
        self.benchmark_miner_uid = benchmark_miner_uid
        self._last_period_end: Optional[datetime] = None

    async def init(self) -> None:
        """Load last checkpoint from DB."""
        row = await self.pg_client.fetchrow(
            "SELECT period_end FROM checkpoints ORDER BY created_at DESC LIMIT 1"
        )
        if row:
            self._last_period_end = row["period_end"]
            logger.info("Loaded last checkpoint: period_end=%s", self._last_period_end.isoformat())
        else:
            logger.info("No previous checkpoint found, start_date=%s", self.start_date.isoformat() if self.start_date else "None")

    async def maybe_create_checkpoint(self) -> None:
        """Create a checkpoint if a full period has elapsed."""
        period = self._next_period()
        if period is None:
            logger.debug("Checkpoint not due yet")
            return

        logger.info("Checkpoint period due, creating...")

        period_start, period_end = period
        checkpoint_id = f"CKP_{period_end.strftime('%Y%m%d_%H%M%S')}"

        rows = await self.pg_client.fetch(
            """
            SELECT l.miner_uid, l.track, l.rank, l.weighted_score,
                   ms.model_name, ms.cruncher_id, ms.cruncher_name, ms.deployment_id
            FROM leaderboard l
            LEFT JOIN model_scores ms ON ms.miner_uid = l.miner_uid
            WHERE l.weighted_score IS NOT NULL AND l.track = 'SIGNAL'
            ORDER BY l.rank
            """
        )

        if not rows:
            logger.debug("No leaderboard data for checkpoint")
            return

        reward_entries = self._compute_rewards(rows)
        total_distributed = sum(e["reward_amount"] for e in reward_entries)
        paid_count = sum(1 for e in reward_entries if e["reward_amount"] > 0)

        await self.pg_client.execute(
            """
            INSERT INTO checkpoints (id, period_start, period_end, status, reward_entries, meta, created_at)
            VALUES ($1, $2, $3, $4, $5::jsonb, $6::jsonb, $7)
            ON CONFLICT (id) DO NOTHING
            """,
            checkpoint_id,
            period_start,
            period_end,
            "PENDING",
            json.dumps(reward_entries),
            json.dumps({
                "reward_pool": self.reward_pool,
                "top_k": self.top_k,
                "alpha": self.alpha,
                "paid_count": paid_count,
                "total_distributed": total_distributed,
            }),
            datetime.now(timezone.utc),
        )

        self._last_period_end = period_end

        logger.info(
            "Created checkpoint %s: %d paid, $%.2f distributed (pool: $%.2f), period %s -> %s",
            checkpoint_id, paid_count, total_distributed, self.reward_pool,
            period_start.isoformat(), period_end.isoformat(),
        )

    def _next_period(self) -> Optional[tuple[datetime, datetime]]:
        now = datetime.now(timezone.utc)
        interval = timedelta(seconds=self.interval_seconds)

        if self._last_period_end:
            period_start = self._last_period_end
            logger.debug("Last checkpoint period_end: %s", period_start.isoformat())
        elif self.start_date:
            period_start = self.start_date
            logger.debug("No previous checkpoint, using start_date: %s", period_start.isoformat())
        else:
            period_start = now - interval
            logger.debug("No start_date, using now - interval: %s", period_start.isoformat())

        period_end = period_start + interval
        logger.debug("Next period_end: %s, now: %s, due: %s",
                      period_end.isoformat(), now.isoformat(), period_end <= now)

        if period_end > now:
            return None

        return period_start, period_end

    def _compute_rewards(self, rows) -> list[dict]:
        entries = [dict(r) for r in rows]
        total_weight = sum(math.exp(self.alpha / r) for r in range(1, self.top_k + 1))

        # Step 1: keep best miner per cruncher, rerank
        seen_crunchers = set()
        deduped = []
        for e in entries:
            cid = e.get("cruncher_id")
            if cid and cid in seen_crunchers:
                continue
            if cid:
                seen_crunchers.add(cid)
            deduped.append(e)

        for i, e in enumerate(deduped, 1):
            e["original_rank"] = e["rank"]
            e["rank"] = i

        # Step 2: find benchmark score
        benchmark_score = None
        if self.benchmark_miner_uid is not None:
            for e in deduped:
                if e["miner_uid"] == self.benchmark_miner_uid:
                    benchmark_score = e["weighted_score"]
                    break

        # Step 3: top K get exponential rewards, zero if worse than benchmark
        reward_entries = []
        for entry in deduped[:self.top_k]:
            weight = math.exp(self.alpha / entry["rank"])
            reward_fraction = weight / total_weight
            reward_amount = reward_fraction * self.reward_pool

            if benchmark_score is not None and entry["weighted_score"] >= benchmark_score:
                reward_amount = 0.0

            reward_entries.append({
                "model_id": str(entry["miner_uid"]),
                "model_name": entry.get("model_name"),
                "player_id": entry.get("cruncher_id"),
                "player_name": entry.get("cruncher_name"),
                "deployment_id": entry.get("deployment_id"),
                "track": entry["track"],
                "rank": entry["rank"],
                "original_rank": entry["original_rank"],
                "weighted_score": entry["weighted_score"],
                "weight": round(weight, 6),
                "reward_fraction": round(reward_fraction, 6),
                "reward_amount": round(reward_amount, 2),
            })

        return reward_entries