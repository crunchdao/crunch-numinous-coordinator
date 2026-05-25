"""
ScoringService — computes rolling averages per pool, weighted scores, and leaderboard.

All computations run against PostgreSQL. Brier scores from scores.event_score,
reasoning scores from scores.reasoning_scores JSONB.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from neurons.validator.utils.logger.logger import NuminousLogger

from crunch_node.clients.pg_client import PgClient


class TopicFilter(Enum):
    EXCLUDE_GEOPOLITICS = "exclude_geopolitics"
    GEOPOLITICS_ONLY = "geopolitics_only"


# Pool configuration (SIGNAL track only)
POOL_CONFIG = [
    {"pool": "global_brier",      "track": "SIGNAL", "emission": 0.60, "max_events": 600, "min_events": 200, "top_miner_min_events": 300},
    {"pool": "geopolitics_brier", "track": "SIGNAL", "emission": 0.25, "max_events": 200, "min_events": 100, "top_miner_min_events": None},
    {"pool": "reasoning",         "track": "SIGNAL", "emission": 0.10, "max_events": 200, "min_events": 100, "top_miner_min_events": None},
]


@dataclass
class PoolScore:
    pool: str
    track: str
    rolling_avg: float | None  # normalized [0,1], 0=best; None if < min_events
    event_count: int
    emission: float


@dataclass
class RollingScore:
    avg: float
    count: int


@dataclass
class LeaderboardEntry:
    miner_uid: int
    weighted_score: float | None
    event_count: int
    global_brier: float | None
    global_brier_count: int
    geopolitics_brier: float | None
    geopolitics_brier_count: int
    reasoning: float | None


class ScoringService:
    def __init__(self, pg_client: PgClient, logger: NuminousLogger):
        self.pg_client = pg_client
        self.logger = logger

    async def compute_all(self) -> None:
        """Full pipeline: rolling averages -> weighted scores -> leaderboard."""
        miner_pool_scores = await self._compute_rolling_averages()
        if not miner_pool_scores:
            self.logger.debug("No scores to compute")
            return

        now = datetime.now(timezone.utc)
        model_scores = await self._upsert_model_scores(miner_pool_scores, now)
        await self._upsert_leaderboard(model_scores, now)

    async def _compute_rolling_averages(self) -> dict[int, list[PoolScore]]:
        """Compute rolling averages for every (miner, pool, track) combination.

        Fetches per pool config entry to respect each (pool, track) max_events.
        Returns {miner_uid: [PoolScore, ...]} with one entry per pool config.
        """
        # Fetch all rolling data and discover all miners
        pool_data: list[dict[int, RollingScore]] = []
        all_miners: set[int] = set()

        for cfg in POOL_CONFIG:
            pool = cfg["pool"]
            track = cfg["track"]
            max_events = cfg["max_events"]

            if pool == "global_brier":
                data = await self._fetch_brier_rolling(TopicFilter.EXCLUDE_GEOPOLITICS, track, max_events)
            elif pool == "geopolitics_brier":
                data = await self._fetch_brier_rolling(TopicFilter.GEOPOLITICS_ONLY, track, max_events)
            elif pool == "reasoning":
                data = await self._fetch_reasoning_rolling(track, max_events)
            else:
                raise ValueError(f"Unknown pool: {pool}")

            pool_data.append(data)
            all_miners.update(data.keys())

        # Build 6 PoolScores per miner (one per pool config entry)
        result: dict[int, list[PoolScore]] = {uid: [] for uid in all_miners}

        for cfg, data in zip(POOL_CONFIG, pool_data):
            pool = cfg["pool"]
            track = cfg["track"]
            min_events = cfg["min_events"]

            for miner_uid in all_miners:
                raw = data.get(miner_uid)

                if raw is None:
                    result[miner_uid].append(PoolScore(
                        pool=pool, track=track,
                        rolling_avg=None, event_count=0,
                        emission=cfg["emission"],
                    ))
                    continue

                if raw.count < min_events:
                    rolling_avg = None
                else:
                    if pool == "reasoning":
                        rolling_avg = (5.0 - raw.avg) / 4.0
                    else:
                        rolling_avg = raw.avg

                result[miner_uid].append(PoolScore(
                    pool=pool, track=track,
                    rolling_avg=rolling_avg, event_count=raw.count,
                    emission=cfg["emission"],
                ))

        return result

    async def _fetch_brier_rolling(
        self, topic_filter: TopicFilter, track: str, max_events: int
    ) -> dict[int, RollingScore]:
        """Fetch rolling Brier averages for a specific track, limited to last max_events.

        Returns {miner_uid: (avg_score, event_count)}.
        """
        if topic_filter == TopicFilter.GEOPOLITICS_ONLY:
            topic_clause = "AND e.metadata @> $1::jsonb"
        else:
            topic_clause = "AND NOT (e.metadata @> $1::jsonb)"

        query = f"""
            SELECT miner_uid, AVG(event_score) AS avg_score, COUNT(*) AS event_count
            FROM (
                SELECT
                    s.miner_uid,
                    s.event_score,
                    ROW_NUMBER() OVER (PARTITION BY s.miner_uid ORDER BY s.scored_at DESC) AS rn
                FROM scores s
                JOIN events e ON e.event_id = s.event_id
                WHERE s.event_score IS NOT NULL
                  AND s.track = $2
                  {topic_clause}
            ) sub
            WHERE rn <= $3
            GROUP BY miner_uid
        """

        topic_json = json.dumps({"topics": ["Geopolitics"]})
        rows = await self.pg_client.fetch(query, topic_json, track, max_events)

        return {row["miner_uid"]: RollingScore(avg=float(row["avg_score"]), count=int(row["event_count"])) for row in rows}

    async def _fetch_reasoning_rolling(
        self, track: str, max_events: int
    ) -> dict[int, RollingScore]:
        """Fetch rolling reasoning score averages for a specific track, limited to last max_events.

        Returns {miner_uid: (avg_total_score, event_count)}.
        """
        query = """
            SELECT miner_uid, AVG(total) AS avg_score, COUNT(*) AS event_count
            FROM (
                SELECT
                    miner_uid,
                    (
                        (reasoning_scores->>'sources')::float +
                        (reasoning_scores->>'evidence')::float +
                        (reasoning_scores->>'weighting')::float +
                        (reasoning_scores->>'uncertainties')::float +
                        (reasoning_scores->>'mapping')::float
                    ) / 5.0 AS total,
                    ROW_NUMBER() OVER (PARTITION BY miner_uid ORDER BY scored_at DESC) AS rn
                FROM scores
                WHERE reasoning_scores IS NOT NULL
                  AND track = $1
            ) sub
            WHERE rn <= $2
            GROUP BY miner_uid
        """

        rows = await self.pg_client.fetch(query, track, max_events)

        return {row["miner_uid"]: RollingScore(avg=float(row["avg_score"]), count=int(row["event_count"])) for row in rows}

    @staticmethod
    def _compute_weighted_scores_by_track(pool_scores: list[PoolScore]) -> dict[str, float]:
        """Compute weighted score per track.

        All 3 pools must have valid rolling_avg for a track to get a score.
        Returns {track: weighted_score} (only tracks where all pools are valid).
        """
        by_track: dict[str, list[PoolScore]] = {}
        for ps in pool_scores:
            by_track.setdefault(ps.track, []).append(ps)

        result = {}
        for track, pools in by_track.items():
            if any(ps.rolling_avg is None for ps in pools):
                continue

            total_weight = sum(ps.emission for ps in pools)
            result[track] = sum(ps.emission * ps.rolling_avg for ps in pools) / total_weight

        return result

    async def _upsert_model_scores(
        self, miner_scores: dict[int, list[PoolScore]], now: datetime
    ) -> list[dict]:
        """Upsert model_scores and return entries as dicts for leaderboard."""
        entries = []
        for miner_uid, pool_scores in miner_scores.items():
            weighted = self._compute_weighted_scores_by_track(pool_scores)
            pools = [
                {
                    "pool": ps.pool,
                    "track": ps.track,
                    "rolling_avg": ps.rolling_avg,
                    "event_count": ps.event_count,
                    "emission": ps.emission,
                }
                for ps in pool_scores
            ]
            entries.append({
                "miner_uid": miner_uid,
                "weighted_scores": weighted,
                "scores_by_pool": pools,
            })

        await self.pg_client.executemany(
            """
            INSERT INTO model_scores (miner_uid, weighted_scores, scores_by_pool, computed_at)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (miner_uid) DO UPDATE SET
                weighted_scores = EXCLUDED.weighted_scores,
                scores_by_pool = EXCLUDED.scores_by_pool,
                computed_at = EXCLUDED.computed_at
            """,
            [
                (e["miner_uid"], json.dumps(e["weighted_scores"]), json.dumps(e["scores_by_pool"]), now)
                for e in entries
            ],
        )

        self.logger.info("Upserted model scores", extra={"count": len(entries)})
        return entries

    async def _upsert_leaderboard(self, model_score_entries: list[dict], now: datetime) -> None:
        """Build per-track leaderboard from model_score entries."""
        by_track: dict[str, list[LeaderboardEntry]] = {}

        for entry in model_score_entries:
            miner_uid = entry["miner_uid"]
            pools = entry["scores_by_pool"]
            weighted = entry["weighted_scores"]

            # Collect all tracks this miner has data for
            tracks_seen = {p["track"] for p in pools if p["event_count"] > 0}

            for track in tracks_seen:
                track_pools = [p for p in pools if p["track"] == track]
                by_track.setdefault(track, []).append(LeaderboardEntry(
                    miner_uid=miner_uid,
                    weighted_score=weighted.get(track),
                    event_count=sum(p["event_count"] for p in track_pools),
                    global_brier=next((p["rolling_avg"] for p in track_pools if p["pool"] == "global_brier"), None),
                    global_brier_count=next((p["event_count"] for p in track_pools if p["pool"] == "global_brier"), 0),
                    geopolitics_brier=next((p["rolling_avg"] for p in track_pools if p["pool"] == "geopolitics_brier"), None),
                    geopolitics_brier_count=next((p["event_count"] for p in track_pools if p["pool"] == "geopolitics_brier"), 0),
                    reasoning=next((p["rolling_avg"] for p in track_pools if p["pool"] == "reasoning"), None),
                ))

        rows = []
        for track, entries in by_track.items():
            entries.sort(key=lambda e: (e.weighted_score if e.weighted_score is not None else float("inf"), -e.event_count))
            for rank, e in enumerate(entries, 1):
                rows.append((e.miner_uid, track, rank, e.weighted_score, e.event_count,
                             e.global_brier, e.global_brier_count,
                             e.geopolitics_brier, e.geopolitics_brier_count,
                             e.reasoning, now))

        async with self.pg_client.transaction() as conn:
            await conn.execute("DELETE FROM leaderboard")
            if rows:
                await conn.executemany(
                    """
                    INSERT INTO leaderboard (miner_uid, track, rank, weighted_score,
                        event_count, global_brier, global_brier_count,
                        geopolitics_brier, geopolitics_brier_count,
                        reasoning, computed_at)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
                    """,
                    rows,
                )

        self.logger.info("Leaderboard computed", extra={"entries": len(rows)})