"""
Report worker — FastAPI app serving data from PostgreSQL.

Reads predictions, scores, agent runs directly from PG
and exposes them via REST endpoints.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import List

import asyncpg
from fastapi import FastAPI, HTTPException, Query

from crunch_node.config import CrunchNodeConfig

_EVENT_STATUS_PENDING = 2

config = CrunchNodeConfig()
_pool: asyncpg.Pool | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pool
    _pool = await asyncpg.create_pool(config.pg_dsn, min_size=2, max_size=10)

    try:
        yield
    finally:
        await _pool.close()
        _pool = None


app = FastAPI(
    title="Numinous Crunch Node — Report Worker",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/predictions")
async def get_predictions(
    event_id: str | None = None,
    limit: int = 100,
):
    query = "SELECT * FROM predictions"
    args = []
    if event_id:
        query += " WHERE unique_event_id = $1"
        args.append(event_id)
    query += f" ORDER BY submitted_at DESC LIMIT {min(limit, 1000)}"

    rows = await _pool.fetch(query, *args)
    return [dict(r) for r in rows]


@app.get("/scores")
async def get_scores(
    event_id: str | None = None,
    limit: int = 100,
):
    query = "SELECT * FROM scores"
    args = []
    if event_id:
        query += " WHERE event_id = $1"
        args.append(event_id)
    query += f" ORDER BY scored_at DESC LIMIT {min(limit, 1000)}"

    rows = await _pool.fetch(query, *args)
    return [dict(r) for r in rows]


@app.get("/agent-runs")
async def get_agent_runs(
    event_id: str | None = None,
    limit: int = 100,
):
    query = "SELECT * FROM agent_runs"
    args = []
    if event_id:
        query += " WHERE unique_event_id = $1"
        args.append(event_id)
    query += f" ORDER BY created_at DESC LIMIT {min(limit, 1000)}"

    rows = await _pool.fetch(query, *args)
    return [dict(r) for r in rows]


@app.get("/leaderboard")
async def get_leaderboard():
    rows = await _pool.fetch(
        """
        SELECT miner_uid, track, rank, weighted_score,
               event_count, global_brier, global_brier_count,
               geopolitics_brier, geopolitics_brier_count, reasoning, computed_at
        FROM leaderboard
        WHERE track = 'SIGNAL'
        ORDER BY rank
        """
    )

    # Cumulative rewards from all checkpoints
    reward_rows = await _pool.fetch(
        """
        SELECT
            (e->>'model_id')::int AS miner_uid,
            e->>'track' AS track,
            SUM((e->>'reward_amount')::float) AS total_reward
        FROM checkpoints c,
             jsonb_array_elements(c.reward_entries::jsonb) AS e
        WHERE (e->>'reward_amount')::float > 0
        GROUP BY (e->>'model_id')::int, e->>'track'
        """
    )
    rewards = {(r["miner_uid"], r["track"]): r["total_reward"] for r in reward_rows}

    result = []
    for r in rows:
        entry = dict(r)
        entry["reward"] = rewards.get((r["miner_uid"], r["track"]), 0.0)
        result.append(entry)

    return result


@app.get("/model/active-events")
async def get_model_active_events(
    miner_uids: List[int] = Query(..., alias="projectIds"),
    start_date: datetime = Query(..., alias="start"),
    end_date: datetime = Query(..., alias="end"),
    track: str | None = Query(None, alias="targetName"),
):
    if end_date < start_date:
        raise HTTPException(status_code=400, detail="end must be >= start")
    if (end_date - start_date) > timedelta(days=32):
        raise HTTPException(status_code=400, detail="Date range must not exceed 32 days")
    if track is not None and track not in ("MAIN", "SIGNAL"):
        raise HTTPException(status_code=400, detail="targetName must be MAIN or SIGNAL")

    rows = await _pool.fetch(
        """
        SELECT
            e.unique_event_id,
            e.event_id,
            e.title,
            e.cutoff,
            e.run_days_before_cutoff,
            e.registered_date,
            CASE WHEN e.metadata @> '{"topics": ["Geopolitics"]}'::jsonb
                 THEN 'geopolitics' ELSE 'global' END AS topic,
            p.track,
            p.miner_uid,
            p.prediction,
            p.submitted_at
        FROM events e
        JOIN predictions p
            ON p.unique_event_id = e.unique_event_id
            AND p.miner_uid = ANY($1::int[])
            AND ($2::text IS NULL OR p.track = $2)
        WHERE e.registered_date >= $3
          AND e.registered_date < $4
          AND e.status = $5
        ORDER BY e.cutoff ASC, p.track
        """,
        miner_uids, track, start_date, end_date, _EVENT_STATUS_PENDING,
    )
    return [dict(r) for r in rows]


@app.get("/model/scored-events")
async def get_model_scored_events(
    miner_uids: List[int] = Query(..., alias="projectIds"),
    start_date: datetime = Query(..., alias="start"),
    end_date: datetime = Query(..., alias="end"),
    track: str | None = Query(None, alias="targetName"),
):
    if end_date < start_date:
        raise HTTPException(status_code=400, detail="end must be >= start")
    if (end_date - start_date) > timedelta(days=32):
        raise HTTPException(status_code=400, detail="Date range must not exceed 32 days")
    if track is not None and track not in ("MAIN", "SIGNAL"):
        raise HTTPException(status_code=400, detail="targetName must be MAIN or SIGNAL")

    rows = await _pool.fetch(
        """
        SELECT
            e.unique_event_id,
            e.event_id,
            e.title,
            e.outcome,
            e.cutoff,
            e.run_days_before_cutoff,
            e.registered_date,
            CASE WHEN e.metadata @> '{"topics": ["Geopolitics"]}'::jsonb
                 THEN 'geopolitics' ELSE 'global' END AS topic,
            p.miner_uid,
            p.track,
            p.prediction,
            p.submitted_at,
            s.event_score,
            (s.reasoning_scores->>'sources')::int      AS reasoning_sources,
            (s.reasoning_scores->>'evidence')::int     AS reasoning_evidence,
            (s.reasoning_scores->>'uncertainties')::int AS reasoning_uncertainties,
            (s.reasoning_scores->>'mapping')::int      AS reasoning_mapping,
            (s.reasoning_scores->>'weighting')::int    AS reasoning_weighting
        FROM events e
        JOIN predictions p
            ON p.unique_event_id = e.unique_event_id
            AND p.miner_uid = ANY($1::int[])
            AND ($4::text IS NULL OR p.track = $4)
        JOIN scores s
            ON s.event_id = e.event_id
            AND s.miner_uid = p.miner_uid
            AND s.track = p.track
        WHERE e.registered_date >= $2
          AND e.registered_date < $3
        ORDER BY e.registered_date DESC, p.track
        """,
        miner_uids, start_date, end_date, track,
    )
    return [dict(r) for r in rows]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
