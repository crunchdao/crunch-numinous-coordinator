"""
Report worker — FastAPI app serving data from PostgreSQL.

Reads predictions, scores, agent runs directly from PG
and exposes them via REST endpoints.
"""

import base64
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Annotated, List, Optional

import asyncpg
from crunch_node.config import CrunchNodeConfig
from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

_EVENT_STATUS_PENDING = 2

# API key for the paginated predictions endpoint
PREDICTIONS_API_KEY = os.getenv("PREDICTIONS_API_KEY", "")

# Rate limit applied to protected endpoints (format: "N/period", e.g. "180/minute")
RATE_LIMIT = os.getenv("RATE_LIMIT", "180/minute")

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


limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="Numinous Crunch Node — Report Worker",
    lifespan=lifespan,
)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    from starlette.responses import JSONResponse
    return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})


# ------------------------------------------------------------------------------
# API key auth
# ------------------------------------------------------------------------------

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _verify_key(api_key: str | None, expected: str) -> None:
    if not expected:
        return
    if api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )


def _verify_predictions_api_key(api_key: Annotated[str | None, Depends(_api_key_header)]) -> None:
    _verify_key(api_key, PREDICTIONS_API_KEY)


# ------------------------------------------------------------------------------
# Cursor helpers for paginated predictions
# ------------------------------------------------------------------------------

def _encode_cursor(submitted_at: datetime, unique_event_id: str, track: str, interval_start_minutes: int) -> str:
    raw = f"{submitted_at.isoformat()}|{unique_event_id}|{track}|{interval_start_minutes}"
    return base64.urlsafe_b64encode(raw.encode()).decode()


def _decode_cursor(cursor: str) -> tuple[datetime, str, str, int]:
    try:
        raw = base64.urlsafe_b64decode(cursor.encode()).decode()
        ts, uid, trk, mins = raw.split("|", 3)
        return datetime.fromisoformat(ts), uid, trk, int(mins)
    except Exception:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid cursor")


# ------------------------------------------------------------------------------
# Response models
# ------------------------------------------------------------------------------

class PredictionDetailResponse(BaseModel):
    unique_event_id: str
    miner_uid: int
    track: str
    provider_type: Optional[str]
    prediction: Optional[float]
    interval_start_minutes: Optional[int] = Field(None, description="Daily scoring bucket: minutes elapsed since 2024-01-01 UTC, snapped to midnight boundaries (multiples of 1440).")
    interval_datetime: Optional[datetime]
    submitted_at: Optional[datetime]
    run_id: Optional[str]
    version_id: Optional[str]


class PaginatedPredictionsResponse(BaseModel):
    data: List[PredictionDetailResponse]
    next_cursor: Optional[str]


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


def _parse_checkpoint(row) -> dict:
    d = dict(row)
    if isinstance(d.get("reward_entries"), str):
        d["reward_entries"] = json.loads(d["reward_entries"])
    if isinstance(d.get("meta"), str):
        d["meta"] = json.loads(d["meta"])
    return d


@app.get("/checkpoints/latest")
async def get_checkpoint_latest():
    row = await _pool.fetchrow(
        "SELECT id, period_start, period_end, status, reward_entries, meta, created_at FROM checkpoints ORDER BY created_at DESC LIMIT 1"
    )
    if not row:
        raise HTTPException(status_code=404, detail="No checkpoints found")
    return _parse_checkpoint(row)


@app.get("/checkpoints")
async def get_checkpoints(status: str | None = None):
    if status:
        rows = await _pool.fetch(
            "SELECT id, period_start, period_end, status, reward_entries, meta, created_at FROM checkpoints WHERE status = $1 ORDER BY created_at DESC",
            status,
        )
    else:
        rows = await _pool.fetch(
            "SELECT id, period_start, period_end, status, reward_entries, meta, created_at FROM checkpoints ORDER BY created_at DESC"
        )
    return [_parse_checkpoint(r) for r in rows]


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
            s.scored_at,
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


@app.get("/predictions/{miner_uid}", response_model=PaginatedPredictionsResponse)
@limiter.limit(RATE_LIMIT)
async def get_predictions_for_miner(
    request: Request,
    miner_uid: int,
    _: Annotated[None, Depends(_verify_predictions_api_key)],
    start: Annotated[Optional[datetime], Query(description="Filter predictions submitted on or after this date (ISO 8601).")] = None,
    end: Annotated[Optional[datetime], Query(description="Filter predictions submitted before this date (ISO 8601).")] = None,
    event_id: Annotated[Optional[str], Query(description="Filter by unique_event_id.")] = None,
    track: Annotated[Optional[str], Query(description="Filter by track (MAIN or SIGNAL).")] = None,
    cursor: Annotated[Optional[str], Query(description="Pagination cursor returned as next_cursor from the previous page.")] = None,
    limit: Annotated[int, Query(ge=1, le=500, description="Number of items per page (default 100, max 500).")] = 100,
    order: Annotated[str, Query(pattern="^(asc|desc)$", description="Sort order by submitted_at: asc (oldest first) or desc (newest first, default).")] = "desc",
):
    """
    Get predictions for a miner with cursor-based pagination.

    **Tip:** Miner UIDs can be found via the /leaderboard endpoint.

    **Tip:** use `order=desc&limit=1` with an `event_id` filter to fetch the latest
    prediction for a specific event.
    """
    conditions: list[str] = ["miner_uid = $1"]
    args: list = [miner_uid]
    p = 2  # next positional parameter index

    if start is not None:
        conditions.append(f"submitted_at >= ${p}")
        args.append(start)
        p += 1

    if end is not None:
        conditions.append(f"submitted_at < ${p}")
        args.append(end)
        p += 1

    if event_id is not None:
        conditions.append(f"unique_event_id = ${p}")
        args.append(event_id)
        p += 1

    if track is not None:
        conditions.append(f"track = ${p}")
        args.append(track)
        p += 1

    if cursor is not None:
        cur_ts, cur_uid, cur_track, cur_mins = _decode_cursor(cursor)
        op = "<" if order == "desc" else ">"
        conditions.append(
            f"(submitted_at, unique_event_id, track, interval_start_minutes) "
            f"{op} (${p}, ${p+1}, ${p+2}, ${p+3})"
        )
        args.extend([cur_ts, cur_uid, cur_track, cur_mins])
        p += 4

    where = " AND ".join(conditions)
    dir_ = "DESC" if order == "desc" else "ASC"
    query = (
        f"SELECT unique_event_id, miner_uid, track, provider_type, prediction, "
        f"interval_start_minutes, interval_datetime, submitted_at, run_id, version_id "
        f"FROM predictions "
        f"WHERE {where} "
        f"ORDER BY submitted_at {dir_}, unique_event_id {dir_}, track {dir_}, interval_start_minutes {dir_} "
        f"LIMIT {limit}"
    )

    rows = await _pool.fetch(query, *args)
    data = [PredictionDetailResponse(**dict(r)) for r in rows]

    next_cursor: Optional[str] = None
    if len(data) == limit:
        last = rows[-1]
        if last["submitted_at"] is not None and last["interval_start_minutes"] is not None:
            next_cursor = _encode_cursor(
                last["submitted_at"],
                last["unique_event_id"],
                last["track"],
                last["interval_start_minutes"],
            )

    return PaginatedPredictionsResponse(data=data, next_cursor=next_cursor)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
