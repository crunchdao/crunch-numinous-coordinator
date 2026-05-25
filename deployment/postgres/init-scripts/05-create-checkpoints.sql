CREATE TABLE IF NOT EXISTS checkpoints (
    id                TEXT PRIMARY KEY,
    period_start      TIMESTAMPTZ NOT NULL,
    period_end        TIMESTAMPTZ NOT NULL,
    status            TEXT NOT NULL DEFAULT 'PENDING',
    reward_entries    JSONB,
    meta              JSONB,
    created_at        TIMESTAMPTZ DEFAULT NOW()
);