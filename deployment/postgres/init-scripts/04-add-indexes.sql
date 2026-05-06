-- Indexes for query performance

-- events: filter active events by status
CREATE INDEX IF NOT EXISTS idx_events_status ON events (status);

-- events: filter scored events by registered_date
CREATE INDEX IF NOT EXISTS idx_events_registered_date ON events (registered_date);

-- events: metadata GIN for topic filtering
CREATE INDEX IF NOT EXISTS idx_events_metadata ON events USING GIN (metadata);

-- predictions: JOIN on (unique_event_id, miner_uid, track)
CREATE INDEX IF NOT EXISTS idx_predictions_event_miner_track ON predictions (unique_event_id, miner_uid, track);

-- scores: JOIN from predictions
CREATE INDEX IF NOT EXISTS idx_scores_event_miner_track ON scores (event_id, miner_uid, track);

-- reasoning: scoring task — only fetch unscored
CREATE INDEX IF NOT EXISTS idx_reasoning_scored ON reasoning (reasoning_scored) WHERE reasoning_scored = false;

-- reasoning: cache lookup — find identical reasoning already scored (hash for exact match on long text)
CREATE INDEX IF NOT EXISTS idx_reasoning_text_scored ON reasoning USING hash (reasoning) WHERE reasoning_scored = true;