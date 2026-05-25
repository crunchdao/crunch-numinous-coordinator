ALTER TABLE model_scores ADD COLUMN IF NOT EXISTS model_name TEXT;
ALTER TABLE model_scores ADD COLUMN IF NOT EXISTS cruncher_id TEXT;
ALTER TABLE model_scores ADD COLUMN IF NOT EXISTS cruncher_name TEXT;
ALTER TABLE model_scores ADD COLUMN IF NOT EXISTS deployment_id TEXT;