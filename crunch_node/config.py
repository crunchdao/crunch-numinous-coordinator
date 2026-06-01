import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, Optional

from dotenv import load_dotenv

load_dotenv()

NuminousEnvType = Literal["test", "prod"]


@dataclass(frozen=True)
class CrunchNodeConfig:
    # Core
    numinous_env: NuminousEnvType = field(
        default_factory=lambda: os.getenv("NUMINOUS_ENV", "test")
    )
    api_key: str = field(
        default_factory=lambda: os.environ["NUMINOUS_API_KEY"]
    )
    database_path: str = field(
        default_factory=lambda: os.getenv("DATABASE_PATH", "./crunch_node.db")
    )
    pg_dsn: str = field(
        default_factory=lambda: (
            os.getenv("PG_DSN")
            or "postgresql://{user}:{password}@{host}:{port}/{db}".format(
                user=os.getenv("POSTGRES_USER", "numinous"),
                password=os.getenv("POSTGRES_PASSWORD", "numinous"),
                host=os.getenv("POSTGRES_HOST", "localhost"),
                port=os.getenv("POSTGRES_PORT", "5432"),
                db=os.getenv("POSTGRES_DB", "numinous"),
            )
        )
    )

    # Sandbox
    run_registry_dir: str = field(
        default_factory=lambda: os.getenv("RUN_REGISTRY_DIR", "./deployment/gateway/run_registry/")
    )

    # Model Runner Client
    mrc_crunch_id: str = field(
        default_factory=lambda: os.environ["MRC_CRUNCH_ID"]
    )
    mrc_host: str = field(
        default_factory=lambda: os.getenv("MRC_HOST", "localhost")
    )
    mrc_port: int = field(
        default_factory=lambda: int(os.getenv("MRC_PORT", "8765"))
    )
    mrc_base_classname: str = field(
        default_factory=lambda: os.environ["MRC_BASE_CLASSNAME"]
    )
    mrc_secure_credentials_dir: str = field(
        default_factory=lambda: os.getenv("MRC_SECURE_CREDENTIALS_DIR", "")
    )

    # Task intervals (seconds)
    pull_events_interval: float = field(
        default_factory=lambda: float(os.getenv("PULL_EVENTS_INTERVAL", "50.0"))
    )
    resolve_events_interval: float = field(
        default_factory=lambda: float(os.getenv("RESOLVE_EVENTS_INTERVAL", "900.0"))
    )
    delete_events_interval: float = field(
        default_factory=lambda: float(os.getenv("DELETE_EVENTS_INTERVAL", "1800.0"))
    )
    run_models_interval: float = field(
        default_factory=lambda: float(os.getenv("RUN_MODELS_INTERVAL", "600.0"))
    )
    register_models_interval: float = field(
        default_factory=lambda: float(os.getenv("REGISTER_MODELS_INTERVAL", "300.0"))
    )
    scoring_interval: float = field(
        default_factory=lambda: float(os.getenv("SCORING_INTERVAL", "307.0"))
    )
    export_scores_interval: float = field(
        default_factory=lambda: float(os.getenv("EXPORT_SCORES_INTERVAL", "373.0"))
    )
    export_predictions_interval: float = field(
        default_factory=lambda: float(os.getenv("EXPORT_PREDICTIONS_INTERVAL", "180.0"))
    )
    export_agent_runs_interval: float = field(
        default_factory=lambda: float(os.getenv("EXPORT_AGENT_RUNS_INTERVAL", "300.0"))
    )
    export_agent_run_logs_interval: float = field(
        default_factory=lambda: float(os.getenv("EXPORT_AGENT_RUN_LOGS_INTERVAL", "600.0"))
    )
    db_cleaner_interval: float = field(
        default_factory=lambda: float(os.getenv("DB_CLEANER_INTERVAL", "53.0"))
    )
    db_vacuum_interval: float = field(
        default_factory=lambda: float(os.getenv("DB_VACUUM_INTERVAL", "300.0"))
    )

    # Logging
    log_level: int = field(
        default_factory=lambda: getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
    )

    # Batch sizes
    export_batch_size: int = field(
        default_factory=lambda: int(os.getenv("EXPORT_BATCH_SIZE", "500"))
    )
    scoring_page_size: int = field(
        default_factory=lambda: int(os.getenv("SCORING_PAGE_SIZE", "100"))
    )
    run_models_timeout: int = field(
        default_factory=lambda: int(os.getenv("RUN_MODELS_TIMEOUT", "240"))
    )
    mrc_max_consecutive_timeouts: int = field(
        default_factory=lambda: int(os.getenv("MRC_MAX_CONSECUTIVE_TIMEOUTS", "20"))
    )
    mrc_report_failure: bool = field(
        default_factory=lambda: os.getenv("MRC_REPORT_FAILURE", "true").lower() == "true"
    )
    event_processing_cooldown: float = field(
        default_factory=lambda: float(os.getenv("EVENT_PROCESSING_COOLDOWN", "3"))
    )

    # OpenAI (reasoning scoring)
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    openai_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-5.4")
    )

    # Reasoning scoring + leaderboard intervals
    score_reasoning_interval: float = field(
        default_factory=lambda: float(os.getenv("SCORE_REASONING_INTERVAL", "300.0"))
    )
    export_reasoning_interval: float = field(
        default_factory=lambda: float(os.getenv("EXPORT_REASONING_INTERVAL", "373.0"))
    )
    compute_leaderboard_interval: float = field(
        default_factory=lambda: float(os.getenv("COMPUTE_LEADERBOARD_INTERVAL", "600.0"))
    )

    # Checkpoint (reward distribution)
    checkpoint_reward_pool: float = field(
        default_factory=lambda: float(os.getenv("CHECKPOINT_REWARD_POOL", "577.0"))
    )
    checkpoint_top_k: int = field(
        default_factory=lambda: int(os.getenv("CHECKPOINT_TOP_K", "10"))
    )
    checkpoint_alpha: float = field(
        default_factory=lambda: float(os.getenv("CHECKPOINT_ALPHA", "1.0"))
    )
    checkpoint_interval_seconds: int = field(
        default_factory=lambda: int(os.getenv("CHECKPOINT_INTERVAL_SECONDS", str(7 * 24 * 3600)))
    )
    checkpoint_start_date: datetime = field(
        default_factory=lambda: (
            datetime.fromisoformat(os.environ["CHECKPOINT_START_DATE"]).replace(tzinfo=timezone.utc)
            if os.getenv("CHECKPOINT_START_DATE")
            else datetime(2026, 5, 18, 12, 0, 0, tzinfo=timezone.utc)
        )
    )
    checkpoint_benchmark_miner_uid: Optional[int] = field(
        default_factory=lambda: (
            int(os.environ["CHECKPOINT_BENCHMARK_MINER_UID"])
            if os.getenv("CHECKPOINT_BENCHMARK_MINER_UID") else 14253
        )
    )
