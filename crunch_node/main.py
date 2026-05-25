"""
numinous crunch-node entry point.

Wires: config -> bt_compat mocks -> DB + PG + DynamicSubclassModelConcurrentRunner -> tasks -> scheduler
"""

if True:
    # bt_compat MUST be imported before any neurons.validator.* module
    import crunch_node.bt_compat  # noqa: F401

import asyncio
import sqlite3
import sys

from model_runner_client.model_concurrent_runners import DynamicSubclassModelConcurrentRunner
from model_runner_client.security.credentials import SecureCredentials
from neurons.validator.db.client import DatabaseClient
from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.sandbox import SandboxManager
from neurons.validator.scheduler.tasks_scheduler import TasksScheduler
from neurons.validator.tasks.db_cleaner import DbCleaner
from neurons.validator.tasks.db_vacuum import DbVacuum
from neurons.validator.tasks.delete_events import DeleteEvents
from neurons.validator.tasks.pull_events import PullEvents
from neurons.validator.tasks.resolve_events import ResolveEvents
from neurons.validator.utils.logger.logger import NuminousLogger, create_logger

from crunch_node._logging import ExtraFormatter
from crunch_node.bt_compat.numinous_client import CrunchNodeNuminousClient
from crunch_node.bt_compat.sandbox_manager import CrunchNodeSandboxManager
from crunch_node.bt_compat.scoring import CrunchNodeScoring
from crunch_node.clients.pg_client import PgClient
from crunch_node.config import CrunchNodeConfig
from crunch_node.migrations import CustomMigrations
from crunch_node.tasks.export_agent_run_logs_pg import ExportAgentRunLogsPg
from crunch_node.tasks.export_reasoning_pg import ExportReasoningPg
from crunch_node.tasks.export_agent_runs_pg import ExportAgentRunsPg
from crunch_node.tasks.export_events_pg import ExportEventsPg
from crunch_node.tasks.export_predictions_pg import ExportPredictionsPg
from crunch_node.tasks.export_scores_pg import ExportScoresPg
from crunch_node.tasks.register_models import RegisterModels
from crunch_node.tasks.run_models import RunModels


async def main():
    config = CrunchNodeConfig()

    # Logger
    logger: NuminousLogger = create_logger("crunch-node")
    ExtraFormatter.install(logger, config.log_level)
    logger.start_session()

    # Database (SQLite)
    db_client = DatabaseClient(db_path=config.database_path, logger=logger)
    db_operations = DatabaseOperations(db_client=db_client, logger=logger)
    await db_client.migrate()

    # Custom migrations on top of submodule's alembic
    await CustomMigrations(db_client).run()

    # PostgreSQL
    pg_client = PgClient(dsn=config.pg_dsn)
    await pg_client.connect()

    # Numinous API client (Bearer auth)
    api_client = CrunchNodeNuminousClient(
        env=config.numinous_env,
        logger=logger,
        api_key=config.api_key,
    )

    sandbox_manager = CrunchNodeSandboxManager(
        run_registry_dir=config.run_registry_dir,
        logger=logger,
    )

    # Model Runner Client
    secure_credentials = None
    if config.mrc_secure_credentials_dir:
        secure_credentials = SecureCredentials.from_directory(config.mrc_secure_credentials_dir)

    concurrent_runner = DynamicSubclassModelConcurrentRunner(
        timeout=config.run_models_timeout,
        crunch_id=config.mrc_crunch_id,
        host=config.mrc_host,
        port=config.mrc_port,
        base_classname=config.mrc_base_classname,
        secure_credentials=secure_credentials,
        max_consecutive_timeouts=config.mrc_max_consecutive_timeouts,
        report_failure=config.mrc_report_failure,
    )

    await concurrent_runner.init()
    await asyncio.sleep(20)  # give time to the model (init)
    model_cluster = concurrent_runner.model_cluster

    # ── Tasks ────────────────────────────────────────────────────────

    pull_events_task = PullEvents(
        interval_seconds=config.pull_events_interval,
        page_size=50,
        db_operations=db_operations,
        api_client=api_client,
    )

    resolve_events_task = ResolveEvents(
        interval_seconds=config.resolve_events_interval,
        db_operations=db_operations,
        api_client=api_client,
        page_size=100,
        logger=logger,
    )

    delete_events_task = DeleteEvents(
        interval_seconds=config.delete_events_interval,
        db_operations=db_operations,
        api_client=api_client,
        page_size=100,
        logger=logger,
    )

    register_models_task = RegisterModels(
        interval_seconds=config.register_models_interval,
        db_operations=db_operations,
        model_cluster=model_cluster,
        pg_client=pg_client,
        logger=logger,
    )

    run_models_task = RunModels(
        interval_seconds=config.run_models_interval,
        db_operations=db_operations,
        concurrent_runner=concurrent_runner,
        sandbox_manager=sandbox_manager,
        logger=logger,
        event_processing_cooldown=config.event_processing_cooldown,
    )

    scoring_task = CrunchNodeScoring(
        interval_seconds=config.scoring_interval,
        db_operations=db_operations,
        logger=logger,
        page_size=config.scoring_page_size,
    )

    export_scores_task = ExportScoresPg(
        interval_seconds=config.export_scores_interval,
        page_size=config.export_batch_size,
        db_operations=db_operations,
        pg_client=pg_client,
        logger=logger,
    )

    export_predictions_task = ExportPredictionsPg(
        interval_seconds=config.export_predictions_interval,
        db_operations=db_operations,
        pg_client=pg_client,
        batch_size=min(config.export_batch_size, 300),
        logger=logger,
    )

    export_agent_runs_task = ExportAgentRunsPg(
        interval_seconds=config.export_agent_runs_interval,
        batch_size=config.export_batch_size,
        db_operations=db_operations,
        pg_client=pg_client,
        logger=logger,
    )

    export_events_task = ExportEventsPg(
        interval_seconds=config.pull_events_interval,
        batch_size=config.export_batch_size,
        db_client=db_client,
        pg_client=pg_client,
        logger=logger,
    )

    export_agent_run_logs_task = ExportAgentRunLogsPg(
        interval_seconds=config.export_agent_run_logs_interval,
        batch_size=config.export_batch_size,
        db_operations=db_operations,
        pg_client=pg_client,
        logger=logger,
    )

    export_reasoning_task = ExportReasoningPg(
        interval_seconds=config.export_reasoning_interval,
        batch_size=config.export_batch_size,
        db_operations=db_operations,
        pg_client=pg_client,
        logger=logger,
    )

    db_cleaner_task = DbCleaner(
        interval_seconds=config.db_cleaner_interval,
        db_operations=db_operations,
        batch_size=4000,
        logger=logger,
    )

    vacuum_task = DbVacuum(
        interval_seconds=config.db_vacuum_interval,
        db_operations=db_operations,
        logger=logger,
        pages=500,
    )

    # ── Scheduler ────────────────────────────────────────────────────

    scheduler = TasksScheduler(logger=logger)

    scheduler.add(task=pull_events_task)
    scheduler.add(task=resolve_events_task)
    scheduler.add(task=delete_events_task)
    scheduler.add(task=register_models_task)
    scheduler.add(task=run_models_task)
    scheduler.add(task=scoring_task)
    scheduler.add(task=export_predictions_task)
    scheduler.add(task=export_scores_task)
    scheduler.add(task=export_agent_runs_task)
    scheduler.add(task=export_events_task)
    scheduler.add(task=export_agent_run_logs_task)
    scheduler.add(task=export_reasoning_task)
    scheduler.add(task=db_cleaner_task)
    scheduler.add(task=vacuum_task)

    logger.info(
        "Crunch node started",
        extra={
            "numinous_env": config.numinous_env,
            "database_path": config.database_path,
            "pg_dsn": config.pg_dsn.split("@")[-1] if "@" in config.pg_dsn else "***",
            "python": sys.version,
            "sqlite": sqlite3.sqlite_version,
        },
    )

    try:
        await asyncio.gather(
            asyncio.create_task(concurrent_runner.sync()),
            asyncio.create_task(scheduler.start()),
        )
    finally:
        await pg_client.close()


def entrypoint():
    asyncio.run(main())


if __name__ == "__main__":
    entrypoint()
