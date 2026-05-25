"""
RegisterModels task — reads models from ModelCluster and upserts them
into the miners SQLite table so that scoring/export can reference them.
"""

from datetime import datetime, timezone

from model_runner_client.model_cluster import ModelCluster
from model_runner_client.model_runners import ModelRunner
from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.models.miner_agent import MinerAgentsModel
from neurons.validator.scheduler.task import AbstractTask
from neurons.validator.utils.logger.logger import NuminousLogger


def map_miner_properties(model: ModelRunner):
    miner_uid = int(model.model_id)

    return miner_uid, *to_miner_properties(miner_uid)


def to_miner_properties(miner_uid: int):
    miner_hotkey = f"hotkey-{miner_uid}"
    version_id = f"version-{miner_uid}"

    return miner_hotkey, version_id


class RegisterModels(AbstractTask):
    def __init__(
        self,
        interval_seconds: float,
        db_operations: DatabaseOperations,
        model_cluster: ModelCluster,
        pg_client,
        logger: NuminousLogger,
    ):
        self.interval = interval_seconds
        self.db_operations = db_operations
        self.model_cluster = model_cluster
        self.pg_client = pg_client
        self.logger = logger

    @property
    def name(self) -> str:
        return "register-models"

    @property
    def interval_seconds(self) -> float:
        return self.interval

    async def run(self) -> None:
        models = self.model_cluster.models_run

        if not models:
            self.logger.warning("No models to register")
            return

        miners_data = []
        miner_agents = []
        now = datetime.now(timezone.utc).isoformat()

        for model in models.values():
            miner_uid, miner_hotkey, version_id = map_miner_properties(model)
            node_ip = f"{model.ip}:{model.port}"
            blocktime = "0"
            is_validating = False
            validator_permit = False

            # Format: [miner_uid, miner_hotkey, node_ip, registered_date, blocktime,
            #          is_validating, validator_permit, node_ip(update), blocktime(update)]
            miners_data.append([
                miner_uid,
                miner_hotkey,
                node_ip,
                now,
                blocktime,
                is_validating,
                validator_permit,
                node_ip,
                blocktime,
            ])

            miner_agents.append(MinerAgentsModel(
                version_id=version_id,
                miner_uid=miner_uid,
                miner_hotkey=miner_hotkey,
                track="MAIN",
                agent_name="default",
                version_number="1",
                file_path="/dev/null",
                pulled_at=datetime.now(timezone.utc),
                created_at=datetime.now(timezone.utc),
            ))

        if miners_data:
            await self.db_operations.upsert_miners(miners_data)
            await self.db_operations.upsert_miner_agents(miner_agents)

            # Upsert model metadata in PG (model_scores table)
            pg_rows = []
            for model in models.values():
                miner_uid = int(model.model_id)
                infos = model.infos or {}
                pg_rows.append((
                    miner_uid,
                    model.model_name,
                    infos.get("cruncher_id"),
                    infos.get("cruncher_name"),
                    model.deployment_id,
                ))

            await self.pg_client.executemany(
                """
                INSERT INTO model_scores (miner_uid, model_name, cruncher_id, cruncher_name, deployment_id)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (miner_uid) DO UPDATE SET
                    model_name = EXCLUDED.model_name,
                    cruncher_id = EXCLUDED.cruncher_id,
                    cruncher_name = EXCLUDED.cruncher_name,
                    deployment_id = EXCLUDED.deployment_id
                """,
                pg_rows,
            )

            self.logger.info(
                "Registered models as miners",
                extra={"n_models": len(miners_data)},
            )
