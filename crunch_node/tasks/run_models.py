"""
RunModels task — replaces RunAgents.

Instead of Docker sandboxes, calls models via model-runner-client gRPC.
Uses DynamicSubclassModelConcurrentRunner.call() which handles concurrency
via asyncio.gather internally.
Stores predictions + agent_runs in SQLite (same schema as RunAgents).
"""

import asyncio
import json
import uuid
from time import time

from model_runner_client.grpc.generated.commons_pb2 import Argument, Variant, VariantType
from model_runner_client.model_concurrent_runners import DynamicSubclassModelConcurrentRunner
from model_runner_client.model_concurrent_runners.model_concurrent_runner import ModelPredictResult
from model_runner_client.model_runners import ModelRunner
from model_runner_client.utils.datatype_transformer import encode_data
from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.models.agent_runs import AgentRunsModel, AgentRunStatus
from neurons.validator.models.event import EventsModel
from neurons.validator.models.prediction import PredictionsModel
from neurons.validator.models.reasoning import MAX_REASONING_CHARS, MISSING_REASONING_PREFIX
from neurons.validator.models.track import TrackEnum
from neurons.validator.sandbox.manager import SandboxManager
from neurons.validator.scheduler.task import AbstractTask
from neurons.validator.utils.common.interval import get_interval_start_minutes
from neurons.validator.utils.logger.logger import NuminousLogger

from crunch_node.tasks.register_models import map_miner_properties, to_miner_properties

TITLE_SEPARATOR = " ==Further Information==: "
MAX_LOG_CHARS = 25_000

# SIGNAL first: prevents models from caching MAIN results (full API access)
# and reusing them during the SIGNAL run (restricted access).
TRACK_ORDER = [TrackEnum.SIGNAL, TrackEnum.MAIN]


class RunModels(AbstractTask):
    def __init__(
        self,
        interval_seconds: float,
        db_operations: DatabaseOperations,
        concurrent_runner: DynamicSubclassModelConcurrentRunner,
        sandbox_manager: SandboxManager,
        logger: NuminousLogger,
        event_processing_cooldown: float = 0,
    ):
        self.interval = interval_seconds
        self.db_operations = db_operations
        self.concurrent_runner = concurrent_runner
        self.sandbox_manager = sandbox_manager
        self.logger = logger
        self.event_processing_cooldown = event_processing_cooldown

    @property
    def name(self) -> str:
        return "run-models"

    @property
    def interval_seconds(self) -> float:
        return self.interval

    def _parse_event_description(self, full_description: str) -> tuple[str, str]:
        if TITLE_SEPARATOR in full_description:
            parts = full_description.split(TITLE_SEPARATOR, 1)
            return parts[0], parts[1]
        return full_description, full_description

    def _build_event_data(self, event: EventsModel) -> dict:
        title = event.title
        description = event.description
        if not title:
            title, description = self._parse_event_description(description)

        metadata = json.loads(event.metadata) if isinstance(event.metadata, str) else event.metadata

        return {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "title": title,
            "description": description,
            "cutoff": event.cutoff.isoformat(),
            "metadata": metadata,
        }

    @staticmethod
    def _model_infos(model: ModelRunner) -> tuple[int, str, str]:
        """Extract (miner_uid, miner_hotkey, version_id) from a ModelRunner."""
        return map_miner_properties(model)

    async def run(self) -> None:
        events = await self.db_operations.get_events_to_predict()
        if not events:
            self.logger.debug("No events to predict")
            return

        models = self.concurrent_runner.model_cluster.models_run
        if not models:
            self.logger.warning("No models available for execution")
            return

        interval_start_minutes = get_interval_start_minutes()

        self.logger.info(
            "Starting to run models",
            extra={
                "total_events": len(events),
                "total_models": len(models),
                "interval_start_minutes": interval_start_minutes,
            },
        )

        for event in events:
            started_at = time()

            await self._process_event(event, interval_start_minutes)

            ended_at = time()
            took = ended_at - started_at

            wait_time = max(0.0, self.event_processing_cooldown - took)
            if wait_time > 0:
                self.logger.info(
                    "Waiting before next event",
                    extra={
                        "time_took": took,
                        "allowed_time": self.concurrent_runner.timeout,
                        "wait_time_seconds": wait_time,
                    },
                )

                await asyncio.sleep(wait_time)

    async def _process_event(self, event: EventsModel, interval_start_minutes: int) -> None:
        """Process a single event across all its tracks (SIGNAL first, then MAIN)."""
        models = self.concurrent_runner.model_cluster.models_run

        tracks = {TrackEnum(t) if isinstance(t, str) else t for t in event.tracks}
        if TrackEnum.SIGNAL not in tracks:
            return

        tracks = [TrackEnum.SIGNAL] # MAIN doesn't have enough emissions, we force for now only SIGNAL

        total_counts = {"success": 0, "failed": 0, "timeout": 0}
        for track in tracks:
            counts = await self._process_event_track(event, track, interval_start_minutes)
            for k in total_counts:
                total_counts[k] += counts[k]

        absent_count = await self._mark_absent_miners(event, models, tracks)

        self.logger.info(
            "Processed event",
            extra={
                "event_id": event.unique_event_id,
                "tracks": [t.value for t in tracks],
                **total_counts,
                "absent_count": absent_count,
            },
        )

    async def _process_event_track(
        self, event: EventsModel, track: TrackEnum, interval_start_minutes: int
    ) -> dict:
        """Process a single (event, track) pair: filter models, call, store results."""
        models = self.concurrent_runner.model_cluster.models_run
        event_id = event.unique_event_id
        models_to_call = []

        for model in models.values():
            miner_uid, miner_hotkey, version_id = self._model_infos(model)

            existing_prediction = (
                await self.db_operations.get_latest_prediction_for_event_and_miner(
                    unique_event_id=event_id,
                    miner_uid=miner_uid,
                    miner_hotkey=miner_hotkey,
                    track=track.value,
                )
            )

            if existing_prediction is not None:
                if existing_prediction.interval_start_minutes == interval_start_minutes:
                    self.logger.debug(
                        "Skipping — prediction exists",
                        extra={"event_id": event_id, "miner_uid": miner_uid, "track": track.value},
                    )
                    continue

                # Replicate existing prediction to current interval
                new_pred = PredictionsModel(
                    unique_event_id=event_id,
                    miner_uid=miner_uid,
                    miner_hotkey=miner_hotkey,
                    track=track.value,
                    latest_prediction=existing_prediction.latest_prediction,
                    interval_start_minutes=interval_start_minutes,
                    interval_agg_prediction=existing_prediction.latest_prediction,
                    interval_count=1,
                    run_id=existing_prediction.run_id,
                    version_id=existing_prediction.version_id,
                )
                await self.db_operations.upsert_predictions([new_pred])
                continue

            models_to_call.append(model)

        if not models_to_call:
            return {"success": 0, "failed": 0, "timeout": 0}

        event_data = self._build_event_data(event)

        # Register a unique run_id per model for track filtering
        model_run_ids: dict[int, str] = {}
        for model in models_to_call:
            run_id = str(uuid.uuid4())
            model_run_ids[id(model)] = run_id
            self.sandbox_manager.register_run(run_id, track)

        def prepare_args(model_runner: ModelRunner):
            data = {**event_data, "run_id": model_run_ids[id(model_runner)], "track": track.value}
            return (
                [Argument(position=1, data=Variant(type=VariantType.JSON, value=encode_data(VariantType.JSON, data)))],
                [],
            )

        counts = {"success": 0, "failed": 0, "timeout": 0}
        try:
            results = await self.concurrent_runner.call(
                method_name="predict",
                arguments=prepare_args,
                model_runs=models_to_call,
            )

            for model_runner, predict_result in results.items():
                status = await self._store_result(
                    event, model_runner, predict_result, interval_start_minutes, track
                )
                if status == AgentRunStatus.SUCCESS:
                    counts["success"] += 1
                elif status == AgentRunStatus.SANDBOX_TIMEOUT:
                    counts["timeout"] += 1
                else:
                    counts["failed"] += 1
        finally:
            for run_id in model_run_ids.values():
                self.sandbox_manager.unregister_run(run_id)

        return counts

    async def _mark_absent_miners(
        self,
        event: EventsModel,
        active_models: dict,
        tracks: list[TrackEnum],
    ) -> int:
        """Create failed runs for registered miners whose model is not in models_run."""
        event_id = event.unique_event_id

        active_keys = set()
        for model in active_models.values():
            miner_uid, _, _ = self._model_infos(model)
            active_keys.add(miner_uid)

        all_miners = await self.db_operations.get_miners_last_registration()
        if not all_miners:
            return 0

        absent_count = 0
        for miner in all_miners:
            miner_uid = int(miner.miner_uid)
            if miner_uid in active_keys:
                continue

            miner_hotkey, version_id = to_miner_properties(int(miner.miner_uid))

            for track in tracks:
                run_id = str(uuid.uuid4())
                agent_run = AgentRunsModel(
                    run_id=run_id,
                    unique_event_id=event_id,
                    agent_version_id=version_id,
                    miner_uid=miner_uid,
                    miner_hotkey=miner_hotkey,
                    track=track.value,
                    status=AgentRunStatus.INTERNAL_AGENT_ERROR,
                    exported=False,
                    is_final=True,
                )
                await self.db_operations.upsert_agent_runs([agent_run])
                absent_count += 1

        return absent_count

    async def _store_result(
        self,
        event: EventsModel,
        model_runner: ModelRunner,
        predict_result: ModelPredictResult,
        interval_start_minutes: int,
        track: TrackEnum,
    ) -> AgentRunStatus:
        """Store agent run, logs, reasoning and prediction from a ModelPredictResult."""
        miner_uid, miner_hotkey, version_id = self._model_infos(model_runner)
        event_id = event.unique_event_id

        prediction_value = None
        logs = ""
        reasoning_text = f"{MISSING_REASONING_PREFIX} - not run]"

        if predict_result.status == ModelPredictResult.Status.SUCCESS:
            result = predict_result.result
            if isinstance(result, dict):
                prediction_value = result.get("prediction")  # todo check
                logs = result.get("logs", "")
                raw_reasoning = result.get("reasoning", "")

                if prediction_value is not None and isinstance(prediction_value, (int, float)):
                    prediction_value = float(prediction_value)
                    status = AgentRunStatus.SUCCESS
                    reasoning_text = (
                        str(raw_reasoning)[:MAX_REASONING_CHARS]
                        if raw_reasoning
                        else f"{MISSING_REASONING_PREFIX} - {status.value}]"
                    )
                else:
                    status = AgentRunStatus.INVALID_SANDBOX_OUTPUT
                    self.logger.warning(
                        "Invalid prediction from model",
                        extra={"event_id": event_id, "miner_uid": miner_uid, "result": str(result)},
                    )
            else:
                status = AgentRunStatus.INVALID_SANDBOX_OUTPUT
                self.logger.warning(
                    "Model returned non-dict result",
                    extra={"event_id": event_id, "miner_uid": miner_uid},
                )
        elif predict_result.status == ModelPredictResult.Status.TIMEOUT:
            status = AgentRunStatus.SANDBOX_TIMEOUT
            logs = "Model execution timeout"
            self.logger.warning(
                "Model execution timeout",
                extra={"event_id": event_id, "miner_uid": miner_uid},
            )
        else:
            status = AgentRunStatus.INTERNAL_AGENT_ERROR
            logs = f"Model execution failed (status={predict_result.status.value})"
            self.logger.warning(
                "Model execution failed",
                extra={"event_id": event_id, "miner_uid": miner_uid},
            )

        # Truncate logs
        if len(logs) > MAX_LOG_CHARS:
            logs = f"[TRUNCATED: {len(logs)} chars]\n\n" + logs[-MAX_LOG_CHARS:]

        # Store agent run
        run_id = str(uuid.uuid4())
        agent_run = AgentRunsModel(
            run_id=run_id,
            unique_event_id=event_id,
            agent_version_id=version_id,
            miner_uid=miner_uid,
            miner_hotkey=miner_hotkey,
            track=track.value,
            status=status,
            exported=False,
            is_final=True,
        )

        await self.db_operations.upsert_agent_runs([agent_run])

        # Store logs
        try:
            await self.db_operations.insert_agent_run_log(run_id, logs)
        except Exception as exc:
            self.logger.error(
                "Failed to store agent run log",
                extra={"run_id": run_id, "error": str(exc)},
            )

        # Store prediction and reasoning if successful
        if status == AgentRunStatus.SUCCESS and prediction_value is not None:
            clipped = max(0.0, min(1.0, prediction_value))
            prediction = PredictionsModel(
                unique_event_id=event_id,
                miner_uid=miner_uid,
                miner_hotkey=miner_hotkey,
                track=track.value,
                latest_prediction=clipped,
                interval_start_minutes=interval_start_minutes,
                interval_agg_prediction=clipped,
                run_id=run_id,
                version_id=version_id,
            )
            await self.db_operations.upsert_predictions([prediction])

            try:
                await self.db_operations.insert_reasoning(run_id, reasoning_text)
            except Exception as exc:
                self.logger.error(
                    "Failed to store reasoning",
                    extra={"run_id": run_id, "error": str(exc)},
                )

            self.logger.debug(
                "Stored prediction",
                extra={"event_id": event_id, "miner_uid": miner_uid, "prediction": clipped},
            )

        return status
