#!/usr/bin/env python3
"""
Payout script for numinous-coordinator weekly payouts.

Fetches the latest checkpoint from the coordinator API, converts USDC amounts
to microusdc, generates a JSON file in coordinator-cli format, and optionally
executes the coordinator-cli checkpoint-create command.

Prerequisites:
    Install the coordinator CLI: npm install -g @crunchdao/coordinator-cli
"""

import argparse
import decimal
import json
import logging
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_API_URL = "http://localhost:8000"
SCRIPT_DIR = Path(__file__).resolve().parent
PROCESSED_CHECKPOINTS_FILE = SCRIPT_DIR / ".processed_checkpoints.json"
SQUADS_URL = "https://app.squads.so/squads/GyNbz9cfYaSPJTVnWyUVwb3SJonFDhEZAqh2qPATwQPg/transactions"


def load_processed_checkpoints() -> dict:
    if PROCESSED_CHECKPOINTS_FILE.exists():
        try:
            with open(PROCESSED_CHECKPOINTS_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed to load processed checkpoints file: %s", e)
    return {}


def save_processed_checkpoint(checkpoint_id: str, num_prizes: int, total_amount_usdc: float):
    processed = load_processed_checkpoints()
    processed[checkpoint_id] = {
        "num_prizes": num_prizes,
        "total_amount_usdc": total_amount_usdc,
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        with open(PROCESSED_CHECKPOINTS_FILE, "w") as f:
            json.dump(processed, f, indent=2)
        logger.info("Saved processed checkpoint record for %s", checkpoint_id)
    except Exception as e:
        logger.error("Failed to save processed checkpoint record: %s", e)


def is_checkpoint_already_processed(checkpoint_id: str, force: bool = False) -> bool:
    if force:
        logger.info("Force flag enabled, skipping duplicate check")
        return False

    processed = load_processed_checkpoints()
    if checkpoint_id in processed:
        existing = processed[checkpoint_id]
        logger.warning(
            "Checkpoint %s already processed on %s",
            checkpoint_id, existing.get("processed_at"),
        )
        return True
    return False


def fetch_checkpoint(api_url: str, checkpoint_id: str | None = None) -> dict:
    if checkpoint_id:
        url = f"{api_url}/checkpoints?status=PENDING"
        logger.info("Fetching checkpoints from %s (looking for %s)", url, checkpoint_id)
        response = requests.get(url, headers={"accept": "application/json"}, timeout=30)
        response.raise_for_status()
        for cp in response.json():
            if cp["id"] == checkpoint_id:
                logger.info(
                    "Found checkpoint %s (status=%s, %d reward entries)",
                    cp["id"], cp["status"], len(cp.get("reward_entries", [])),
                )
                return cp
        raise ValueError(f"Checkpoint {checkpoint_id} not found")
    else:
        url = f"{api_url}/checkpoints/latest"
        logger.info("Fetching latest checkpoint from %s", url)
        response = requests.get(url, headers={"accept": "application/json"}, timeout=30)
        response.raise_for_status()

    checkpoint = response.json()
    logger.info(
        "Fetched checkpoint %s (status=%s, %d reward entries)",
        checkpoint["id"],
        checkpoint["status"],
        len(checkpoint.get("reward_entries", [])),
    )
    return checkpoint


def aggregate_rewards_by_model(reward_entries: list[dict]) -> dict[str, dict]:
    """Aggregate reward amounts per model_id (no horizons in numinous, but keeps the pattern)."""
    aggregated: dict[str, dict] = defaultdict(
        lambda: {"reward_amount": decimal.Decimal("0"), "model_name": None, "player_name": None}
    )

    for entry in reward_entries:
        model_id = entry["model_id"]
        amount = decimal.Decimal(str(entry["reward_amount"]))
        aggregated[model_id]["reward_amount"] += amount
        aggregated[model_id]["model_name"] = entry.get("model_name")
        aggregated[model_id]["player_name"] = entry.get("player_name")

    return aggregated


def generate_payout_file(checkpoint: dict, output_file: Path) -> tuple[int, Path, float]:
    reward_entries = checkpoint.get("reward_entries", [])
    if not reward_entries:
        logger.error("No reward entries in checkpoint %s", checkpoint["id"])
        return 0, output_file, 0.0

    created_at = checkpoint.get("created_at", "")
    try:
        ts = int(datetime.fromisoformat(created_at).timestamp() * 1000)
    except (ValueError, TypeError):
        ts = int(datetime.now(timezone.utc).timestamp() * 1000)

    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    timestamped_file = output_file.parent / f"{output_file.stem}_{date_str}{output_file.suffix}"

    aggregated = aggregate_rewards_by_model(reward_entries)
    prizes_written = 0
    total_amount_usdc = decimal.Decimal("0")

    with open(timestamped_file, "w") as f:
        for model_id, info in aggregated.items():
            amount_microusdc = int(
                (info["reward_amount"] * decimal.Decimal("1000000")).to_integral_value()
            )
            if amount_microusdc <= 0:
                continue

            record = {
                "timestamp": ts,
                "model": model_id,
                "prize": amount_microusdc,
            }
            f.write(json.dumps(record) + "\n")
            prizes_written += 1
            total_amount_usdc += info["reward_amount"]

            logger.info(
                "  %s (%s / %s): $%.2f -> %d microusdc",
                model_id,
                info["model_name"],
                info["player_name"],
                info["reward_amount"],
                amount_microusdc,
            )

    logger.info(
        "Wrote %d prizes totaling $%s USDC to %s",
        prizes_written, total_amount_usdc, timestamped_file,
    )
    return prizes_written, timestamped_file, float(total_amount_usdc)


def run_check_prize_atas(file_path: Path, crunch_name: str, wallet_path: str, multisig_address: str) -> bool:
    cmd = [
        "crunch-coordinator", "crunch", "check-prize-atas",
        "--wallet", wallet_path,
        "--multisig", multisig_address,
        crunch_name, str(file_path),
        "--create",
    ]

    logger.info("Executing: %s", " ".join(cmd))

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info("check-prize-atas executed successfully")
        logger.info("Output: %s", result.stdout)
        if result.stderr:
            logger.warning("Stderr: %s", result.stderr)
        return True
    except subprocess.CalledProcessError:
        logger.exception("check-prize-atas failed")
        return False
    except FileNotFoundError:
        logger.error("crunch-coordinator not found. Ensure it's installed and in PATH.")
        return False


def run_coordinator_cli(
    file_path: Path, crunch_name: str, wallet_path: str, multisig_address: str, silent: bool = False
) -> bool:
    cmd = [
        "crunch-coordinator", "crunch", "checkpoint-create",
        "--wallet", wallet_path,
        "--multisig", multisig_address,
        crunch_name, str(file_path),
    ]

    logger.info("Executing: %s", " ".join(cmd))

    try:
        if silent:
            result = subprocess.run(cmd, text=True, check=False, input="y\n")
        else:
            result = subprocess.run(cmd, text=True, check=False)

        if result.returncode == 0:
            logger.info("coordinator-cli executed successfully")
            return True
        else:
            logger.warning("coordinator-cli cancelled or failed with return code: %s", result.returncode)
            return False
    except subprocess.CalledProcessError:
        logger.exception("coordinator-cli failed")
        return False
    except FileNotFoundError:
        logger.error("crunch-coordinator not found. Ensure it's installed and in PATH.")
        return False


def send_slack_notification(
    slack_webhook_url: str, crunch_name: str, num_prizes: int, total_amount_usdc: float, file_path: Path
) -> bool:
    try:
        message = {
            "text": "New checkpoint created for %s" % crunch_name,
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": (
                            ":trophy: :money_with_wings: *New Checkpoint Created*\n\n"
                            "*Crunch:* %s\n"
                            "*Number of prizes:* %d\n"
                            "*Total Amount:* $%.2f USDC\n"
                            "*File:* %s\n\n"
                            "<%s|Approve here>"
                        )
                        % (crunch_name, num_prizes, total_amount_usdc, file_path.name, SQUADS_URL),
                    },
                }
            ],
        }

        response = requests.post(
            slack_webhook_url, json=message,
            headers={"Content-Type": "application/json"}, timeout=10,
        )

        if response.status_code == 200:
            logger.info("Slack notification sent successfully")
            return True
        else:
            logger.error("Slack notification failed: %s %s", response.status_code, response.text)
            return False
    except Exception:
        logger.exception("Error sending Slack notification")
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate numinous-coordinator checkpoint payout file")

    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="Coordinator API base URL")
    parser.add_argument("--output", default=str(SCRIPT_DIR / "prizes_payout.json"), help="Output JSON file path")
    parser.add_argument("--crunch", default="numinous", help="Crunch name for coordinator-cli")
    parser.add_argument("--wallet", required=True, help="Path to wallet file")
    parser.add_argument("--multisig", required=True, help="Multisig address for checkpoint")
    parser.add_argument("--checkpoint-id", help="Fetch a specific checkpoint by ID")
    parser.add_argument("--no-execute", action="store_true", help="Generate file only")
    parser.add_argument("--slack-webhook", help="Slack webhook URL for notifications")
    parser.add_argument("--force", action="store_true", help="Force processing even if already processed")
    parser.add_argument("--silent", action="store_true", help="Run without prompts (for automation)")

    args = parser.parse_args()
    logger.info("Starting numinous checkpoint payout with args: %s", vars(args))

    try:
        checkpoint = fetch_checkpoint(args.api_url, args.checkpoint_id)
    except requests.RequestException:
        logger.exception("Failed to fetch checkpoint from API")
        sys.exit(1)

    checkpoint_id = checkpoint["id"]

    if is_checkpoint_already_processed(checkpoint_id, args.force):
        logger.error("Checkpoint %s already processed. Use --force to override.", checkpoint_id)
        sys.exit(1)

    output_path = Path(args.output)
    num_prizes, actual_output_path, total_amount_usdc = generate_payout_file(checkpoint, output_path)

    if num_prizes == 0:
        logger.error("No prizes to process")
        sys.exit(1)

    if args.no_execute:
        logger.info("Done (--no-execute)")
        sys.exit(0)

    logger.info("Generated payout file: %s", actual_output_path)
    logger.info("Number of prizes: %d", num_prizes)
    logger.info("Total amount: $%.2f USDC", total_amount_usdc)

    should_execute = args.silent
    if not args.silent:
        should_execute = input("\nExecute crunch-coordinator now? (y/N): ").lower().strip() == "y"

    if not should_execute:
        logger.info("Skipped. Run manually when ready.")
        sys.exit(0)

    logger.info("Step 1: Checking and creating prize ATAs...")
    if not run_check_prize_atas(actual_output_path, args.crunch, args.wallet, args.multisig):
        logger.error("check-prize-atas failed")
        sys.exit(1)

    logger.info("Step 2: Creating checkpoint...")
    if not run_coordinator_cli(actual_output_path, args.crunch, args.wallet, args.multisig, args.silent):
        logger.error("coordinator-cli failed")
        sys.exit(1)

    logger.info("Step 3: Saving processed checkpoint record...")
    save_processed_checkpoint(checkpoint_id, num_prizes, total_amount_usdc)

    if args.slack_webhook:
        logger.info("Step 4: Sending Slack notification...")
        send_slack_notification(args.slack_webhook, args.crunch, num_prizes, total_amount_usdc, actual_output_path)

    logger.info("Done")


if __name__ == "__main__":
    main()