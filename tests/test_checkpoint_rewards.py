from crunch_node.services.checkpoint_service import CheckpointService


def _make_service(**kwargs):
    return CheckpointService(pg_client=None, **kwargs)


def _row(miner_uid, rank, score, track="SIGNAL", cruncher_id=None):
    return {"miner_uid": miner_uid, "track": track, "rank": rank,
            "weighted_score": score, "cruncher_id": cruncher_id,
            "model_name": None, "cruncher_name": None, "deployment_id": None}


def test_basic_top_k():
    """Top 3 miners get exponential rewards, rest get nothing."""
    svc = _make_service(reward_pool=1000, top_k=3, alpha=1.0)
    rows = [_row(i, i, 0.1 * i) for i in range(1, 6)]
    entries = svc._compute_rewards(rows)

    assert len(entries) == 3
    assert all(e["reward_amount"] > 0 for e in entries)
    assert entries[0]["reward_amount"] > entries[1]["reward_amount"] > entries[2]["reward_amount"]

    total = sum(e["reward_amount"] for e in entries)
    assert total <= 1000


def test_benchmark_cutoff():
    """Miners worse than benchmark get reward=0, keeping their rank."""
    svc = _make_service(reward_pool=1000, top_k=5, alpha=1.0, benchmark_miner_uid=99)
    rows = [
        _row(1, 1, 0.10),
        _row(2, 2, 0.20),
        _row(99, 3, 0.25),  # benchmark
        _row(4, 4, 0.30),
        _row(5, 5, 0.40),
    ]
    entries = svc._compute_rewards(rows)

    assert len(entries) == 5
    paid = [e for e in entries if e["reward_amount"] > 0]
    assert len(paid) == 2
    assert paid[0]["model_id"] == "1"
    assert paid[1]["model_id"] == "2"

    benchmark_entry = next(e for e in entries if e["model_id"] == "99")
    assert benchmark_entry["reward_amount"] == 0


def test_duplicate_cruncher():
    """Same cruncher with 2 miners: duplicate stays in top K but gets reward=0."""
    svc = _make_service(reward_pool=1000, top_k=5, alpha=1.0)
    rows = [
        _row(1, 1, 0.10, cruncher_id="alice"),
        _row(2, 2, 0.15, cruncher_id="bob"),
        _row(3, 3, 0.20, cruncher_id="alice"),  # dup -> reward=0
        _row(4, 4, 0.25, cruncher_id="charlie"),
        _row(5, 5, 0.30, cruncher_id="bob"),  # dup -> reward=0
    ]
    entries = svc._compute_rewards(rows)

    assert len(entries) == 5  # all stay in top K
    assert entries[0]["model_id"] == "1" and entries[0]["reward_amount"] > 0
    assert entries[1]["model_id"] == "2" and entries[1]["reward_amount"] > 0
    assert entries[2]["model_id"] == "3" and entries[2]["reward_amount"] == 0  # dup alice
    assert entries[3]["model_id"] == "4" and entries[3]["reward_amount"] > 0
    assert entries[4]["model_id"] == "5" and entries[4]["reward_amount"] == 0  # dup bob


def test_benchmark_and_duplicate_combined():
    """Benchmark + duplicate cruncher: both zero out, no extension."""
    svc = _make_service(reward_pool=1000, top_k=5, alpha=1.0, benchmark_miner_uid=99)
    rows = [
        _row(1, 1, 0.10, cruncher_id="alice"),
        _row(2, 2, 0.15, cruncher_id="alice"),  # dup -> reward=0
        _row(99, 3, 0.20),  # benchmark -> reward=0
        _row(4, 4, 0.25, cruncher_id="bob"),  # worse than benchmark -> reward=0
        _row(5, 5, 0.30, cruncher_id="charlie"),  # worse than benchmark -> reward=0
    ]
    entries = svc._compute_rewards(rows)

    assert len(entries) == 5
    paid = [e for e in entries if e["reward_amount"] > 0]
    assert len(paid) == 1
    assert paid[0]["model_id"] == "1"


def test_no_benchmark():
    """Without benchmark, all top K get rewards."""
    svc = _make_service(reward_pool=1000, top_k=3, alpha=1.0)
    rows = [_row(i, i, 0.1 * i) for i in range(1, 6)]
    entries = svc._compute_rewards(rows)

    paid = [e for e in entries if e["reward_amount"] > 0]
    assert len(paid) == 3


def test_ranks_preserved():
    """Ranks stay fixed after benchmark/dedup zeroing — no shifting."""
    svc = _make_service(reward_pool=1000, top_k=5, alpha=1.0, benchmark_miner_uid=99)
    rows = [
        _row(1, 1, 0.10),
        _row(99, 2, 0.15),
        _row(3, 3, 0.20),
        _row(4, 4, 0.25),
        _row(5, 5, 0.30),
    ]
    entries = svc._compute_rewards(rows)

    assert entries[0]["rank"] == 1
    assert entries[0]["reward_amount"] > 0
    assert entries[1]["rank"] == 2
    assert entries[1]["reward_amount"] == 0


def test_output_format():
    """Verify output fields match condorgame checkpoint format."""
    svc = _make_service(reward_pool=1000, top_k=2, alpha=1.0)
    rows = [
        _row(42, 1, 0.10, cruncher_id="player-abc"),
    ]
    rows[0]["model_name"] = "my-model"
    rows[0]["cruncher_name"] = "alice"
    rows[0]["deployment_id"] = "dep-123"

    entries = svc._compute_rewards(rows)
    e = entries[0]

    assert e["model_id"] == "42"
    assert e["model_name"] == "my-model"
    assert e["player_id"] == "player-abc"
    assert e["player_name"] == "alice"
    assert e["deployment_id"] == "dep-123"
    assert e["track"] == "SIGNAL"
    assert e["rank"] == 1
    assert "weighted_score" in e
    assert "weight" in e
    assert "reward_fraction" in e
    assert "reward_amount" in e