import hashlib
import json
from pathlib import Path

from crossinterest.baselines.google import evaluate as google_evaluate
from crossinterest.baselines.google.prompts import cluster_output, render_full_history_generation_prompt, render_generation_prompt, render_training_messages, select_recent_unique_clusters
from crossinterest.baselines.google.build_dataset import build_train_rows, frozen_rebuild_matches, recent_unique_pair
from crossinterest.baselines.google.evaluate_ranking import load_labels, seed_resume_details, validate_full_history_rows
from crossinterest.training.evaluate_ranking import score_ranking


LABEL_A = "Bath > Bath Rugs"
LABEL_B = "Kitchen & Dining > Bakeware"
LABEL_C = "Furniture > Bedroom Furniture"


def test_recent_unique_clusters_preserve_chronological_order() -> None:
    history = [LABEL_A, "unsupported", LABEL_B, LABEL_A, LABEL_C, LABEL_C]
    assert select_recent_unique_clusters(history, {LABEL_A, LABEL_B, LABEL_C}) == [
        LABEL_A,
        LABEL_C,
    ]


def test_generation_prompt_is_k2_candidate_free_and_paper_shaped() -> None:
    prompt = render_generation_prompt([LABEL_A, LABEL_B])
    assert "Cluster 1: {Bath, Bath Rugs}" in prompt
    assert "Cluster 2: {Kitchen & Dining, Bakeware}." in prompt
    assert "less than 30 words" in prompt
    assert 'prefix that says "New cluster: "' in prompt
    assert "Candidates:" not in prompt


def test_full_history_ranking_prompt_preserves_every_cluster_and_has_no_candidates() -> None:
    history = [LABEL_A, LABEL_B, LABEL_A, LABEL_C]
    titles = ["rug one", "baking tin", "rug two", "bed frame"]
    prompt = render_full_history_generation_prompt(titles, history)
    expected_items = [
        f"1. [{LABEL_A}] rug one",
        f"2. [{LABEL_B}] baking tin",
        f"3. [{LABEL_A}] rug two",
        f"4. [{LABEL_C}] bed frame",
    ]
    positions = [prompt.index(item) for item in expected_items]
    assert positions == sorted(positions)
    assert "Candidates:" not in prompt


def test_full_history_ranking_dev_covers_all_frozen_samples() -> None:
    path = Path('crossinterest.training/data/eval/dev_2k_behavior_interest.parquet')
    rows = google_evaluate.pq.read_table(
        path,
        columns=[
            "sample_id",
            "historical_titles",
            "historical_interests",
            "history_length",
            "input_type",
        ],
    ).to_pylist()
    assert len(rows) == 2_000
    assert len({row["sample_id"] for row in rows}) == 2_000
    assert all(
        len(row["historical_titles"])
        == len(row["historical_interests"])
        == row["history_length"]
        for row in rows
    )
    validate_full_history_rows(rows, 2_000)


def test_full_history_ranking_rejects_k2_rows() -> None:
    rows = google_evaluate.pq.read_table(
        'crossinterest.baselines.google/data/dev_2k_pair.parquet'
    ).to_pylist()
    try:
        validate_full_history_rows(rows, len(rows))
    except ValueError as error:
        assert "behavior_interest" in str(error)
    else:
        raise AssertionError("K=2 rows must not be accepted by full-history ranking")


def test_training_messages_enforce_novel_target() -> None:
    row = {"cluster_pair": [LABEL_A, LABEL_B], "target_cluster": LABEL_C}
    assert render_training_messages(row)[1]["content"] == cluster_output(LABEL_C)

    row["target_cluster"] = LABEL_A
    try:
        render_training_messages(row)
    except ValueError as error:
        assert "appears" in str(error)
    else:
        raise AssertionError("a repeated target must be rejected")


def test_recent_unique_pair_skips_unsupported_and_duplicate_clusters() -> None:
    history = [LABEL_A, "unsupported", LABEL_B, LABEL_A, LABEL_C, LABEL_C]
    assert recent_unique_pair(history, {LABEL_A, LABEL_B, LABEL_C}) == (
        LABEL_A,
        LABEL_C,
    )


def test_top_pairs_use_frequency_then_lexicographic_tie_break() -> None:
    pair_a = (LABEL_A, LABEL_B)
    pair_b = (LABEL_B, LABEL_A)
    other_pairs = {
        (LABEL_A, f"Furniture > Label {index:02d}"): 10 - index
        for index in range(8)
    }
    counts = {LABEL_C: {pair_b: 20, pair_a: 20, **other_pairs}}
    representatives = {
        (LABEL_C, pair): f"source-{index}"
        for index, pair in enumerate(counts[LABEL_C])
    }
    rows, _ = build_train_rows(
        [LABEL_C],
        counts,
        representatives,
        "google-interest-v2",
    )
    assert rows[0]["cluster_pair"] == list(pair_a)
    assert rows[1]["cluster_pair"] == list(pair_b)
    assert [row["target_pair_rank"] for row in rows] == list(range(1, 11))
    assert all(row["target_cluster"] not in row["cluster_pair"] for row in rows)


def test_ranking_metrics_accept_google_cluster_outputs() -> None:
    ranking = [
        {
            "rank": 1,
            "raw_text": cluster_output(LABEL_C),
            "predicted_output": cluster_output(LABEL_C),
            "prediction": LABEL_C,
            "format_valid": True,
            "sequence_score": -0.1,
        }
    ]
    scored = score_ranking(ranking, LABEL_C)
    assert scored["target_rank"] == 1
    assert scored["recall_at_10"] == 1.0
    assert scored["ndcg_at_20"] == 1.0


def test_ranking_loads_labels_from_baseline_train() -> None:
    labels = load_labels(
        Path('crossinterest.baselines.google/data/train_top10_pairs.parquet'),
        112,
    )
    assert len(labels) == 112
    assert labels == sorted(labels)


def test_frozen_rebuild_requires_bytes_and_manifest() -> None:
    comparison = {
        "train_top10_pairs.parquet": {"byte_sha256_match": True},
        "dev_2k_pair.parquet": {"byte_sha256_match": True},
        "pair_grid.parquet": {"byte_sha256_match": True},
        "MANIFEST.json": {"stable_fields_match": True},
    }
    assert frozen_rebuild_matches(comparison)
    comparison["dev_2k_pair.parquet"]["byte_sha256_match"] = False
    assert not frozen_rebuild_matches(comparison)
    comparison["dev_2k_pair.parquet"]["byte_sha256_match"] = True
    comparison["MANIFEST.json"]["stable_fields_match"] = False
    assert not frozen_rebuild_matches(comparison)


def test_category_model_loader_adapts_shared_signature(monkeypatch) -> None:
    calls = []
    expected_model = object()

    def fake_load_model(*args):
        calls.append(args)
        return expected_model

    monkeypatch.setattr(google_evaluate, "load_sft_model", fake_load_model)
    adapter = Path("adapter")
    tokenizer = object()
    model = google_evaluate.load_category_model("qwen35_9b", "model", adapter, tokenizer)

    assert model is expected_model
    assert calls == [("qwen35_9b", "model", adapter, tokenizer, None, None)]


def test_ranking_resume_requires_identical_contract(tmp_path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    contract = {"fingerprint": "run-fingerprint", "num_beams": 20}
    (source / "run_contract.json").write_text(json.dumps(contract))
    details = [
        {"sample_id": "sample-1", "run_fingerprint": "run-fingerprint"},
        {"sample_id": "sample-2", "run_fingerprint": "run-fingerprint"},
    ]
    (source / "details.jsonl").write_text(
        "".join(json.dumps(detail) + "\n" for detail in details)
    )

    seed_resume_details(source, target, contract, expected_resume_rows=2, resume_adapter_sha256=None)
    assert (target / "details.jsonl").read_text() == (source / "details.jsonl").read_text()

    try:
        seed_resume_details(source, tmp_path / "other", {**contract, "num_beams": 21}, 2, None)
    except ValueError as error:
        assert "contract differs" in str(error)
    else:
        raise AssertionError("a changed resume contract must be rejected")


def test_ranking_resume_accepts_verified_adapter_path_alias(tmp_path) -> None:
    source = tmp_path / "source"
    source_run = source / "ranking"
    source_adapter = source / "train" / "final_adapter"
    target = tmp_path / "target"
    source_run.mkdir(parents=True)
    source_adapter.mkdir(parents=True)
    target.mkdir()
    adapter_weights = source_adapter / "adapter_model.safetensors"
    adapter_weights.write_bytes(b"verified adapter weights")
    adapter_sha256 = hashlib.sha256(adapter_weights.read_bytes()).hexdigest()
    source_contract = {
        "fingerprint": "source-fingerprint",
        "adapter": "/mnt/output/source/train/final_adapter",
        "num_beams": 20,
    }
    current_contract = {
        **source_contract,
        "fingerprint": "target-fingerprint",
        "adapter": str(source_adapter),
    }
    (source_run / "run_contract.json").write_text(json.dumps(source_contract))
    (source_run / "details.jsonl").write_text(
        json.dumps({"sample_id": "sample-1", "run_fingerprint": "source-fingerprint"}) + "\n"
    )

    seed_resume_details(source_run, target, current_contract, 1, adapter_sha256)

    resumed = json.loads((target / "details.jsonl").read_text())
    provenance = json.loads((target / "resume_provenance.json").read_text())
    assert resumed["run_fingerprint"] == "target-fingerprint"
    assert provenance["source_fingerprint"] == "source-fingerprint"
    assert provenance["adapter_model_sha256"] == adapter_sha256