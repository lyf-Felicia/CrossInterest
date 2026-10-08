from __future__ import annotations

import math
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from crossinterest.baselines.mera.core import build_cooccurrence_matrix, build_label_space, home_metrics, paper_metrics, query_set_metrics, relation_infonce, sample_relation_rows, sibling_mask, target_frequency_summary, threshold_cooccurrence, validate_label_space
from crossinterest.baselines.mera.data import build_candidates, expand_unified_eval, parse_official_history, render_prompt, uniform_target_resample
from crossinterest.baselines.mera.modeling import category_representations, category_token_mapping
from crossinterest.baselines.mera.prepare_data import official_label_space
from crossinterest.baselines.mera.prepare_official_home import INPUT_PREFIX, child_name, to_official_record
from crossinterest.baselines.mera.evaluate import unified_metrics
from crossinterest.baselines.mera.evaluate_official_home import validated_home_method
from crossinterest.baselines.mera.evaluate_official_strict import official_formula_metrics, validated_strict_method
from crossinterest.baselines.mera.finalize_crossmodel_metrics import validated_crossmodel_method
from crossinterest.baselines.mera.adapt_official_model import assert_official_semantics, patch_inference, patch_train
from crossinterest.baselines.mera.apply_training_protocol import apply_protocol as apply_training_protocol


HERE = Path(__file__).resolve().parents[1]
SFT_DATA = HERE.parent / 'crossinterest.training' / "data"


def test_real_home_label_space_and_splits() -> None:
    train = pq.read_table(
        SFT_DATA / "train.parquet",
        columns=["user_id", "target_interest"],
    )
    dev = pq.read_table(
        SFT_DATA / "dev.parquet",
        columns=["user_id", "target_interest"],
    )
    labels = set(train["target_interest"].to_pylist())
    label_space = validate_label_space(build_label_space(labels), expected_count=112)
    assert len(set(label_space["category_to_token"].values())) == 112
    assert set(dev["target_interest"].to_pylist()) == labels
    assert set(train["user_id"].to_pylist()).isdisjoint(dev["user_id"].to_pylist())


def test_context_space_can_strictly_contain_prediction_space() -> None:
    label_space = build_label_space(
        ["A > seen-only", "A > target", "B > other"],
        prediction_categories=["A > target", "B > other"],
    )
    validate_label_space(label_space)
    assert label_space["label_count"] == 3
    assert label_space["prediction_label_count"] == 2
    assert "A > seen-only" not in label_space["prediction_labels"]


def test_semantic_text_is_separate_from_structural_path() -> None:
    label_space = build_label_space(
        ["Parent > Child"],
        semantic_texts={"Parent > Child": "Child"},
    )
    validate_label_space(label_space)
    assert label_space["category_tree"] == {"Parent": ["Parent > Child"]}
    assert label_space["category_to_semantic_text"] == {"Parent > Child": "Child"}


def test_target_frequency_summary_includes_zero_count_labels() -> None:
    summary = target_frequency_summary(
        [
            {"targets": ["a"]},
            {"targets": ["a", "b"]},
        ],
        ["a", "b", "c"],
    )
    assert summary["gold_count"] == 3
    assert summary["observed_label_count"] == 2
    assert summary["zero_count_label_count"] == 1
    assert summary["min_nonzero_count"] == 1
    assert summary["max_count"] == 2
    assert summary["count_by_label"] == {"a": 2, "b": 1, "c": 0}


def test_real_home_eval_contract() -> None:
    rows = pq.read_table(SFT_DATA / "eval" / "dev_2k_interest.parquet").to_pylist()
    assert len(rows) == 2_000
    assert len({row["sample_id"] for row in rows}) == 2_000
    assert len({row["target_interest"] for row in rows}) == 112
    assert {len(row["candidate_interests"]) for row in rows} == {10}
    assert all(
        row["candidate_interests"][row["gold_index"]] == row["target_interest"]
        for row in rows
    )
    assert all(
        row["target_interest"] not in set(row["full_seen_interests"])
        for row in rows
    )
    label_space = build_label_space(
        sorted(
            set(
                pq.read_table(SFT_DATA / "train.parquet", columns=["target_interest"])[
                    "target_interest"
                ].to_pylist()
            )
        )
    )
    adapted = [
        {
            "sample_id": row["sample_id"],
            "history": row["historical_interests"],
            "full_seen": row["full_seen_interests"],
            "targets": [row["target_interest"]],
            "candidates": row["candidate_interests"],
        }
        for row in rows
    ]
    expanded = expand_unified_eval(adapted, label_space)
    assert all(
        item["candidates"] == source["candidate_interests"]
        and item["gold_index"] == source["gold_index"]
        and item["candidate_source"] == "frozen"
        for item, source in zip(expanded, rows, strict=True)
    )


def test_relation_construction_and_sampling() -> None:
    labels = ["A > x", "A > y", "B > z", "B > w"]
    label_space = build_label_space(labels)
    assert sibling_mask(label_space).tolist() == [
        [False, True, False, False],
        [True, False, False, False],
        [False, False, False, True],
        [False, False, True, False],
    ]
    counts = build_cooccurrence_matrix(
        [["A > x", "A > y"], ["A > x", "B > z"]],
        label_space["labels"],
    )
    probabilities = threshold_cooccurrence(counts, threshold=0.6)
    assert probabilities[label_space["labels"].index("A > x")].sum() == 0
    rows = sample_relation_rows(
        torch.tensor(sibling_mask(label_space), dtype=torch.float32),
        negative_count=2,
        generator=torch.Generator().manual_seed(7),
    )
    assert rows.shape == (4, 4)
    assert all(sibling_mask(label_space)[anchor, positive] for anchor, positive in rows[:, :2])
    assert all(
        not sibling_mask(label_space)[anchor, negative]
        for anchor, *_, negative_a, negative_b in rows.tolist()
        for negative in (negative_a, negative_b)
    )


def test_relation_infonce_prefers_aligned_positive() -> None:
    representations = torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    good = relation_infonce(representations, torch.tensor([[0, 1, 2]]))
    bad = relation_infonce(representations, torch.tensor([[0, 2, 1]]))
    assert good < bad


def test_category_representations_unwraps_distributed_model() -> None:
    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_embeddings = torch.nn.Embedding(4, 3)
            self.output_embeddings = torch.nn.Linear(3, 4, bias=False)

        def get_input_embeddings(self):
            return self.input_embeddings

        def get_output_embeddings(self):
            return self.output_embeddings

    class DistributedWrapper(torch.nn.Module):
        def __init__(self, module: torch.nn.Module) -> None:
            super().__init__()
            self.module = module

    model = TinyModel()
    token_ids = torch.tensor([1, 3])
    expected = category_representations(model, token_ids)
    actual = category_representations(DistributedWrapper(model), token_ids)
    assert all(torch.equal(left, right) for left, right in zip(expected, actual))


def test_paper_and_home_metrics() -> None:
    paper = paper_metrics(
        [["a", "x", "b"], ["d", "c", "x"]],
        [["a", "b"], ["c"]],
    )
    assert paper["recall_at_1"] == 0.25
    assert paper["precision_at_2"] == 0.5
    home = home_metrics(
        [["a", "b", "c"], ["c", "b", "a"]],
        ["a", "a"],
        [["a", "b"], ["a", "c"]],
    )
    assert home["macro_accuracy_10way"] == 0.5
    assert home["macro_recall_at_10"] == 1.0
    assert math.isclose(home["macro_ndcg_at_10"], 0.75)


def test_official_and_project_paper_metric_implementations_match() -> None:
    rankings = [["a", "x", "b", "y"], ["d", "c", "x", "y"]]
    targets = [["a", "b"], ["c"]]
    official = official_formula_metrics(rankings, targets)
    project = paper_metrics(rankings, targets, cutoffs=(1, 2, 3, 4))
    assert official == project


def test_official_crossmodel_patch_preserves_method_semantics() -> None:
    checkout = Path("/tmp/mera-official-strict-local")
    if not all((checkout / name).is_file() for name in ("train.py", "inference.py")):
        return
    train_source = patch_train((checkout / "train.py").read_text(encoding="utf-8"))
    inference_source = patch_inference(
        (checkout / "inference.py").read_text(encoding="utf-8")
    )
    assert_official_semantics(train_source, inference_source)
    compile(train_source, "train.py", "exec")
    compile(inference_source, "inference.py", "exec")
    train_code = "\n".join(
        line for line in train_source.splitlines() if not line.lstrip().startswith("#")
    )
    inference_code = "\n".join(
        line
        for line in inference_source.splitlines()
        if not line.lstrip().startswith("#")
    )
    assert "prepare_model_for_int8_training" not in train_code
    assert "self.model.model.model.embed_tokens(" not in train_code
    assert "torch.save(model.model.model.embed_tokens.weight.data" not in train_code
    assert "151643" not in train_code
    assert "151643" not in inference_code
    assert "model-native chat template" not in train_source
    assert "max_steps=max_steps" in train_source
    assert 'os.environ.get("MERA_SUPPRESS_BNB_CAST_LOG") == "1"' in train_source
    assert 'logging.getLogger("bitsandbytes.autograd._functions")' in train_source
    assert (
        'train_sampling_strategy="group_by_length" if group_by_length else "random"'
        in train_source
    )
    assert "report_to=[]" in train_source
    assert "structure_loss.to(loss.device)" in train_source
    assert "collaborative_loss.to(loss.device)" in train_source
    assert "LoRA targets must cover every decoder linear projection" in train_source
    assert "PEFT LoRA module mismatch" in train_source
    assert "input embedding and LM head must both use ModulesToSaveWrapper" in train_source
    assert '"optimizer_class": type(runtime_optimizer).__name__' in train_source
    assert '"scheduler_class": type(runtime_scheduler).__name__' in train_source


def test_colleague_evaluation_protocol_forwards_attention_mask(tmp_path: Path) -> None:
    from crossinterest.baselines.mera.apply_evaluation_protocol import apply_protocol

    inference = tmp_path / "inference.py"
    inference.write_text(
        "def evaluate(model, input_ids, inputs):\n"
        "    if True:\n"
        "        with __import__('contextlib').nullcontext():\n"
        "            output = model(input_ids).logits\n",
        encoding="utf-8",
    )
    contract = apply_protocol(inference, "colleague")
    source = inference.read_text(encoding="utf-8")
    assert contract["base_loading"] == "8-bit"
    assert contract["attention_mask_forwarded"] is True
    assert "attention_mask=inputs.attention_mask" in source
    assert "output = model(input_ids).logits" not in source


def test_author_clarified_training_protocol_matches_author_reply(tmp_path: Path) -> None:
    train = tmp_path / "train.py"
    train.write_text(
        "def train():\n"
        "    def generate_and_tokenize_prompt(data_point):\n"
        "        generate_prompt = generate_prompt_with_prompt\n"
        "        return generate_prompt(data_point)\n"
        "    args = dict(\n"
        "            load_best_model_at_end=True,\n"
        '            save_strategy="steps",\n'
        "    )\n"
        "    trainer = dict(\n"
        "        callbacks = [EarlyStoppingCallback(early_stopping_patience=2)],\n"
        "    )\n"
        "    trainer.train(resume_from_checkpoint=resume_from_checkpoint)\n"
        "    model.save_pretrained(output_dir)\n"
        "\n"
        "def generate_prompt_with_prompt(data_point):\n"
        "    return data_point['output']\n",
        encoding="utf-8",
    )

    contract = apply_training_protocol(
        train,
        "author_clarified",
        disable_periodic_checkpoints=True,
    )
    source = train.read_text(encoding="utf-8")

    assert contract == {
        "schema_version": 1,
        "protocol": "author_clarified",
        "source": "author clarification",
        "training_prompt": "generate_prompt_without_prompt",
        "training_sequence": "concatenated quoted history categories + output category",
        "early_stopping": False,
        "load_best_model_at_end": False,
        "completion_contract": True,
        "checkpoint_policy": "final_adapter_only",
        "periodic_checkpoints": False,
        "final_adapter_save": True,
        "source_changed": True,
    }
    assert "generate_prompt = generate_prompt_without_prompt" in source
    assert "generate_prompt = generate_prompt_with_prompt" not in source
    assert "load_best_model_at_end=False" in source
    assert 'save_strategy="no"' in source
    assert 'save_strategy="steps"' not in source
    assert "callbacks = []" in source
    assert '"global_step": trainer.state.global_step' in source
    assert '"max_steps": trainer.state.max_steps' in source
    assert '"epoch": trainer.state.epoch' in source
    assert '"num_train_epochs": trainer.args.num_train_epochs' in source
    namespace: dict = {}
    exec(source, namespace)
    prefixes = {
        "arts": (
            "The user has given high ratings to the following categories of art "
            "crafts before: "
        ),
        "grocery": (
            "The user has given high ratings to the following categories of "
            "groceries before: "
        ),
        "home": (
            "The user has given high ratings to the following categories of home "
            "goods before: "
        ),
    }
    assert {name: len(prefix) for name, prefix in prefixes.items()} == {
        "arts": 82,
        "grocery": 81,
        "home": 82,
    }
    for prefix in prefixes.values():
        formatted = namespace["generate_prompt_without_prompt"](
            {
                "input": prefix + '"Painting", "Drawing"\n',
                "output": '"Craft Supplies"',
            }
        )
        assert formatted == '"Painting""Drawing""Craft Supplies"'


def test_official_training_protocol_does_not_change_source(tmp_path: Path) -> None:
    train = tmp_path / "train.py"
    original = "generate_prompt = generate_prompt_with_prompt\n"
    train.write_text(original, encoding="utf-8")

    contract = apply_training_protocol(train, "official")

    assert train.read_text(encoding="utf-8") == original
    assert contract["training_prompt"] == "generate_prompt_with_prompt"
    assert contract["training_sequence"] == "instruction + input + output"
    assert contract["completion_contract"] is False
    assert contract["checkpoint_policy"] == "periodic_and_final_adapter"
    assert contract["periodic_checkpoints"] is True
    assert contract["final_adapter_save"] is True
    assert contract["source_changed"] is False


def test_author_clarified_qwen2_runner_contract(tmp_path: Path) -> None:
    runner = (HERE / "run_official_strict.sh").read_text(encoding="utf-8")
    arts_config = (HERE / "amlt/amlt_author_clarified_arts.yaml").read_text(
        encoding="utf-8"
    )
    grocery_config = (HERE / "amlt/amlt_author_clarified_qwen2.yaml").read_text(
        encoding="utf-8"
    )

    assert 'TRAIN_PROTOCOL=${TRAIN_PROTOCOL:-official}' in runner
    assert 'FORMAL_TRAIN_EPOCHS=5' in runner
    assert '"training_prompt": (' in runner
    assert '"early_stopping_patience": None if author_clarified else 2' in runner
    assert '"load_best_model_at_end": not author_clarified' in runner
    assert "training-completion.json" in runner
    assert 'completion["global_step"] != expected_max_steps' in runner
    assert '{"arts": 400, "grocery": 690}[sys.argv[3]]' in runner
    assert "TRAINING_PROTOCOL_OPTIONS+=(--disable-periodic-checkpoints)" in runner
    assert (
        'if [[ "${PREFLIGHT_ONLY:-0}" != "1" && "$TRAIN_PROTOCOL" == '
        '"author_clarified" ]]; then'
    ) in runner
    assert '"negative_sample": int(negatives)' in runner
    assert '"contrastive_module": "both"' in runner
    assert arts_config.count("TRAIN_PROTOCOL=author_clarified") == 2
    assert grocery_config.count("TRAIN_PROTOCOL=author_clarified") == 2
    assert arts_config.count("EVAL_PROTOCOL=official") == 2
    assert grocery_config.count("EVAL_PROTOCOL=official") == 2
    assert "PREFLIGHT_ONLY=1 DATASET=arts" in arts_config
    assert "PREFLIGHT_ONLY=1 DATASET=grocery" in grocery_config

    contract_marker = '"$AMLT_OUTPUT_DIR/train_contract.json" <<\'PY\'\n'
    contract_source = runner.split(contract_marker, 1)[1].split("\nPY\n", 1)[0]
    contract_code = compile(
        contract_source, "run_official_strict.sh:train_contract", "exec"
    )
    output = tmp_path / "train_contract.json"
    previous_argv = sys.argv
    try:
        sys.argv = [
            "-",
            "arts",
            "Qwen/Qwen2-1.5B",
            "8a16abf2848eda07cc5253dec660bf1ce007ad7a",
            "143",
            "5",
            "0.01",
            "0.01",
            "author_clarified",
            "official",
            "1",
            "5",
            str(output),
        ]
        exec(contract_code, {})
    finally:
        sys.argv = previous_argv
    contract = json.loads(output.read_text(encoding="utf-8"))
    assert contract["epochs"] == 5
    assert contract["training_prompt"] == "generate_prompt_without_prompt"
    assert contract["early_stopping_patience"] is None
    assert contract["load_best_model_at_end"] is False
    assert contract["checkpoint_policy"] == "final_adapter_only"
    assert contract["save_strategy"] == "no"
    assert contract["save_steps"] is None
    assert contract["negative_sample"] == 5
    assert contract["structure_weight"] == 0.01
    assert contract["collaborative_weight"] == 0.01
    assert contract["evaluation_protocol"] == "official"
    assert contract["author_reference_metrics"]["recall_at_1"] == 0.1209

    output = tmp_path / "grocery_train_contract.json"
    previous_argv = sys.argv
    try:
        sys.argv = [
            "-",
            "grocery",
            "Qwen/Qwen2-1.5B",
            "8a16abf2848eda07cc5253dec660bf1ce007ad7a",
            "172",
            "10",
            "0.1",
            "0.1",
            "author_clarified",
            "official",
            "1",
            "5",
            str(output),
        ]
        exec(contract_code, {})
    finally:
        sys.argv = previous_argv
    grocery_contract = json.loads(output.read_text(encoding="utf-8"))
    assert grocery_contract["dataset"] == "grocery"
    assert grocery_contract["epochs"] == 5
    assert "author_reference_metrics" not in grocery_contract
    assert validated_strict_method(grocery_contract["method"]) == grocery_contract["method"]


def test_author_clarified_crossmodel_three_dataset_contract(tmp_path: Path) -> None:
    runner = (HERE / "run_official_crossmodel.sh").read_text(encoding="utf-8")
    config = (HERE / "amlt/amlt_author_clarified_crossmodel.yaml").read_text(
        encoding="utf-8"
    )

    assert 'TRAIN_PROTOCOL=${TRAIN_PROTOCOL:-official}' in runner
    assert 'FORMAL_TRAIN_EPOCHS=5' in runner
    assert "apply_training_protocol" in runner
    assert '--num_epochs "$FORMAL_TRAIN_EPOCHS"' in runner
    assert config.count("TRAIN_PROTOCOL=author_clarified") == 8
    assert config.count("EVAL_PROTOCOL=official") == 8
    assert config.count("DATASET=arts LOSS_CONDITION=both") == 2
    assert config.count("DATASET=grocery LOSS_CONDITION=both") == 2
    assert config.count("DATASET=home LOSS_CONDITION=both") == 4
    assert "LOSS_CONDITION=sft" not in config
    assert "LOSS_CONDITION=structure" not in config
    assert "LOSS_CONDITION=collaborative" not in config
    assert "PREDICTION_LABEL_COUNT=112" in runner
    assert "evaluate_official_home" in runner
    assert "training-completion.json" in runner
    assert 'completion["epoch"]' in runner
    assert "TRAINING_PROTOCOL_OPTIONS+=(--disable-periodic-checkpoints)" in runner
    assert (
        'if [[ "$EVAL_ONLY" != "1" && "${PREFLIGHT_ONLY:-0}" != "1" && '
        '"$TRAIN_PROTOCOL" == "author_clarified" ]]; then'
    ) in runner

    contract_marker = (
        '"$DISABLE_PERIODIC_CHECKPOINTS" "$FORMAL_TRAIN_EPOCHS" \\\n'
        '  "$AMLT_OUTPUT_DIR/train_contract.json" <<\'PY\'\n'
    )
    contract_source = runner.split(contract_marker, 1)[1].split("\nPY\n", 1)[0]
    contract_code = compile(
        contract_source, "run_official_crossmodel.sh:train_contract", "exec"
    )
    output = tmp_path / "home_train_contract.json"
    previous_argv = sys.argv
    try:
        sys.argv = [
            "-",
            "home",
            "both",
            "qwen35_9b",
            "Qwen/Qwen3.5-9B",
            "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
            "136",
            "112",
            "10",
            "0.05",
            "0.1",
            "248",
            '["q_proj"]',
            "author_clarified",
            "official",
            "1",
            "5",
            str(output),
        ]
        exec(contract_code, {})
    finally:
        sys.argv = previous_argv
    contract = json.loads(output.read_text(encoding="utf-8"))
    assert contract["dataset"] == "home"
    assert contract["loss_condition"] == "both"
    assert contract["epochs"] == 5
    assert contract["training_prompt"] == "generate_prompt_without_prompt"
    assert contract["early_stopping_patience"] is None
    assert contract["load_best_model_at_end"] is False
    assert contract["checkpoint_policy"] == "final_adapter_only"
    assert contract["save_strategy"] == "no"
    assert contract["save_steps"] is None
    assert contract["category_count"] == 136
    assert contract["prediction_category_count"] == 112
    assert contract["structure_weight"] == 0.05
    assert contract["collaborative_weight"] == 0.1
    assert contract["method"] == "MeRA author-clarified Home extension"
    assert "136-context/112-prediction" in contract["equivalence_scope"]
    assert validated_home_method(contract["method"]) == contract["method"]
    assert (
        validated_crossmodel_method("MeRA author-clarified cross-model reproduction")
        == "MeRA author-clarified cross-model reproduction"
    )


def test_official_crossmodel_runner_requires_training_level_preflight() -> None:
    source = (HERE / "run_official_crossmodel.sh").read_text(encoding="utf-8")
    requirements = (HERE / "requirements_amlt.txt").read_text()
    assert "bitsandbytes==0.50.2" in requirements
    assert "fire==0.7.1" in requirements
    assert "datasets==4.4.1" in requirements
    assert "gradio==6.26.0" in requirements
    assert 'for module in ("fire", "datasets", "gradio")' in source
    assert "EXPECTED_DECODER_LINEAR_COUNT=248" in source
    assert "EXPECTED_DECODER_LINEAR_COUNT=224" in source
    assert "in_proj_qkv" in source and "in_proj_a" in source
    assert 'MAX_STEPS_ARGS=(--max_steps 1)' in source
    assert 'MERA_RUNTIME_CONTRACT_PATH=' in source
    assert '(runtime["structure_loss_calls"] > 0) != expected_structure' in source
    assert '(runtime["collaborative_loss_calls"] > 0) != expected_collaborative' in source
    assert '(runtime["structure_loss_device"] is not None) != expected_structure' in source
    assert '(runtime["collaborative_loss_device"] is not None) != expected_collaborative' in source
    assert 'runtime["main_loss_device"] != runtime["combined_loss_device"]' in source
    assert 'runtime["train_sampling_strategy"] != "group_by_length"' in source
    assert 'runtime["report_to"] != []' in source
    assert 'PeftModel.from_pretrained(base, adapter_dir)' in source
    assert 'completion_path = input_dir / "training-completion.json"' in source
    assert 'completion["global_step"] != completion["max_steps"]' in source
    assert 'completion["num_train_epochs"], expected_epochs' in source
    recovery_source = source.split(
        '"$AMLT_OUTPUT_DIR/recovery-contract.json" <<\'PY\'\n', 1
    )[1].split("\nPY\n", 1)[0]
    assert "import math" in recovery_source
    assert "official cross-model training preflight passed" in source


def test_official_home_data_and_loss_factorial_contract() -> None:
    import ast

    full_to_child = {
        "Bath > Bath Rugs": "Bath Rugs",
        "Kitchen & Dining > Bakeware": "Bakeware",
    }
    row = {
        "historical_interests": ["Kitchen & Dining > Bakeware"],
        "target_interest": "Bath > Bath Rugs",
    }
    record = to_official_record(row, full_to_child)
    assert len(INPUT_PREFIX) == 82
    assert ast.literal_eval(record["input"][81:].strip("\n")) == "Bakeware"
    assert record["output"] == '"Bath Rugs"'
    assert child_name("Bath > Bath Rugs") == "Bath Rugs"

    runner = (HERE / "run_official_crossmodel.sh").read_text(encoding="utf-8")
    config = (HERE / "amlt/amlt_official_home_ablations.yaml").read_text(
        encoding="utf-8"
    )
    assert 'elif [[ "$DATASET" == "home" ]]' in runner
    assert "PREDICTION_LABEL_COUNT=112" in runner
    assert "STRUCTURE_WEIGHT=0.05" in runner
    assert "COLLABORATIVE_WEIGHT=0.1" in runner
    assert 'DATA_ROOT="$AMLT_OUTPUT_DIR/home-assets"' in runner
    assert "evaluate_official_home" in runner
    prepare_source = (HERE / "prepare_official_home.py").read_text(encoding="utf-8")
    assert ").astype(np.float32)" in prepare_source
    assert '"cooccurrence_dtype": str(cooccurrence.dtype)' in prepare_source
    expected_jobs = {
        "qwen35-home-official-preflight",
        "llama31-home-official-preflight",
        "qwen35-home-sft",
        "qwen35-home-structure",
        "qwen35-home-collaborative",
        "qwen35-home-both",
        "llama31-home-sft",
        "llama31-home-structure",
        "llama31-home-collaborative",
        "llama31-home-both",
    }
    assert all(f"name: {job}" in config for job in expected_jobs)
    assert config.count("LOSS_CONDITION=sft ") == 2
    assert config.count("LOSS_CONDITION=structure ") == 2
    assert config.count("LOSS_CONDITION=collaborative ") == 2
    assert config.count("LOSS_CONDITION=both ") == 4
    assert "<placeholder>" not in config


def test_official_strict_runner_supports_validated_eval_recovery() -> None:
    source = (HERE / "run_official_strict.sh").read_text(encoding="utf-8")
    requirements = (HERE / "requirements_official_strict.txt").read_text()
    recovery = (HERE / "amlt/amlt_official_strict_recovery.yaml").read_text()
    assert "gradio==3.50.2" in requirements
    assert '"gradio": "3.50.2"' in source
    assert 'if [[ "${EVAL_ONLY:-0}" == "1" ]]' in source
    assert (
        'if [[ "${EVAL_ONLY:-0}" == "1" && -z "${TRAIN_PROTOCOL:-}" ]]'
        in source
    )
    assert '["training_protocol"]' in source
    assert "mapped training contract differs" in source
    assert "mapped final adapter is incomplete" in source
    assert 'completion = load(source / "training-completion.json")' in source
    assert 'source_train["checkpoint_policy"] == "final_adapter_only"' in source
    assert '{"arts": 400, "grocery": 690}[source_train["dataset"]]' in source
    assert 'completion["global_step"] != expected_max_steps' in source
    recovery_source = source.split(
        '"$AMLT_OUTPUT_DIR/recovery-contract.json" <<\'PY\'\n', 1
    )[1].split("\nPY\n", 1)[0]
    compile(recovery_source, "run_official_strict.sh:recovery", "exec")
    assert '"training_reused": True' in source
    assert "official strict evaluation-only recovery passed" in source
    assert "$$AMLT_MAP_INPUT_DIR" not in recovery
    assert "EVAL_ONLY=1" in recovery
    assert recovery.count("TRAIN_PROTOCOL=author_clarified") == 2
    assert "eval_interest_id_minionerec/" in recovery
    assert "80G1-A100" in recovery


def test_multi_gold_set_metrics_are_normalized_and_source_weighted() -> None:
    rankings = [
        ["a", "b", "c", "d", "e"],
        ["x", "y", "z", "q", "target"],
    ]
    targets = [["a", "b", "c", "d"], ["target"]]
    metrics = query_set_metrics(rankings, targets, cutoffs=(4,))
    assert metrics["mean_recall_at_4"] == 0.5
    assert metrics["mean_ndcg_at_4"] == 0.5
    assert metrics["source_sample_count"] == 2


def test_set_metrics_reduce_to_single_gold_home_formula() -> None:
    rankings = [["a", "b", "c"], ["c", "b", "a"]]
    targets = [["a"], ["a"]]
    metrics = query_set_metrics(rankings, targets, cutoffs=(10,))
    assert metrics["mean_recall_at_10"] == 1.0
    assert math.isclose(metrics["mean_ndcg_at_10"], 0.75)


def test_unified_multi_gold_averages_accuracy_within_source() -> None:
    labels = [
        f"{parent} > {parent}{index}"
        for parent in ("A", "B", "C")
        for index in range(5)
    ]
    label_space = build_label_space(labels)
    rows = [
        {
            "sample_id": "multi",
            "history": [],
            "full_seen": [],
            "targets": ["A > A0", "B > B0"],
        },
        {
            "sample_id": "single",
            "history": [],
            "full_seen": [],
            "targets": ["C > C0"],
        },
    ]
    rankings = [
        ["A > A0", *[label for label in labels if label != "A > A0"]],
        ["C > C0", *[label for label in labels if label != "C > C0"]],
    ]
    metrics, details = unified_metrics(rows, rankings, label_space)
    assert metrics["query_mean"]["source_sample_count"] == 2
    assert len(details) == 2
    assert metrics["category_macro_single_gold_only"] is None
    assert all("gold_details" in detail for detail in details)


def test_official_parser_and_atomic_history_tokens() -> None:
    history = parse_official_history(
        "The user has given high ratings to the following categories of groceries "
        'before: "Tea", "Cookies"\n'
    )
    assert history == ["Tea", "Cookies"]
    label_space = build_label_space(
        history,
        category_tree={"Food": history},
        preserve_order=True,
    )
    prompt = render_prompt(
        {"history": history},
        label_space["category_to_token"],
    )
    assert "<category_000>" in prompt
    assert "<category_001>" in prompt
    assert "Tea" not in prompt


def test_qwen2_reference_uses_official_token_and_prompt_contract() -> None:
    history = ["Tea", "Cookies"]
    label_space = build_label_space(
        history,
        category_tree={"Food": history},
        preserve_order=True,
    )
    tokens = category_token_mapping(label_space, "official_quoted_name")
    row = {
        "history": history,
        "official_instruction": "Predict a novel category.",
        "official_input": 'History: "Tea", "Cookies"\n',
    }
    training_prompt = render_prompt(
        row,
        tokens,
        model_key="qwen2_1_5b",
        training=True,
    )
    evaluation_prompt = render_prompt(
        row,
        tokens,
        model_key="qwen2_1_5b",
        training=False,
    )
    assert tokens == {"Tea": '"Tea"', "Cookies": '"Cookies"'}
    assert "### Instruction:\nPredict a novel category." in training_prompt
    assert "### Input:\nHistory:" in training_prompt
    assert training_prompt.endswith("### Response:\n")
    assert evaluation_prompt == '"Tea""Cookies"'


def test_uniform_target_resample_preserves_budget_and_balances_observed_labels() -> None:
    rows = [
        {"sample_id": f"a:{index}", "targets": ["a"]}
        for index in range(5)
    ] + [
        {"sample_id": "b:0", "targets": ["b"]},
        {"sample_id": "c:0", "targets": ["c"]},
    ]
    balanced = uniform_target_resample(rows, seed=2024)
    counts = target_frequency_summary(balanced, ["a", "b", "c"])
    assert len(balanced) == len(rows)
    assert counts["min_nonzero_count"] == 2
    assert counts["max_count"] == 3
    assert balanced == uniform_target_resample(rows, seed=2024)


def test_unified_eval_expands_multigold_without_false_negatives() -> None:
    labels = [
        "A > a",
        "A > b",
        "A > c",
        "A > d",
        "B > e",
        "B > f",
        "B > g",
        "B > h",
        "C > i",
        "C > j",
        "C > k",
        "C > l",
        "D > m",
        "D > n",
    ]
    label_space = build_label_space(labels)
    row = {
        "sample_id": "test:0",
        "history": ["A > a"],
        "full_seen": ["A > a"],
        "targets": ["A > b", "B > e"],
    }
    expanded = expand_unified_eval([row], label_space, seed=7)
    assert len(expanded) == 2
    assert {item["targets"][0] for item in expanded} == {"A > b", "B > e"}
    for item in expanded:
        target = item["targets"][0]
        other_gold = ({"A > b", "B > e"} - {target}).pop()
        assert len(item["candidates"]) == 10
        assert target in item["candidates"]
        assert "A > a" not in item["candidates"]
        assert other_gold not in item["candidates"]
        assert item["candidates"] == build_candidates(
            row,
            target,
            label_space,
            seed=7,
        )[0]


def test_official_label_space_accepts_fetched_data_root(tmp_path) -> None:
    source = tmp_path / "Arts"
    source.mkdir()
    (source / "layer2_map.txt").write_text("First\t0\nSecond\t1\n")
    (source / "category_tree.json").write_text(
        '{"Parent": ["First", "Second"]}'
    )
    label_space, resolved_source, _ = official_label_space(tmp_path, "arts")
    assert resolved_source == source
    assert label_space["labels"] == ["First", "Second"]
