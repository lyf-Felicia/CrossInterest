
from crossinterest.paths import source_workspace as _source_workspace
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from crossinterest.training.label_space import build_label_mapping
from crossinterest.training.prompts import render_training_messages
from crossinterest.methods.alignment.data import ALIGNMENT_TASKS, RECOMMENDATION_TASKS, alignment_rows, recommendation_rows, render_messages


class FrequencyDataTests(unittest.TestCase):
    def test_frequency_collection_requires_final_checkpoint_and_preserved_smoke(self):
        from crossinterest.methods.frequency.audit import select_inventory, verify_preserved_smoke
        from crossinterest.methods.alignment.data import sha256_file
        names = {"EXECUTION_CONTRACT.json", "AUDIT_RECOVERY_CONTRACT.json", "smoke/RESULT_AUDIT.json",
                 "smoke/novel/STAGE_AUDIT.json", "smoke/novel/checkpoint-7/trainer_state.json",
                 "smoke/novel/checkpoint-7/adapter_model.safetensors"}
        inventory = [{"name": name} for name in sorted(names)]
        selected = select_inventory(inventory + [{"name": "formal/novel/train_metrics.json"},
                                                {"name": "smoke/novel/checkpoint-7/optimizer.pt"}], True)
        self.assertEqual({entry["name"] for entry in selected}, names)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            select_inventory(inventory[:-1], True)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            select_inventory(inventory, False)
        with self.assertRaisesRegex(ValueError, "unsafe"):
            select_inventory(inventory + [{"name": "../secret.json"}], True)
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            paths = {"execution_contract_sha256": "EXECUTION_CONTRACT.json",
                     "original_metrics_sha256": "smoke/novel/train_metrics.json",
                     "final_adapter_sha256": "smoke/novel/final_adapter/adapter_model.safetensors"}
            for relative in paths.values():
                path = directory / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"preserved")
            preservation = {"conditions": {"once": {key: sha256_file(directory / relative)
                                                    for key, relative in paths.items()}}}
            verify_preserved_smoke(directory, "once", preservation)
            for relative in paths.values():
                with self.subTest(file=relative):
                    (directory / relative).write_bytes(b"changed")
                    with self.assertRaisesRegex(ValueError, "original smoke evidence changed"):
                        verify_preserved_smoke(directory, "once", preservation)
                    (directory / relative).write_bytes(b"preserved")

    def test_frequency_evaluation_cli_and_balanced_loading(self):
        from unittest.mock import patch
        from crossinterest.training import evaluate, evaluate_ranking
        arguments = ["--model-key", "qwen35_9b", "--model", "model", "--adapter", "/tmp/adapter",
                     "--view", "behavior_interest", "--label-mode", "interest_id", "--run-dir", "/tmp/eval"]
        for module in (evaluate, evaluate_ranking):
            self.assertEqual(module.parse_args(arguments).device_map, "cuda")
            self.assertEqual(module.parse_args(arguments + ["--device-map", "balanced"]).device_map, "balanced")
        ranking = evaluate_ranking.parse_args(arguments + ["--num-beams", "20", "--decoding", "direct_beam_search"])
        self.assertEqual((ranking.num_beams, ranking.decoding), (20, "direct_beam_search"))
        with patch.object(evaluate.AutoModelForCausalLM, "from_pretrained") as base, \
             patch("peft.PeftModel.from_pretrained") as adapter:
            model = evaluate.load_model("qwen35_9b", "model", Path("/tmp/adapter"), None, None, None,
                                        device_map="balanced")
        self.assertEqual(base.call_args.kwargs["device_map"], "balanced")
        self.assertIs(model, adapter.return_value)

    def test_audit_recovery_preserves_old_contract_and_rejects_training_drift(self):
        import json
        from crossinterest.methods.frequency.runner import execution_contract
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            old = {"files": {"train": {"rows": 112000}}, "condition": "once", "code_sha256": {
                "eval_interest_id_minionerec/crossfit.py": "059ff2ea7be0907f5faf98f72c0f1e231801cd9dfb844da58e91787af2a448cc",
                "eval_interest_id_minionerec/frequency_run.py": "a95519f389a0f2a1108accb2db8ff535bd431ccafd0cb11aa80f0565f4456d4e",
                'crossinterest.training/evaluate.py': "2bf39a81d8e882ba90bd135648ce4725d5b985293195bb95d4ac2b75f56a0d6f",
                'crossinterest.training/evaluate_ranking.py': "dfcd1033de7f268df1fa0b4a9497f5f3f4962355b98795c97fc24a0995f761ff",
                'crossinterest.training/train.py': "unchanged"}}
            path = output / "EXECUTION_CONTRACT.json"
            path.write_text(json.dumps(old))
            before = path.read_bytes()
            new = old | {"code_sha256": {key: value if "train.py" in key else "patched" for key, value in old["code_sha256"].items()}}
            execution_contract(output, new)
            execution_contract(output, new)
            self.assertEqual(path.read_bytes(), before)
            self.assertTrue((output / "AUDIT_RECOVERY_CONTRACT.json").is_file())
            with self.assertRaisesRegex(ValueError, "training or data contract"):
                execution_contract(output, new | {"files": {"train": {"rows": 1000}}})
            with self.assertRaisesRegex(ValueError, "beyond the audited"):
                execution_contract(output, new | {"code_sha256": new["code_sha256"] | {'crossinterest.training/train.py': "changed"}})

    def test_completion_metrics_require_matching_completed_checkpoint(self):
        import json
        from crossinterest.methods.crossfit.train import completion_metrics
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            checkpoint = directory / "checkpoint-7"
            final = directory / "final_adapter"
            checkpoint.mkdir()
            final.mkdir()
            state = {"global_step": 7, "max_steps": 7, "epoch": 1.0}
            (checkpoint / "trainer_state.json").write_text(json.dumps(state))
            (checkpoint / "adapter_model.safetensors").write_bytes(b"identical weights")
            (final / "adapter_model.safetensors").write_bytes(b"identical weights")
            metrics = {"epoch": 1.0, "sample_count": 224}
            self.assertEqual(completion_metrics(directory, metrics, 7), metrics | state)
            self.assertNotIn("global_step", metrics)
            self.assertEqual(completion_metrics(directory, metrics | {"global_step": 6}, 7)["global_step"], 6)
            (final / "adapter_model.safetensors").write_bytes(b"different weights")
            with self.assertRaisesRegex(ValueError, "differs from completed checkpoint"):
                completion_metrics(directory, metrics, 7)
            (final / "adapter_model.safetensors").write_bytes(b"identical weights")
            (checkpoint / "trainer_state.json").write_text(json.dumps(state | {"global_step": 6}))
            with self.assertRaisesRegex(ValueError, "global_step"):
                completion_metrics(directory, metrics, 7)

    def test_frequency_training_is_fresh_base_n_only_with_equal_budget(self):
        from crossinterest.methods.frequency.runner import training_command
        prepared = {"files": {arm: {"path": f"frequency/{arm}.parquet", "rows": 112000}
                              for arm in ("once", "at_least_five")}}
        commands = [training_command("model", arm, Path("output"), False, prepared)
                    for arm in ("once", "at_least_five")]
        for command in commands:
            self.assertIn('crossinterest.training.train', command)
            self.assertNotIn("--init-adapter", command)
            self.assertNotIn("--stage", command)
            self.assertNotIn("--selection-file", command)
            self.assertEqual(command[command.index("--expected-train-rows") + 1], "112000")
            self.assertEqual(command[command.index("--epochs") + 1], "1")
            self.assertEqual(command[command.index("--expected-effective-batch-size") + 1], "32")
        path_index = commands[0].index("--train-file") + 1
        self.assertNotEqual(commands[0][path_index], commands[1][path_index])
        commands[1][path_index] = commands[0][path_index]
        self.assertEqual(*commands)

    def test_paired_replacement_reserves_identical_disjoint_background(self):
        from crossinterest.methods.frequency.data import paired_rows
        original = [{"id": f"{category}{number}", "target_interest": category}
                    for category in ("A", "B", "C") for number in range(4)]
        once = [{"id": f"{category}{number}", "target_interest": category,
                 "future_target_count_including_first": 1}
                for category in ("A", "B", "C") for number in (0, 1, 2, 4)]
        five = [{"id": "B3", "target_interest": "B", "future_target_count_including_first": 5}]
        five += [{"id": f"C{number}", "target_interest": "C", "future_target_count_including_first": 5}
                 for number in (3, 5, 6, 7)]
        arms, slots = paired_rows(original, once, five, {"A", "B", "C"}, cap=4)
        self.assertEqual(len(arms["once"]), 12)
        self.assertEqual(sum(slot["role"] == "background" for slot in slots), 7)
        self.assertEqual(sum(slot["role"] == "replacement" for slot in slots), 5)
        self.assertIn("B4", {row["id"] for row in arms["once"]})
        for left, right, slot in zip(arms["once"], arms["at_least_five"], slots):
            self.assertEqual(left["target_interest"], right["target_interest"])
            if slot["role"] == "background":
                self.assertEqual(left, right)
        self.assertEqual((arms, slots), paired_rows(list(reversed(original)), list(reversed(once)),
                                                  list(reversed(five)), {"A", "B", "C"}, cap=4))

    def test_frequency_threshold_and_balancing_preserve_original_rows(self):
        from crossinterest.methods.frequency.data import BalancedPool, frequency_arm
        self.assertEqual([frequency_arm(count) for count in (0, 1, 2, 4, 5, 9)],
                         [None, "once", None, None, "at_least_five", "at_least_five"])
        candidates = [{"id": f"{category}{number}", "user_id": "shared-user", "target_interest": category}
                      for category in ("A", "B") for number in range(3)]
        original_ids = {"A2", "B2"}
        pool = BalancedPool({"A", "B"}, original_ids, cap=2)
        reverse = BalancedPool({"A", "B"}, original_ids, cap=2)
        for row in candidates:
            pool.add(row)
        for row in reversed(candidates):
            reverse.add(row)
        rows, quota = pool.rows()
        self.assertEqual(quota, 2)
        self.assertEqual(len(rows), 4)
        self.assertTrue(original_ids <= {row["id"] for row in rows})
        self.assertEqual(pool.rows(), reverse.rows())
        missing = BalancedPool({"A", "B", "C"}, original_ids)
        for row in candidates:
            missing.add(row)
        self.assertEqual(missing.rows(), ([], 0))

    def test_counts_full_future_without_changing_prediction_input(self):
        from crossinterest.methods.frequency.data import DAY_MS, counted_user_records
        from crossinterest.datasets.amazon.scripts.build import build_user_records
        products = {f"history{number}": (f"title{number}", "root", "old") for number in range(5)}
        history = [(number * DAY_MS, f"history{number}", 5.0) for number in range(5)]
        for count in (1, 4, 5):
            with self.subTest(count=count):
                catalog = products | {f"target{number}": (f"future{number}", "root", "new") for number in range(count)}
                events = history + [((number + 5) * DAY_MS, f"target{number}", 5.0) for number in range(count)]
                original = build_user_records("user", events, catalog, "train", 20260806, 50)
                counted = counted_user_records("user", events, catalog, "train", 20260806)
                self.assertEqual(len(counted), 1)
                self.assertEqual({key: counted[0][key] for key in original[0]}, original[0])
                self.assertEqual(counted[0]["future_target_count_including_first"], count)
                self.assertEqual(counted[0]["future_target_count_strictly_after_timestamp"], count - 1)
                self.assertEqual(counted[0]["future_target_span_days"], count - 1)
                self.assertFalse(any("future" in title for title in counted[0]["historical_titles"]))

    def test_timestamp_ties_and_product_dedup_match_original_builder(self):
        from crossinterest.methods.frequency.data import counted_user_records
        products = {str(number): (f"title{number}", "root", "old" if number < 5 else "new") for number in range(8)}
        events = [(number, str(number), 5.0) for number in range(5)]
        events += [(5, "5", 5.0), (5, "6", 5.0), (6, "5", 5.0), (7, "7", 5.0), (8, "unknown", 5.0)]
        row = counted_user_records("user", events, products, "train", 20260806)[0]
        self.assertEqual(row["future_target_count_including_first"], 3)
        self.assertEqual(row["future_target_count_excluding_selected_event"], 2)
        self.assertEqual(row["future_target_count_strictly_after_timestamp"], 1)
        self.assertEqual(row["history_timestamps"], list(range(5)))
        self.assertEqual(row["future_target_distinct_products"], 3)

    def test_old_category_outside_visible_history_is_not_new(self):
        from crossinterest.methods.frequency.data import counted_user_records
        products = {str(number): (f"title{number}", "root", "A" if number in (0, 7) else "B") for number in range(8)}
        events = [(number, str(number), 5.0) for number in range(8)]
        self.assertEqual(counted_user_records("user", events, products, "train", 20260806, history_limit=5), [])


class CIRDataTests(unittest.TestCase):
    def test_external_view_never_uses_future_label_or_hidden_items_as_input(self):
        from crossinterest.methods.cir.data import category_view
        row = {"id": "train:user:8", "user_id": "user", "timestamp": 8,
               "history_timestamps": list(range(7)), "historical_titles": [f"title{number}" for number in range(7)],
               "historical_interests": ["A", "B", "A", "B", "B", "B", "B"], "target_interest": "future"}
        view = category_view(row, "A", {"A", "B"})
        self.assertEqual(view["historical_titles"], ["title1", "title3", "title4", "title5", "title6"])
        self.assertEqual(view["hidden_titles"], ["title0", "title2"])
        self.assertEqual(view, category_view(row | {"target_interest": "different"}, "A", {"A", "B"}))
        with self.assertRaisesRegex(ValueError, "strictly precede"):
            category_view(row | {"timestamp": 6}, "A", {"A", "B"})

    def test_joint_balancing_recovers_rare_class_without_repeating_users(self):
        from crossinterest.methods.cir.data import balanced_assignment
        candidates = [{"id": signature, "user_id": user, "target_interest": category, "source_sample_ids": [signature]}
                      for user, signature, category in [("shared", "sA", "A"), ("shared", "sB", "B"),
                                                        ("onlyA", "aA", "A"), ("otherA", "bA", "A")]]
        selected, audit = balanced_assignment(candidates, {"A", "B"}, 1000)
        self.assertEqual(audit["quota"], 1)
        self.assertEqual({row["user_id"] for row in selected if row["target_interest"] == "B"}, {"shared"})
        self.assertEqual(len({row["user_id"] for row in selected}), 2)

    def test_balancing_excludes_conflicts_and_joint_capacity_limits_quota(self):
        from crossinterest.methods.cir.data import balanced_assignment
        candidates = [{"id": f"{user}-{category}", "user_id": user, "target_interest": category,
                       "source_sample_ids": [user]} for user in ["one", "two", "three"] for category in ["A", "B"]]
        candidates.extend([{"id": "conflict", "user_id": f"extra-{category}", "target_interest": category,
                            "source_sample_ids": [category]} for category in ["A", "B"]])
        selected, audit = balanced_assignment(candidates, {"A", "B"})
        self.assertEqual(audit["quota"], 1)
        self.assertEqual(audit["conflicting_inputs_excluded"], 1)
        self.assertNotIn("conflict", {row["id"] for row in selected})
        with self.assertRaisesRegex(ValueError, "no valid candidate"):
            balanced_assignment(candidates, {"A", "B", "C"})


class CIRSemanticTests(unittest.TestCase):
    def test_semantic_projection_is_not_loaded_in_novel_command(self):
        from crossinterest.methods.cir_semantic.runner import training_command
        prepared = {"model_key": "qwen35_9b", "variant": "semantic_true", "files": {
            "auxiliary": {"path": "aux", "rows": 83888}, "novel": {"path": "novel", "rows": 112000}}}
        auxiliary = training_command("model", "auxiliary", Path("aux"), False, prepared)
        novel = training_command("model", "novel", Path("novel"), False, prepared, Path("aux/final_adapter"))
        self.assertIn('crossinterest.methods.cir_semantic.train', auxiliary)
        self.assertIn('crossinterest.methods.cir.train', novel)
        self.assertNotIn('crossinterest.methods.cir_semantic.train', novel)
        self.assertEqual(novel[novel.index("--init-adapter") + 1], "aux/final_adapter")

    def test_semantic_trainer_scales_updates_and_restores_projection(self):
        import torch
        from unittest.mock import Mock
        from torch.utils.data import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import Qwen2Config, Qwen2ForCausalLM, TrainingArguments
        from crossinterest.methods.cir_semantic.train import SemanticTrainer
        class TinySemantic(Dataset):
            def __len__(self):
                return 24
            def __getitem__(self, index):
                return {"input_ids": [1, 4, 5, 28, 2], "labels": [-100, -100, -100, 28, 2],
                        "attention_mask": [1] * 5, "positive_mask": [True, False],
                        "semantic_target": [1.0, 0.0, 0.0]}
        base = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2))
        model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, task_type="CAUSAL_LM",
                                              target_modules=["q_proj", "v_proj", "lm_head"]))
        model.enable_input_require_grads()
        tokenizer = Mock()
        tokenizer.pad_token_id = 0
        with TemporaryDirectory() as temporary:
            trainer = SemanticTrainer(model=model, train_dataset=TinySemantic(), processing_class=tokenizer,
                args=TrainingArguments(output_dir=temporary, max_steps=3, per_device_train_batch_size=1,
                    gradient_accumulation_steps=8, learning_rate=0.01, save_steps=2, report_to="none", use_cpu=True,
                    remove_unused_columns=False, gradient_checkpointing=True,
                    gradient_checkpointing_kwargs={"use_reentrant": False}),
                stage="auxiliary", relation=False, init_adapter=None, sparse_embedding_save=True,
                label_token_ids=[28, 29], semantic_dimension=3, semantic_identity={"control": "true"})
            batch = trainer.data_collator([TinySemantic()[0]])
            trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            raw = trainer.compute_loss(model, batch).detach()
            scaled = trainer.training_step(model, batch, num_items_in_batch=torch.tensor(16))
            torch.testing.assert_close(scaled, raw / 8)
            trainer.optimizer = None
            trainer.train()
            self.assertGreater(trainer.semantic_controller.head[1].weight.count_nonzero().item(), 0)
            self.assertTrue(any(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()))
            trainer.semantic_controller.head[1].weight.data.fill_(100)
            trainer._load_from_checkpoint(str(Path(temporary) / "checkpoint-2"))
            self.assertLess(trainer.semantic_controller.head[1].weight.abs().max().item(), 1)
            self.assertFalse((Path(temporary) / "checkpoint-2/interest_relation.safetensors").exists())
            trainer.semantic_controller.close()

    def test_residuals_and_deranged_control_preserve_class_marginals(self):
        import numpy as np
        from crossinterest.methods.cir_semantic.data import residual_targets
        embeddings = np.arange(36, dtype=np.float32).reshape(6, 6)
        categories = ["A", "B", "A", "B", "A", "B"]
        ids = [f"row{index}" for index in range(6)]
        arrays, audit = residual_targets(embeddings, categories, ids)
        repeated, _ = residual_targets(embeddings, categories, ids)
        np.testing.assert_array_equal(arrays["permutation"], repeated["permutation"])
        self.assertEqual(audit["permutation_fixed_points"], 0)
        self.assertAlmostEqual(audit["mean_squared_target_norm"], 1, places=6)
        for category in set(categories):
            indices = [index for index, label in enumerate(categories) if label == category]
            np.testing.assert_allclose(arrays["targets"][indices].mean(axis=0), 0, atol=1e-7)
            self.assertEqual(set(arrays["permutation"][indices]), set(indices))
        shifted, _ = residual_targets(embeddings + np.array([100 if label == "A" else -100 for label in categories])[:, None], categories, ids)
        np.testing.assert_allclose(arrays["targets"], shifted["targets"], atol=1e-7)
        with self.assertRaisesRegex(ValueError, "no usable variance"):
            residual_targets(np.ones((6, 6)), categories, ids)


class BidirectionalDataTests(unittest.TestCase):
    def test_inverse_removes_whole_A_and_only_inverse_gets_B(self):
        from crossinterest.methods.directionality.data import make_view
        row = {"id": "train:user:10", "user_id": "user", "timestamp": 10,
               "historical_titles": ["a1", "r1", "a2", "r2", "r3", "r4", "r5"],
               "historical_interests": ["A", "R", "A", "R", "R", "R", "R"],
               "history_timestamps": list(range(7)), "full_seen_interests": ["A", "R"],
               "target_interest": "B", "target_title": "new product"}
        forward = make_view(row)
        ordinary = make_view(row, "A")
        paired = make_view(row, "A", row)
        self.assertNotIn("new product", forward["historical_titles"])
        self.assertNotIn("A", paired["historical_interests"])
        self.assertEqual(paired["historical_titles"][:-1], ordinary["historical_titles"])
        self.assertEqual(paired["historical_interests"][-1], "B")
        self.assertEqual(paired["target_interest"], "A")
        with self.assertRaisesRegex(ValueError, "leaks"):
            make_view(row, "A", {**row, "target_interest": "A"})

    def test_joint_flow_and_donor_shuffle_preserve_both_margins(self):
        from crossinterest.methods.directionality.data import assign_pairs, donor_permutation
        categories = {"A", "B", "C", "D"}
        rows = []
        for label_index, label in enumerate(sorted(categories)):
            for index in range(12):
                hidden = sorted(categories)[(label_index + 1) % 4]
                rows.append({"id": f"{label}:{index}", "user_id": f"{label}:{index}",
                             "target_interest": label, "full_seen_interests": [hidden, "other"],
                             "historical_titles": ["hidden"] + ["keep"] * 5,
                             "historical_interests": [hidden] + ["other"] * 5})
        pairs, quota = assign_pairs(rows, categories, cap=3)
        self.assertEqual(quota, 3)
        self.assertEqual(len(pairs), 12)
        permutation = donor_permutation(pairs, categories)
        self.assertEqual(sorted(permutation), list(range(12)))
        for index, donor_index in enumerate(permutation):
            row, target = pairs[index]
            donor = pairs[donor_index][0]
            self.assertNotEqual(row["user_id"], donor["user_id"])
            self.assertNotIn(donor["target_interest"], set(row["full_seen_interests"]) | {row["target_interest"], target})


class CIRTrainingTests(unittest.TestCase):
    def test_three_epoch_auxiliary_keeps_novel_one_epoch_and_separate_freeze(self):
        from crossinterest.methods.cir.runner import training_command, preflight_path
        prepared = {"model_key": "qwen35_9b", "variant": "external_balanced", "auxiliary_epochs": 3,
                    "files": {"auxiliary": {"path": "aux.parquet", "rows": 83888, "steps": 7866, "epochs": 3},
                              "novel": {"path": "n.parquet", "rows": 112000, "steps": 3500}}}
        for stage, epochs in (("auxiliary", "3"), ("novel", "1")):
            command = training_command("model", stage, Path("output"), False, prepared,
                                       Path("aux/final_adapter") if stage == "novel" else None)
            self.assertEqual(command[command.index("--epochs") + 1], epochs)
            self.assertIn("external_balanced_ep3", command[command.index("--cir-preflight") + 1])
        self.assertNotEqual(preflight_path("qwen35_9b", "external_balanced"),
                            preflight_path("qwen35_9b", "external_balanced", 3))

    def test_collector_distinguishes_smoke_from_complete_and_excludes_checkpoints(self):
        from crossinterest.methods.cir.audit import select_inventory
        inventory = [{"name": name} for name in ["EXECUTION_CONTRACT.json", "smoke/RESULT_AUDIT.json",
            "smoke/auxiliary/final_adapter/adapter_model.safetensors", "formal/novel/checkpoint-1000/optimizer.pt",
            "formal/novel/final_adapter/adapter_model.safetensors"]]
        selected = select_inventory(inventory, True)
        self.assertEqual(len(selected), 3)
        self.assertFalse(any(entry["name"].startswith("formal/") for entry in selected))
        with self.assertRaisesRegex(ValueError, "has not completed"):
            select_inventory(inventory, False)
        inventory.extend([{"name": "COMPLETE.json"}, {"name": "formal/RESULT_AUDIT.json"}])
        self.assertFalse(any("checkpoint-" in entry["name"] for entry in select_inventory(inventory, False)))
        projection = {"name": "smoke/auxiliary/final_adapter/cir_semantic_projection.safetensors"}
        self.assertIn(projection, select_inventory(inventory + [projection], True))
        checkpoint_projection = {"name": "smoke/auxiliary/checkpoint-5/final_adapter/cir_semantic_projection.safetensors"}
        self.assertNotIn(checkpoint_projection, select_inventory(inventory + [checkpoint_projection], True))
        with self.assertRaisesRegex(ValueError, "unsafe"):
            select_inventory(inventory + [{"name": "smoke/../../outside.json"}], True)

    def test_model_specific_command_and_stage_contract(self):
        from unittest.mock import patch
        from crossinterest.training.train import parse_args
        from crossinterest.methods.cir.runner import training_command, MODELS
        from crossinterest.methods.cir.train import validate_stage
        for model_key, (_, revision) in MODELS.items():
            with self.subTest(model=model_key), TemporaryDirectory() as temporary:
                info = {"path": "aux.parquet", "rows": 83888, "steps": 2622, "sha256": "freeze"}
                prepared = {"files": {"auxiliary": info}, "model_key": model_key, "model_revision": revision,
                            "variant": "external_balanced", "mapping_sha256": "freeze"}
                command = training_command(f"/snapshots/{revision}", "auxiliary", Path(temporary), False, prepared)
                self.assertEqual("--require-qwen-fast-path" in command, model_key == "qwen35_9b")
                start = command.index('crossinterest.methods.cir.train') + 1
                end = command.index("--stage")
                args = parse_args(command[start:end])
                with patch("eval_interest_id_minionerec.cir_train.sha256_file", return_value="freeze"):
                    metadata = validate_stage(args, prepared, "auxiliary", False, Path("preflight"))
                    self.assertFalse(metadata["relation_head"])
                    self.assertFalse(metadata["smoke"])
                    args.init_adapter = Path("smoke/final_adapter")
                    with self.assertRaisesRegex(ValueError, "starts fresh"):
                        validate_stage(args, prepared, "auxiliary", False, Path("preflight"))


class MechanismDataTests(unittest.TestCase):
    def setUp(self):
        self.row = {
            "id": "source:1", "historical_titles": ["hidden1", "keep1", "hidden2", "keep2", "keep3", "keep4", "keep5"],
            "historical_interests": ["A", "B", "A", "B", "B", "B", "B"],
            "target_interest": "future", "history_summary": "must not survive",
            "full_seen_interests": ["A", "B", "other past"],
        }

    def test_holdout_removes_whole_class_without_future_fields(self):
        from crossinterest.methods.mechanism.data import holdout_view
        view = holdout_view(self.row, {"A", "B"})
        self.assertEqual(view["positive_interests"], ["A"])
        self.assertEqual(view["historical_titles"], [f"keep{number}" for number in range(1, 6)])
        self.assertEqual(view["full_seen_interests"], ["B"])
        self.assertNotIn("history_summary", view)
        self.assertEqual(view, holdout_view(self.row | {"target_interest": "different"}, {"A", "B"}))

    def test_holdout_isolation_and_positive_set_merging(self):
        from crossinterest.methods.mechanism.data import construct_views, history_signature
        other = self.row | {"id": "source:2", "historical_interests": [
            "C" if category == "A" else category for category in self.row["historical_interests"]]}
        views, counts = construct_views([self.row, other], {"A", "B", "C"}, set())
        self.assertEqual(len(views), 1)
        self.assertEqual(views[0]["positive_interests"], ["A", "C"])
        self.assertEqual(counts["merged_rows"], 1)
        blocked, counts = construct_views([self.row, other], {"A", "B", "C"}, {history_signature(views[0])})
        self.assertEqual(blocked, [])
        self.assertEqual(counts["isolated_rows"], 2)

    def test_holdout_rejects_short_or_unpaired_histories(self):
        from crossinterest.methods.mechanism.data import holdout_view
        self.assertIsNone(holdout_view(self.row | {"historical_titles": ["only"], "historical_interests": ["A"]}, {"A"}))
        with self.assertRaisesRegex(ValueError, "unpaired"):
            holdout_view(self.row | {"historical_titles": []}, {"A", "B"})


class MechanismModelTests(unittest.TestCase):
    def test_auxiliary_loss_is_scaled_for_gradient_accumulation(self):
        import torch
        from unittest.mock import Mock
        from transformers import Qwen2Config, Qwen2ForCausalLM, TrainingArguments
        from crossinterest.methods.mechanism.train import MechanismTrainer, MechanismCollator
        model = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2))
        tokenizer = Mock()
        tokenizer.pad_token_id = 0
        with TemporaryDirectory() as temporary:
            trainer = MechanismTrainer(model=model, processing_class=tokenizer,
                args=TrainingArguments(output_dir=temporary, use_cpu=True, report_to="none", gradient_accumulation_steps=8),
                stage="auxiliary", relation=False, init_adapter=None, label_token_ids=[28, 29])
            trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            batch = MechanismCollator(0)([{"input_ids": [1, 4, 5, 28, 2], "labels": [-100] * 3 + [28, 2],
                                          "attention_mask": [1] * 5, "positive_mask": [True, False]}])
            raw = trainer.compute_loss(model, batch).detach()
            reported = trainer.training_step(model, batch, num_items_in_batch=torch.tensor(16))
            self.assertFalse(trainer.model_accepts_loss_kwargs)
            torch.testing.assert_close(reported, raw / 8)

    def test_mechanism_comparison_reproduces_historical_group_and_paired_diagnostics(self):
        import json
        import pyarrow.parquet as pq
        from crossinterest.evaluation.common import latest_details
        from crossinterest.training.evaluate_ranking import score_ranking
        from crossinterest.methods.analysis.followup import RUNS, DATA, BASE, HERE
        from crossinterest.methods.mechanism.audit import ranking_diagnostics
        rows = pq.read_table(BASE / "eval/dev_2k_behavior_interest.parquet").to_pylist()
        seen_labels = set(json.loads((DATA / "MANIFEST.json").read_text())["seen_labels"])
        historical = json.loads((HERE / "curriculum/outputs/FULLNOVEL_DIAGNOSTICS_20260915.json").read_text())
        raw = {}
        for name, directory in RUNS.items():
            contract = json.loads((directory / "run_contract.json").read_text())
            details = latest_details(directory / "details.jsonl", contract["fingerprint"])
            raw[name] = [dict(details[row["sample_id"]]) for row in rows]
            for detail in raw[name]:
                detail.update(score_ranking(detail["ranking"], detail["target_interest"]))
            actual = ranking_diagnostics(raw[name], raw["n_only"], rows, seen_labels)
            for key in ("metrics", "groups"):
                self.assertEqual(actual[key], historical["conditions"][name][key])
            if name != "n_only":
                self.assertEqual(actual["macro_ndcg_at_20_delta_components"], historical["conditions"][name]["macro_ndcg_at_20_delta_components"])
        with self.assertRaisesRegex(ValueError, "identity differs"):
            ranking_diagnostics(raw["shuffle"][::-1], raw["n_only"], rows, seen_labels)

    def test_mechanism_result_inventory_excludes_checkpoints_and_path_escape(self):
        from crossinterest.methods.mechanism.audit import wanted
        for name in ("COMPLETE.json", "EXECUTION_CONTRACT.json", "formal/novel/final_adapter/adapter_model.safetensors",
                     "smoke/auxiliary/initial_relation/interest_relation.safetensors", "formal/ranking/details.jsonl"):
            self.assertTrue(wanted(name), name)
        for name in ("formal/novel/checkpoint-500/adapter_model.safetensors", "formal/novel/final_adapter/tokenizer.json",
                     "logs/other.txt", "formal/novel/training_args.bin"):
            self.assertFalse(wanted(name), name)
        for name in ("../other.json", "/absolute.json", "formal/../../other.json"):
            with self.assertRaisesRegex(ValueError, "unsafe"):
                wanted(name)

    def test_mechanism_download_reuses_only_verified_unchanged_artifacts(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from crossinterest.methods.mechanism.audit import download_artifact
        blob = Mock()
        blob.get_blob_properties.return_value = SimpleNamespace(etag="version1", size=4)
        blob.download_blob.return_value.readinto.side_effect = lambda handle: handle.write(b"data")
        with TemporaryDirectory() as temporary:
            destination = Path(temporary) / "artifact.bin"
            receipt = download_artifact(blob, destination, {})
            self.assertEqual(download_artifact(blob, destination, receipt), receipt)
            self.assertEqual(blob.download_blob.call_count, 1)
            destination.write_bytes(b"bad!")
            self.assertEqual(download_artifact(blob, destination, receipt), receipt)
            self.assertEqual(blob.download_blob.call_count, 2)
            blob.get_blob_properties.return_value.etag = "version2"
            updated = download_artifact(blob, destination, receipt)
            self.assertEqual(updated["etag"], "version2")
            self.assertEqual(blob.download_blob.call_count, 3)
            blob.get_blob_properties.return_value.size = 5
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                download_artifact(blob, destination, updated)
            self.assertEqual(destination.read_bytes(), b"data")

    def test_mechanism_monitor_detects_failure_without_waiting_for_other_jobs(self):
        from datetime import datetime, timezone
        from crossinterest.methods.runtime.performance import classify_observation, progress_line, has_smoke_audit
        self.assertEqual(classify_observation({"status": "Failed"}, None, datetime.now(timezone.utc), 1800), "job_failure")
        self.assertEqual(progress_line("30%| | 855/2848 [00:01]"), "30%| | 855/2848 [00:01]")
        self.assertEqual(progress_line("Loading weights: 100%\r20%| | 524/2622 [00:01]"), "20%| | 524/2622 [00:01]")
        self.assertTrue(has_smoke_audit('progress\n{"passed": true, "smoke": true, "stage_audits": {}}'))
        self.assertFalse(has_smoke_audit('{"passed": true, "smoke": false, "stage_audits": {}}'))
        self.assertFalse(has_smoke_audit('broken "stage_audits"'))
        self.assertTrue(has_smoke_audit('{"passed": true, "smoke": true, "training": {"passed": true}}'))
        self.assertFalse(has_smoke_audit('{"passed": true, "smoke": true, "training": {"passed": false}}'))

    def test_history_frame_respects_tokenizer_unicode_normalization(self):
        from types import SimpleNamespace
        from tokenizers.normalizers import NFC
        from crossinterest.methods.mechanism.model import history_frame
        class CharacterTokenizer:
            backend_tokenizer = SimpleNamespace(normalizer=NFC())
            def decode(self, ids, **kwargs):
                return "".join(chr(value) for value in ids)
            def __call__(self, text, **kwargs):
                return {"input_ids": [ord(character) for character in text],
                        "offset_mapping": [(index, index + 1) for index in range(len(text))]}
        row = {"id": "unicode", "historical_titles": ["cafe\u0301"], "historical_interests": ["A"]}
        text = NFC().normalize_str("prefix\n1. [A] cafe\u0301\npredict")
        frame = history_frame(CharacterTokenizer(), [ord(character) for character in text], row, len(text) - 1)
        self.assertEqual(frame["positions"], [text.index("\npredict") - 1])

    def test_mechanism_trainer_optimizes_and_restores_relation_checkpoint(self):
        import torch
        from unittest.mock import Mock
        from torch.utils.data import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import Qwen2Config, Qwen2ForCausalLM, TrainingArguments
        from crossinterest.methods.mechanism.train import MechanismTrainer

        mapping = {"special_tokens": ["<interest_000>", "<interest_001>"],
                   "id_to_category": {"<interest_000>": "A", "<interest_001>": "B"}, "mapping_fingerprint": "test"}
        class TinyHistory(Dataset):
            def __len__(self):
                return 4
            def __getitem__(self, index):
                target = 28 + index % 2
                return {"input_ids": [1, 4, 5, 6, target, 2], "labels": [-100] * 4 + [target, 2],
                        "attention_mask": [1] * 6, "positive_mask": [target == 28, target == 29],
                        "history_frame": {"positions": [1, 2], "categories": ["A", "B"], "prediction_position": 3}}
        dataset = TinyHistory()
        dataset.mapping = mapping
        tokenizer = Mock()
        tokenizer.pad_token_id = 0
        tokenizer.convert_tokens_to_ids.side_effect = lambda token: mapping["special_tokens"].index(token) + 28
        tokenizer.encode.side_effect = lambda text, **kwargs: [4] if text == "A" else [5]
        base = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2))
        model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, task_type="CAUSAL_LM",
                                              target_modules=["q_proj", "v_proj", "lm_head"]))
        model.enable_input_require_grads()
        with TemporaryDirectory() as temporary:
            trainer = MechanismTrainer(model=model, train_dataset=dataset, processing_class=tokenizer,
                args=TrainingArguments(output_dir=temporary, max_steps=3, per_device_train_batch_size=1,
                    learning_rate=0.01, save_steps=2, logging_steps=1, report_to="none", use_cpu=True,
                    remove_unused_columns=False, gradient_checkpointing=True,
                    gradient_checkpointing_kwargs={"use_reentrant": False}),
                stage="auxiliary", relation=True, init_adapter=None,
                sparse_embedding_save=True, label_token_ids=[28, 29])
            before = trainer.controller.head.readout.weight.detach().clone()
            trainer.train()
            self.assertNotEqual(trainer.controller.head.beta.item(), 0)
            self.assertFalse(torch.equal(before, trainer.controller.head.readout.weight))
            trainer.controller.head.beta.data.fill_(100)
            trainer._load_from_checkpoint(str(Path(temporary) / "checkpoint-2"))
            self.assertLess(abs(trainer.controller.head.beta.item()), 1)
            trainer.controller.close()

    def test_mechanism_commands_preserve_formal_budget_and_handoff(self):
        from crossinterest.methods.mechanism.runner import training_command
        prepared = {"files": {"novel": {"path": "original.parquet", "rows": 112000}}}
        command = training_command("base", "combined", "novel", Path("run"), False, prepared, Path("aux/final_adapter"))
        self.assertIn('crossinterest.methods.mechanism.train', command)
        for flag, expected in (("--expected-train-rows", "112000"), ("--gradient-accumulation-steps", "8"),
                               ("--per-device-batch-size", "1"), ("--epochs", "1"),
                               ("--init-adapter", "aux/final_adapter")):
            self.assertEqual(command[command.index(flag) + 1], expected)
        self.assertNotIn("--max-steps", command)
        self.assertNotIn("--limit", command)

    def test_auxiliary_set_loss_and_collator(self):
        import torch
        from crossinterest.methods.mechanism.train import positive_set_loss, MechanismCollator
        logits = torch.tensor([[1., 2., 3.], [3., 2., 1.]], requires_grad=True)
        mask = torch.tensor([[False, True, False], [True, False, False]])
        torch.testing.assert_close(positive_set_loss(logits, mask), torch.nn.functional.cross_entropy(logits, torch.tensor([1, 0])))
        mask[0, 2] = True
        loss = positive_set_loss(logits, mask)
        self.assertTrue(loss.isfinite())
        loss.backward()
        self.assertTrue(logits.grad.isfinite().all())
        features = [{"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2],
                     "history_frame": {"positions": [0]}, "positive_mask": [True, False]},
                    {"input_ids": [1], "attention_mask": [1], "labels": [1],
                     "history_frame": {"positions": [0]}, "positive_mask": [False, True]}]
        batch = MechanismCollator(0)(features)
        self.assertEqual(batch["input_ids"].tolist(), [[1, 2], [1, 0]])
        self.assertEqual(batch["labels"].tolist(), [[-100, 2], [1, -100]])
        self.assertEqual(len(batch["history_frames"]), 2)
        self.assertEqual(batch["positive_mask"].shape, (2, 2))

    def test_relation_gradients_permutation_and_zero_residual(self):
        import torch
        from crossinterest.methods.mechanism.model import RelationHead
        torch.manual_seed(19)
        head = RelationHead(8, torch.randn(4, 8), dimension=6)
        states = torch.randn(5, 8, requires_grad=True)
        source = torch.randn(2, 8)
        groups = [0, 1, 0, 1, 1]
        self.assertTrue(torch.equal(head(states, groups, source), torch.zeros(4)))
        with torch.no_grad():
            head.beta.fill_(0.1)
        scores = head(states, groups, source)
        torch.testing.assert_close(scores, head(states, [1 - group for group in groups], source.flip(0)))
        scores[1].backward()
        for name, parameter in head.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(parameter.grad.isfinite().all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
        self.assertGreater(states.grad.abs().sum().item(), 0)

    def test_relation_real_causal_model_save_reload_and_beams(self):
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM
        from crossinterest.methods.mechanism.model import RelationController
        from unittest.mock import Mock
        torch.manual_seed(13)
        model = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=16,
            intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=2, eos_token_id=2, pad_token_id=0)).eval()
        tokenizer = Mock()
        tokenizer.convert_tokens_to_ids.side_effect = lambda token: {"<interest_000>": 28, "<interest_001>": 29}[token]
        tokenizer.encode.side_effect = lambda text, **kwargs: [4] if text == "A" else [5]
        mapping = {"special_tokens": ["<interest_000>", "<interest_001>"],
            "id_to_category": {"<interest_000>": "A", "<interest_001>": "B"}, "mapping_fingerprint": "test"}
        inputs = torch.tensor([[1, 4, 5, 6, 7]])
        original = model(inputs).logits.detach()
        frame = {"positions": [1, 2], "categories": ["A", "B"], "prediction_position": 4}
        controller = RelationController(model, tokenizer, mapping, dimension=8)
        controller.set_frames([frame])
        torch.testing.assert_close(model(inputs).logits, original, rtol=0, atol=0)
        with torch.no_grad():
            controller.head.beta.fill_(2.0)
        changed = model(inputs).logits
        torch.testing.assert_close(changed[:, :4], original[:, :4], rtol=0, atol=0)
        torch.testing.assert_close(changed[:, 4, :28], original[:, 4, :28], rtol=0, atol=0)
        self.assertFalse(torch.equal(changed[:, 4, 28:30], original[:, 4, 28:30]))
        changed[0, 4, 28].backward()
        self.assertGreater(controller.head.readout.weight.grad.abs().sum().item(), 0)
        with TemporaryDirectory() as temporary:
            controller.save(Path(temporary))
            controller.head.beta.data.zero_()
            controller.load(Path(temporary))
            torch.testing.assert_close(model(inputs).logits, changed, rtol=0, atol=0)
        before_calls = controller.applied_calls
        with torch.inference_mode():
            model.generate(inputs, attention_mask=torch.ones_like(inputs), num_beams=3,
                           num_return_sequences=3, max_new_tokens=3, do_sample=False)
        self.assertEqual(controller.applied_calls - before_calls, 1)
        controller.close()


class TaskTests(unittest.TestCase):
    def setUp(self):
        self.mapping = build_label_mapping(["Parent > Seen", "Parent > New"])
        self.row = {
            "id": "train:example",
            "historical_titles": ["Observed product"],
            "historical_interests": ["Parent > Seen"],
            "full_seen_interests": ["Parent > Seen"],
            "target_interest": "Parent > New",
        }

    def test_warmup_is_balanced_and_contains_no_recommendation(self):
        rows = alignment_rows(self.mapping)
        self.assertEqual(len(rows), 4)
        self.assertEqual({row["task_type"] for row in rows}, set(ALIGNMENT_TASKS))
        for row in rows:
            self.assertFalse(row["source_sample_id"])
            self.assertFalse(row["historical_interests"])
            messages = render_messages(row, self.mapping)
            category = row["target_interest"]
            interest_id = self.mapping["category_to_id"][category]
            answer = interest_id if row["task_type"] == "name_to_id" else category
            self.assertEqual(messages[-1]["content"], answer)
            self.assertNotIn(answer, messages[0]["content"])

    def test_recommendation_preserves_existing_prompts_and_novelty(self):
        rows = recommendation_rows(self.row, self.mapping)
        self.assertEqual({row["task_type"] for row in rows}, set(RECOMMENDATION_TASKS))
        for row, mode in zip(rows, ("interest_id", "category"), strict=True):
            self.assertEqual(row["source_sample_id"], self.row["id"])
            self.assertEqual(render_messages(row, self.mapping), render_training_messages(self.row, "interest", mode, self.mapping))

    def test_seen_target_is_rejected(self):
        self.row["full_seen_interests"].append("Parent > New")
        with self.assertRaisesRegex(ValueError, "not novel"):
            recommendation_rows(self.row, self.mapping)

    def test_behavior_interest_changes_only_recommendation_input(self):
        for row in alignment_rows(self.mapping):
            self.assertEqual(render_messages(row, self.mapping, "behavior_interest"),
                             render_messages(row, self.mapping))
        for row, mode in zip(recommendation_rows(self.row, self.mapping),
                             ("interest_id", "category"), strict=True):
            messages = render_messages(row, self.mapping, "behavior_interest")
            self.assertEqual(messages, render_training_messages(
                self.row, "behavior_interest", mode, self.mapping))
            self.assertIn("Observed product", messages[0]["content"])
            self.assertIn("Parent > Seen", messages[0]["content"])
            self.assertNotIn("Observed product", render_messages(row, self.mapping)[0]["content"])
            self.assertEqual(messages[-1], render_messages(row, self.mapping)[-1])
        with self.assertRaisesRegex(ValueError, "input view"):
            render_messages(row, self.mapping, "behavior")

    def test_backbone_choices_and_thinking_are_consistent(self):
        from unittest.mock import MagicMock
        from crossinterest.training import evaluate, evaluate_ranking
        from crossinterest.methods.backbones.runner import BACKBONE_KEYS

        self.assertEqual(len(BACKBONE_KEYS), 9)
        self.assertEqual({evaluate.MODEL_NAMES[key] for key in BACKBONE_KEYS}, {
            f"Qwen/Qwen2.5-{size}B-Instruct" for size in ("0.5", "1.5", "3", "7", "14", "32")
        } | {"Qwen/Qwen3-8B", "Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-27B"})
        self.assertNotIn("qwen3_14b", BACKBONE_KEYS)
        self.assertNotIn("qwen3_32b", BACKBONE_KEYS)
        self.assertIn("qwen3_8b", evaluate.THINKING_MODEL_KEYS)
        self.assertIn("qwen35_27b", evaluate.THINKING_MODEL_KEYS)
        row = {
            **self.row, "source_sample_id": self.row["id"], "sample_id": "eval:example",
            "candidate_interests": ["Parent > Seen", "Parent > New"],
        }
        for model_key, model_name in evaluate.MODEL_NAMES.items():
            with self.subTest(model_key=model_key):
                arguments = [
                    "--model-key", model_key, "--model", model_name, "--adapter", "/tmp/adapter",
                    "--view", "behavior_interest", "--run-dir", "/tmp/evaluation",
                ]
                for module in (evaluate, evaluate_ranking):
                    self.assertEqual(module.parse_args(arguments).model_key, model_key)
                    tokenizer = MagicMock()
                    tokenizer.apply_chat_template.return_value = {"input_ids": [[3, 4]]}
                    self.assertEqual(module.render_prompt_ids(
                        tokenizer, row, model_key, "behavior_interest", "interest_id"), [3, 4])
                    call = tokenizer.apply_chat_template.call_args
                    self.assertIn("Observed product", call.args[0][0]["content"])
                    if model_key in evaluate.THINKING_MODEL_KEYS:
                        self.assertIs(call.kwargs["enable_thinking"], False)
                    else:
                        self.assertNotIn("enable_thinking", call.kwargs)
                evaluate_ranking.apply_generation_template(tokenizer, [], model_key)
                self.assertEqual(
                    tokenizer.apply_chat_template.call_args.kwargs.get("enable_thinking"),
                    False if model_key in evaluate.THINKING_MODEL_KEYS else None,
                )

    def test_backbone_device_map_is_optional_and_forwarded(self):
        from unittest.mock import patch
        from crossinterest.training import evaluate, evaluate_ranking

        arguments = [
            "--model-key", "qwen25_32b", "--model", "Qwen/Qwen2.5-32B-Instruct",
            "--adapter", "/tmp/adapter", "--view", "behavior_interest",
            "--run-dir", "/tmp/evaluation",
        ]
        for module in (evaluate, evaluate_ranking):
            self.assertEqual(module.parse_args(arguments).device_map, "cuda")
            self.assertEqual(module.parse_args(arguments + ["--device-map", "balanced"]).device_map,
                             "balanced")
        with patch.object(evaluate.AutoModelForCausalLM, "from_pretrained") as load_base, \
             patch("peft.PeftModel.from_pretrained") as load_adapter:
            restored = evaluate.load_model(
                "qwen25_32b", "Qwen/Qwen2.5-32B-Instruct", Path("/tmp/adapter"),
                None, None, None, device_map="balanced",
            )
        self.assertEqual(load_base.call_args.kwargs["device_map"], "balanced")
        self.assertIs(restored, load_adapter.return_value)
        restored.eval.assert_called_once_with()

    def test_tied_output_lora_is_aligned_as_a_whole_during_balanced_evaluation(self):
        import torch
        from unittest.mock import patch
        from peft import LoraConfig, get_peft_model
        from transformers import Qwen2Config, Qwen2ForCausalLM
        from crossinterest.training import evaluate
        from crossinterest.training.semantic_tokens import prepare_semantic_output_head

        base = Qwen2ForCausalLM(Qwen2Config(vocab_size=64, hidden_size=16,
            intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=2, tie_word_embeddings=True)).eval()
        inputs = torch.tensor([[1, 2, 3]])
        before = base(inputs).logits.detach()

        def initialize(model, *unused):
            return {"output_head_detached_for_lora": prepare_semantic_output_head(model)}

        def load_adapter(model, *unused):
            return get_peft_model(model, LoraConfig(r=2, lora_alpha=4,
                                  target_modules=["lm_head"], task_type="CAUSAL_LM"))

        with patch.object(evaluate.AutoModelForCausalLM, "from_pretrained", return_value=base), \
             patch.object(evaluate, "initialize_semantic_token_embeddings", side_effect=initialize), \
             patch("peft.PeftModel.from_pretrained", side_effect=load_adapter):
            restored = evaluate.load_model("qwen25_05b", "base", Path("/tmp/adapter"),
                                           None, {"test": True}, {}, device_map="balanced")
        head = restored.get_output_embeddings()
        self.assertTrue(hasattr(head, "lora_A"))
        self.assertEqual(head._hf_hook.execution_device, head.weight.device)
        self.assertTrue(head._hf_hook.io_same_device)
        torch.testing.assert_close(restored(inputs).logits, before, rtol=0, atol=0)

    def test_backbone_evaluation_repair_cannot_change_training_contract(self):
        from unittest.mock import patch
        import crossinterest.methods.backbones.runner as backbone

        original = {name: "original" for name in backbone.CODE_FILES}
        updated = original | {"eval_interest_id_minionerec/backbone.py": "new-runner",
                              'crossinterest.training/evaluate.py': "device-hook"}
        manifest = {"code_sha256": original}
        with patch.object(backbone, "code_hashes", return_value=updated):
            with self.assertRaisesRegex(ValueError, "explicit"):
                backbone.validate_code_contract(manifest)
            manifest["evaluation_repair"] = {"original_code_sha256": original,
                                              "repaired_code_sha256": updated}
            backbone.validate_code_contract(manifest)
            updated['crossinterest.training/train.py'] = "forbidden"
            with self.assertRaisesRegex(ValueError, "training"):
                backbone.validate_code_contract(manifest)

    def test_backbone_smoke_audit_gates_formal_and_keeps_budgets_separate(self):
        from unittest.mock import patch
        import crossinterest.methods.backbones.runner as backbone

        manifest = {"model_key": "qwen25_7b", "model_repo": "Qwen/Qwen2.5-7B-Instruct",
                    "model_revision": "a" * 40, "code_sha256": {}, "smoke_sha256": "smoke"}
        with TemporaryDirectory() as temporary, patch.object(backbone.subprocess, "run") as run, \
             patch.object(backbone, "audit_run", side_effect=ValueError("failed smoke")):
            with self.assertRaisesRegex(ValueError, "failed smoke"):
                backbone.run_stages(manifest, "/tmp/base", Path(temporary))
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args.kwargs["env"]["EXPECTED_TRAIN_ROWS"], "224")
        with TemporaryDirectory() as temporary, patch.object(backbone.subprocess, "run") as run, \
             patch.object(backbone, "audit_run", return_value={"passed": True}) as audit:
            backbone.run_stages(manifest, "/tmp/base", Path(temporary))
            self.assertEqual(run.call_count, 2)
            smoke_env, formal_env = [call.kwargs["env"] for call in run.call_args_list]
            self.assertEqual(smoke_env["EVAL_LIMIT"], "4")
            self.assertNotIn("EVAL_LIMIT", formal_env)
            self.assertEqual(formal_env["EXPECTED_TRAIN_ROWS"], "112000")
            self.assertNotEqual(smoke_env["AMLT_OUTPUT_DIR"], formal_env["AMLT_OUTPUT_DIR"])
            self.assertEqual(formal_env["MODEL_NAME"], "/tmp/base")
            self.assertEqual(formal_env["GRADIENT_CHECKPOINTING"], "1")
            self.assertEqual([call.args[-1] for call in audit.call_args_list], [True, False])

    def test_qwen35_sizes_require_fast_path_and_keep_original_sft_budget(self):
        from unittest.mock import patch
        import crossinterest.methods.backbones.runner as backbone

        for model_key in ("qwen25_7b", "qwen35_9b", "qwen35_27b"):
            with self.subTest(model_key=model_key), TemporaryDirectory() as temporary, \
                 patch.object(backbone.subprocess, "run") as run, \
                 patch.object(backbone, "audit_run", return_value={"passed": True}):
                manifest = {"model_key": model_key, "model_repo": backbone.evaluate.MODEL_NAMES[model_key],
                            "model_revision": "a" * 40, "code_sha256": {}, "smoke_sha256": "smoke"}
                backbone.run_stages(manifest, "/tmp/base", Path(temporary))
                self.assertEqual(run.call_count, 2)
                for call, expected_rows in zip(run.call_args_list, ("224", "112000"), strict=True):
                    environment = call.kwargs["env"]
                    self.assertEqual(environment["REQUIRE_QWEN_FAST_PATH"],
                                     "1" if model_key.startswith("qwen35_") else "0")
                    self.assertEqual(environment["EXPECTED_TRAIN_ROWS"], expected_rows)
                    self.assertEqual(environment["PER_DEVICE_BATCH_SIZE"], "1")
                    self.assertEqual(environment["GRADIENT_ACCUMULATION_STEPS"], "8")
                    self.assertEqual(environment["NPROC"], "4")
                    self.assertEqual(environment["RANKING_NUM_BEAMS"], "20")
                    self.assertNotIn("TRAIN_SELECTION_FILE", environment)

    def test_performance_watch_uses_latest_retry_and_ignores_old_interruptions(self):
        from datetime import datetime, timedelta, timezone
        from crossinterest.methods.runtime.performance import classify_observation, latest_stdout, progress_line

        names = ["user_logs/std_log.txt", "user_logs/retry_002/std_log.txt",
                 "user_logs/retry_010/std_log.txt", "system_logs/lifecycler.log"]
        self.assertEqual(latest_stdout(names), "user_logs/retry_010/std_log.txt")
        self.assertIsNone(latest_stdout([]))
        self.assertEqual(progress_line("10/3500\rcompleted=480/2000\n"), "completed=480/2000")
        self.assertEqual(progress_line("train_runtime=206\r842/4954 [1:31:57<8:02:50]\n"),
                 "842/4954 [1:31:57<8:02:50]")
        now = datetime(2026, 9, 12, 15, tzinfo=timezone.utc)
        current = {"status": "Running", "interruption": "old eviction",
                   "last_active_at": (now - timedelta(minutes=31)).isoformat()}
        self.assertIsNone(classify_observation(current, None, now, 1800))
        self.assertEqual(classify_observation(current, current, now, 1800), "no_progress")
        active = current | {"last_active_at": now.isoformat()}
        self.assertIsNone(classify_observation(active, current, now, 1800))
        self.assertEqual(classify_observation(active | {"interruption": "new eviction"},
                                             current, now, 1800), "scheduling_interruption")
        self.assertEqual(classify_observation(current | {"status": "Completed"},
                                             current, now, 1800), "terminal")

    def test_crossfit_groups_shared_users_and_inputs_and_scores_only_target(self):
        import torch
        from crossinterest.methods.crossfit.train import assign_folds, target_nll, teacher_command
        from crossinterest.training.train import parse_args

        command = teacher_command("/tmp/model", Path("/tmp/train.parquet"), Path("/tmp/train"), 89635)
        arguments = parse_args(command[command.index('crossinterest.training.train') + 1:])
        self.assertEqual(arguments.expected_train_rows, 89635)
        self.assertEqual(arguments.epochs, 1)
        self.assertIsNone(arguments.init_adapter)
        self.assertIsNone(arguments.selection_file)
        self.assertEqual(arguments.per_device_batch_size * arguments.gradient_accumulation_steps * 4, 32)
        self.assertTrue(arguments.gradient_checkpointing)
        self.assertTrue(arguments.require_qwen_fast_path)

        rows = [{**self.row, "id": f"row-{index:02d}", "user_id": f"user-{index}",
                 "historical_titles": [f"product-{index}"],
                 "target_interest": "label-a" if index % 2 else "label-b"} for index in range(24)]
        rows[1]["user_id"] = rows[0]["user_id"]
        rows[2]["historical_titles"] = rows[1]["historical_titles"]
        records, audit = assign_folds(rows, n_splits=3)
        self.assertEqual(len({record["fold"] for record in records[:3]}), 1)
        self.assertEqual(len(records), 24)
        self.assertEqual(audit["coverage_count_per_row"], 1)
        self.assertEqual(audit["user_and_input_leakage"], 0)
        self.assertEqual(audit["duplicate_input_excess_rows"], 1)
        logits = torch.zeros((1, 4, 7))
        logits[0, 1, 4] = 2
        encoded = {"labels": [-100, -100, 4, 6]}
        expected = float(torch.logsumexp(logits[0, 1], 0) - logits[0, 1, 4])
        self.assertAlmostEqual(target_nll(logits, encoded, 4), expected)
        logits[0, 2, 6] = 100
        self.assertAlmostEqual(target_nll(logits, encoded, 4), expected)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            target_nll(logits, {"labels": [-100, 4, 4, 6]}, 4)

    def test_crossfit_class_weights_retain_all_rows_and_preserve_class_mass(self):
        import math
        import pyarrow as pa
        import pyarrow.parquet as pq
        from crossinterest.methods.crossfit.train import class_normalized_weights
        from crossinterest.training.train import load_selection

        records = [{"id": str(index), "target_interest": "first" if index < 2 else "second",
                    "nll": nll, "score_count": 1}
                   for index, nll in enumerate((0.0, math.log(3), 1000.0, 1000.0))]
        weighted, statistics = class_normalized_weights(records)
        self.assertEqual(len(weighted), 4)
        self.assertEqual([row["nested_rank"] for row in weighted], [1, 2, 3, 4])
        for actual, expected in zip(weighted, (1.5, 0.5, 1.0, 1.0)):
            self.assertAlmostEqual(actual["sample_weight"], expected)
        for stats in statistics.values():
            self.assertAlmostEqual(stats["mean"], 1.0)
            self.assertGreater(stats["effective_sample_size"], 0)
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "weights.parquet"
            pq.write_table(pa.Table.from_pylist(weighted), path)
            ids, weights = load_selection(path, 4, "sample_weight")
            self.assertEqual(ids, {row["id"] for row in records})
            self.assertAlmostEqual(sum(weights.values()), 4)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            class_normalized_weights(records + [records[0]])
        with self.assertRaisesRegex(ValueError, "finite"):
            class_normalized_weights([{**records[0], "nll": float("nan")}])
        with self.assertRaisesRegex(ValueError, "unique"):
            class_normalized_weights([{**records[0], "score_count": 2}])

    def test_crossfit_score_chunks_resume_without_loading_and_reject_wrong_fold(self):
        import json
        import torch
        import pyarrow as pa
        import pyarrow.parquet as pq
        from types import SimpleNamespace
        from unittest.mock import MagicMock, patch
        from crossinterest.methods.crossfit.train import score_fold

        rows = [{**self.row, "id": "sample-a"}, {**self.row, "id": "sample-b"}]
        memberships = {row["id"]: {"id": row["id"], "fold": 1, "group_id": index,
                                   "target_interest": row["target_interest"]}
                       for index, row in enumerate(rows)}
        token = self.mapping["category_to_id"][self.row["target_interest"]]
        model = MagicMock(device="cpu")
        model.return_value = SimpleNamespace(logits=torch.zeros((1, 3, 8)))
        identity = {"fold": 1, "teacher_adapter_sha256": "adapter"}
        with TemporaryDirectory() as temporary, \
             patch("transformers.AutoTokenizer.from_pretrained"), \
             patch('crossinterest.training.label_space.load_label_mapping', return_value=self.mapping), \
             patch('crossinterest.training.semantic_tokens.register_semantic_tokens',
                   return_value={"token_ids": {token: 4}}), \
             patch('crossinterest.training.train.tokenize_assistant_only',
                   return_value={"input_ids": [1, 4, 6], "labels": [-100, 4, 6]}), \
             patch('crossinterest.training.evaluate.load_model', return_value=model) as loader:
            output = Path(temporary) / "scores"
            score_fold("/tmp/base", Path("/tmp/train"), rows, memberships, output, identity)
            score_fold("/tmp/base", Path("/tmp/train"), rows, memberships, output, identity)
            self.assertEqual(loader.call_count, 1)
            self.assertEqual(model.call_count, 2)
            self.assertEqual(json.loads((output.parent / "RESULT_AUDIT.json").read_text())["scored_rows"], 2)
            part = output / "part-00000.parquet"
            saved = pq.read_table(part).to_pylist()
            saved[0]["fold"] = 2
            pq.write_table(pa.Table.from_pylist(saved), part)
            with self.assertRaisesRegex(ValueError, "fold"):
                score_fold("/tmp/base", Path("/tmp/train"), rows, memberships, output, identity)

    def test_backbone_smoke_is_distinct_balanced_and_includes_longest_inputs(self):
        from crossinterest.methods.backbones.runner import longest_balanced_indices, tokenized_hash

        rows = [{"id": f"row-{index}", "target_interest": "old" if index < 3 else "new"}
                for index in range(6)]
        lengths = [8, 100, 9, 20, 7, 20]
        selected = longest_balanced_indices(rows, lengths)
        self.assertEqual(selected, [1, 2, 3, 5])
        self.assertEqual(len({rows[index]["id"] for index in selected}), 4)
        self.assertEqual(longest_balanced_indices(rows, lengths, per_class=1), [1, 3])
        with self.assertRaisesRegex(ValueError, "insufficient"):
            longest_balanced_indices(rows, lengths, per_class=4)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            longest_balanced_indices(rows + [rows[0]], lengths + [8])
        self.assertNotEqual(
            tokenized_hash({"input_ids": [1, 2], "labels": [-100, 2]}),
            tokenized_hash({"input_ids": [1, 2], "labels": [1, 2]}),
        )

    def test_curriculum_uses_true_later_events_and_distinct_task_prompts(self):
        from collections import Counter
        from crossinterest.methods.curriculum.model import user_records, render_messages

        products = {f"asin-{index}": (f"product-{index}", "Parent", "Parent > Seen")
                    for index in range(7)}
        products["asin-6"] = ("future product", "Parent", "Parent > New")
        interactions = [(index + 1, f"asin-{index}", 5.0) for index in range(7)]
        interactions.append((6, "asin-5", 4.0))
        stats = Counter()
        rows = list(user_records("user", interactions, products, set(self.mapping["category_to_id"]), stats))
        self.assertEqual(stats["duplicate_source_events"], 1)
        self.assertEqual([row["task_type"] for row in rows], ["regular", "seen", "regular"])
        self.assertEqual(rows[-1]["full_history_length"], 6)
        self.assertFalse(rows[-1]["target_was_seen"])
        self.assertEqual(rows[1]["target_interest"], "Parent > Seen")
        for row in rows:
            self.assertNotIn(row["target_parent_asin"], row["history_parent_asins"])
            self.assertTrue(all(timestamp < row["timestamp"] for timestamp in row["history_timestamps"]))
            messages = render_messages(row, self.mapping)
            self.assertNotIn("future product", messages[0]["content"])
            self.assertEqual(messages[-1]["content"], self.mapping["category_to_id"][row["target_interest"]])
        novel = {**rows[-1], "task_type": "novel"}
        self.assertEqual(render_messages(novel, self.mapping), render_training_messages(
            novel, "behavior_interest", "interest_id", self.mapping))
        with self.assertRaisesRegex(ValueError, "absent"):
            render_messages({**rows[-1], "task_type": "seen"}, self.mapping)

    def test_curriculum_balances_common_classes_and_only_changes_order(self):
        from collections import Counter
        from crossinterest.methods.curriculum.model import balanced_pools

        sources = {}
        for task in ("seen", "novel"):
            rows = []
            for label, count in (("common", 4), ("rare", 1 if task == "seen" else 4)):
                for index in range(count):
                    identity = f"{task}-{label}-{index}"
                    rows.append({**self.row, "id": identity, "historical_titles": [identity],
                                 "target_interest": label,
                                 "full_seen_interests": [label] if task == "seen" else []})
            sources[task] = rows
        result = balanced_pools(sources["seen"], sources["novel"], 3)
        self.assertEqual(result["labels"], ["common"])
        self.assertEqual(set(result["orders"]), {"seen_then_novel", "shuffle"})
        ordered = result["orders"]["seen_then_novel"]
        self.assertEqual([row["task_type"] for row in ordered], ["seen"] * 3 + ["novel"] * 3)
        self.assertEqual({row["id"] for row in ordered},
                         {row["id"] for row in result["orders"]["shuffle"]})
        self.assertEqual(result, balanced_pools(sources["seen"][::-1], sources["novel"][::-1], 3))
        for rows in result["pools"].values():
            self.assertEqual(Counter(row["target_interest"] for row in rows), {"common": 3})
        with self.assertRaisesRegex(ValueError, "duplicate source"):
            balanced_pools(sources["seen"] * 2, sources["novel"], 3)
        with self.assertRaisesRegex(ValueError, "no common"):
            balanced_pools(sources["seen"], sources["novel"], 5)

    def test_curriculum_full_novel_preserves_all_labels_and_filters_shared_history(self):
        from collections import Counter
        from crossinterest.methods.curriculum.model import balanced_pools

        novel = [{**self.row, "id": f"novel-{label}-{index}",
                  "historical_titles": [f"novel-{label}-{index}"],
                  "target_interest": label, "full_seen_interests": []}
                 for label in ("common", "novel-only") for index in range(4)]
        seen = [{**self.row, "id": f"seen-{index}", "historical_titles": [f"seen-{index}"],
                 "target_interest": "common", "full_seen_interests": ["common"]}
                for index in range(4)]
        conflict = {**novel[-1], "id": "conflicting-seen", "target_interest": "common",
                    "full_seen_interests": ["common"]}
        same_label = {**novel[0], "id": "same-label-seen", "full_seen_interests": ["common"]}
        result = balanced_pools(seen + [conflict, same_label], novel, 3, full_novel=True)
        self.assertEqual(result["labels"], ["common", "novel-only"])
        self.assertEqual(result["seen_labels"], ["common"])
        self.assertEqual(Counter(row["target_interest"] for row in result["pools"]["novel"]),
                         {"common": 4, "novel-only": 4})
        self.assertEqual(result["isolation"]["excluded_shared_history"], 2)
        self.assertEqual(result["isolation"]["excluded_conflicting_label"], 1)
        self.assertEqual(len(result["pools"]["seen"]), 3)
        original = {row["id"]: row for row in novel}
        for row in result["pools"]["novel"]:
            self.assertEqual({key: row[key] for key in original[row["id"]]}, original[row["id"]])
        self.assertEqual({row["id"] for row in result["orders"]["shuffle"]},
                         {row["id"] for row in result["orders"]["seen_then_novel"]})
        self.assertEqual([row["task_type"] for row in result["orders"]["seen_then_novel"]],
                         ["seen"] * 3 + ["novel"] * 8)
        with self.assertRaisesRegex(ValueError, "frozen novel input"):
            balanced_pools(seen, novel + [{**novel[0], "id": "duplicate"}], 3, full_novel=True)

    def test_curriculum_sampler_shards_preserve_global_order_without_duplicates(self):
        from accelerate.data_loader import BatchSamplerShard
        from torch.utils.data import BatchSampler
        from crossinterest.methods.curriculum.train import CurriculumTrainer

        trainer = object.__new__(CurriculumTrainer)
        trainer.train_dataset = list(range(36))
        sampler = trainer._get_train_sampler()
        batches = BatchSampler(sampler, batch_size=1, drop_last=False)
        shards = [list(BatchSamplerShard(batches, num_processes=4, process_index=rank,
                                       split_batches=False, even_batches=True)) for rank in range(4)]
        restored = [item for group in zip(*shards, strict=True) for batch in group for item in batch]
        self.assertEqual(restored, list(range(36)))
        self.assertEqual(len(set(restored)), 36)
        self.assertEqual(list(trainer._get_train_sampler(["a", "b"])), [0, 1])

    def test_curriculum_training_metadata_keeps112_ids_for93_balanced_classes(self):
        from crossinterest.methods.curriculum.runner import DATA, train_command
        from crossinterest.methods.curriculum.train import training_metadata
        from crossinterest.training.train import parse_args

        for arm in ("seen_then_novel", "shuffle"):
            command = train_command("base", DATA / f"{arm}.parquet", Path("/tmp/course"), 93000, arm, False)
            start = command.index('crossinterest.methods.curriculum.train') + 1
            end = command.index("--arm")
            args = parse_args(command[start:end])
            metadata = training_metadata(args, arm, DATA / "MANIFEST.json", False)
            self.assertEqual(metadata["training_label_count"], 93)
            self.assertEqual(metadata["semantic_label_count"], 112)
            self.assertFalse(metadata["optimizer_reset_between_tasks"])
            args.init_adapter = Path("/tmp/old")
            with self.assertRaisesRegex(ValueError, "fresh base"):
                training_metadata(args, arm, DATA / "MANIFEST.json", False)

    def test_curriculum_full_novel_metadata_retains112k_and_unequal_task_boundary(self):
        from crossinterest.methods.curriculum.runner import DATA, train_command
        from crossinterest.methods.curriculum.train import training_metadata
        from crossinterest.training.train import parse_args

        data = DATA.parent / "seen500-novel112k"
        for arm in ("seen_then_novel", "shuffle"):
            command = train_command("base", data / f"{arm}.parquet", Path("/tmp/course"),
                                    158500, arm, False, data)
            start = command.index('crossinterest.methods.curriculum.train') + 1
            args = parse_args(command[start:command.index("--arm")])
            metadata = training_metadata(args, arm, data / "MANIFEST.json", False)
            self.assertEqual(metadata["training_label_count"], 112)
            self.assertEqual(metadata["semantic_label_count"], 112)
            self.assertFalse(metadata["optimizer_reset_between_tasks"])
            self.assertEqual(command[command.index("--course-manifest") + 1], str(data / "MANIFEST.json"))
            args.expected_train_rows = 93000
            with self.assertRaisesRegex(ValueError, "row budget"):
                training_metadata(args, arm, data / "MANIFEST.json", False)

    def test_two_stage_recovery_checkpointing_preserves_microbatch_and_warmup(self):
        import os
        import shlex
        import subprocess
        from crossinterest.methods.alignment.data import HERE

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            helpers = root / "helpers.sh"
            helpers.write_text(
                "torchrun() { printf 'TRAIN'; printf ' %s' \"$@\"; printf '\\n'; }\n"
                "python() { :; }\nnvidia-smi() { :; }\n"
            )
            environment = os.environ | {
                "BASH_ENV": str(helpers), "MODEL_KEY": "llama31_8b", "MODEL_NAME": "base",
                "VIEW": "behavior_interest", "NPROC": "4", "PER_DEVICE_BATCH_SIZE": "2",
                "GRADIENT_ACCUMULATION_STEPS": "4", "AMLT_OUTPUT_DIR": str(root), "SMOKE": "0",
            }
            for enabled in ("1", "0"):
                result = subprocess.run(["bash", str(HERE / "run_train_eval.sh")],
                                        cwd=HERE.parent, capture_output=True, text=True, check=True,
                                        env=environment | {"RECOMMENDATION_GRADIENT_CHECKPOINTING": enabled})
                warmup, recommendation = [shlex.split(line) for line in result.stdout.splitlines()]
                self.assertNotIn("--gradient-checkpointing", warmup)
                self.assertEqual("--gradient-checkpointing" in recommendation, enabled == "1")
                for arguments in (warmup, recommendation):
                    self.assertEqual(arguments[arguments.index("--per-device-batch-size") + 1], "2")
                    self.assertEqual(arguments[arguments.index("--gradient-accumulation-steps") + 1], "4")
                self.assertEqual(recommendation[recommendation.index("--epochs") + 1], "1")
                self.assertIn("224000", recommendation)

    def test_result_contract_rejects_missing_and_mismatched_fields(self):
        from crossinterest.methods.alignment.audit import require_fields

        require_fields({"complete": True, "count": 2000}, {"complete": True, "count": 2000}, "test")
        with self.assertRaisesRegex(ValueError, "count"):
            require_fields({"complete": True}, {"count": 2000}, "test")
        with self.assertRaisesRegex(ValueError, "complete"):
            require_fields({"complete": False}, {"complete": True}, "test")

    def test_placement_uses_both_authorized_targets(self):
        from crossinterest.methods.alignment.runner import placement

        self.assertEqual(placement({"Feeds": 7, "Recall": 9}),
                         {"qwen35": "Recall", "llama31": "Feeds"})
        self.assertEqual(set(placement({"Feeds": 16, "Recall": 0}).values()), {"Feeds"})

    def test_full_result_audit_accepts_valid_fixture_and_rejects_corruption(self):
        import json
        import pyarrow.parquet as pq
        from crossinterest.evaluation.common import metrics_from_details
        from crossinterest.training.evaluate_ranking import metrics_from_rankings, score_ranking
        from crossinterest.methods.alignment.audit import audit, MODELS
        from crossinterest.methods.alignment.data import BASE, HERE, sha256_file

        manifest_path = HERE / "data" / "MANIFEST.json"
        manifest = json.loads(manifest_path.read_text())
        dataset = BASE / "eval" / "dev_2k_interest.parquet"
        rows = pq.read_table(dataset).to_pylist()[:4]
        model_key = "qwen35_9b"
        with TemporaryDirectory() as directory:
            root = Path(directory)

            def write(relative, record):
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(json.dumps(record) + "\n")

            for stage in ("warmup", "recommendation"):
                metadata = {
                    "stage": stage, "smoke": True, "max_steps": 2,
                    "manifest_sha256": sha256_file(manifest_path),
                    "task_counts": manifest[stage]["task_counts"],
                }
                contract = {
                    "model": MODELS[model_key], "view": "interest", "label_mode": "interest_id",
                    "label_map_sha256": manifest["mapping_sha256"],
                    "train_sha256": manifest[stage]["sha256"], "epochs": manifest[stage]["epochs"],
                    "effective_batch_size": 32, "learning_rate": 1e-4, "seed": 20260825,
                    "semantic_token_registration": {}, "experiment_metadata": metadata,
                }
                if stage == "recommendation":
                    contract["init_adapter_model_sha256"] = "warmup-weights"
                    metadata["handoff"] = {
                        "adapter_model_sha256": "warmup-weights",
                        "train_contract_sha256": sha256_file(root / "warmup" / "train_contract.json"),
                        "optimizer_reset": True, "scheduler_reset": True,
                    }
                write(f"{stage}/train_contract.json", contract)
                write(f"{stage}/train_metrics.json", {
                    **contract, "sample_count": manifest[stage]["rows"], "train_loss": 1.0,
                    "epoch": 0.01,
                })
                write(f"{stage}/stage_audit.json", {
                    "passed": True, "stage": stage, "smoke": True,
                    "adapter_model_sha256": f"{stage}-weights",
                })
            for evaluation, beams, decoding in (
                ("classification", 10, "candidate_constrained_beam_search"),
                ("ranking", 20, "candidate_free_direct_beam_search"),
            ):
                contract = {
                    "model_key": model_key, "model": MODELS[model_key], "view": "interest",
                    "label_mode": "interest_id", "label_map_sha256": manifest["mapping_sha256"],
                    "dataset_sha256": sha256_file(dataset), "sample_limit": 4, "num_beams": beams,
                    "decoding": decoding, "do_sample": False, "semantic_token_registration": {},
                    "decoder_label_constraint": False, "candidates_in_prompt": False,
                    "label_file_sha256": manifest["train_sha256"], "fingerprint": "fixture",
                }
                write(f"{evaluation}/run_contract.json", contract)
                details = []
                for row in rows:
                    detail = {
                        "sample_id": row["sample_id"], "source_sample_id": row["source_sample_id"],
                        "target_interest": row["target_interest"], "run_fingerprint": "fixture",
                        "prediction": row["target_interest"], "correct": True,
                        "format_valid": True, "resolved": True,
                    }
                    if evaluation == "ranking":
                        detail["ranking"] = [
                            {"rank": rank, "prediction": row["target_interest"] if rank == 1 else None,
                             "format_valid": rank == 1}
                            for rank in range(1, 21)
                        ]
                        detail.update(score_ranking(detail["ranking"], row["target_interest"]))
                    details.append(detail)
                (root / evaluation / "details.jsonl").write_text(
                    "".join(json.dumps(detail) + "\n" for detail in details)
                )
                metrics = (metrics_from_details(details, rows) if evaluation == "classification"
                           else metrics_from_rankings(details, rows, 20))
                write(f"{evaluation}/metrics.json", {
                    **metrics, "model_key": model_key, "input_type": "interest",
                    "label_mode": "interest_id", "decoding": decoding,
                })
            self.assertTrue(audit(root, model_key, True)["passed"])
            path = root / "ranking" / "metrics.json"
            original = json.loads(path.read_text())
            for key, value in (("macro_recall_at_20", 0.0), ("recorded_sample_count", 3)):
                with self.subTest(field=key):
                    write("ranking/metrics.json", {**original, key: value})
                    with self.assertRaises(ValueError):
                        audit(root, model_key, True)

    def test_monitor_existing_never_submits_or_lists_experiments(self):
        import io
        import json
        import subprocess
        from unittest.mock import MagicMock, patch
        import crossinterest.methods.alignment.runner as execution

        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "outputs").mkdir()
            state_path = root / "outputs" / "EXECUTION_STATE.json"
            state_path.write_text(json.dumps({"formal_experiment": execution.FORMAL_EXPERIMENT}))
            (root / "EXPERIMENT_REPORT.md").write_text("pending")
            statuses = [{"job_name": f":{prefix}-two-stage-formal"} for prefix in execution.JOBS]
            process = MagicMock()
            process.stdout = io.StringIO('{"event":"completed","exit_code":0}\n')
            process.wait.return_value = 0
            process.__enter__.return_value = process
            with patch.object(execution, "HERE", root), \
                 patch("sys.argv", ["execute", "--monitor-existing"]), \
                 patch.object(execution.subprocess, "run", side_effect=[
                     subprocess.CompletedProcess([], 0, json.dumps(statuses), ""),
                     subprocess.CompletedProcess([], 0, "", ""),
                     subprocess.CompletedProcess([], 0, "", ""),
                 ]) as run, \
                 patch.object(execution.subprocess, "Popen", return_value=process) as popen, \
                 patch.object(execution, "audit", return_value={"passed": True}) as audit_mock, \
                 patch.object(execution, "render_formal_report", return_value="audited"):
                for prefix in execution.JOBS:
                    (root / "outputs" / "formal" / execution.FORMAL_EXPERIMENT /
                     f"{prefix}-two-stage-formal").mkdir(parents=True)
                execution.main()
            calls = [call.args[0][1:] for call in run.call_args_list]
            self.assertEqual(calls[0], ["--json-tables", "status", execution.FORMAL_EXPERIMENT])
            self.assertTrue(all(call[:2] == ["results", "download"] for call in calls[1:]))
            self.assertEqual(popen.call_args.args[0][1], "watch")
            self.assertEqual(audit_mock.call_count, 2)
            self.assertEqual(json.loads(state_path.read_text())["status"], "formal_results_audited")

    def test_report_populates_only_audited_formal_results(self):
        from copy import deepcopy
        import xml.etree.ElementTree as ET
        from crossinterest.methods.alignment.data import HERE
        from crossinterest.methods.alignment.runner import render_formal_report, JOBS

        summary = {
            "smoke": False, "passed": True,
            "stage_audits": {stage: {"train_loss": 1.0} for stage in ("warmup", "recommendation")},
            "results": {
                "classification": {"complete": True, "recorded_sample_count": 2000,
                                   "class_count": 112, "macro_accuracy": 0.125,
                                   "strict_format_rate": 1.0},
                "ranking": {"complete": True, "recorded_sample_count": 2000, "class_count": 112,
                            "macro_recall_at_10": 0.2, "macro_recall_at_20": 0.3,
                            "macro_ndcg_at_10": 0.1, "macro_ndcg_at_20": 0.12,
                            "valid_beam_rate": 0.9, "strict_beam_format_rate": 0.9},
            },
        }
        summaries = {key: deepcopy(summary) for key in JOBS.values()}
        text = (HERE / "EXPERIMENT_REPORT.md").read_text()
        rendered = render_formal_report(text, summaries)
        self.assertEqual(render_formal_report(rendered, summaries), rendered)
        for pending in ("新方法当前无正式结果，不做优劣判断。", "新方案当前无正式结果，不做优劣判断。"):
            pending_text = rendered.replace("两模型正式结果已完成合同审计，并从原始预测重算一致。", pending)
            self.assertEqual(render_formal_report(pending_text, summaries), rendered)
        table = ET.fromstring(rendered[rendered.index("<table>"):rendered.index("</table>") + 8])
        self.assertEqual([cell.text for cell in table.find("thead").find("tr")],
                         ["模型", "方法", "Macro Recall@10 ↑", "Macro Recall@20 ↑",
                          "Macro nDCG@10 ↑", "Macro nDCG@20 ↑"])
        rows = table.find("tbody").findall("tr")
        self.assertEqual([len(row) for row in rows], [6, 5, 6, 5])
        self.assertEqual([cell.text for cell in list(rows[1])[1:]],
                         ["0.2000", "0.3000", "0.1000", "0.1200"])
        summaries["qwen35_9b"]["smoke"] = True
        with self.assertRaisesRegex(ValueError, "smoke"):
            render_formal_report(text, summaries)

    def test_behavior_interest_monitor_uses_separate_experiment_and_state(self):
        import io
        import json
        import subprocess
        from unittest.mock import MagicMock, patch
        import crossinterest.methods.alignment.runner as execution

        with TemporaryDirectory() as directory:
            root = Path(directory)
            experiment_root = root / "behavior_interest"
            (experiment_root / "outputs").mkdir(parents=True)
            formal = execution.EXPERIMENTS["behavior_interest"][1]
            state_path = experiment_root / "outputs" / "EXECUTION_STATE.json"
            state_path.write_text(json.dumps({"formal_experiment": formal, "view": "behavior_interest"}))
            statuses = [{"job_name": f":{prefix}-two-stage-formal"} for prefix in execution.JOBS]
            process = MagicMock()
            process.stdout = io.StringIO("")
            process.wait.return_value = 12
            process.__enter__.return_value = process
            with patch.object(execution, "HERE", root), \
                 patch("sys.argv", ["execute", "--monitor-existing", "--view", "behavior_interest"]), \
                 patch.object(execution.subprocess, "run", return_value=
                              subprocess.CompletedProcess([], 0, json.dumps(statuses), "")) as run, \
                 patch.object(execution.subprocess, "Popen", return_value=process) as popen:
                with self.assertRaisesRegex(RuntimeError, "exit 12"):
                    execution.main()
            run.assert_called_once()
            self.assertEqual(run.call_args.args[0][1:], ["--json-tables", "status", formal])
            self.assertEqual(run.call_args.kwargs["cwd"], experiment_root)
            self.assertEqual(popen.call_args.args[0][1:3], ["watch", formal])
            self.assertFalse((root / "outputs").exists())
            self.assertEqual(json.loads(state_path.read_text())["status"], "stopped_with_error")

    def test_smoke_recovery_selects_replacement_without_registry_or_resubmission(self):
        import io
        import json
        import subprocess
        from unittest.mock import MagicMock, patch
        import crossinterest.methods.alignment.runner as execution

        with TemporaryDirectory() as directory:
            root = Path(directory)
            experiment_root = root / "behavior_interest"
            output = experiment_root / "outputs"
            output.mkdir(parents=True)
            smoke, formal = execution.EXPERIMENTS["behavior_interest"]
            state = {"view": "behavior_interest", "smoke_experiment": smoke,
                     "formal_experiment": formal, "formal_submissions": [],
                     "status": "stopped_with_error", "error": "amlt watch: exit 12; see execution.log"}
            state_path = output / "EXECUTION_STATE.json"
            state_path.write_text(json.dumps(state))
            names = ["qwen35-two-stage-smoke-micro1", "llama31-two-stage-smoke"]
            for name in names:
                (output / "smoke" / smoke / name).mkdir(parents=True)
            process = MagicMock()
            process.stdout = io.StringIO("")
            process.wait.return_value = 0
            process.__enter__.return_value = process
            argv = ["execute", "--view", "behavior_interest", "--resume-smoke",
                    "--smoke-job", f"qwen35={names[0]}"]
            with patch.object(execution, "HERE", root), patch("sys.argv", argv), \
                 patch.object(execution.subprocess, "Popen", return_value=process) as popen, \
                 patch.object(execution.subprocess, "run", side_effect=[
                     subprocess.CompletedProcess([], 0, "", ""),
                     subprocess.CompletedProcess([], 0, "", ""),
                     RuntimeError("stop after smoke audit"),
                 ]) as run, patch.object(execution, "audit", return_value={"passed": True}) as audit_mock:
                with self.assertRaisesRegex(RuntimeError, "stop after smoke audit"):
                    execution.main()
            self.assertEqual(popen.call_args.args[0][1:],
                             ["watch", smoke, *(f":{name}" for name in names), "--json", "--no-notify"])
            for call in run.call_args_list[:2]:
                self.assertEqual(call.args[0][1:6],
                                 ["results", "download", smoke, *(f":{name}" for name in names)])
            self.assertEqual(run.call_args_list[2].args[0][1:], ["--json-tables", "target", "list", "sing"])
            self.assertEqual([call.args[0].name for call in audit_mock.call_args_list], names)
            self.assertTrue(all(call.args[2:] == (True, "behavior_interest")
                                for call in audit_mock.call_args_list))
            self.assertEqual(json.loads(state_path.read_text())["failed_at"], "smoke_audited")
            for field, value in (("placement", {}), ("monitor_pid", 123),
                                 ("formal_submissions", [{"job": "already-submitted"}]),
                                 ("failed_at", "submitting_formal"), ("view", "interest")):
                with self.subTest(field=field):
                    state_path.write_text(json.dumps({**state, field: value}))
                    with patch.object(execution, "HERE", root), patch("sys.argv", argv), \
                         patch.object(execution.subprocess, "run") as run:
                        with self.assertRaisesRegex(ValueError, "smoke recovery"):
                            execution.main()
                        run.assert_not_called()

    def test_ranking_output_repair_preserves_backup_and_valid_records(self):
        import json
        from crossinterest.methods.backbones.runner import repair_ranking_output

        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "details.jsonl"
            record = {"sample_id": "sample-1", "run_fingerprint": "fixed", "ranking": []}
            original = (json.dumps(record) + "\ntruncated-record\n").encode()
            path.write_bytes(original)
            receipt = repair_ranking_output(root)
            self.assertEqual(receipt["retained_records"], 1)
            self.assertEqual(receipt["invalid_lines"], [2])
            self.assertTrue(receipt["missing_predictions_must_be_regenerated"])
            self.assertEqual(Path(receipt["backup"]).read_bytes(), original)
            self.assertEqual(json.loads(path.read_text()), record)
            self.assertIsNone(repair_ranking_output(root))
            path.write_text('{"unexpected": true}\n')
            with self.assertRaisesRegex(ValueError, "schema"):
                repair_ranking_output(root)

    def test_weighted_backbone_preserves_two_stage_command_contract(self):
        from unittest.mock import patch
        import crossinterest.methods.backbones.runner as backbone
        from crossinterest.methods.alignment.data import sha256_file

        with TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "weights.parquet"
            weights.write_bytes(b"weight-file-hash-fixture")
            stage = {"file": str(weights), "sha256": sha256_file(weights)}
            manifest = {"model_key": "qwen35_9b", "model_repo": "fixed-repo",
                        "model_revision": "fixed-revision", "code_sha256": {},
                        "smoke_sha256": "fixed-smoke", "weighting": {"smoke": stage, "formal": stage}}
            with patch.object(backbone.subprocess, "run") as run, \
                 patch.object(backbone, "audit_run", return_value={"passed": True}):
                backbone.run_stages(manifest, "fixed-model", root / "output")
            self.assertEqual(run.call_count, 2)
            for index, call in enumerate(run.call_args_list):
                environment = call.kwargs["env"]
                self.assertEqual(environment["TRAIN_SELECTION_SIZE"], "224" if index == 0 else "112000")
                self.assertEqual(environment["TRAIN_SELECTION_FILE"], str(weights))
                self.assertEqual(environment["TRAIN_SELECTION_WEIGHT_COLUMN"], "sample_weight")
                self.assertEqual(environment["GRADIENT_ACCUMULATION_STEPS"], "8")
                self.assertEqual(environment["PER_DEVICE_BATCH_SIZE"], "1")
                self.assertEqual(environment["NPROC"], "4")
                self.assertEqual(environment["RANKING_NUM_BEAMS"], "20")
                self.assertEqual("EVAL_LIMIT" in environment, index == 0)
            self.assertEqual(backbone.selection_fields({}, False), {"selection_file": None})
            weights.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "weights changed"):
                backbone.selection_fields(manifest, False)

    def test_warmup_rejects_user_history(self):
        row = alignment_rows(self.mapping)[0]
        row["historical_interests"] = ["Parent > Seen"]
        with self.assertRaisesRegex(ValueError, "alignment"):
            render_messages(row, self.mapping)

    def test_unknown_task_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported task"):
            render_messages({**self.row, "task_type": "recommend_next_item"}, self.mapping)

    def test_stage_contract_and_wrong_mixture_rejection(self):
        import json
        from crossinterest.training.train import parse_args
        from crossinterest.methods.alignment.train import stage_metadata

        data = _source_workspace('eval_interest_id_minionerec') / "data"
        args = parse_args([
            "--model", "test-model", "--view", "interest", "--label-mode", "interest_id",
            "--train-file", str(data / "warmup.parquet"), "--output-dir", "/tmp/unused",
            "--expected-train-rows", "224", "--epochs", "10",
        ])
        metadata = stage_metadata(args, "warmup", data / "MANIFEST.json", False)
        self.assertEqual(metadata["stage"], "warmup")
        self.assertIsNone(metadata["handoff"])
        args.view = "behavior_interest"
        with self.assertRaisesRegex(ValueError, "view must match"):
            stage_metadata(args, "warmup", data / "MANIFEST.json", False)
        with TemporaryDirectory() as directory:
            manifest = json.loads((data / "MANIFEST.json").read_text())
            manifest["view"] = "behavior_interest"
            path = Path(directory) / "MANIFEST.json"
            path.write_text(json.dumps(manifest))
            self.assertEqual(stage_metadata(args, "warmup", path, False)["stage"], "warmup")
        args.view = "interest"
        args.train_file = data / "recommendation.parquet"
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            stage_metadata(args, "warmup", data / "MANIFEST.json", False)

    def test_dataset_forwards_behavior_interest_view_to_tokenizer(self):
        from unittest.mock import patch
        import pyarrow as pa
        from crossinterest.methods.alignment.train import TaskDataset

        dataset = TaskDataset.__new__(TaskDataset)
        row = recommendation_rows(self.row, self.mapping)[0]
        dataset.table = pa.Table.from_pylist([row])
        dataset.mapping = self.mapping
        dataset.view = "behavior_interest"
        dataset.tokenizer = object()
        dataset.max_length = 8192
        with patch("eval_interest_id_minionerec.train.baseline.tokenize_assistant_only",
                   return_value={"input_ids": [1], "labels": [1]}) as tokenize:
            self.assertEqual(dataset[0]["input_ids"], [1])
        tokenize.assert_called_once_with(dataset.tokenizer,
                                        render_messages(row, self.mapping, "behavior_interest"))

    def test_sparse_embeddings_and_lora_survive_trainable_handoff(self):
        import torch
        from peft import LoraConfig
        from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM
        from crossinterest.training.semantic_tokens import semantic_lora_target_modules
        from crossinterest.training.train import attach_or_load_lora

        cases = (
            (LlamaConfig, LlamaForCausalLM, False),
            (Qwen2Config, Qwen2ForCausalLM, True),
        )
        for config_type, model_type, tied in cases:
            with self.subTest(model_type=model_type.__name__, tied=tied):
                config = config_type(vocab_size=64, hidden_size=32, intermediate_size=64,
                                     num_hidden_layers=1, num_attention_heads=2,
                                     num_key_value_heads=2, tie_word_embeddings=tied)
                original_config = config.to_dict()
                base = model_type(config)
                base.eval()
                pristine = {key: value.detach().clone() for key, value in base.state_dict().items()}
                inputs = torch.tensor([[3, 60, 4]])
                with torch.no_grad():
                    original_logits = base(input_ids=inputs).logits
                lora = LoraConfig(r=2, lora_alpha=4, lora_dropout=0,
                                  target_modules=semantic_lora_target_modules(base, False),
                                  trainable_token_indices=[60, 61], task_type="CAUSAL_LM")
                model = attach_or_load_lora(base, lora, None)
                model.eval()
                with torch.no_grad():
                    torch.testing.assert_close(model(input_ids=inputs).logits,
                                               original_logits, rtol=0, atol=0)
                labels = torch.tensor([[-100, -100, 61]])
                optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
                model(input_ids=inputs, labels=labels).loss.backward()
                optimizer.step()
                model.eval()
                with torch.no_grad():
                    expected = model(input_ids=inputs).logits
                with TemporaryDirectory() as directory:
                    model.save_pretrained(directory, save_embedding_layers=False)
                    reloaded_base = model_type(config_type.from_dict(original_config))
                    reloaded_base.load_state_dict(pristine)
                    restored = attach_or_load_lora(reloaded_base, lora, Path(directory))
                    restored.eval()
                    with torch.no_grad():
                        actual = restored(input_ids=inputs).logits
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertTrue(any("trainable_tokens" in name and value.requires_grad
                                        for name, value in restored.named_parameters()))
                    self.assertTrue(any("lora_B" in name and value.requires_grad
                                        for name, value in restored.named_parameters()))


if __name__ == "__main__":
    unittest.main()