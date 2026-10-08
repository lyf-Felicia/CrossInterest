#!/usr/bin/env python3
"""Apply audited model-compatibility patches to the pinned official MeRA source."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def replace_exact(source: str, old: str, new: str, *, count: int = 1) -> str:
    actual = source.count(old)
    if actual != count:
        raise ValueError(f"expected {count} occurrences, found {actual}: {old[:80]!r}")
    return source.replace(old, new)


def patch_train(source: str) -> str:
    source = replace_exact(
        source,
        "    num_epochs: int = 3,\n    learning_rate: float = 3e-4,\n",
        "    num_epochs: int = 3,\n    max_steps: int = -1,\n    learning_rate: float = 3e-4,\n",
    )
    source = replace_exact(
        source,
        "    prepare_model_for_int8_training,\n",
        "    prepare_model_for_kbit_training,\n",
    )
    source = replace_exact(
        source,
        "from transformers import AutoTokenizer, AutoModelForCausalLM\n",
        '''import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

if os.environ.get("MERA_SUPPRESS_BNB_CAST_LOG") == "1":
    logging.getLogger("bitsandbytes.autograd._functions").setLevel(logging.ERROR)
''',
    )
    source = replace_exact(
        source,
        """    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = (
        151643  # unk. we want this to be different from the eos token
    )
""",
        """    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
""",
    )
    source = replace_exact(
        source,
        """    max_valid_token_id = max(tokenizer.all_special_ids)
    num_old_tokens = max_valid_token_id + 1
    tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
""",
        """    num_old_tokens = len(tokenizer)
    tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
""",
    )
    source = replace_exact(
        source,
        "    old_token_embeddings = model.model.embed_tokens\n",
        "    old_token_embeddings = model.get_input_embeddings()\n",
    )
    source = replace_exact(
        source,
        """    new_token_embeddings = new_token_embeddings.cuda()

    new_token_embeddings_all = torch.nn.Embedding(num_old_tokens + len(new_tokens), num_dims)
    new_token_embeddings_all.weight.data = torch.cat((old_token_embeddings.weight.data[:num_old_tokens], new_token_embeddings.weight.data), 0)

    model.model.embed_tokens = new_token_embeddings_all.cuda()
""",
        """    embedding_device = old_token_embeddings.weight.device
    embedding_dtype = old_token_embeddings.weight.dtype
    new_token_embeddings = new_token_embeddings.to(
        device=embedding_device, dtype=embedding_dtype
    )

    new_token_embeddings_all = torch.nn.Embedding(
        num_old_tokens + len(new_tokens), num_dims,
        device=embedding_device, dtype=embedding_dtype,
    )
    new_token_embeddings_all.weight.data = torch.cat((old_token_embeddings.weight.data[:num_old_tokens], new_token_embeddings.weight.data), 0)

    model.set_input_embeddings(new_token_embeddings_all)
""",
    )
    source = replace_exact(
        source,
        "    model = prepare_model_for_int8_training(model)\n",
        "    model = prepare_model_for_kbit_training(model)\n",
    )
    source = replace_exact(
        source,
        """    model = prepare_model_for_kbit_training(model)


    config = LoraConfig(
""",
        """    model = prepare_model_for_kbit_training(model)

    decoder_linear_modules = sorted(
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and ".layers." in name
    )
    target_matches = lambda name: any(
        name == target or name.endswith("." + target)
        for target in lora_target_modules
    )
    targeted_linear_modules = sorted(
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and target_matches(name)
    )
    if targeted_linear_modules != decoder_linear_modules:
        missing = sorted(set(decoder_linear_modules) - set(targeted_linear_modules))
        extra = sorted(set(targeted_linear_modules) - set(decoder_linear_modules))
        raise ValueError(
            f"LoRA targets must cover every decoder linear projection: missing={missing}, extra={extra}"
        )

    config = LoraConfig(
""",
    )
    source = replace_exact(
        source,
        """    model = get_peft_model(model, config)

    # initialize lm_head for new tokens using normal distribution (model.model.lm_head.weight[num_old_tokens:] = normal(0, 0.02))
""",
        """    model = get_peft_model(model, config)

    def active_module(module):
        modules_to_save = getattr(module, "modules_to_save", None)
        if modules_to_save is None:
            return module
        active_adapter = getattr(module, "active_adapter", None)
        if isinstance(active_adapter, (list, tuple)):
            active_adapter = active_adapter[0]
        if active_adapter in modules_to_save:
            return modules_to_save[active_adapter]
        if len(modules_to_save) == 1:
            return next(iter(modules_to_save.values()))
        raise ValueError("cannot resolve active modules_to_save adapter")

    def input_embedding_rows(active_model, token_ids):
        module = active_module(active_model.get_input_embeddings())
        return module(token_ids.to(module.weight.device))

    def output_embedding_rows(active_model, token_ids):
        module = active_module(active_model.get_output_embeddings())
        return module.weight[token_ids.to(module.weight.device)]

    actual_lora_modules = []
    for name, module in model.named_modules():
        if not hasattr(module, "lora_A"):
            continue
        matches = [expected for expected in decoder_linear_modules if name.endswith(expected)]
        if len(matches) != 1:
            raise ValueError(f"cannot map PEFT LoRA module to decoder projection: {name}")
        actual_lora_modules.append(matches[0])
    actual_lora_modules.sort()
    if actual_lora_modules != decoder_linear_modules:
        raise ValueError(
            f"PEFT LoRA module mismatch: {actual_lora_modules} != {decoder_linear_modules}"
        )
    configured_modules_to_save = sorted(model.peft_config["default"].modules_to_save)
    if configured_modules_to_save != ["embed_tokens", "lm_head"]:
        raise ValueError(f"unexpected modules_to_save: {configured_modules_to_save}")
    input_wrapper = model.get_input_embeddings()
    output_wrapper = model.get_output_embeddings()
    if not hasattr(input_wrapper, "modules_to_save") or not hasattr(output_wrapper, "modules_to_save"):
        raise ValueError("input embedding and LM head must both use ModulesToSaveWrapper")
    active_input = active_module(input_wrapper)
    active_output = active_module(output_wrapper)
    if not active_input.weight.requires_grad or not active_output.weight.requires_grad:
        raise ValueError("active input embedding and LM head must both be trainable")
    if active_input.weight.shape[0] != len(tokenizer) or active_output.weight.shape[0] != len(tokenizer):
        raise ValueError("trainable input/output vocabulary does not match tokenizer")

    runtime_contract_path = os.environ.get("MERA_RUNTIME_CONTRACT_PATH")
    runtime_contract = {
        "schema_version": 1,
        "model_class": type(model.get_base_model()).__name__,
        "loaded_in_8bit": bool(getattr(model.get_base_model(), "is_loaded_in_8bit", False)),
        "lora_target_modules": sorted(lora_target_modules),
        "decoder_linear_modules": decoder_linear_modules,
        "actual_lora_modules": actual_lora_modules,
        "modules_to_save": configured_modules_to_save,
        "input_embedding_shape": list(active_input.weight.shape),
        "output_embedding_shape": list(active_output.weight.shape),
        "input_embedding_trainable": active_input.weight.requires_grad,
        "output_embedding_trainable": active_output.weight.requires_grad,
    }

    def write_runtime_contract():
        if runtime_contract_path:
            with open(runtime_contract_path, "w", encoding="utf-8") as handle:
                json.dump(runtime_contract, handle, indent=2)
                handle.write("\\n")

    write_runtime_contract()

    # initialize lm_head for new tokens using normal distribution (model.model.lm_head.weight[num_old_tokens:] = normal(0, 0.02))
""",
    )
    source = replace_exact(
        source,
        "    print(model.model.lm_head.weight.data[num_old_tokens:])\n",
        "    print(active_module(model.get_output_embeddings()).weight.data[num_old_tokens:])\n",
    )
    source = replace_exact(
        source,
        "self.model.model.model.embed_tokens(contrastive_data)",
        "input_embedding_rows(self.unwrap_model(self.model), contrastive_data)",
        count=4,
    )
    source = replace_exact(
        source,
        "self.model.model.lm_head.weight[contrastive_data]",
        "output_embedding_rows(self.unwrap_model(self.model), contrastive_data)",
        count=4,
    )
    source = replace_exact(
        source,
        """                loss = -torch.log_softmax(similarity_tokens, dim=1)[:, 0].mean() + -torch.log_softmax(similarity_lm_head, dim=1)[:, 0].mean()
""",
        """                token_loss = -torch.log_softmax(similarity_tokens, dim=1)[:, 0].mean()
                head_loss = -torch.log_softmax(similarity_lm_head, dim=1)[:, 0].mean()
                loss = token_loss.to(head_loss.device) + head_loss
""",
        count=2,
    )
    source = replace_exact(
        source,
        "        def compute_loss(self, model, inputs, return_outputs=False):\n",
        "        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):\n",
    )
    source = replace_exact(
        source,
        """            self.contrastive_module = contrastive_module

            self.brothers = torch.zeros(len(new_tokens), len(new_tokens)).to(self.model.device)
""",
        """            self.contrastive_module = contrastive_module
            self.structure_loss_calls = 0
            self.collaborative_loss_calls = 0
            self.main_loss_device = None
            self.structure_loss_device = None
            self.collaborative_loss_device = None
            self.combined_loss_device = None

            self.brothers = torch.zeros(len(new_tokens), len(new_tokens)).to(self.model.device)
""",
    )
    source = replace_exact(
        source,
        """        def compute_collaborative_loss(self):
            \"\"\"
            Using infonce loss to enforce the collaborative of the new tokens
            \"\"\"
""",
        """        def compute_collaborative_loss(self):
            \"\"\"
            Using infonce loss to enforce the collaborative of the new tokens
            \"\"\"
            self.collaborative_loss_calls += 1
""",
    )
    source = replace_exact(
        source,
        """        def compute_structure_loss(self):
            \"\"\"
            Using infonce loss to enforce the structure of the new tokens
            \"\"\"
""",
        """        def compute_structure_loss(self):
            \"\"\"
            Using infonce loss to enforce the structure of the new tokens
            \"\"\"
            self.structure_loss_calls += 1
""",
    )
    source = replace_exact(
        source,
        "                shift_logits = shift_logits.view(-1, model.config.vocab_size)\n",
        "                shift_logits = shift_logits.view(-1, shift_logits.size(-1))\n",
    )
    source = replace_exact(
        source,
        """                loss = loss_fct(shift_logits, shift_labels) 
                if self.structure_weight > 0:
                    loss += self.structure_weight * self.compute_structure_loss()
                if self.collaborative_weight > 0:
                    loss += self.collaborative_weight * self.compute_collaborative_loss()
""",
        """                loss = loss_fct(shift_logits, shift_labels)
                self.main_loss_device = str(loss.device)
                if self.structure_weight > 0:
                    structure_loss = self.compute_structure_loss()
                    self.structure_loss_device = str(structure_loss.device)
                    loss = loss + self.structure_weight * structure_loss.to(loss.device)
                if self.collaborative_weight > 0:
                    collaborative_loss = self.compute_collaborative_loss()
                    self.collaborative_loss_device = str(collaborative_loss.device)
                    loss = loss + self.collaborative_weight * collaborative_loss.to(loss.device)
                self.combined_loss_device = str(loss.device)
""",
    )
    source = replace_exact(
        source,
        "            evaluation_strategy=\"steps\",\n",
        "            eval_strategy=\"steps\",\n",
    )
    source = replace_exact(
        source,
        "            group_by_length=group_by_length,\n",
        "            train_sampling_strategy=\"group_by_length\" if group_by_length else \"random\",\n",
    )
    source = replace_exact(
        source,
        "            report_to=None,\n",
        "            report_to=[],\n",
    )
    source = replace_exact(
        source,
        "            num_train_epochs=num_epochs,\n            learning_rate=learning_rate,\n",
        "            num_train_epochs=num_epochs,\n            max_steps=max_steps,\n            learning_rate=learning_rate,\n",
    )
    source = replace_exact(
        source,
        "    trainer.train(resume_from_checkpoint=resume_from_checkpoint)\n",
        """    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    runtime_optimizer = trainer.optimizer
    while hasattr(runtime_optimizer, "optimizer"):
        runtime_optimizer = runtime_optimizer.optimizer
    runtime_scheduler = trainer.lr_scheduler
    while hasattr(runtime_scheduler, "scheduler"):
        runtime_scheduler = runtime_scheduler.scheduler
    runtime_contract.update({
        "global_step": trainer.state.global_step,
        "structure_loss_calls": trainer.structure_loss_calls,
        "collaborative_loss_calls": trainer.collaborative_loss_calls,
        "main_loss_device": trainer.main_loss_device,
        "structure_loss_device": trainer.structure_loss_device,
        "collaborative_loss_device": trainer.collaborative_loss_device,
        "combined_loss_device": trainer.combined_loss_device,
        "train_sampling_strategy": trainer.args.train_sampling_strategy,
        "report_to": trainer.args.report_to,
        "optimizer_class": type(runtime_optimizer).__name__,
        "scheduler_class": type(runtime_scheduler).__name__,
    })
    write_runtime_contract()
""",
    )
    source = replace_exact(
        source,
        """    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    torch.save(model.model.model.embed_tokens.weight.data, os.path.join(output_dir, "new_token_embeddings.pt"))
""",
        """    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        torch.save(active_module(model.get_input_embeddings()).weight.data, os.path.join(output_dir, "new_token_embeddings.pt"))
""",
    )
    return source


def patch_inference(source: str) -> str:
    source = replace_exact(
        source,
        "from transformers import AutoTokenizer, AutoModelForCausalLM\n",
        "from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig\n",
    )
    source = replace_exact(
        source,
        "    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)\n",
        "    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)\n",
    )
    source = replace_exact(
        source,
        "    additional_special_token_ids = tokenizer.additional_special_tokens_ids\n",
        """    added_token_ids = sorted(tokenizer.get_added_vocab().values())
    additional_special_token_ids = added_token_ids[-num_beams:]
    expected_token_ids = list(range(
        additional_special_token_ids[0],
        additional_special_token_ids[0] + num_beams,
    )) if additional_special_token_ids else []
    category_tokens = tokenizer.convert_ids_to_tokens(additional_special_token_ids)
    if (
        len(additional_special_token_ids) != num_beams
        or additional_special_token_ids != expected_token_ids
        or any(
            not isinstance(token, str)
            or len(token) < 2
            or token[0] != '"'
            or token[-1] != '"'
            for token in category_tokens
        )
    ):
        raise ValueError(
            f"expected {num_beams} contiguous quoted category tokens at the end "
            "of the added vocabulary"
        )
""",
    )
    source = replace_exact(
        source,
        """        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
""",
        """        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=(
                BitsAndBytesConfig(load_in_8bit=True) if load_8bit else None
            ),
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
""",
    )
    source = replace_exact(
        source,
        """                torch_dtype=torch.float16,
                device_map={'': 0}
""",
        """                torch_dtype=torch.float16,
                device_map="auto"
""",
    )
    source = replace_exact(
        source,
        """    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 151643  # unk
    model.config.bos_token_id = 151643
    model.config.eos_token_id = 151643
""",
        """    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
""",
    )
    return source


def assert_official_semantics(train_source: str, inference_source: str) -> None:
    required_train = [
        "positive_sample = torch.multinomial(self.positive_prob, 1, replacement=False)",
        "negative_sample = torch.multinomial(self.negative_prob, self.negative_samples, replacement=False)",
        "positive_pairs = random.sample(self.category_structure[first_level_category], 2)",
        "similarity = torch.bmm(rest_embedding, first_embedding.unsqueeze(2)).squeeze(2)",
        "shift_logits = logits[..., :-1, :].contiguous()",
        "shift_labels = labels[..., 1:].contiguous()",
        "warmup_steps=20",
        'optim="adamw_torch"',
        "EarlyStoppingCallback(early_stopping_patience=2)",
        "gradient_accumulation_steps = batch_size // micro_batch_size",
        "generate_prompt_with_prompt",
        'train_sampling_strategy="group_by_length" if group_by_length else "random"',
        "report_to=[]",
        "structure_loss.to(loss.device)",
        "collaborative_loss.to(loss.device)",
    ]
    required_inference = [
        "generate_prompt_without_prompt",
        "output = output[:, -1, additional_special_token_ids]",
        "torch.topk(output, dim=-1, k=num_beams)",
    ]
    missing = [item for item in required_train if item not in train_source]
    missing += [item for item in required_inference if item not in inference_source]
    if missing:
        raise ValueError(f"official MeRA semantics were not preserved: {missing}")


def patch_checkout(checkout: Path, manifest_path: Path) -> dict:
    train_path = checkout / "train.py"
    inference_path = checkout / "inference.py"
    train_path.write_text(patch_train(train_path.read_text(encoding="utf-8")), encoding="utf-8")
    inference_path.write_text(
        patch_inference(inference_path.read_text(encoding="utf-8")), encoding="utf-8"
    )
    train_source = train_path.read_text(encoding="utf-8")
    inference_source = inference_path.read_text(encoding="utf-8")
    assert_official_semantics(train_source, inference_source)
    for path in (train_path, inference_path):
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    changed = subprocess.run(
        ["git", "-C", str(checkout), "diff", "--name-only"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if changed != ["inference.py", "train.py"]:
        raise ValueError(f"unexpected official source changes: {changed}")
    diff = subprocess.run(
        ["git", "-C", str(checkout), "diff", "--", "train.py", "inference.py"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    manifest = {
        "schema_version": 1,
        "classification": "official MeRA with model-architecture compatibility only",
        "changed_files": changed,
        "allowed_differences": [
            "dynamic tokenizer pad token without overriding native BOS/EOS",
            "pre-addition vocabulary size from len(tokenizer)",
            "public input/output embedding APIs",
            "device-aware embedding rows for single-process model parallelism",
            "current k-bit loading and PEFT APIs",
            "current Trainer API keyword and compute_loss signature",
            "group_by_length mapped to current train_sampling_strategy API",
            "disabled reporting mapped from None to the current empty-list API",
            "relation-loss scalars moved to the main-loss device for model parallelism",
            "dynamic logits vocabulary dimension",
            "trust_remote_code model loading",
            "inference adapter device_map auto",
            "category token IDs recovered from the public added-vocabulary API",
            "official inference .indice typo repair",
        ],
        "preserved_semantics": [
            "official quoted category tokens and semantic initialization",
            "official Alpaca training and token-only inference prompts",
            "official full shifted cross entropy including EOS",
            "official raw-dot structure and collaborative relation losses",
            "official relation sampling and cooccurrence matrix",
            "official full embed_tokens and lm_head modules_to_save",
            "official AdamW, linear scheduler, warmup 20, and early stopping",
        ],
        "diff": diff,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = patch_checkout(args.checkout_dir.resolve(), args.manifest.resolve())
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()