"""Use the frozen CIR/N objectives with independent interest encoding."""

import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import pyarrow.parquet as pq
import torch

from crossinterest.evaluation.common import write_json_atomic
from crossinterest.training import train as baseline
from crossinterest.methods.composition.model import CompositionController, encode_example
from crossinterest.methods.mechanism.train import positive_set_loss


def write_parameters_atomic(path, parameters):
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                            prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(parameters, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary.replace(path)


class CompositionDataset(baseline.AmazonSftDataset):
    def __init__(self, *args, stage, grouping, encoded_file=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage = stage
        self.grouping = grouping
        self.encoded = pq.read_table(encoded_file, memory_map=True) if encoded_file else None
        if self.encoded is not None and self.encoded["id"].to_pylist() != self.table["id"].to_pylist():
            raise ValueError("cached token rows must exactly match source order")
        if stage == "auxiliary":
            positives = pq.read_table(args[0], columns=["positive_interests"])
            self.table = self.table.append_column("positive_interests", positives["positive_interests"])
        self.category_order = [self.mapping["id_to_category"][token] for token in self.mapping["special_tokens"]]

    def __getitem__(self, index):
        if self.encoded is not None:
            encoded = self.encoded.slice(index, 1).to_pylist()[0]
            encoded.pop("id")
            return encoded
        row = self.table.slice(index, 1).to_pylist()[0]
        encoded = encode_example(row, self.tokenizer, self.mapping, self.stage, self.grouping,
                                 max_length=self.max_length)
        if self.stage == "auxiliary":
            encoded["positive_mask"] = [category in row["positive_interests"] for category in self.category_order]
        return encoded


class CompositionCollator(baseline.AssistantOnlyCollator):
    def __call__(self, features):
        batch = super().__call__(features)
        maximum = batch["input_ids"].shape[1]
        batch["memory_mask"] = torch.tensor([
            feature["memory_mask"] + [False] * (maximum - len(feature["memory_mask"]))
            for feature in features], dtype=torch.bool)
        groups = [group for feature in features for group in feature["groups"]]
        longest = max(map(len, groups))
        batch["group_input_ids"] = torch.tensor([
            group + [self.pad_token_id] * (longest - len(group)) for group in groups], dtype=torch.long)
        batch["group_attention_mask"] = torch.tensor([
            [1] * len(group) + [0] * (longest - len(group)) for group in groups], dtype=torch.long)
        if "positive_mask" in features[0]:
            batch["positive_mask"] = torch.tensor([feature["positive_mask"] for feature in features], dtype=torch.bool)
        return batch


class CompositionTrainer(baseline.AmazonSftTrainer):
    def __init__(self, *args, stage, grouping, init_adapter=None, **kwargs):
        self.controller = CompositionController(kwargs["model"], grouping, directory=init_adapter)
        self.controller.attach_training()
        kwargs["data_collator"] = CompositionCollator(kwargs["processing_class"].pad_token_id)
        super().__init__(*args, **kwargs)
        self.stage = stage
        if stage == "auxiliary":
            self.model_accepts_loss_kwargs = False
        if self.is_world_process_zero():
            parameters_path = Path(self.args.output_dir) / "trainable_parameters.json"
            if parameters_path.exists():
                parameters = json.loads(parameters_path.read_text())
                trainable = [(name, value) for name, value in self.model.named_parameters() if value.requires_grad]
                parameters.update(count=sum(value.numel() for _, value in trainable),
                                  tensor_count=len(trainable), names=[name for name, _ in trainable])
                write_parameters_atomic(parameters_path, parameters)
            write_json_atomic(Path(self.args.output_dir) / "COMPOSITION_RUNTIME.json", {
                **self.controller.metadata, "stage": stage,
                "model_accepts_loss_kwargs": self.model_accepts_loss_kwargs,
                "bridge_trainable_parameters": sum(parameter.numel() for parameter in self.controller.bridge.parameters()),
                "total_trainable_parameters": sum(parameter.numel() for parameter in self.model.parameters()
                                                  if parameter.requires_grad)})

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.stage == "novel":
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        inputs = dict(inputs)
        positives = inputs.pop("positive_mask")
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        positions = (labels != -100).long().argmax(dim=-1) - 1
        target_logits = outputs.logits[torch.arange(len(labels), device=labels.device), positions]
        loss = positive_set_loss(target_logits[:, self.label_token_ids], positives)
        return (loss, outputs) if return_outputs else loss

    def _save(self, output_dir=None, state_dict=None):
        super()._save(output_dir, state_dict)
        self.controller.save(Path(output_dir or self.args.output_dir))

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        super()._load_from_checkpoint(resume_from_checkpoint, model)
        self.controller.load(Path(resume_from_checkpoint))