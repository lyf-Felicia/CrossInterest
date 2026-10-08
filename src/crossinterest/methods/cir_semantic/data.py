"""Cache frozen hidden-history embeddings and matched within-class controls."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq

from crossinterest.evaluation.common import write_json_atomic
from crossinterest.methods.alignment.data import HERE, sha256_file
from crossinterest.methods.mechanism.data import SEED


ENCODER = "sentence-transformers/all-mpnet-base-v2"
REVISION = "e8c3b32edf5434bc2275fc9bab85f82640a19130"
DATA = HERE / "cir/data/external_balanced"
ROOT = HERE / "cir/data/semantic"


def text_hash(values: list[str]) -> str:
    return hashlib.sha256(json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def residual_targets(embeddings: np.ndarray, categories: list[str], ids: list[str]) -> tuple[dict, dict]:
    embeddings = np.asarray(embeddings, dtype=np.float64)
    if embeddings.ndim != 2 or len(embeddings) != len(categories) or len(ids) != len(categories):
        raise ValueError("unaligned semantic records")
    if not np.isfinite(embeddings).all() or len(set(ids)) != len(ids):
        raise ValueError("invalid semantic embeddings or repeated IDs")
    groups = defaultdict(list)
    for index, category in enumerate(categories):
        groups[category].append(index)
    centered = embeddings.copy()
    means = []
    permutation = np.arange(len(ids), dtype=np.int64)
    for category in sorted(groups):
        indices = groups[category]
        if len(indices) < 2:
            raise ValueError("semantic control requires at least two rows per class")
        mean = embeddings[indices].mean(axis=0)
        means.append(mean)
        centered[indices] -= mean
        order = sorted(indices, key=lambda index: text_hash([str(SEED), category, ids[index]]))
        shift = 1 + int(text_hash([str(SEED), category, "shift"]), 16) % (len(order) - 1)
        permutation[order] = np.roll(order, shift)
    scale = float(np.sqrt(np.mean(np.sum(centered ** 2, axis=1))))
    if not np.isfinite(scale) or scale < 1e-8:
        raise ValueError("semantic residuals have no usable variance")
    targets = (centered / scale).astype(np.float32)
    if np.any(permutation == np.arange(len(ids))) or set(permutation) != set(range(len(ids))):
        raise ValueError("semantic permutation is not a derangement")
    if any(categories[index] != categories[source] for index, source in enumerate(permutation)):
        raise ValueError("semantic permutation crossed a class boundary")
    identical = np.linalg.norm(targets - targets[permutation], axis=1) < 1e-7
    arrays = {"targets": targets, "class_means": np.asarray(means, dtype=np.float32),
              "permutation": permutation, "history_embeddings": embeddings.astype(np.float32)}
    audit = {"rows": len(ids), "dimension": targets.shape[1], "classes": len(groups),
             "class_order": sorted(groups), "row_order_sha256": text_hash(ids), "global_residual_rms_norm": scale,
             "mean_squared_target_norm": float(np.mean(np.sum(targets ** 2, axis=1))),
             "max_class_mean_abs": float(max(np.abs(targets[indices].mean(axis=0)).max() for indices in groups.values())),
             "permutation_fixed_points": 0, "permutation_cross_class": 0,
             "shuffled_identical_vector_pairs": int(identical.sum()), "seed": SEED,
             "normalization": "class_center_then_single_global_rms_norm", "teacher_stop_gradient": True}
    return arrays, audit


def build(output: Path = ROOT, batch_size: int = 32, device: str = "cpu") -> dict:
    import torch
    import transformers
    from safetensors import safe_open
    from safetensors.numpy import load_file, save_file
    from transformers import AutoModel, AutoTokenizer

    started = time.monotonic()
    data_path = DATA / "train.parquet"
    source_audit = json.loads((DATA / "INDEPENDENT_AUDIT.json").read_text())
    source_sha = sha256_file(data_path)
    if not source_audit["passed"] or source_sha != source_audit["train_sha256"]:
        raise ValueError("semantic targets require independently audited balanced CIR")
    identity = {"encoder": ENCODER, "encoder_revision": REVISION, "data_sha256": source_sha,
                "builder_sha256": sha256_file(Path(__file__)), "torch_version": torch.__version__,
                "transformers_version": transformers.__version__, "device": device, "batch_size": batch_size}
    manifest_path = output / "MANIFEST.json"
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text())
        if prior["identity"] != identity or sha256_file(output / "targets.safetensors") != prior["targets_sha256"]:
            raise ValueError("existing semantic cache identity differs")
        return prior
    rows = pq.read_table(data_path, columns=["id", "hidden_titles", "target_interest"]).to_pylist()
    titles = sorted({title for row in rows for title in row["hidden_titles"]})
    if not titles or any(not row["hidden_titles"] for row in rows):
        raise ValueError("semantic targets require nonempty removed historical titles")
    tokenizer = AutoTokenizer.from_pretrained(ENCODER, revision=REVISION)
    lengths = [len(tokens) for start in range(0, len(titles), 4096)
               for tokens in tokenizer(titles[start:start + 4096], truncation=False)["input_ids"]]
    if max(lengths) > 384:
        raise ValueError(f"hidden title exceeds encoder window: {max(lengths)}; no silent truncation")
    model = AutoModel.from_pretrained(ENCODER, revision=REVISION).to(device).eval()
    model.requires_grad_(False)
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "title_cache"
    cache.mkdir(exist_ok=True)
    embeddings = np.empty((len(titles), model.config.hidden_size), dtype=np.float32)
    chunk_identity = json.dumps(identity, sort_keys=True)
    for start in range(0, len(titles), 4096):
        end = min(start + 4096, len(titles))
        path = cache / f"{start:08d}.safetensors"
        metadata = {"identity": chunk_identity, "titles_sha256": text_hash(titles[start:end])}
        if path.exists():
            with safe_open(path, framework="np") as handle:
                if handle.metadata() != metadata:
                    raise ValueError("partial title cache differs")
            embeddings[start:end] = load_file(path)["embeddings"]
            continue
        for offset in range(start, end, batch_size):
            stop = min(offset + batch_size, end)
            inputs = tokenizer(titles[offset:stop], padding=True, truncation=False, return_tensors="pt").to(device)
            with torch.inference_mode():
                hidden = model(**inputs).last_hidden_state.float()
                mask = inputs["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
                pooled = torch.nn.functional.normalize(pooled, dim=-1)
            embeddings[offset:stop] = pooled.cpu().numpy()
        temporary = path.with_suffix(".temporary")
        save_file({"embeddings": embeddings[start:end].copy()}, temporary, metadata=metadata)
        temporary.replace(path)
        print(json.dumps({"event": "semantic_encode", "titles": end, "total_titles": len(titles)}), flush=True)
    title_index = {title: index for index, title in enumerate(titles)}
    history = np.stack([embeddings[[title_index[title] for title in row["hidden_titles"]]].mean(axis=0) for row in rows])
    arrays, audit = residual_targets(history, [row["target_interest"] for row in rows], [row["id"] for row in rows])
    target_path = output / "targets.safetensors"
    temporary = target_path.with_suffix(".temporary")
    save_file(arrays, temporary, metadata={"data_sha256": source_sha, "row_order_sha256": audit["row_order_sha256"]})
    temporary.replace(target_path)
    result = {"passed": True, "identity": identity, "audit": audit, "unique_hidden_titles": len(titles),
              "max_title_tokens": max(lengths), "title_truncations": 0, "teacher_dtype": "float32",
              "history_pooling": "mean_of_unit_title_embeddings_with_event_multiplicity",
              "targets_sha256": sha256_file(target_path), "preparation_wall_seconds": time.monotonic() - started,
              "future_titles_used": False, "dev_used": False, "student_input_changed": False}
    write_json_atomic(manifest_path, result)
    print(json.dumps(result), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    build(args.output_dir, args.batch_size, args.device)


if __name__ == "__main__":
    main()