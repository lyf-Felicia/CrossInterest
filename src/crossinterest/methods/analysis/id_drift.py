"""Offline weight-space diagnostics, not evidence of semantic damage."""

from __future__ import annotations

from crossinterest.paths import source_workspace as _source_workspace

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import threading

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch
import torch.nn.functional as functional
from urllib3.util.retry import Retry


HERE = _source_workspace('eval_interest_id_minionerec')
REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
MODEL_URL = f"https://huggingface.co/Qwen/Qwen3.5-9B/resolve/{REVISION}"
MODEL_CACHE = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots" / REVISION
STAGES = HERE / "bidirectional/outputs/results_envfix/qwen-cir-3ep/formal"
OUTPUT = HERE / "cir_factorization/outputs/interest_id_drift"
MAPPING = HERE / 'cir_factorization/package/crossinterest.training/data/interest_id_semantic_mapping.json'
CONNECTIONS = threading.local()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def row_metrics(current: torch.Tensor, reference: torch.Tensor) -> dict[str, torch.Tensor]:
    current, reference = current.double(), reference.double()
    if current.shape != reference.shape or current.ndim != 2:
        raise ValueError("row matrices must have identical shapes")
    if not torch.isfinite(current).all() or not torch.isfinite(reference).all():
        raise ValueError("nonfinite weights")
    reference_norm = reference.norm(dim=1)
    if (reference_norm == 0).any() or (current.norm(dim=1) == 0).any():
        raise ValueError("zero weight row")
    return {
        "relative_l2": (current - reference).norm(dim=1) / reference_norm,
        "cosine": functional.cosine_similarity(current, reference, dim=1),
        "norm_ratio": current.norm(dim=1) / reference_norm,
    }


def describe(values: torch.Tensor) -> dict[str, float]:
    array = values.detach().double().numpy()
    return {
        "min": float(array.min()), "mean": float(array.mean()),
        "median": float(np.median(array)), "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def centered(matrix: torch.Tensor) -> torch.Tensor:
    return matrix.double() - matrix.double().mean(dim=0, keepdim=True)


def common_shift_energy(delta: torch.Tensor) -> float:
    delta = delta.double()
    total = delta.square().sum()
    return float(delta.shape[0] * delta.mean(dim=0).square().sum() / total) if total else 0.0


def output_delta(factor_left: torch.Tensor, factor_right: torch.Tensor, scale: float) -> torch.Tensor:
    return (factor_left.double() @ factor_right.double()) * scale


def byte_range(url: str, start: int, end: int) -> tuple[bytes, dict]:
    if not hasattr(CONNECTIONS, "session"):
        CONNECTIONS.session = requests.Session()
        retries = Retry(total=4, connect=4, read=4, status=4, backoff_factor=0,
                        status_forcelist=(429, 500, 502, 503, 504), respect_retry_after_header=False)
        CONNECTIONS.session.mount("https://", HTTPAdapter(max_retries=retries))
    try:
        response = CONNECTIONS.session.get(
            url, params={"interest_drift_range": f"{start}-{end}"},
            headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
            stream=True, timeout=(30, 120))
    except requests.RequestException as error:
        raise RuntimeError(f"model range request failed after finite retries: {type(error).__name__}") from None
    with response:
        response.raise_for_status()
        expected = f"bytes {start}-{end}/"
        if response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(expected):
            raise ValueError("server did not honor exact range; refusing full-shard download")
        data = response.raw.read(end - start + 2, decode_content=False)
        if len(data) != end - start + 1:
            raise ValueError("partial model byte range")
        return data, {"range": response.headers["Content-Range"],
                      "etag": response.headers.get("ETag"), "bytes": len(data),
                      "sha256": hashlib.sha256(data).hexdigest()}


def load_base_rows(token_ids: list[int], output: Path) -> tuple[dict[str, torch.Tensor], dict]:
    rows_path, receipt_path = output / "base_rows.safetensors", output / "BASE_ROWS_RECEIPT.json"
    if rows_path.exists() and receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if (receipt["revision"] != REVISION or receipt["token_ids"] != token_ids
                or receipt["rows_sha256"] != sha256(rows_path)):
            raise ValueError("cached base rows identity mismatch")
        return load_file(rows_path), receipt
    response = requests.get(f"{MODEL_URL}/model.safetensors.index.json", timeout=90)
    response.raise_for_status()
    index = response.json()["weight_map"]
    names = {"input": "model.language_model.embed_tokens.weight", "output": "lm_head.weight"}
    shards = {index[name] for name in names.values()}
    if len(shards) != 1:
        raise ValueError("expected pinned input/output rows in one shard")
    url = f"{MODEL_URL}/{shards.pop()}"
    size_bytes, size_receipt = byte_range(url, 0, 7)
    header_size = int.from_bytes(size_bytes, "little")
    if not 0 < header_size < 10_000_000:
        raise ValueError("unexpected safetensors header size")
    header_bytes, header_receipt = byte_range(url, 8, 7 + header_size)
    header = json.loads(header_bytes)
    groups = []
    for token_id in token_ids:
        if groups and token_id - groups[-1][-1] <= 64:
            groups[-1].append(token_id)
        else:
            groups.append([token_id])
    tasks = []
    for kind, name in names.items():
        metadata = header[name]
        if metadata["dtype"] != "BF16" or metadata["shape"] != [248320, 4096]:
            raise ValueError("unexpected base vocabulary tensor layout")
        offset = 8 + header_size + metadata["data_offsets"][0]
        for group in groups:
            tasks.append((kind, group, offset + group[0] * 8192, offset + (group[-1] + 1) * 8192 - 1))

    def fetch(task):
        kind, group, start, end = task
        data, record = byte_range(url, start, end)
        matrix = torch.frombuffer(bytearray(data), dtype=torch.bfloat16).reshape(-1, 4096)
        selected = matrix[[token_id - group[0] for token_id in group]].clone()
        return kind, group, selected, record

    matrices = {kind: torch.empty(len(token_ids), 4096, dtype=torch.bfloat16) for kind in names}
    positions = {token_id: position for position, token_id in enumerate(token_ids)}
    records = [size_receipt, header_receipt]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for number, (kind, group, selected, record) in enumerate(pool.map(fetch, tasks), 1):
            matrices[kind][[positions[token_id] for token_id in group]] = selected
            records.append(record)
            if number % 40 == 0 or number == len(tasks):
                print(f"base row ranges {number}/{len(tasks)}", flush=True)
    if len({record["etag"] for record in records}) != 1 or not records[0]["etag"]:
        raise ValueError("base shard changed between range requests")
    save_file(matrices, rows_path)
    receipt = {"revision": REVISION, "url": url, "token_ids": token_ids,
               "index_sha256": hashlib.sha256(response.content).hexdigest(),
               "tensor_metadata": {kind: header[name] for kind, name in names.items()},
               "ranges": records, "downloaded_bytes": sum(record["bytes"] for record in records),
               "rows_sha256": sha256(rows_path), "entire_shard_sha256_verified": False}
    save_json(receipt_path, receipt)
    return matrices, receipt


def alignment(current: torch.Tensor, reference: torch.Tensor) -> dict:
    from scipy.stats import spearmanr

    current = functional.normalize(current.double(), dim=1)
    reference = functional.normalize(reference.double(), dim=1)
    nearest = (current @ reference.T).argmax(dim=1)
    triangle = torch.triu_indices(len(current), len(current), offset=1)
    before = (reference @ reference.T)[triangle[0], triangle[1]].numpy()
    after = (current @ current.T)[triangle[0], triangle[1]].numpy()
    return {"own_initial_anchor_top1": int((nearest == torch.arange(len(current))).sum()),
            "pairwise_cosine_spearman": float(spearmanr(before, after).statistic)}


def plot(records: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    figure, axes = pyplot.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    panels = [("output_vs_init", "relative_l2", "Output change from initialization (%)", 100),
              ("input_vs_init", "relative_l2", "Input change from initialization (%)", 100),
              ("output_vs_init_centered", "cosine", "Centered cosine to initialization", 1),
              ("output_vs_current_name", "relative_l2", "ID vs current category-name mean (%)", 100),
              ("output_vs_current_name_centered", "relative_l2", "Same name gap, common shifts removed (%)", 100),
              ("output_vs_init", "cosine", "Output cosine to initialization", 1)]
    colors = {"initialization": "#777777", "auxiliary": "#167a78", "novel": "#c54c38"}
    labels = {"initialization": "Initialization", "auxiliary": "After CIR (3 epochs)", "novel": "After CIR + N (1 epoch)"}
    for axis, (metric, statistic, title, multiplier) in zip(axes.flat, panels):
        for stage, measurements in records.items():
            values = np.sort(measurements[metric][statistic].numpy()) * multiplier
            axis.step(values, np.arange(1, len(values) + 1) / len(values), where="post",
                      label=labels[stage], color=colors[stage], linewidth=2, alpha=0.85)
        axis.set(xlabel=title, ylabel="Cumulative fraction of 112 categories", ylim=(0, 1.03))
        axis.grid(alpha=0.2)
    axes[0, 0].legend(loc="lower right", fontsize=9)
    figure.suptitle("Interest ID weight drift: full 112-category census\nWeight changes are not evidence of semantic damage", fontsize=14)
    figure.savefig(output / "interest_id_drift.png", dpi=170)
    pyplot.close(figure)


def analyze(output: Path) -> dict:
    from transformers import AutoTokenizer

    torch.set_num_threads(4)
    output.mkdir(parents=True, exist_ok=True)
    mapping = json.loads(MAPPING.read_text())
    assert sha256(MAPPING) == "5d9d9e1863b54f458ea175eac90a1c7a537240e706d05d210188778d238be5da"
    setup = json.loads((STAGES / "auxiliary/semantic_token_setup.json").read_text())
    tokens = mapping["special_tokens"]
    assert len(tokens) == len(set(tokens)) == 112 and not setup["tied_word_embeddings"]
    identifiers = [setup["token_ids"][token] for token in tokens]
    source_ids = [setup["source_token_ids"][token] for token in tokens]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CACHE, local_files_only=True)
    for token, sources in zip(tokens, source_ids):
        assert tokenizer(mapping["token_to_text"][token], add_special_tokens=False)["input_ids"] == sources
    unique_ids = sorted({token_id for sources in source_ids for token_id in sources})
    base, receipt = load_base_rows(unique_ids, output)
    positions = {token_id: position for position, token_id in enumerate(unique_ids)}
    groups = [[positions[token_id] for token_id in sources] for sources in source_ids]
    initial = {kind: torch.stack([matrix[group].mean(dim=0) for group in groups]).double()
               for kind, matrix in base.items()}
    source_mean = torch.stack([base["output"][group].double().mean(dim=0) for group in groups])
    result = {"scope": "G0 completed CIR3 -> original N, 112 categories, weight geometry only",
              "revision": REVISION, "mapping_sha256": sha256(MAPPING),
              "base_rows_sha256": receipt["rows_sha256"], "downloaded_bytes": receipt["downloaded_bytes"],
              "reconstruction": "BF16 source mean initialization; saved sparse input rows replace rows; W_eff = W0 + (alpha/r) B A in FP64",
              "limitations": ["No semantic-damage or causal performance claim", "No full-model activation or prediction analysis",
                              "FP64 effective linear operator excludes runtime BF16 matmul rounding", "Pinned HTTPS range reads, not full base-shard checksum"],
              "stages": {}}
    records, csv_rows, arrays = {}, [], {}
    previous_output, auxiliary_sha = None, None
    for stage in ("initialization", "auxiliary", "novel"):
        current_input, current_output = initial["input"], initial["output"]
        current_name = source_mean
        metadata = {}
        if stage != "initialization":
            directory = STAGES / stage
            adapter = directory / "final_adapter/adapter_model.safetensors"
            config = json.loads((directory / "final_adapter/adapter_config.json").read_text())
            audit = json.loads((directory / "STAGE_AUDIT.json").read_text())
            metrics = json.loads((directory / "train_metrics.json").read_text())
            current_setup = json.loads((directory / "semantic_token_setup.json").read_text())
            assert audit["passed"] and sha256(adapter) == audit["adapter_sha256"]
            assert current_setup["token_ids"] == setup["token_ids"] and current_setup["source_token_ids"] == setup["source_token_ids"]
            assert config["trainable_token_indices"] == identifiers
            assert config["r"] == 16 and config["lora_alpha"] == 32 and not config["modules_to_save"]
            assert not any(config.get(key) for key in ("fan_in_fan_out", "use_rslora", "use_dora", "rank_pattern", "alpha_pattern", "lora_bias"))
            if stage == "novel":
                assert metrics["init_adapter_model_sha256"] == auxiliary_sha
            else:
                auxiliary_sha = audit["adapter_sha256"]
            with safe_open(adapter, framework="pt", device="cpu") as handle:
                sparse_keys = [key for key in handle.keys() if key.endswith("trainable_tokens_delta")]
                assert len(sparse_keys) == 1
                current_input = handle.get_tensor(sparse_keys[0]).double()
                factor_right = handle.get_tensor("base_model.model.lm_head.lora_A.weight")
                factor_left = handle.get_tensor("base_model.model.lm_head.lora_B.weight")
            assert current_input.shape == initial["input"].shape
            delta = output_delta(factor_left[identifiers], factor_right, 2)
            lexical_delta = output_delta(factor_left[unique_ids], factor_right, 2)
            current_output = initial["output"] + delta
            current_name = source_mean + torch.stack([lexical_delta[group].mean(dim=0) for group in groups])
            metadata = {"adapter_path": str(adapter.relative_to(HERE)), "adapter_sha256": audit["adapter_sha256"],
                        "setup_sha256": sha256(directory / "semantic_token_setup.json"),
                        "steps": audit["steps"], "rows": audit["rows"],
                        "unique_source_word_output_update": {key: describe(values) for key, values in
                            row_metrics(base["output"].double() + lexical_delta, base["output"]).items()}}
        measurements = {
            "input_vs_init": row_metrics(current_input, initial["input"]),
            "output_vs_init": row_metrics(current_output, initial["output"]),
            "output_vs_init_centered": row_metrics(centered(current_output), centered(initial["output"])),
            "output_vs_current_name": row_metrics(current_output, current_name),
            "output_vs_current_name_centered": row_metrics(centered(current_output), centered(current_name)),
        }
        if previous_output is not None:
            measurements["output_vs_previous"] = row_metrics(current_output, previous_output)
        summaries = {name: {key: describe(values) for key, values in measurement.items()}
                     for name, measurement in measurements.items()}
        metadata.update(summaries)
        metadata.update(
            input_rows_changed=int((current_input != initial["input"]).any(dim=1).sum()),
            input_coordinates_changed_fraction=float((current_input != initial["input"]).double().mean()),
            output_rows_changed=int((current_output != initial["output"]).any(dim=1).sum()),
            common_output_shift_energy=common_shift_energy(current_output - initial["output"]),
            raw_anchor_alignment=alignment(current_output, initial["output"]),
            centered_anchor_alignment=alignment(centered(current_output), centered(initial["output"])),
        )
        name_delta = current_name - source_mean
        id_delta = current_output - initial["output"]
        metadata["displacement_decomposition"] = {
            "id_relative_frobenius": float(id_delta.norm() / initial["output"].norm()),
            "name_relative_frobenius": float(name_delta.norm() / source_mean.norm()),
            "name_delta_common_energy_fraction": common_shift_energy(name_delta),
            "id_minus_name_delta_common_energy_fraction": common_shift_energy(id_delta - name_delta),
        }
        result["stages"][stage] = metadata
        records[stage] = measurements
        arrays[f"{stage}_input"] = current_input.numpy()
        arrays[f"{stage}_output"] = current_output.numpy()
        arrays[f"{stage}_current_name"] = current_name.numpy()
        for position, token in enumerate(tokens):
            row = {"stage": stage, "interest_id": token, "category_name": mapping["token_to_text"][token]}
            for name, measurement in measurements.items():
                for key, values in measurement.items():
                    row[f"{name}_{key}"] = float(values[position])
            csv_rows.append(row)
        previous_output = current_output
        print(json.dumps({"stage": stage, "input_changed": metadata["input_rows_changed"],
                          "output_relative_l2": summaries["output_vs_init"]["relative_l2"],
                          "output_cosine": summaries["output_vs_init"]["cosine"],
                          "centered_cosine": summaries["output_vs_init_centered"]["cosine"],
                          "current_name_gap": summaries["output_vs_current_name"]["relative_l2"],
                          "common_shift_energy": metadata["common_output_shift_energy"]}), flush=True)
    columns = list(dict.fromkeys(key for row in csv_rows for key in row))
    with (output / "per_category.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(csv_rows)
    np.savez_compressed(output / "reconstructed_rows.npz", **arrays)
    plot(records, output)
    result["analysis_script_sha256"] = sha256(Path(__file__))
    result["artifacts_sha256"] = {name: sha256(output / name) for name in
                                  ("per_category.csv", "reconstructed_rows.npz", "interest_id_drift.png")}
    save_json(output / "SUMMARY.json", result)
    return result


def spectral_summary(matrix: torch.Tensor) -> dict:
    matrix = matrix.double()
    if matrix.ndim != 2 or not torch.isfinite(matrix).all():
        raise ValueError("spectrum requires a finite matrix")
    singular_values = torch.linalg.svdvals(matrix)
    energy = singular_values.square()
    total = float(energy.sum())
    fractions = energy / total if total else energy
    return {
        "shape": list(matrix.shape), "frobenius_squared": total,
        "numerical_rank": int((singular_values > singular_values[0] * 1e-10).sum()),
        "participation_rank": float(1 / fractions.square().sum()) if total else 0.0,
        "energy_by_rank": {str(rank): float(fractions[:rank].sum()) for rank in (1, 4, 8, 16)},
    }


def screen_output_adaptation(output: Path) -> dict:
    torch.set_num_threads(4)
    summary_path = output / "SUMMARY.json"
    arrays_path = output / "reconstructed_rows.npz"
    destination = output / "OUTPUT_ADAPTATION_SCREEN.json"
    if destination.exists():
        raise FileExistsError("preserve the existing screen receipt")
    summary = json.loads(summary_path.read_text())
    if sha256(arrays_path) != summary["artifacts_sha256"][arrays_path.name]:
        raise ValueError("reconstructed row identity changed")
    mapping = json.loads(MAPPING.read_text())
    if sha256(MAPPING) != summary["mapping_sha256"]:
        raise ValueError("mapping identity changed")
    setup = json.loads((STAGES / "auxiliary/semantic_token_setup.json").read_text())
    identifiers = [setup["token_ids"][token] for token in mapping["special_tokens"]]
    with np.load(arrays_path, allow_pickle=False) as arrays:
        initial = torch.from_numpy(arrays["initialization_output"].copy())
        matrices = {stage: torch.from_numpy(arrays[f"{stage}_output"].copy())
                    for stage in ("auxiliary", "novel")}
    result = {
        "scope": "G0 output adaptation geometry; no training, predictions or Dev selection",
        "source_summary_sha256": sha256(summary_path),
        "source_arrays_sha256": sha256(arrays_path),
        "analysis_script_sha256": sha256(Path(__file__)),
        "initial_centered_head": spectral_summary(centered(initial)), "stages": {},
        "limitations": ["Only the LoRA correction is rank-limited, not the full initialized head",
                        "Spectral utilization does not establish an optimization bottleneck or expected uplift",
                        "No hidden states or loss gradients are measured"],
    }
    for stage, matrix in matrices.items():
        directory = STAGES / stage
        adapter = directory / "final_adapter/adapter_model.safetensors"
        config = json.loads((directory / "final_adapter/adapter_config.json").read_text())
        if sha256(adapter) != summary["stages"][stage]["adapter_sha256"]:
            raise ValueError("source adapter identity changed")
        with safe_open(adapter, framework="pt", device="cpu") as handle:
            factor_right = handle.get_tensor("base_model.model.lm_head.lora_A.weight")
            factor_left = handle.get_tensor("base_model.model.lm_head.lora_B.weight")[identifiers]
        delta = output_delta(factor_left, factor_right, config["lora_alpha"] / config["r"])
        torch.testing.assert_close(matrix - initial, delta, rtol=1e-12, atol=1e-14)
        result["stages"][stage] = {
            "adapter_sha256": summary["stages"][stage]["adapter_sha256"],
            "common_shift_energy": common_shift_energy(delta),
            "centered_correction": spectral_summary(centered(delta)),
            "centered_effective_head": spectral_summary(centered(matrix)),
        }
    result["n_handoff_centered_change"] = spectral_summary(centered(matrices["novel"] - matrices["auxiliary"]))
    save_json(destination, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def self_test() -> None:
    from peft.tuners.lora.layer import Linear

    generator = torch.Generator().manual_seed(20260920)
    reference = torch.randn(112, 64, generator=generator, dtype=torch.float64)
    unchanged = row_metrics(reference, reference)
    torch.testing.assert_close(unchanged["relative_l2"], torch.zeros(112, dtype=torch.float64))
    torch.testing.assert_close(unchanged["cosine"], torch.ones(112, dtype=torch.float64))
    shift = torch.randn(1, 64, generator=generator, dtype=torch.float64)
    moved = reference + shift
    assert row_metrics(moved, reference)["relative_l2"].min() > 0
    torch.testing.assert_close(centered(moved), centered(reference))
    assert abs(common_shift_energy(moved - reference) - 1) < 1e-12
    factor_left = torch.randn(112, 16, generator=generator, dtype=torch.float64)
    factor_right = torch.randn(16, 64, generator=generator, dtype=torch.float64)
    hidden = torch.randn(5, 64, generator=generator, dtype=torch.float64)
    delta = output_delta(factor_left, factor_right, 2)
    torch.testing.assert_close(
        functional.linear(hidden, reference + delta),
        functional.linear(hidden, reference) + functional.linear(functional.linear(hidden, factor_right), factor_left) * 2,
    )
    base_layer = torch.nn.Linear(64, 112, bias=False, dtype=torch.float64)
    layer = Linear(base_layer, "default", r=16, lora_alpha=32, lora_dropout=0)
    with torch.no_grad():
        base_layer.weight.copy_(reference)
        layer.lora_A["default"].weight.copy_(factor_right)
        layer.lora_B["default"].weight.copy_(factor_left)
    torch.testing.assert_close(layer(hidden), functional.linear(hidden, reference + delta))
    torch.testing.assert_close(
        functional.linear(hidden, moved).log_softmax(dim=-1),
        functional.linear(hidden, reference).log_softmax(dim=-1),
    )
    assert spectral_summary(torch.zeros(4, 8))["numerical_rank"] == 0
    identity_spectrum = spectral_summary(torch.eye(4, dtype=torch.float64))
    assert identity_spectrum["numerical_rank"] == 4
    assert identity_spectrum["participation_rank"] == 4
    assert identity_spectrum["energy_by_rank"]["1"] == 0.25
    assert spectral_summary(centered(torch.ones(4, 8)))["frobenius_squared"] == 0
    assert spectral_summary(centered(delta))["numerical_rank"] == 16
    assert spectral_summary(centered(reference + delta))["numerical_rank"] == 64
    print("PASS: identity, common-shift removal, conditional-softmax invariance and actual PEFT LoRA output")
    print("PASS: zero/flat/low-rank spectra; full head is not limited to correction rank")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--screen-output", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.screen_output:
        screen_output_adaptation(args.output.resolve())
    else:
        analyze(args.output.resolve())


if __name__ == "__main__":
    main()