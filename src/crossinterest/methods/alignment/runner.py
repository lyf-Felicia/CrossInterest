#!/usr/bin/env python3
"""Gate formal submissions on both smoke audits, then collect audited results."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from crossinterest.methods.alignment.audit import audit
from crossinterest.methods.alignment.data import HERE, VIEWS


SMOKE_EXPERIMENT = "home112-minionerec-two-stage-smoke-20260910"
FORMAL_EXPERIMENT = "home112-minionerec-two-stage-formal-20260910"
EXPERIMENTS = {
    "interest": (SMOKE_EXPERIMENT, FORMAL_EXPERIMENT),
    "behavior_interest": (
        "home112-minionerec-bi-two-stage-smoke-20260911",
        "home112-minionerec-bi-two-stage-formal-20260911",
    ),
}
JOBS = {"qwen35": "qwen35_9b", "llama31": "llama31_8b"}


def placement(headroom: dict[str, int]) -> dict[str, str]:
    remaining = dict(headroom)
    result = {}
    for prefix in JOBS:
        target = max(remaining, key=lambda name: (remaining[name], name))
        result[prefix] = target
        remaining[target] -= 4
    return result


def render_formal_report(text: str, summaries: dict, formal_experiment: str = FORMAL_EXPERIMENT) -> str:
    if set(summaries) != set(JOBS.values()):
        raise ValueError("report requires both audited formal models")
    beginning = text.index("<table>")
    ending = text.index("</table>", beginning) + len("</table>")
    table = ET.fromstring(text[beginning:ending])
    rows = table.find("tbody").findall("tr")
    if len(rows) != 4:
        raise ValueError("unexpected report table structure")
    for row, model_key in zip((rows[1], rows[3]), JOBS.values(), strict=True):
        summary = summaries[model_key]
        if summary["smoke"] or not summary["passed"]:
            raise ValueError("smoke or failed results cannot populate the formal report")
        classification = summary["results"]["classification"]
        ranking = summary["results"]["ranking"]
        for metrics in (classification, ranking):
            if not metrics["complete"] or metrics["recorded_sample_count"] != 2000 or metrics["class_count"] != 112:
                raise ValueError("report requires complete Dev-2K results")
        values = [f"{ranking[key]:.4f}" for key in (
            "macro_recall_at_10", "macro_recall_at_20", "macro_ndcg_at_10", "macro_ndcg_at_20",
        )]
        for cell, value in zip(list(row)[1:], values, strict=True):
            cell.text = value
    text = text[:beginning] + ET.tostring(table, encoding="unicode", method="html") + text[ending:]
    pending = "新方法当前无正式结果，不做优劣判断。"
    text = text.replace("新方案当前无正式结果，不做优劣判断。", pending)
    completed = "两模型正式结果已完成合同审计，并从原始预测重算一致。"
    if text.count(pending) + text.count(completed) != 1:
        raise ValueError("report result marker changed; review before replacing")
    text = text.replace(pending, completed)
    status_start = text.index("## 7.")
    status_end = text.index("## 8.", status_start)
    section = [
        "## 7. 执行状态与限制", "",
        "两个模型均完成非推荐预热、新兴趣多任务 SFT、完整 Dev-2K 分类和排序。正式结果均通过权重交接、数据与模型合同和原始预测重算检查。", "",
        "| 模型 | 预热 loss | 推荐 loss | 分类格式率 | 排序合法 beam 率 | 排序严格格式率 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model_key, summary in summaries.items():
        classification = summary["results"]["classification"]
        ranking = summary["results"]["ranking"]
        section.append(
            f"| {model_key} | {summary['stage_audits']['warmup']['train_loss']:.4f} | "
            f"{summary['stage_audits']['recommendation']['train_loss']:.4f} | "
            f"{classification['strict_format_rate']:.4f} | {ranking['valid_beam_rate']:.4f} | "
            f"{ranking['strict_beam_format_rate']:.4f} |"
        )
    section.extend([
        "", "loss 为 Trainer 写出的 train_loss 字段。断点恢复后的累计口径可能不覆盖恢复前的全部损失，不把该字段作为跨模型的完整 epoch 平均损失比较。",
        "", f"正式实验：`{formal_experiment}`。",
        "机器可读结果见 [outputs/formal/AUDITED_RESULTS.json](outputs/formal/AUDITED_RESULTS.json)，逐模型原始预测和合同保存在同目录下的实验子目录。",
        "", "本轮仅两个模型、一个 seed 和 Home 单域，不推断跨域或种子稳定性。相对旧基线同时增加预热和类别名预测监督，训练预算从 3,500 增至 7,070 step，不能把效果差异单独归因于任务顺序。无效自由生成保持无效，格式率差异必须纳入解释。", "",
    ])
    return text[:status_start] + "\n".join(section) + "\n" + text[status_end:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amlt", default="amlt")
    parser.add_argument("--view", choices=VIEWS, default="interest")
    parser.add_argument("--smoke-job", action="append", default=[], metavar="MODEL_PREFIX=JOB_NAME")
    recovery = parser.add_mutually_exclusive_group()
    recovery.add_argument("--resume-unsubmitted", action="store_true")
    recovery.add_argument("--resume-smoke", action="store_true")
    recovery.add_argument("--monitor-existing", action="store_true")
    args = parser.parse_args()
    root = HERE if args.view == "interest" else HERE / args.view
    smoke_experiment, formal_experiment = EXPERIMENTS[args.view]
    output = root / "outputs"
    output.mkdir(exist_ok=True)
    state_path = output / "EXECUTION_STATE.json"
    if state_path.exists() and not (args.resume_unsubmitted or args.resume_smoke or args.monitor_existing):
        raise ValueError("execution state already exists; inspect existing jobs before restarting")
    state = {"smoke_experiment": smoke_experiment, "formal_experiment": formal_experiment,
             "view": args.view, "formal_submissions": [], "status": "starting"}
    if args.monitor_existing:
        state = json.loads(state_path.read_text())
        if state["formal_experiment"] != formal_experiment or state.get("view", "interest") != args.view:
            raise ValueError("execution state refers to a different formal experiment")
    if args.resume_unsubmitted:
        state = json.loads(state_path.read_text())
        if state["status"] != "stopped_with_error" or state["formal_submissions"]:
            raise ValueError("resume is only allowed after an unsubmitted failure")
        state.setdefault("previous_errors", []).append(state.pop("error"))
    if args.resume_smoke:
        state = json.loads(state_path.read_text())
        if (state.get("view", "interest") != args.view
                or state.get("smoke_experiment") != smoke_experiment
                or state.get("formal_experiment") != formal_experiment
                or state.get("status") != "stopped_with_error"
                or state.get("formal_submissions") != []
                or "placement" in state or "monitor_pid" in state
                or state.get("failed_at", "waiting_for_smoke") != "waiting_for_smoke"
                or state.get("error") != "amlt watch: exit 12; see execution.log"):
            raise ValueError("smoke recovery requires a failed smoke watch before any formal submission")
        state.setdefault("previous_errors", []).append(state.pop("error"))
    smoke_jobs = state.get("smoke_jobs", {prefix: f"{prefix}-two-stage-smoke" for prefix in JOBS})
    for specification in args.smoke_job:
        prefix, separator, name = specification.partition("=")
        if (not separator or prefix not in JOBS or not name.startswith(f"{prefix}-two-stage-smoke")
                or not all(character.isalnum() or character in "-_" for character in name)):
            raise ValueError("smoke job must be MODEL_PREFIX=JOB_NAME with the matching model prefix")
        smoke_jobs[prefix] = name
    state["smoke_jobs"] = smoke_jobs

    def save_state(status: str) -> None:
        state.update(status=status, updated_at=datetime.now(timezone.utc).isoformat())
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(state_path)
        print(json.dumps({"status": status, "updated_at": state["updated_at"]}), flush=True)

    def command(*arguments: str) -> str:
        print("amlt " + " ".join(arguments), flush=True)
        if arguments[0] == "watch":
            with (output / "execution.log").open("a", buffering=1) as handle:
                handle.write("amlt " + " ".join(arguments) + "\n")
                with subprocess.Popen(
                    [args.amlt, *arguments], cwd=root,
                    env={**os.environ, "PAGER": "cat", "AMLT_SKIP_GROUP_POLICIES": "0"},
                    text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
                ) as process:
                    for line in process.stdout:
                        handle.write(line)
                        print(line, end="", flush=True)
                    returncode = process.wait()
                if returncode:
                    raise RuntimeError(f"amlt watch: exit {returncode}; see execution.log")
            return ""
        result = subprocess.run([args.amlt, *arguments], cwd=root,
                                env={**os.environ, "PAGER": "cat", "AMLT_SKIP_GROUP_POLICIES": "0"},
                                text=True, capture_output=True)
        with (output / "execution.log").open("a") as handle:
            handle.write("amlt " + " ".join(arguments) + "\n" + result.stdout + result.stderr + "\n")
        if result.returncode:
            raise RuntimeError(f"amlt {' '.join(arguments[:3])}: exit {result.returncode}; see execution.log")
        return result.stdout

    def collect(experiment: str, smoke: bool) -> dict:
        destination = output / ("smoke" if smoke else "formal")
        names = smoke_jobs if smoke else {prefix: f"{prefix}-two-stage-formal" for prefix in JOBS}
        for pattern in ("*.json", "*.jsonl"):
            command("results", "download", experiment, *(f":{name}" for name in names.values()),
                    "-I", pattern, "-o", str(destination))
        summaries = {}
        for prefix, model_key in JOBS.items():
            directory = destination / experiment / names[prefix]
            summary = audit(directory, model_key, smoke, args.view)
            (directory / "RESULT_AUDIT.json").write_text(json.dumps(summary, indent=2) + "\n")
            summaries[model_key] = summary
        (destination / "AUDITED_RESULTS.json").write_text(json.dumps(summaries, indent=2) + "\n")
        return summaries

    def finish_formal() -> None:
        statuses = json.loads(command("--json-tables", "status", formal_experiment))
        expected = {f":{prefix}-two-stage-formal" for prefix in JOBS}
        if len(statuses) != len(expected) or {row["job_name"] for row in statuses} != expected:
            raise ValueError("backend did not confirm exactly the two formal jobs")
        state["monitor_pid"] = os.getpid()
        save_state("formal_running_or_queued")
        command("watch", formal_experiment, "--json", "--no-notify")
        summaries = collect(formal_experiment, False)
        report = root / "EXPERIMENT_REPORT.md"
        rendered = render_formal_report(report.read_text(), summaries, formal_experiment)
        report.write_text(rendered, encoding="utf-8")
        state.pop("error", None)
        save_state("formal_results_audited")

    try:
        if args.monitor_existing:
            finish_formal()
            return
        if args.resume_unsubmitted:
            existing = json.loads(command("--json-tables", "list", "-e"))
            if any(row["_experiment_name"] == formal_experiment for row in existing):
                raise ValueError("formal jobs already exist; refusing duplicate submission")
        smoke_selectors = [f":{name}" for name in smoke_jobs.values()]
        save_state("waiting_for_smoke")
        command("watch", smoke_experiment, *smoke_selectors, "--json", "--no-notify")
        collect(smoke_experiment, True)
        save_state("smoke_audited")
        targets = json.loads(command("--json-tables", "target", "list", "sing"))
        if not {"Feeds", "Recall"} <= {item["target_name"] for item in targets}:
            raise ValueError("authorized Feeds/Recall targets are not discoverable")
        capacities = {}
        for target in ("Feeds", "Recall"):
            rows = json.loads(command("--json-tables", "target", "info", "sing", "-t", target, "-v"))
            matches = [row for row in rows if row["accelerator"]["type"] == "A100"
                       and row["accelerator"]["memory_gb"] == 80]
            if len(matches) != 1:
                raise ValueError(f"ambiguous A100 capacity for {target}")
            quota = matches[0]["quotas_(used/limit)"]["standard"]
            capacities[target] = quota["limit"] - quota["used"]
        state["quota_headroom_not_physical_capacity"] = capacities
        selected = placement(capacities)
        state["placement"] = selected
        save_state("submitting_formal")
        for target in sorted(set(selected.values())):
            names = [f":{prefix}-two-stage-formal" for prefix in JOBS if selected[prefix] == target]
            command("run", "amlt_formal.yaml", *names, formal_experiment, "-t", target,
                    "-y", "-d", "Home112 alignment-only warmup then novel-interest multitask SFT; both smoke audits passed")
            state["formal_submissions"].extend({"job": name[1:], "target": target} for name in names)
            save_state("submitting_formal")
        finish_formal()
    except Exception as error:
        state["failed_at"] = state["status"]
        state["error"] = str(error)
        save_state("stopped_with_error")
        raise


if __name__ == "__main__":
    main()