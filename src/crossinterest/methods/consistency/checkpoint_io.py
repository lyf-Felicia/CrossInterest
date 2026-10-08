"""Stage PyTorch checkpoint archives locally before sequential blob-mount writes."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory, mkdtemp
import time

import torch


ARCHIVE = re.compile(r"(?:optimizer\.pt|scheduler\.pt|scaler\.pt|rng_state(?:_\d+)?\.pth|training_args\.bin)")


def checkpoint_archive(path, root: Path) -> bool:
    if not isinstance(path, (str, os.PathLike)):
        return False
    destination = Path(path).absolute()
    try:
        relative = destination.relative_to(root.absolute())
    except ValueError:
        return False
    return bool(ARCHIVE.fullmatch(destination.name) and any(
        re.fullmatch(r"checkpoint-\d+", part) for part in relative.parts[:-1]))


def staged_save(original, value, destination, *args, root: Path, **kwargs):
    if not checkpoint_archive(destination, root):
        return original(value, destination, *args, **kwargs)
    destination = Path(destination)
    started = time.monotonic()
    with TemporaryDirectory(prefix="dual-checkpoint-", dir="/tmp") as temporary:
        local = Path(temporary) / destination.name
        original(value, local, *args, **kwargs)
        serialized = time.monotonic()
        size = local.stat().st_size
        with local.open("rb") as source, destination.open("wb") as target:
            shutil.copyfileobj(source, target, length=16 * 1024 * 1024)
        if destination.stat().st_size != size:
            raise OSError("checkpoint archive size differs after sequential copy")
    print(json.dumps({"event": "local_checkpoint_archive_saved", "rank": os.environ.get("RANK"),
                      "file": destination.name, "checkpoint": destination.parent.name, "bytes": size,
                      "serialize_seconds": serialized - started,
                      "copy_seconds": time.monotonic() - serialized}), flush=True)


def install(root: Path) -> None:
    original = torch.save
    if getattr(original, "_dual_staged_save", False):
        return

    def save(value, destination, *args, **kwargs):
        return staged_save(original, value, destination, *args, root=root, **kwargs)

    save._dual_staged_save = True
    torch.save = save
    print(json.dumps({"event": "dual_local_checkpoint_serialization_enabled",
                      "rank": os.environ.get("RANK"), "root": str(root)}), flush=True)


def validate_checkpoint(path: Path, world_size: int) -> dict:
    required = {"adapter_model.safetensors", "adapter_config.json", "optimizer.pt", "scheduler.pt",
                "trainer_state.json", "training_args.bin", *[f"rng_state_{rank}.pth" for rank in range(world_size)]}
    if any(not (path / name).is_file() or (path / name).stat().st_size == 0 for name in required):
        raise ValueError("checkpoint lacks complete model/optimizer/scheduler/all-rank RNG files")
    state = json.loads((path / "trainer_state.json").read_text())
    step = int(path.name.removeprefix("checkpoint-"))
    if state["global_step"] != step or step <= 0:
        raise ValueError("checkpoint step identity differs")
    optimizer = torch.load(path / "optimizer.pt", map_location="cpu", weights_only=False)
    steps = {int(value["step"]) for value in optimizer["state"].values() if "step" in value}
    if steps != {step}:
        raise ValueError("optimizer update count differs from trainer state")
    scheduler = torch.load(path / "scheduler.pt", map_location="cpu", weights_only=False)
    if scheduler["last_epoch"] != step:
        raise ValueError("scheduler update count differs")
    for rank in range(world_size):
        rng = torch.load(path / f"rng_state_{rank}.pth", map_location="cpu", weights_only=False)
        if not {"python", "numpy", "cpu", "cuda"} <= set(rng) or not isinstance(rng["cpu"], torch.Tensor):
            raise ValueError("checkpoint RNG state is incomplete")
    return {"passed": True, "step": step, "world_size": world_size,
            "optimizer_steps": sorted(steps), "scheduler_step": scheduler["last_epoch"]}


def install_resume(train) -> None:
    original = train.latest_checkpoint

    def latest(output_dir):
        output_dir = Path(output_dir)
        if output_dir.parts[-2:] != ("formal", "novel"):
            return original(output_dir)
        distributed = torch.distributed.is_initialized()
        rank = torch.distributed.get_rank() if distributed else 0
        world_size = torch.distributed.get_world_size() if distributed else int(os.environ.get("WORLD_SIZE", "8"))
        selection = [None]
        if rank == 0:
            try:
                candidates = sorted((path for path in output_dir.glob("checkpoint-*")
                                     if re.fullmatch(r"checkpoint-\d+", path.name)),
                                    key=lambda path: int(path.name.split("-")[1]), reverse=True)
                for candidate in candidates:
                    if not (candidate / "trainer_state.json").is_file():
                        continue
                    local = Path(mkdtemp(prefix="dual-resume-", dir="/tmp")) / candidate.name
                    local.mkdir()
                    try:
                        for source in candidate.iterdir():
                            if source.is_file():
                                with source.open("rb") as stream, (local / source.name).open("wb") as target:
                                    shutil.copyfileobj(stream, target, length=16 * 1024 * 1024)
                        record = validate_checkpoint(local, world_size)
                    except Exception as error:
                        shutil.rmtree(local.parent)
                        print(json.dumps({"event": "incomplete_checkpoint_skipped", "checkpoint": candidate.name,
                                          "error_type": type(error).__name__}), flush=True)
                        continue
                    selection[0] = {"path": str(local), "source": str(candidate), **record}
                    break
                if selection[0] is None:
                    raise ValueError("no complete checkpoint; refusing to restart formal N")
                (output_dir / "IO_RESUME_AUDIT.json").write_text(json.dumps(selection[0], indent=2) + "\n")
            except Exception as error:
                selection[0] = {"error": f"{type(error).__name__}: {error}"}
        if distributed:
            torch.distributed.broadcast_object_list(selection, src=0)
        if "error" in selection[0]:
            raise RuntimeError(selection[0]["error"])
        print(json.dumps({"event": "dual_complete_checkpoint_selected", "rank": rank, **selection[0]}), flush=True)
        return Path(selection[0]["path"])

    train.latest_checkpoint = latest


def self_test() -> None:
    import io

    original = torch.save
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        destination = root / "formal/novel/checkpoint-100/optimizer.pt"
        destination.parent.mkdir(parents=True)
        value = {"state": {0: {"step": torch.tensor(100), "exp_avg": torch.arange(512).float()}},
                 "param_groups": [{"lr": 0.0001, "params": [0]}]}
        rng = torch.get_rng_state().clone()
        staged_save(original, value, destination, root=root)
        restored = torch.load(destination, weights_only=False)
        torch.testing.assert_close(restored["state"][0]["exp_avg"], value["state"][0]["exp_avg"])
        assert restored["param_groups"] == value["param_groups"]
        torch.testing.assert_close(torch.get_rng_state(), rng)
        direct = root / "direct/optimizer.pt"
        direct.parent.mkdir()
        original(value, direct)
        assert direct.read_bytes() == destination.read_bytes()
        assert not checkpoint_archive(root / "other/optimizer.pt", root)
        assert not checkpoint_archive(destination, root / "unrelated")
        buffer = io.BytesIO()
        staged_save(original, value, buffer, root=root)
        buffer.seek(0)
        assert torch.load(buffer, weights_only=False)["param_groups"] == value["param_groups"]
        for rank in range(8):
            path = destination.with_name(f"rng_state_{rank}.pth")
            assert checkpoint_archive(path, root)
            staged_save(original, {"cpu": rng, "python": (), "numpy": (), "cuda": []}, path, root=root)
            torch.testing.assert_close(torch.load(path, weights_only=False)["cpu"], rng)
        for name in ("adapter_model.safetensors", "adapter_config.json", "training_args.bin"):
            (destination.parent / name).write_bytes(b"fixture")
        (destination.parent / "trainer_state.json").write_text(json.dumps({"global_step": 100}))
        original({"last_epoch": 100}, destination.parent / "scheduler.pt")
        assert validate_checkpoint(destination.parent, 8)["step"] == 100
        (destination.parent / "rng_state_7.pth").unlink()
        try:
            validate_checkpoint(destination.parent, 8)
        except ValueError:
            pass
        else:
            raise AssertionError("partial all-rank checkpoint was accepted")
    print("PASS: identical archives, optimizer/RNG round-trip, unchanged RNG, scoped fallback and partial-checkpoint rejection")


if __name__ == "__main__":
    self_test()