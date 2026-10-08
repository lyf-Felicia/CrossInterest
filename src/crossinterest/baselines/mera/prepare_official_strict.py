#!/usr/bin/env python3
"""Prepare and audit the pinned official MeRA source for strict execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from crossinterest.baselines.mera.core import write_json
from crossinterest.baselines.mera.evaluate_official_strict import OFFICIAL_COMMIT, OFFICIAL_REPOSITORY, QWEN2_REVISION


EXPECTED_SHA256 = {
    "train.py": "a3cec1c6528097ff9c5e639c18b4ee0d051fe19a81986d45a9a43314247c4278",
    "inference.py": "db9def0a4a53dd6ed45b0144aebfe31e49ab47f36c68448d2dcadbdfcacc05ff",
    "analyze_output.py": "cc0fc8699c6d899bb319d6b4a040bb091adf83e2c0635f1cde7dc8aa05c7478a",
    "train_arts.sh": "e4a8ab54d0aa37dc7439d09218031792e87a71945cec7a702da0f3f31e6cff03",
    "train_grocery.sh": "464ef11658f2db1a1bd9d61c4f7d0bb8bb96f8e1bf11ae4ce550dab8fe287f22",
    "inference_arts.sh": "081939cb8ae4d47aeea44909f03ce6c3e4263a48f805256171922c82fb1ee75e",
    "inference_grocery.sh": "e62ff67d99aa3cbaa1d71610f14c07b62197fe35f5a1cdee44de029763e3f573",
    "data/Arts/category_tree.json": "8a155213c710703bd2dc518d4d830bb8f9566fb262881fab169f83ee0b58d737",
    "data/Arts/category_cooccurrence.npy": "d77bf0078a579d8db73387def9b283acf6a7ec4c0112b8ed06d757463e335a75",
    "data/Arts/layer2_map.txt": "03c39ed9c270082ddf177d3c218d9653f062f2b70bfcec372dc25b96c34a935a",
    "data/Arts/test.json": "90bdd93c026864fd80355603bb0c26b38cdf84561401e010453c126209b032cc",
    "data/Arts/train.json": "2dde39423708fbf2ad938633ae9d8f1e414a10189d01069e4a22fc465de78b39",
    "data/Arts/valid.json": "451c809a1437e99d3d82f63e316a4a7bceaa4075269a148f2448fc91121f1d15",
    "data/Grocery_and_Gourmet_Food/category_tree_new.json": "0e5a4c04409804f8bb8f19cb4e8e715d25adb2979e6c70dab4679a6daaf43cb2",
    "data/Grocery_and_Gourmet_Food/category_cooccurrence.npy": "9782b5471607643908f363d2b017b4de65122116424d7eee80cf859c29b0991e",
    "data/Grocery_and_Gourmet_Food/layer2_map.txt": "fb03715e274ecac5e32ba84e4abdebd991daab544742f0c2d39d11618b7ec0ae",
    "data/Grocery_and_Gourmet_Food/test.json": "ed0f9b769c2bae03f4c27b1f07f69088a354b28848500d4d38bcf224ce445b94",
    "data/Grocery_and_Gourmet_Food/train.json": "0e4da7761afde99e0aaac2443c1a03aa1f18c9766301f5561d507ac8ab1c9233",
    "data/Grocery_and_Gourmet_Food/valid.json": "d10f18828061a740cae18511f5193aa5661eda106b86dbaf50bd34eef4571b40",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checkout(checkout: Path) -> dict[str, dict[str, int | str]]:
    actual_commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual_commit != OFFICIAL_COMMIT:
        raise ValueError(f"official commit mismatch: {actual_commit} != {OFFICIAL_COMMIT}")
    files = {}
    for relative, expected_sha in EXPECTED_SHA256.items():
        path = checkout / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise ValueError(
                f"official source SHA mismatch for {relative}: "
                f"{actual_sha} != {expected_sha}"
            )
        files[relative] = {"sha256": actual_sha, "bytes": path.stat().st_size}
    return files


def prepare_checkout(checkout: Path) -> dict:
    if checkout.exists():
        shutil.rmtree(checkout)
    subprocess.run(
        ["git", "clone", "--filter=blob:none", OFFICIAL_REPOSITORY, str(checkout)],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(checkout), "checkout", "--detach", OFFICIAL_COMMIT],
        check=True,
    )
    files = verify_checkout(checkout)
    inference_path = checkout / "inference.py"
    source = inference_path.read_text(encoding="utf-8")
    old = "        output = output.indice\n"
    new = "        output = output.indices\n"
    if source.count(old) != 1:
        raise ValueError("the pinned inference typo no longer has the expected form")
    inference_path.write_text(source.replace(old, new), encoding="utf-8")
    patch = subprocess.run(
        ["git", "-C", str(checkout), "diff", "--", "inference.py"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    changed = subprocess.run(
        ["git", "-C", str(checkout), "diff", "--name-only"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if changed != ["inference.py"]:
        raise ValueError(f"unexpected official source changes: {changed}")
    return {
        "schema_version": 1,
        "strictness": "fixed-public-code execution",
        "repository": OFFICIAL_REPOSITORY,
        "commit": OFFICIAL_COMMIT,
        "files_before_patch": files,
        "patches": [
            {
                "file": "inference.py",
                "change": "torch.return_types.topk.indice -> .indices",
                "semantic_effect": "repair AttributeError only",
                "diff": patch,
            }
        ],
        "unavailable_private_assets": [
            "authors' local Qwen2-1.5B directory checksum",
            "authors' container image digest and trainer state",
        ],
        "public_model": "Qwen/Qwen2-1.5B",
        "public_model_revision": QWEN2_REVISION,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = prepare_checkout(args.checkout_dir.resolve())
    write_json(args.manifest.resolve(), manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()