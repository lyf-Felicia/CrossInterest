#!/usr/bin/env python3
"""Fetch and verify the immutable official MeRA Arts/Grocery data snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from crossinterest.baselines.mera.core import write_json


REPOSITORY = "https://github.com/GabyUSTC/MeRA.git"
COMMIT = "c07bdbe57bf82e5c938b478035baee78d630b17c"
EXPECTED_SHA256 = {
    "Arts/category_cooccurrence.npy": "d77bf0078a579d8db73387def9b283acf6a7ec4c0112b8ed06d757463e335a75",
    "Arts/category_tree.json": "8a155213c710703bd2dc518d4d830bb8f9566fb262881fab169f83ee0b58d737",
    "Arts/layer2_map.txt": "03c39ed9c270082ddf177d3c218d9653f062f2b70bfcec372dc25b96c34a935a",
    "Arts/test.json": "90bdd93c026864fd80355603bb0c26b38cdf84561401e010453c126209b032cc",
    "Arts/train.json": "2dde39423708fbf2ad938633ae9d8f1e414a10189d01069e4a22fc465de78b39",
    "Arts/valid.json": "451c809a1437e99d3d82f63e316a4a7bceaa4075269a148f2448fc91121f1d15",
    "Grocery_and_Gourmet_Food/category_cooccurrence.npy": "9782b5471607643908f363d2b017b4de65122116424d7eee80cf859c29b0991e",
    "Grocery_and_Gourmet_Food/category_tree_new.json": "0e5a4c04409804f8bb8f19cb4e8e715d25adb2979e6c70dab4679a6daaf43cb2",
    "Grocery_and_Gourmet_Food/layer2_map.txt": "fb03715e274ecac5e32ba84e4abdebd991daab544742f0c2d39d11618b7ec0ae",
    "Grocery_and_Gourmet_Food/test.json": "ed0f9b769c2bae03f4c27b1f07f69088a354b28848500d4d38bcf224ce445b94",
    "Grocery_and_Gourmet_Food/train.json": "0e4da7761afde99e0aaac2443c1a03aa1f18c9766301f5561d507ac8ab1c9233",
    "Grocery_and_Gourmet_Food/valid.json": "d10f18828061a740cae18511f5193aa5661eda106b86dbaf50bd34eef4571b40",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def verify(root: Path) -> dict:
    files = {}
    for relative, expected in EXPECTED_SHA256.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"official data SHA mismatch for {relative}: {actual} != {expected}"
            )
        files[relative] = {"sha256": actual, "bytes": path.stat().st_size}
    return files


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    manifest_path = output / "OFFICIAL_DATA_MANIFEST.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("commit") == COMMIT:
            verify(output)
            print(json.dumps(manifest, ensure_ascii=False, indent=2))
            return
    with tempfile.TemporaryDirectory(prefix="mera-official-") as temporary:
        checkout = Path(temporary) / "repository"
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-checkout", REPOSITORY, str(checkout)],
            check=True,
        )
        subprocess.run(["git", "-C", str(checkout), "checkout", COMMIT, "--", "data"], check=True)
        stage = Path(temporary) / "data"
        shutil.copytree(checkout / "data", stage)
        files = verify(stage)
        if output.exists():
            shutil.rmtree(output)
        shutil.copytree(stage, output)
    manifest = {
        "schema_version": 1,
        "repository": REPOSITORY,
        "commit": COMMIT,
        "files": files,
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
