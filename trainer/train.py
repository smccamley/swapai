#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import types
from pathlib import Path

os.environ["NEEDLE_TELEMETRY"] = "0"
os.environ["DO_NOT_TRACK"] = "1"

EXPECTED_NEEDLE_VERSION = "2.0.14"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified_input(job_directory: Path, descriptor: dict, expected_path: str) -> Path:
    if descriptor.get("path") != expected_path:
        raise RuntimeError(f"Unexpected training input path: {expected_path}")
    path = job_directory / expected_path
    if not path.is_file() or sha256(path) != descriptor.get("sha256"):
        raise RuntimeError(f"{expected_path} failed SHA-256 verification")
    return path


def train(job_directory: Path) -> None:
    import needle
    from needle.model.finetune import build_main, finetune_local

    if getattr(needle, "__version__", None) != EXPECTED_NEEDLE_VERSION:
        raise RuntimeError(
            f"cactus-needle {EXPECTED_NEEDLE_VERSION} is required; "
            f"found {getattr(needle, '__version__', None)}"
        )

    manifest = json.loads((job_directory / "job.json").read_text(encoding="utf-8"))
    if manifest.get("format") != 2 or manifest.get("needleVersion") != EXPECTED_NEEDLE_VERSION:
        raise RuntimeError("Unsupported SwapAI training job")
    inputs = manifest.get("inputs", {})
    training_data = verified_input(job_directory, inputs.get("training", {}), "training.jsonl")
    if not training_data.is_file() or training_data.stat().st_size == 0:
        raise RuntimeError("Training data is empty")
    training_count = sum(
        1 for line in training_data.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    if training_count != inputs["training"].get("examples"):
        raise RuntimeError("Training example count does not match job manifest")
    example_count = training_count

    checkpoints = Path("/workspace/swapai-checkpoints")
    checkpoints.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoints / "needle2.pkl"
    adapter = job_directory / "swapai-lora.pkl"
    output = job_directory / "model.cact"
    finetune_local(
        types.SimpleNamespace(
            jsonl_path=str(training_data),
            checkpoint=str(checkpoint),
            epochs=10,
            batch_size=min(16, max(1, (example_count + 9) // 10)),
            lr=1e-4,
            lora_rank=16,
            lora_alpha=32.0,
            max_len=1024,
            val_split=0.0,
            seed=0,
            generate=0,
            model="deepseek/deepseek-v4-flash",
            workers=1,
            checkpoint_dir=str(checkpoints),
            out=str(adapter),
            qat_bits="auto",
        )
    )
    build_main(
        types.SimpleNamespace(
            checkpoint=str(checkpoint),
            lora=str(adapter),
            out=str(output),
            upload=False,
            bits=None,
        )
    )
    if not output.is_file():
        raise RuntimeError("Needle did not create model.cact")
    labels = job_directory / "number-labels.json"
    if labels.is_file():
        verified_input(job_directory, inputs.get("numberLabels", {}), "number-labels.json")
        shutil.copyfile(labels, job_directory / "model.cact.numbers.json")
    result = {
        "format": 2,
        "jobId": manifest["id"],
        "datasetRevisionId": manifest["datasetRevisionId"],
        "needleVersion": (
            f"{EXPECTED_NEEDLE_VERSION}/number-buckets-v1"
            if labels.is_file()
            else EXPECTED_NEEDLE_VERSION
        ),
        "inputs": {
            "trainingSha256": inputs["training"]["sha256"],
        },
        "outputs": {
            "model": {
                "path": "model.cact",
                "sha256": sha256(output),
            },
            **(
                {
                    "numberLabels": {
                        "path": "model.cact.numbers.json",
                        "sha256": sha256(job_directory / "model.cact.numbers.json"),
                    }
                }
                if labels.is_file()
                else {}
            ),
        },
    }
    (job_directory / "result.json").write_text(json.dumps(result), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(prog="swapai-trainer")
    parser.add_argument("--job-directory", required=True)
    args = parser.parse_args()
    train(Path(args.job_directory).resolve())


if __name__ == "__main__":
    main()
