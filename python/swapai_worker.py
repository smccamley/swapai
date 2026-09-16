#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any

os.environ["NEEDLE_TELEMETRY"] = "0"
os.environ["DO_NOT_TRACK"] = "1"

EXPECTED_NEEDLE_VERSION = "2.0.14"


def write_message(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def require_version(needle: Any) -> None:
    version = getattr(needle, "__version__", None)
    if version != EXPECTED_NEEDLE_VERSION:
        raise RuntimeError(
            f"cactus-needle {EXPECTED_NEEDLE_VERSION} is required; found {version}"
        )


@contextmanager
def artifact_write_lock(args: argparse.Namespace):
    lock_database = sqlite3.connect(args.artifact_lock_database, timeout=30)
    main_database: sqlite3.Connection | None = None
    try:
        lock_database.execute("BEGIN IMMEDIATE")
        main_database = sqlite3.connect(f"file:{args.main_database}?mode=ro", uri=True)
        row = main_database.execute(
            "SELECT data_epoch, clear_pending FROM classifiers WHERE name = ?",
            (args.classifier_name,),
        ).fetchone()
        if row is None or row[0] != args.expected_epoch or row[1] != 0:
            raise RuntimeError("Classifier training was superseded by data erasure")
        yield
    finally:
        if main_database is not None:
            main_database.close()
        try:
            lock_database.rollback()
        finally:
            lock_database.close()


def train(args: argparse.Namespace) -> None:
    with artifact_write_lock(args):
        import needle
        from needle.model.finetune import build_main, finetune_local

        require_version(needle)
        checkpoint_dir = Path(args.checkpoint_dir).resolve()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = checkpoint_dir / "needle2.pkl"
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        training_input = sys.stdin.read()
        number_labels: dict[str, Any] | None = None
        if args.numeric_labels_from_stdin:
            labels_json, separator, training_input = training_input.partition("\n")
            if not separator:
                raise RuntimeError("Numeric label map is missing")
            labels = json.loads(labels_json)
            values = labels.get("values") if isinstance(labels, dict) else None
            if (
                not isinstance(labels, dict)
                or labels.get("format") != 1
                or not isinstance(values, list)
                or not values
                or any(
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(value)
                    for value in values
                )
                or values != sorted(set(values))
            ):
                raise RuntimeError("Numeric label map has an invalid format")
            number_labels = labels
        training_data = Path(args.training_data).resolve()
        training_data.write_text(training_input, encoding="utf-8")
        adapter = output.parent / "swapai-lora.pkl"
        with training_data.open("r", encoding="utf-8") as handle:
            example_count = sum(1 for line in handle if line.strip())
        batch_size = min(16, max(1, (example_count + 9) // 10))

        finetune_local(
            types.SimpleNamespace(
                jsonl_path=str(training_data),
                checkpoint=str(checkpoint),
                epochs=args.epochs,
                batch_size=batch_size,
                lr=1e-4,
                lora_rank=16,
                lora_alpha=32.0,
                max_len=1024,
                val_split=0.0,
                seed=0,
                generate=0,
                model="deepseek/deepseek-v4-flash",
                workers=1,
                checkpoint_dir=str(checkpoint_dir),
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
            raise RuntimeError("Needle did not create the requested .cact model")
        if number_labels is not None:
            Path(f"{output}.numbers.json").write_text(
                json.dumps(number_labels, separators=(",", ":")),
                encoding="utf-8",
            )


def classification_result(response: Any) -> Any:
    if not isinstance(response, dict):
        raise RuntimeError("Needle returned an invalid response")
    calls = response.get("function_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        raise RuntimeError("Needle did not return exactly one classify call")
    call = calls[0]
    if not isinstance(call, dict) or call.get("name") != "classify":
        raise RuntimeError("Needle did not return exactly one classify call")
    arguments = call.get("arguments")
    if not isinstance(arguments, dict) or "result" not in arguments:
        raise RuntimeError("Needle classify call did not contain a result")
    return arguments["result"]


def serve(args: argparse.Namespace) -> None:
    import needle

    require_version(needle)
    with open(args.schema, "r", encoding="utf-8") as handle:
        tools = json.load(handle)
    agent = needle.Needle(tools=tools, weights=args.model)
    write_message({"type": "ready", "needleVersion": needle.__version__})
    try:
        for raw_line in sys.stdin:
            request: Any = None
            try:
                request = json.loads(raw_line)
                if request.get("type") == "close":
                    write_message({"type": "closed"})
                    return
                if request.get("type") != "classify":
                    raise ValueError("unknown worker request")
                request_id = request.get("id")
                input_text = request.get("input")
                if not isinstance(request_id, int) or not isinstance(input_text, str):
                    raise ValueError("classify requires an integer id and string input")
                agent.reset()
                response = agent.complete(input_text)
                write_message(
                    {
                        "id": request_id,
                        "ok": True,
                        "result": classification_result(response),
                    }
                )
            except Exception as error:
                request_id = request.get("id") if isinstance(request, dict) else None
                write_message(
                    {
                        "id": request_id,
                        "ok": False,
                        "error": {
                            "code": "classification_failed",
                            "message": str(error),
                        },
                    }
                )
    finally:
        agent.close()


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="swapai_worker")
    commands = root.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train")
    train_parser.add_argument("--training-data", required=True)
    train_parser.add_argument("--output", required=True)
    train_parser.add_argument("--checkpoint-dir", required=True)
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--artifact-lock-database", required=True)
    train_parser.add_argument("--main-database", required=True)
    train_parser.add_argument("--classifier-name", required=True)
    train_parser.add_argument("--expected-epoch", type=int, required=True)
    train_parser.add_argument("--numeric-labels-from-stdin", action="store_true")

    serve_parser = commands.add_parser("serve")
    serve_parser.add_argument("--model", required=True)
    serve_parser.add_argument("--schema", required=True)
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "train":
        train(args)
    else:
        serve(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if len(sys.argv) > 1 and sys.argv[1] == "serve":
            write_message({"type": "error", "message": str(error)})
        else:
            sys.stderr.write(f"SwapAI Needle worker failed: {error}\n")
        raise
