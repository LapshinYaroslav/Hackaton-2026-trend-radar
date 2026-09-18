"""CLI for both collector modes. Live sources require network; tests use FakeAdapter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from collector.api import build_collector
from collector.models import Candidate, Technology


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Weak-signal document collector")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Training mode: history for a labelled technology list")
    train.add_argument("--input", required=True, help="JSON list of Technology objects")
    train.add_argument("--output", required=True, help="JSON file for CollectionResult list")

    recent = sub.add_parser("recent", help="Query mode, Search #1: fresh documents by subqueries")
    recent.add_argument("--subquery", action="append", required=True, dest="subqueries")
    recent.add_argument("--output", required=True)

    history = sub.add_parser("history", help="Query mode, Search #2: 6-year history of a candidate")
    history.add_argument("--candidate", required=True, help="JSON Candidate object")
    history.add_argument("--output", required=True)

    many = sub.add_parser("histories", help="Query mode, Search #2 for a candidate list")
    many.add_argument("--input", required=True, help="JSON list of Candidate objects")
    many.add_argument("--output", required=True)
    many.add_argument("--max-candidates", type=int, default=None)

    args = parser.parse_args(argv)
    collector = build_collector()

    if args.command == "train":
        techs = [Technology.from_dict(item) for item in _load_json(args.input)]
        results = [item.to_contract_dict() for item in collector.collect_training(techs)]
        _dump_json(args.output, results)
        return 0

    if args.command == "recent":
        result = collector.search_recent(args.subqueries)
        _dump_json(args.output, result.to_dict())
        return 0

    if args.command == "history":
        candidate = Candidate.from_dict(_load_json(args.candidate))
        result = collector.collect_history(candidate)
        _dump_json(args.output, result.to_contract_dict())
        return 0

    if args.command == "histories":
        candidates = [Candidate.from_dict(item) for item in _load_json(args.input)]
        results = [
            item.to_contract_dict()
            for item in collector.collect_histories(
                candidates, max_candidates=args.max_candidates
            )
            if not item.skipped_as_mainstream
        ]
        _dump_json(args.output, results)
        return 0

    return 1


def _load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _dump_json(path: str, payload) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
