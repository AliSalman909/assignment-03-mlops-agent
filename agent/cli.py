"""Call a single agent tool directly - useful for debugging and for collecting evidence.

    python -m agent.cli list
    python -m agent.cli dvc_get_dataset_version '{"rev": "data-v1.1"}'
    python -m agent.cli dvc_restore_workspace '{}' --confirm
"""

from __future__ import annotations

import argparse
import json
import sys

from agent.logging_utils import configure_logging
from agent.tools import build_registry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Invoke an MLOps agent tool")
    parser.add_argument("tool", help="tool name, or 'list' to show all tools")
    parser.add_argument("arguments", nargs="?", default="{}", help="JSON object of tool arguments")
    parser.add_argument("--confirm", action="store_true", help="explicitly confirm a state-changing tool")
    args = parser.parse_args(argv)

    configure_logging()
    registry = build_registry()
    if args.tool == "list":
        for spec in registry.specs():
            flag = " [requires confirmation]" if spec.requires_confirmation else ""
            print(f"{spec.name:30} ({spec.module}){flag}\n    {spec.description}\n")
        return 0

    try:
        arguments = json.loads(args.arguments)
    except json.JSONDecodeError as exc:
        print(f"arguments must be a JSON object: {exc}", file=sys.stderr)
        return 2
    result = registry.execute(args.tool, arguments, confirmed=args.confirm)
    print(result.model_dump_json(indent=2))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
