"""Run the five assessed demo scenarios against the running agent API and save transcripts.

    uvicorn api.main:app --port 8100            # in another terminal
    python scripts/run_demo_conversations.py    # writes docs/conversations/*.md and *.json

Scenario 4 injects a real outage into the cluster first and restores it afterwards.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "conversations"
API = os.environ.get("AGENT_API_URL", "http://localhost:8100")
HEADERS = {"X-API-Key": os.environ["AGENT_API_KEY"]} if os.environ.get("AGENT_API_KEY") else {}

SCENARIOS = [
    (
        "01_dataset_provenance",
        "Dataset provenance",
        "Which dataset version was used to train the model that is currently deployed? Verify it against DVC and "
        "explain how I can reproduce that exact dataset.",
    ),
    (
        "02_model_selection",
        "Model selection",
        "Compare the last five training runs and recommend the best candidate based on F1 score.",
    ),
    (
        "03_pipeline_execution",
        "Pipeline execution",
        "Trigger the model training pipeline and report whether execution succeeded.",
    ),
    ("04_deployment_diagnosis", "Deployment diagnosis", "Why is my prediction API unavailable?"),
    (
        "05_multi_tool_reasoning",
        "Multi-tool reasoning (mandatory integration scenario)",
        "Find the best recent model, identify the dataset version used to train it, check the deployment status, "
        "and explain whether retraining is necessary.",
    ),
]


def _bash() -> str:
    # on Windows, plain "bash" can resolve to the WSL stub; prefer Git Bash
    for cand in (os.environ.get("BASH_EXE"), r"C:\Program Files\Git\bin\bash.exe"):
        if cand and Path(cand).exists():
            return cand
    return shutil.which("bash") or "bash"


def sh(*args: str) -> None:
    subprocess.run([_bash(), *args], cwd=ROOT, check=True)


def render(title: str, question: str, responses: list[dict]) -> str:
    lines = [f"# {title}", "", f"**User:** {question}", ""]
    for i, r in enumerate(responses):
        if i:
            lines += ["---", "", f"**User confirms action** `{r.get('_confirmed_action')}`", ""]
        lines += [
            f"_request `{r['request_id']}` · model `{r['model']}` · {r['steps']} step(s)_",
            "",
            "| step | tool | arguments | status | ms |",
            "|---|---|---|---|---|",
        ]
        for c in r["tool_calls"]:
            args = json.dumps(c["arguments"])[:90].replace("|", "\\|")
            lines.append(f"| {c['step']} | `{c['tool']}` | `{args}` | {c['status']} | {c['duration_ms']:.0f} |")
        lines += ["", "**Agent:**", "", r["answer"], ""]
    return "\n".join(lines)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    only = set(sys.argv[1:])
    with httpx.Client(base_url=API, headers=HEADERS, timeout=900) as client:
        client.get("/health").raise_for_status()
        for slug, title, question in SCENARIOS:
            if only and slug not in only:
                continue
            print(f"== {title}")
            if slug.startswith("04"):
                sh("scripts/k8s_inject_failure.sh", "broken-config")
                time.sleep(30)
            try:
                resp = client.post("/agent/query", json={"message": question}).json()
                responses = [resp]
                for action in resp.get("pending_actions", []):
                    # the demo plays the user typing "yes" to the exact action shown
                    confirmed = client.post("/agent/confirm", json={"action_id": action["action_id"]}).json()
                    confirmed["_confirmed_action"] = f"{action['action_id']}: {action['tool']}({action['arguments']})"
                    responses.append(confirmed)
            finally:
                if slug.startswith("04"):
                    sh("scripts/k8s_inject_failure.sh", "restore")
            (OUT / f"{slug}.json").write_text(json.dumps(responses, indent=2) + "\n")
            (OUT / f"{slug}.md").write_text(render(title, question, responses) + "\n", encoding="utf-8")
            for r in responses:
                print("   tools:", [f"{c['tool']}:{c['status']}" for c in r["tool_calls"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
