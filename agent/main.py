"""Interactive command-line chat with the MLOps agent.

    python -m agent.main                      # interactive session
    python -m agent.main "Is the prediction API healthy?"   # one-shot

State-changing actions requested by the agent are shown with their exact arguments and only run
if you answer 'yes' to the confirmation prompt.
"""

from __future__ import annotations

import sys

from agent.logging_utils import configure_logging
from agent.orchestrator import build_agent
from agent.schemas import AgentResponse


def show(resp: AgentResponse) -> None:
    print()
    for c in resp.tool_calls:
        print(
            f"  · [{c.step}] {c.tool}({', '.join(f'{k}={v!r}' for k, v in c.arguments.items())}) -> "
            f"{c.status.value} ({c.duration_ms:.0f} ms)"
        )
    print(f"\n{resp.answer}\n")
    print(f"  [request {resp.request_id} · session {resp.session_id} · {resp.steps} step(s) · {resp.model}]")


def confirm_loop(agent, resp: AgentResponse) -> None:
    for action in resp.pending_actions:
        print(f"\n⚠ The agent wants to run a STATE-CHANGING action:\n    {action.tool}({action.arguments})")
        if input("    Type 'yes' to execute, anything else to cancel: ").strip().lower() == "yes":
            show(agent.confirm(action.action_id))
        else:
            print("    Cancelled - nothing was executed.")


def main(argv: list[str]) -> int:
    configure_logging("WARNING")
    agent = build_agent()
    if argv:
        resp = agent.run(" ".join(argv))
        show(resp)
        if sys.stdin.isatty():
            confirm_loop(agent, resp)
        return 0
    print("MLOps agent - ask about datasets, experiments, pipelines or the deployed API. Ctrl+C to exit.")
    session = None
    while True:
        try:
            msg = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not msg:
            continue
        resp = agent.run(msg, session_id=session)
        session = resp.session_id
        show(resp)
        confirm_loop(agent, resp)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
