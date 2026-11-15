"""Pre-commit / pre-push secret scan (cross-platform). Never prints a secret value - only where it was found.

    python scripts/check_secrets.py

Checks:
  1. exact values: every secret-looking variable in .env (...KEY / ...PASSWORD / ...TOKEN / ...SECRET) and the token in
     .secrets/*.kubeconfig are searched for literally in the staged diff and in all tracked files
  2. patterns: well-known credential shapes (Google API keys, JWTs, private keys, user:password@ URLs)
  3. files that must never be tracked (.env, .secrets/, kubeconfigs, exported model)
Exit code 1 if anything is found.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SECRET_VAR = re.compile(r"^([A-Z0-9_]*(KEY|PASSWORD|TOKEN|SECRET)[A-Z0-9_]*)=(.*)$")
PATTERNS = {
    "Google API key": re.compile(r"AIza[0-9A-Za-z_-]{35}"),
    "JWT": re.compile(r"eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\."),
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "credentials in URL": re.compile(r"://[^/:@\s]+:[^/@\s]{6,}@"),
}
# files that legitimately contain the *patterns* (redaction code, this scanner, tests with fake values)
PATTERN_EXEMPT = ("agent/logging_utils.py", "scripts/check_secrets.py", "tests/")
NEVER_TRACK = re.compile(r"(^|/)\.env$|^\.secrets/|\.kubeconfig$|^serving/model/")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True
    ).stdout


def secret_values() -> dict[str, str]:
    values: dict[str, str] = {}
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            m = SECRET_VAR.match(line.strip())
            if m:
                v = m.group(3).strip().strip('"').strip("'")
                if len(v) >= 8:
                    values[m.group(1)] = v
    for f in (ROOT / ".secrets").glob("*.kubeconfig"):
        for m in re.finditer(r"token:\s*(\S+)", f.read_text(encoding="utf-8")):
            values[f"{f.name} token"] = m.group(1)
    return values


def main() -> int:
    problems: list[str] = []
    values = secret_values()

    staged = git("diff", "--cached")
    for name, v in values.items():
        if v in staged:
            problems.append(f"LEAK: value of {name} appears in the staged changes")

    tracked = [p for p in git("ls-files", "-z").split("\0") if p]
    for path in tracked:
        if NEVER_TRACK.search(path):
            problems.append(f"LEAK: sensitive file is tracked by git: {path}")
            continue
        try:
            text = (ROOT / path).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary or unreadable
        for name, v in values.items():
            if v in text:
                problems.append(f"LEAK: value of {name} found in {path}")
        if not path.startswith(PATTERN_EXEMPT):
            for label, rx in PATTERNS.items():
                if rx.search(text):
                    problems.append(f"SUSPICIOUS: {label} pattern in {path}")

    if problems:
        print("\n".join(problems))
        return 1
    print(
        f"OK: no secrets found ({len(values)} secret value(s) + {len(PATTERNS)} credential patterns checked "
        f"against the staged diff and {len(tracked)} tracked files)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
