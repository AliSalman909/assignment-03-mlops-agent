#!/usr/bin/env python3
"""One-command launcher for the MLOps agent: set up everything that's missing, then start the agent.

    python start.py              set up (idempotent) and open the interactive agent chat
    python start.py --api        set up and serve the HTTP API on http://localhost:8100 (docs at /docs)
    python start.py --status     only report the state of every component
    python start.py --setup-only set up, don't start the agent
    python start.py --no-k8s     skip the Kubernetes parts (the agent runs; k8s tools report "unavailable")

Prerequisites (the script checks them): Python 3.11+, git, Docker (running), and for Kubernetes kubectl + kind.
Uses only the Python standard library, so it runs before the virtualenv exists. Every step is idempotent:
a second run only does what is still missing.
"""

from __future__ import annotations

import argparse
import getpass
import glob
import hashlib
import json
import os
import secrets
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IS_WIN = os.name == "nt"
VENV = ROOT / ".venv"
VPY = VENV / ("Scripts/python.exe" if IS_WIN else "bin/python")
ENV = ROOT / ".env"
CLUSTER, CTX, NS, IMAGE = "mlops", "kind-mlops", "mlops", "churn-api:1.0.0"
KUBECONFIG = ROOT / ".secrets" / "agent.kubeconfig"


# ----------------------------------------------------------------------------- output helpers
def step(msg: str) -> None:
    print(f"\n==> {msg}", flush=True)


def ok(msg: str) -> None:
    print(f"    [ok] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"    [!]  {msg}", flush=True)


def die(msg: str) -> None:
    print(f"\n[x] {msg}", file=sys.stderr)
    sys.exit(1)


def run(
    cmd: list[str],
    *,
    check: bool = True,
    capture: bool = False,
    quiet: bool = False,
    env: dict | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess:
    kw: dict = {
        "cwd": ROOT,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "timeout": timeout,
        "env": {**os.environ, **(env or {})},
    }
    if capture or quiet:
        kw["stdout"] = subprocess.PIPE
        kw["stderr"] = subprocess.PIPE
    proc = subprocess.run(cmd, **kw)
    if check and proc.returncode != 0:
        detail = ((proc.stderr or "") + (proc.stdout or "")).strip()[-1500:] if (capture or quiet) else ""
        die(f"command failed ({proc.returncode}): {' '.join(cmd[:4])}...\n{detail}")
    return proc


def find_exe(name: str) -> str | None:
    exe = shutil.which(name)
    if exe or not IS_WIN:
        return exe
    # winget installs (e.g. kind) may not be on PATH until a new shell is opened
    local = os.environ.get("LOCALAPPDATA", "")
    for pattern in (rf"{local}\Microsoft\WinGet\Links\{name}.exe", rf"{local}\Microsoft\WinGet\Packages\*\{name}.exe"):
        hits = glob.glob(pattern)
        if hits:
            return hits[0]
    return None


def http_ok(url: str, timeout: float = 3) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 - fixed localhost URLs
            return 200 <= r.status < 300
    except Exception:  # noqa: BLE001
        return False


def wait_for(url: str, label: str, seconds: int) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        if http_ok(url):
            ok(f"{label} is up ({url})")
            return
        time.sleep(3)
    die(f"{label} did not become healthy within {seconds}s ({url}). Check: docker compose logs")


# ----------------------------------------------------------------------------- .env handling
def read_env() -> dict[str, str]:
    out = {}
    if ENV.exists():
        for line in ENV.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out


def set_env(key: str, value: str) -> None:
    """Set KEY=value in .env, replacing an existing line or appending."""
    lines = ENV.read_text(encoding="utf-8").splitlines() if ENV.exists() else []
    for i, line in enumerate(lines):
        if line.split("=", 1)[0].strip() == key and not line.lstrip().startswith("#"):
            lines[i] = f"{key}={value}"
            break
    else:
        lines.append(f"{key}={value}")
    ENV.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------- steps
def check_prereqs(use_k8s: bool) -> dict[str, str]:
    step("Checking prerequisites")
    if sys.version_info < (3, 11):  # noqa: UP036 - runs before the venv, possibly on an older Python
        die(f"Python 3.11+ required (this is {sys.version.split()[0]})")
    ok(f"Python {sys.version.split()[0]}")
    tools = {"git": "https://git-scm.com/downloads", "docker": "https://docs.docker.com/get-docker/"}
    if use_k8s:
        tools |= {
            "kubectl": "https://kubernetes.io/docs/tasks/tools/",
            "kind": "https://kind.sigs.k8s.io/docs/user/quick-start/  (Windows: winget install Kubernetes.kind)",
        }
    found, missing = {}, []
    for name, hint in tools.items():
        exe = find_exe(name)
        if exe:
            found[name] = exe
            ok(f"{name}")
        else:
            missing.append(f"{name}: {hint}")
    if missing:
        die("missing tools:\n  " + "\n  ".join(missing) + ("\n(or run with --no-k8s)" if use_k8s else ""))
    if run([found["docker"], "info"], check=False, quiet=True).returncode != 0:
        die("Docker is installed but not running. Start Docker Desktop and re-run.")
    ok("Docker daemon running")
    return found


def setup_venv() -> None:
    step("Python environment (.venv)")
    if not VPY.exists():
        run([sys.executable, "-m", "venv", str(VENV)])
        ok("created .venv")
    marker = VENV / ".installed-pyproject"
    digest = hashlib.sha256((ROOT / "pyproject.toml").read_bytes()).hexdigest()
    if marker.exists() and marker.read_text().strip() == digest:
        ok("dependencies up to date")
        return
    print("    installing dependencies (first run takes a few minutes)...", flush=True)
    run([str(VPY), "-m", "pip", "install", "-q", "--upgrade", "pip"], quiet=True)
    run([str(VPY), "-m", "pip", "install", "-q", "-e", ".[dev]"], quiet=True)
    marker.write_text(digest)
    ok("dependencies installed")


def setup_env(interactive: bool) -> None:
    step("Configuration (.env)")
    if not ENV.exists():
        shutil.copy(ROOT / ".env.example", ENV)
        ok("created .env from .env.example")
    env = read_env()
    if env.get("AIRFLOW_ADMIN_PASSWORD", "") in ("", "change-me"):
        set_env("AIRFLOW_ADMIN_PASSWORD", secrets.token_urlsafe(18))
        ok("generated a random Airflow admin password (stored only in .env)")
    if not env.get("GEMINI_API_KEY"):
        if not interactive:
            die("GEMINI_API_KEY is empty in .env - add your key (https://aistudio.google.com/apikey) and re-run")
        key = getpass.getpass("    Enter your Gemini API key (input hidden; saved only to .env): ").strip()
        if not key:
            die("a Gemini API key is required to run the agent")
        set_env("GEMINI_API_KEY", key)
        ok("Gemini API key saved to .env")
    else:
        ok("Gemini API key present")
    for k, v in {"GEMINI_API_VERSION": "v1beta", "GEMINI_TEMPERATURE": "0.1"}.items():
        if k not in env:
            set_env(k, v)


def setup_data() -> None:
    step("Versioned data (DVC)")
    run([str(VPY), "-m", "dvc", "pull", "-q"], quiet=True, env={"DVC_NO_ANALYTICS": "1"})
    status = run([str(VPY), "-m", "dvc", "status", "-q"], check=False, quiet=True)
    if status.returncode != 0:
        warn("workspace differs from the committed data version (dvc status) - continuing")
    else:
        ok("data/raw/churn.csv and pipeline outputs match their DVC records")


def setup_services(docker: str) -> None:
    step("MLflow + Airflow (docker compose)")
    if http_ok("http://localhost:5000/health") and http_ok("http://localhost:8080/api/v2/monitor/health"):
        ok("already running")
    else:
        print("    starting containers (first run builds the Airflow image, ~3-5 min)...", flush=True)
        run([docker, "compose", "up", "-d", "--build"], quiet=True)
        wait_for("http://localhost:5000/health", "MLflow", 180)
        wait_for("http://localhost:8080/api/v2/monitor/health", "Airflow", 420)
    step("MLflow experiment history")
    out = run([str(VPY), "scripts/seed_experiments.py"], capture=True, env={"MLFLOW_DISABLE_AGENT_HINT": "1"})
    ok("already seeded" if "nothing to do" in out.stdout else "seeded 9 runs + registered 'champion' model")


def setup_kubernetes(tools: dict[str, str]) -> None:
    kind, kubectl, docker = tools["kind"], tools["kubectl"], tools["docker"]
    k = [kubectl, "--context", CTX]
    step("Kubernetes cluster (kind)")
    clusters = run([kind, "get", "clusters"], capture=True).stdout.split()
    if CLUSTER in clusters:
        ok(f"cluster '{CLUSTER}' exists")
    else:
        print("    creating cluster (~1-2 min)...", flush=True)
        run([kind, "create", "cluster", "--config", "kubernetes/kind-cluster.yaml", "--wait", "120s"], quiet=True)
        ok(f"cluster '{CLUSTER}' created")

    step("Prediction API deployment")
    dep = run(k + ["-n", NS, "get", "deployment", "churn-api", "-o", "json"], check=False, capture=True)
    ready = 0
    if dep.returncode == 0:
        st = json.loads(dep.stdout).get("status", {})
        ready = st.get("readyReplicas", 0)
    if ready >= 2:
        ok(f"churn-api running ({ready} ready replicas)")
    else:
        print("    exporting champion model from MLflow, building image, deploying...", flush=True)
        run(
            [str(VPY), "serving/export_model.py", "--alias", "champion"],
            quiet=True,
            env={"MLFLOW_DISABLE_AGENT_HINT": "1"},
        )
        run([docker, "build", "-q", "-t", IMAGE, "serving/"], quiet=True)
        run([kind, "load", "docker-image", IMAGE, "--name", CLUSTER], quiet=True)
        run(k + ["apply", "-f", "kubernetes/namespace.yaml"], quiet=True)
        run(
            k
            + [
                "apply",
                "-f",
                "kubernetes/configmap.yaml",
                "-f",
                "kubernetes/deployment.yaml",
                "-f",
                "kubernetes/service.yaml",
                "-f",
                "kubernetes/agent-rbac.yaml",
            ],
            quiet=True,
        )
        run(k + ["-n", NS, "rollout", "status", "deployment/churn-api", "--timeout=240s"], quiet=True)
        ok("churn-api deployed (2 replicas)")

    step("Restricted agent credentials")
    fresh = KUBECONFIG.exists() and time.time() - KUBECONFIG.stat().st_mtime < 20 * 3600
    if fresh:
        ok("agent kubeconfig valid")
    else:
        run(k + ["apply", "-f", "kubernetes/agent-rbac.yaml"], quiet=True)
        view = [kubectl, "config", "view", "--raw", "-o"]
        server = run(view + [f'jsonpath={{.clusters[?(@.name=="{CTX}")].cluster.server}}'], capture=True).stdout
        ca = run(
            view + [f'jsonpath={{.clusters[?(@.name=="{CTX}")].cluster.certificate-authority-data}}'], capture=True
        ).stdout
        token = run(k + ["-n", NS, "create", "token", "mlops-agent", "--duration=24h"], capture=True).stdout.strip()
        KUBECONFIG.parent.mkdir(exist_ok=True)
        KUBECONFIG.write_text(
            "apiVersion: v1\nkind: Config\n"
            f"clusters:\n- name: mlops\n  cluster:\n    server: {server}\n    certificate-authority-data: {ca}\n"
            f"users:\n- name: mlops-agent\n  user:\n    token: {token}\n"
            "contexts:\n- name: mlops-agent\n  context: {cluster: mlops, user: mlops-agent, namespace: mlops}\n"
            "current-context: mlops-agent\n",
            encoding="utf-8",
        )
        if not IS_WIN:
            KUBECONFIG.chmod(0o600)
        ok("restricted ServiceAccount kubeconfig written to .secrets/ (valid 24h)")
    if read_env().get("K8S_KUBECONFIG") != ".secrets/agent.kubeconfig":
        set_env("K8S_KUBECONFIG", ".secrets/agent.kubeconfig")


def status(tools: dict[str, str]) -> None:
    step("Status")
    env = read_env()
    rows = [
        ("virtualenv", VPY.exists()),
        (".env with Gemini key", bool(env.get("GEMINI_API_KEY"))),
        (
            "data (dvc status clean)",
            VPY.exists() and run([str(VPY), "-m", "dvc", "status", "-q"], check=False, quiet=True).returncode == 0,
        ),
        ("MLflow  http://localhost:5000", http_ok("http://localhost:5000/health")),
        ("Airflow http://localhost:8080", http_ok("http://localhost:8080/api/v2/monitor/health")),
    ]
    if "kubectl" in tools:
        dep = run(
            [tools["kubectl"], "--context", CTX, "-n", NS, "get", "deployment", "churn-api", "-o", "json"],
            check=False,
            capture=True,
        )
        ready = json.loads(dep.stdout).get("status", {}).get("readyReplicas", 0) if dep.returncode == 0 else 0
        rows.append((f"churn-api on Kubernetes ({ready} ready)", ready >= 1))
        rows.append(
            (
                "agent kubeconfig (<20h old)",
                KUBECONFIG.exists() and time.time() - KUBECONFIG.stat().st_mtime < 20 * 3600,
            )
        )
    for label, good in rows:
        print(f"    {'[ok]' if good else '[--]'} {label}")


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description="Set up and start the MLOps agent")
    ap.add_argument("--api", action="store_true", help="serve the HTTP API instead of the interactive chat")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--status", action="store_true", help="only report component status")
    ap.add_argument("--setup-only", action="store_true", help="set everything up but don't start the agent")
    ap.add_argument("--no-k8s", action="store_true", help="skip Kubernetes setup")
    args = ap.parse_args()
    os.chdir(ROOT)

    if args.status:
        status({n: e for n in ("kubectl",) if (e := find_exe(n))})
        return 0

    tools = check_prereqs(use_k8s=not args.no_k8s)
    setup_venv()
    setup_env(interactive=sys.stdin.isatty())
    setup_data()
    setup_services(tools["docker"])
    if not args.no_k8s:
        setup_kubernetes(tools)
    status(tools)

    if args.setup_only:
        print("\nSetup complete. Start the agent with:  python start.py   (or --api)")
        return 0
    env = {"MLFLOW_DISABLE_AGENT_HINT": "1"}
    if args.api:
        step(f"Starting the agent API on http://localhost:{args.port}  (interactive docs: /docs, Ctrl+C to stop)")
        cmd = [str(VPY), "-m", "uvicorn", "api.main:app", "--port", str(args.port)]
    else:
        step("Starting the agent chat (Ctrl+C to exit)")
        cmd = [str(VPY), "-m", "agent.main"]
    try:
        return subprocess.run(cmd, cwd=ROOT, env={**os.environ, **env}).returncode
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\ninterrupted")
        sys.exit(130)
