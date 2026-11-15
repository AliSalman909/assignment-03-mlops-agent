"""Task 4 - Kubernetes Operations module.

Read-only tools
  list_pods                 pods with phase, readiness, restarts and problems
  list_deployments          deployments with desired/ready/updated replicas and rollout state
  get_pod_status            one pod in depth: container states, last termination, probes, conditions
  get_pod_logs              recent (or previous-crash) container logs, redacted
  list_services             services with type, ports, selector and ready endpoint counts
  inspect_service           one service: endpoints, matching pods, live health check through the API server
  find_unhealthy_pods       not-ready / crash-looping / OOM-killed / image-pull failures / restarts
  get_events                namespace or object events (warnings first)
  get_deployed_version      image, app version, model lineage annotations, rollout revision, live /version
  diagnose_service          "why is my prediction API unavailable?" - correlates service, endpoints,
                            pods, probe failures, events and logs into an evidence-based diagnosis
State-changing (explicit confirmation + approved namespace + managed-label Deployment only)
  restart_deployment        rolling restart
  scale_deployment          change replica count (bounded)
  rollback_deployment       roll back to a previous ReplicaSet revision

Defence in depth: the tool layer enforces confirmation and the namespace allow-list, and the
recommended credentials (kubernetes/agent-rbac.yaml) can only read + patch Deployments in `mlops`.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from agent.config import Settings, get_settings
from agent.logging_utils import redact
from agent.tools.base import (
    Tool,
    ToolError,
    ToolForbiddenError,
    ToolInvalidRequestError,
    ToolOutput,
    ToolRegistry,
    ToolTimeoutError,
    ToolUnavailableError,
)

MODULE = "kubernetes"
_DNS_RE = re.compile(r"^[a-z0-9]([-a-z0-9.]{0,251}[a-z0-9])?$")
_SELECTOR_RE = re.compile(r"^[A-Za-z0-9_./\-=!,() ]{1,200}$")
BAD_WAITING = {
    "CrashLoopBackOff": (
        "crash loop",
        "the container starts and exits repeatedly",
        "read previous-container logs for the crash cause; fix code/config, then restart",
    ),
    "ImagePullBackOff": (
        "image pull failure",
        "the container image cannot be pulled (wrong tag/registry/auth)",
        "fix the image reference or roll back to the last working revision",
    ),
    "ErrImagePull": (
        "image pull failure",
        "the container image cannot be pulled (wrong tag/registry/auth)",
        "fix the image reference or roll back to the last working revision",
    ),
    "CreateContainerConfigError": (
        "configuration error",
        "a referenced ConfigMap/Secret key is missing",
        "check the ConfigMap/Secret the pod references",
    ),
    "InvalidImageName": ("image pull failure", "the image name is invalid", "fix the image reference"),
}


# ---------------------------------------------------------------------------- inputs
def _ns(v: str | None) -> str | None:
    if v is not None and not _DNS_RE.match(v):
        raise ValueError(f"invalid namespace '{v}'")
    return v


def _name(v: str, what: str) -> str:
    v = v.strip()
    if not _DNS_RE.match(v):
        raise ValueError(f"invalid {what} '{v}'")
    return v


class _NsInput(BaseModel):
    namespace: str | None = Field(None, description="Kubernetes namespace (default: the application namespace)")

    @field_validator("namespace")
    @classmethod
    def v_ns(cls, v: str | None) -> str | None:
        return _ns(v)


class ListPodsInput(_NsInput):
    label_selector: str | None = Field(None, description="e.g. 'app.kubernetes.io/name=churn-api'")

    @field_validator("label_selector")
    @classmethod
    def v_sel(cls, v: str | None) -> str | None:
        if v is not None and not _SELECTOR_RE.match(v):
            raise ValueError("invalid label_selector")
        return v


class PodInput(_NsInput):
    pod_name: str

    @field_validator("pod_name")
    @classmethod
    def v_pod(cls, v: str) -> str:
        return _name(v, "pod_name")


class PodLogsInput(PodInput):
    container: str | None = None
    tail_lines: int = Field(80, ge=1, le=1000)
    previous: bool = Field(False, description="Logs of the previous (crashed) container instance")

    @field_validator("container")
    @classmethod
    def v_c(cls, v: str | None) -> str | None:
        return v if v is None else _name(v, "container")


class ServiceInput(_NsInput):
    service_name: str = Field("churn-api")
    check_health: bool = Field(True, description="Call the service's readiness endpoint through the API server")

    @field_validator("service_name")
    @classmethod
    def v_svc(cls, v: str) -> str:
        return _name(v, "service_name")


class EventsInput(_NsInput):
    object_name: str | None = Field(None, description="Only events about this pod/deployment/service")
    warnings_only: bool = False
    limit: int = Field(30, ge=1, le=200)

    @field_validator("object_name")
    @classmethod
    def v_obj(cls, v: str | None) -> str | None:
        return v if v is None else _name(v, "object_name")


class DeploymentInput(_NsInput):
    deployment_name: str = Field("churn-api")

    @field_validator("deployment_name")
    @classmethod
    def v_dep(cls, v: str) -> str:
        return _name(v, "deployment_name")


class ScaleInput(DeploymentInput):
    replicas: int = Field(..., ge=0, le=50)


class RollbackInput(DeploymentInput):
    to_revision: int | None = Field(None, ge=1, description="Revision to roll back to; default: the previous one")


# ---------------------------------------------------------------------------- helpers
def _age(ts: str | None) -> str | None:
    if not ts:
        return None
    try:
        delta = datetime.now(UTC) - datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    s = int(delta.total_seconds())
    return (
        f"{s // 86400}d{(s % 86400) // 3600}h"
        if s >= 86400
        else f"{s // 3600}h{(s % 3600) // 60}m"
        if s >= 3600
        else f"{s // 60}m{s % 60}s"
    )


def pod_health(pod: dict[str, Any]) -> dict[str, Any]:
    """Summarise a pod (raw API dict) and list concrete problems."""
    meta, spec, status = pod.get("metadata", {}), pod.get("spec", {}), pod.get("status", {})
    problems: list[str] = []
    containers = []
    for cs in status.get("containerStatuses") or []:
        state = cs.get("state") or {}
        last = (cs.get("lastState") or {}).get("terminated")
        c: dict[str, Any] = {
            "name": cs.get("name"),
            "image": cs.get("image"),
            "ready": cs.get("ready", False),
            "restart_count": cs.get("restartCount", 0),
        }
        if "running" in state:
            c["state"] = "running"
            c["started_at"] = state["running"].get("startedAt")
        elif "waiting" in state:
            w = state["waiting"]
            c["state"] = f"waiting: {w.get('reason')}"
            c["state_message"] = redact((w.get("message") or "")[:300])
            if w.get("reason") in BAD_WAITING or w.get("reason") not in (None, "ContainerCreating", "PodInitializing"):
                problems.append(f"container {cs.get('name')} waiting: {w.get('reason')}")
        elif "terminated" in state:
            t = state["terminated"]
            c["state"] = f"terminated: {t.get('reason')} (exit {t.get('exitCode')})"
            problems.append(f"container {cs.get('name')} terminated: {t.get('reason')}")
        if last:
            c["last_termination"] = {
                "reason": last.get("reason"),
                "exit_code": last.get("exitCode"),
                "finished_at": last.get("finishedAt"),
            }
            if last.get("reason") == "OOMKilled":
                problems.append(f"container {cs.get('name')} was OOMKilled (memory limit too low)")
        if c["restart_count"]:
            problems.append(f"container {cs.get('name')} restarted {c['restart_count']} time(s)")
        if "running" in state and not c["ready"]:
            problems.append(f"container {cs.get('name')} running but not ready (readiness probe failing)")
        containers.append(c)
    phase = status.get("phase")
    if phase == "Pending":
        for cond in status.get("conditions") or []:
            if cond.get("type") == "PodScheduled" and cond.get("status") == "False":
                problems.append(f"unschedulable: {cond.get('reason')} {cond.get('message', '')[:200]}")
    if phase in ("Failed", "Unknown"):
        problems.append(f"pod phase {phase}: {status.get('reason')}")
    ready = any(c.get("type") == "Ready" and c.get("status") == "True" for c in status.get("conditions") or [])
    return {
        "name": meta.get("name"),
        "phase": phase,
        "ready": ready,
        "restarts": sum(c["restart_count"] for c in containers),
        "age": _age(meta.get("creationTimestamp")),
        "node": spec.get("nodeName"),
        "pod_ip": status.get("podIP"),
        "version_label": (meta.get("labels") or {}).get("app.kubernetes.io/version"),
        "containers": containers,
        "problems": problems,
        "healthy": ready and not problems,
    }


class KubernetesTools:
    def __init__(self, settings: Settings | None = None, core=None, apps=None, discovery=None) -> None:
        self.settings = settings or get_settings()
        self._core, self._apps, self._discovery = core, apps, discovery

    # ----- client plumbing
    def _connect(self) -> None:
        try:
            from kubernetes import client, config
        except ImportError as exc:
            raise ToolUnavailableError("kubernetes Python client not installed") from exc
        try:
            if self.settings.k8s_kubeconfig:
                from pathlib import Path

                from agent.config import PROJECT_ROOT

                path = Path(self.settings.k8s_kubeconfig).expanduser()
                path = path if path.is_absolute() else PROJECT_ROOT / path
                api = config.new_client_from_config(config_file=str(path), context=self.settings.k8s_context)
            else:
                try:
                    config.load_incluster_config()
                    api = client.ApiClient()
                except config.ConfigException:
                    api = config.new_client_from_config(context=self.settings.k8s_context)
        except Exception as exc:  # noqa: BLE001
            raise ToolUnavailableError(f"no usable Kubernetes configuration: {type(exc).__name__}") from exc
        self._core = client.CoreV1Api(api)
        self._apps = client.AppsV1Api(api)
        self._discovery = client.DiscoveryV1Api(api)

    @property
    def core(self):
        if self._core is None:
            self._connect()
        return self._core

    @property
    def apps(self):
        if self._apps is None:
            self._connect()
        return self._apps

    @property
    def discovery(self):
        if self._discovery is None:
            self._connect()
        return self._discovery

    def _call(self, fn, *args, raw_text: bool = False, **kwargs):
        """Call the API with raw JSON (no model deserialisation) and map failures to typed errors."""
        kwargs.setdefault("_preload_content", False)
        kwargs.setdefault("_request_timeout", self.settings.k8s_timeout_s)
        try:
            resp = fn(*args, **kwargs)
        except ToolError:
            raise
        except Exception as exc:  # noqa: BLE001
            status = getattr(exc, "status", None)
            name = type(exc).__name__
            body = redact(str(getattr(exc, "body", "") or exc)[:300])
            if status == 404:
                raise ToolInvalidRequestError(f"Kubernetes: not found ({body})") from exc
            if status in (401, 403):
                raise ToolForbiddenError(f"Kubernetes RBAC denied the request: {body}") from exc
            if status is not None and status >= 500:
                raise ToolUnavailableError(f"Kubernetes API error HTTP {status}") from exc
            if "Timeout" in name or "timed out" in str(exc).lower():
                raise ToolTimeoutError("Kubernetes API request timed out") from exc
            if any(
                s in f"{name} {exc}"
                for s in ("MaxRetryError", "NewConnectionError", "Connection refused", "ConnectionError")
            ):
                raise ToolUnavailableError("Kubernetes API server is unreachable") from exc
            raise ToolError(f"Kubernetes API error: {name}: {body}") from exc
        data = resp.data if hasattr(resp, "data") else resp
        if isinstance(data, bytes):
            data = data.decode("utf-8", "replace")
        if raw_text:
            return data
        return json.loads(data) if isinstance(data, str) else data

    def _namespace(self, ns: str | None) -> str:
        ns = ns or self.settings.k8s_default_namespace
        if ns not in self.settings.k8s_approved_namespaces:
            raise ToolForbiddenError(
                f"namespace '{ns}' is not approved for this agent "
                f"(approved: {list(self.settings.k8s_approved_namespaces)})"
            )
        return ns

    def _pods(self, ns: str, selector: str | None = None) -> list[dict[str, Any]]:
        kw = {"label_selector": selector} if selector else {}
        return self._call(self.core.list_namespaced_pod, ns, **kw).get("items", [])

    def _events(
        self, ns: str, object_name: str | None = None, warnings_only: bool = False, limit: int = 30
    ) -> list[dict[str, Any]]:
        kw = {"field_selector": f"involvedObject.name={object_name}"} if object_name else {}
        items = self._call(self.core.list_namespaced_event, ns, **kw).get("items", [])
        out = []
        for e in items:
            if warnings_only and e.get("type") != "Warning":
                continue
            obj = e.get("involvedObject") or {}
            when = e.get("lastTimestamp") or e.get("eventTime") or (e.get("metadata") or {}).get("creationTimestamp")
            out.append(
                {
                    "type": e.get("type"),
                    "reason": e.get("reason"),
                    "object": f"{obj.get('kind')}/{obj.get('name')}",
                    "message": redact((e.get("message") or "")[:300]),
                    "count": e.get("count") or 1,
                    "last_seen": when,
                }
            )
        out.sort(key=lambda e: e["last_seen"] or "", reverse=True)
        return out[:limit]

    def _logs(self, ns: str, pod: str, tail: int, previous: bool = False, container: str | None = None) -> list[str]:
        kw: dict[str, Any] = {"tail_lines": tail, "previous": previous}
        if container:
            kw["container"] = container
        text = self._call(self.core.read_namespaced_pod_log, pod, ns, raw_text=True, **kw)
        return [redact(ln) for ln in (text or "").splitlines() if ln.strip()]

    def _endpoints(self, ns: str, svc: str) -> dict[str, Any]:
        slices = self._call(
            self.discovery.list_namespaced_endpoint_slice, ns, label_selector=f"kubernetes.io/service-name={svc}"
        ).get("items", [])
        ready, not_ready = [], []
        for s in slices:
            for ep in s.get("endpoints") or []:
                target = (ep.get("targetRef") or {}).get("name") or ",".join(ep.get("addresses", []))
                (ready if (ep.get("conditions") or {}).get("ready") else not_ready).append(target)
        return {"ready": ready, "not_ready": not_ready}

    def _health_probe(self, ns: str, svc: str, port_name: str, path: str) -> dict[str, Any]:
        try:
            body = self._call(
                self.core.connect_get_namespaced_service_proxy_with_path, f"{svc}:{port_name}", ns, path, raw_text=True
            )
            try:
                body = json.loads(body)
            except (TypeError, ValueError):
                pass
            return {"path": f"/{path}", "ok": True, "response": body}
        except ToolError as exc:
            return {"path": f"/{path}", "ok": False, "error": str(exc)[:200]}

    def _deployment(self, ns: str, name: str) -> dict[str, Any]:
        return self._call(self.apps.read_namespaced_deployment, name, ns)

    @staticmethod
    def _deployment_summary(d: dict[str, Any]) -> dict[str, Any]:
        meta, spec, st = d.get("metadata", {}), d.get("spec", {}), d.get("status", {})
        conds = {c.get("type"): c for c in st.get("conditions") or []}
        prog = conds.get("Progressing", {})
        containers = spec.get("template", {}).get("spec", {}).get("containers", [])
        desired = spec.get("replicas", 0)
        return {
            "name": meta.get("name"),
            "desired_replicas": desired,
            "ready_replicas": st.get("readyReplicas", 0),
            "updated_replicas": st.get("updatedReplicas", 0),
            "available_replicas": st.get("availableReplicas", 0),
            "unavailable_replicas": st.get("unavailableReplicas", 0),
            "images": [c.get("image") for c in containers],
            "revision": (meta.get("annotations") or {}).get("deployment.kubernetes.io/revision"),
            "rollout": ("stalled: " + prog.get("reason", ""))
            if prog.get("status") == "False"
            else (
                "complete"
                if st.get("updatedReplicas", 0) == desired == st.get("availableReplicas", 0)
                and not st.get("unavailableReplicas")
                else "in progress / degraded"
            ),
            "available": (conds.get("Available") or {}).get("status") == "True",
        }

    def _check_mutable(self, ns: str, name: str) -> dict[str, Any]:
        dep = self._deployment(ns, name)
        key, _, val = self.settings.k8s_managed_label.partition("=")
        if (dep.get("metadata", {}).get("labels") or {}).get(key) != val:
            raise ToolForbiddenError(f"deployment '{name}' is not managed by this agent (missing label {key}={val})")
        return dep

    # ----- read tools
    def list_pods(self, p: ListPodsInput) -> ToolOutput:
        ns = self._namespace(p.namespace)
        pods = [pod_health(x) for x in self._pods(ns, p.label_selector)]
        return ToolOutput(
            data={"namespace": ns, "count": len(pods), "unhealthy": sum(not x["healthy"] for x in pods), "pods": pods},
            evidence_source=[f"GET /api/v1/namespaces/{ns}/pods"],
        )

    def list_deployments(self, p: _NsInput) -> ToolOutput:
        ns = self._namespace(p.namespace)
        deps = [
            self._deployment_summary(d) for d in self._call(self.apps.list_namespaced_deployment, ns).get("items", [])
        ]
        return ToolOutput(
            data={"namespace": ns, "count": len(deps), "deployments": deps},
            evidence_source=[f"GET /apis/apps/v1/namespaces/{ns}/deployments"],
        )

    def get_pod_status(self, p: PodInput) -> ToolOutput:
        ns = self._namespace(p.namespace)
        pod = self._call(self.core.read_namespaced_pod, p.pod_name, ns)
        health = pod_health(pod)
        probes = {}
        for c in pod.get("spec", {}).get("containers", []):
            probes[c["name"]] = {k: c.get(k) for k in ("readinessProbe", "livenessProbe", "startupProbe") if c.get(k)}
            health.setdefault("resources", {})[c["name"]] = c.get("resources")
        health["conditions"] = [
            {k: c.get(k) for k in ("type", "status", "reason", "message")}
            for c in pod.get("status", {}).get("conditions") or []
        ]
        health["probes"] = probes
        health["recent_events"] = self._events(ns, p.pod_name, limit=10)
        return ToolOutput(
            data={"namespace": ns, **health},
            evidence_source=[f"GET pods/{p.pod_name}", f"GET events (involvedObject={p.pod_name})"],
        )

    def get_pod_logs(self, p: PodLogsInput) -> ToolOutput:
        ns = self._namespace(p.namespace)
        lines = self._logs(ns, p.pod_name, p.tail_lines, p.previous, p.container)
        return ToolOutput(
            data={"namespace": ns, "pod": p.pod_name, "previous": p.previous, "line_count": len(lines), "lines": lines},
            evidence_source=[f"GET pods/{p.pod_name}/log?tailLines={p.tail_lines}&previous={p.previous}"],
        )

    def list_services(self, p: _NsInput) -> ToolOutput:
        ns = self._namespace(p.namespace)
        out = []
        for s in self._call(self.core.list_namespaced_service, ns).get("items", []):
            name = s["metadata"]["name"]
            spec = s.get("spec", {})
            eps = self._endpoints(ns, name)
            out.append(
                {
                    "name": name,
                    "type": spec.get("type"),
                    "cluster_ip": spec.get("clusterIP"),
                    "ports": [
                        {"name": x.get("name"), "port": x.get("port"), "target_port": x.get("targetPort")}
                        for x in spec.get("ports", [])
                    ],
                    "selector": spec.get("selector"),
                    "ready_endpoints": len(eps["ready"]),
                    "not_ready_endpoints": len(eps["not_ready"]),
                }
            )
        return ToolOutput(
            data={"namespace": ns, "count": len(out), "services": out},
            evidence_source=[f"GET /api/v1/namespaces/{ns}/services", "GET endpointslices"],
        )

    def inspect_service(self, p: ServiceInput) -> ToolOutput:
        ns = self._namespace(p.namespace)
        svc = self._call(self.core.read_namespaced_service, p.service_name, ns)
        spec = svc.get("spec", {})
        selector = spec.get("selector") or {}
        sel = ",".join(f"{k}={v}" for k, v in selector.items())
        pods = [pod_health(x) for x in self._pods(ns, sel)] if sel else []
        eps = self._endpoints(ns, p.service_name)
        data: dict[str, Any] = {
            "namespace": ns,
            "name": p.service_name,
            "type": spec.get("type"),
            "cluster_ip": spec.get("clusterIP"),
            "ports": spec.get("ports"),
            "selector": selector,
            "matching_pods": len(pods),
            "ready_pods": sum(x["ready"] for x in pods),
            "endpoints": eps,
        }
        sources = [f"GET services/{p.service_name}", "GET endpointslices", f"GET pods?labelSelector={sel}"]
        if p.check_health:
            port = (spec.get("ports") or [{}])[0].get("name") or str((spec.get("ports") or [{}])[0].get("port"))
            data["health_check"] = self._health_probe(ns, p.service_name, port, "health/ready")
            sources.append("GET services/proxy/health/ready")
        data["serving"] = bool(eps["ready"]) and data.get("health_check", {}).get("ok", True)
        return ToolOutput(data=data, evidence_source=sources)

    def find_unhealthy_pods(self, p: _NsInput) -> ToolOutput:
        ns = self._namespace(p.namespace)
        pods = [pod_health(x) for x in self._pods(ns)]
        bad = [x for x in pods if not x["healthy"]]
        restart_events = [
            e
            for e in self._events(ns, limit=200)
            if e["reason"] in ("BackOff", "Killing", "Unhealthy", "OOMKilling", "Failed", "FailedScheduling")
        ]
        return ToolOutput(
            data={
                "namespace": ns,
                "total_pods": len(pods),
                "unhealthy_count": len(bad),
                "unhealthy_pods": bad,
                "pods_with_restarts": [{"name": x["name"], "restarts": x["restarts"]} for x in pods if x["restarts"]],
                "restart_and_probe_events": restart_events[:20],
            },
            evidence_source=[f"GET pods (ns={ns})", f"GET events (ns={ns})"],
        )

    def get_events(self, p: EventsInput) -> ToolOutput:
        ns = self._namespace(p.namespace)
        events = self._events(ns, p.object_name, p.warnings_only, p.limit)
        return ToolOutput(
            data={"namespace": ns, "count": len(events), "events": events},
            evidence_source=[f"GET events (ns={ns}, object={p.object_name})"],
        )

    def get_deployed_version(self, p: DeploymentInput) -> ToolOutput:
        ns = self._namespace(p.namespace)
        dep = self._deployment(ns, p.deployment_name)
        meta = dep.get("metadata", {})
        ann = meta.get("annotations") or {}
        pods = [pod_health(x) for x in self._pods(ns, f"app.kubernetes.io/name={p.deployment_name}")]
        data: dict[str, Any] = {
            "namespace": ns,
            **self._deployment_summary(dep),
            "app_version": (meta.get("labels") or {}).get("app.kubernetes.io/version"),
            "model_lineage": {k.split("/", 1)[1]: v for k, v in ann.items() if k.startswith("mlops.io/")},
            "change_cause": ann.get("kubernetes.io/change-cause"),
            "pod_versions": sorted({f"{x['version_label']} ({c['image']})" for x in pods for c in x["containers"]}),
        }
        sources = [f"GET deployments/{p.deployment_name}", "GET pods"]
        live = self._health_probe(ns, p.deployment_name, "http", "version")
        data["live_version_endpoint"] = live
        if live.get("ok"):
            sources.append("GET services/proxy/version")
            model = (live.get("response") or {}).get("model", {}) if isinstance(live.get("response"), dict) else {}
            declared = data["model_lineage"].get("model-version")
            if declared and model.get("model_version") and str(declared) != str(model.get("model_version")):
                data["lineage_mismatch"] = (
                    f"deployment annotation says model v{declared} but the running "
                    f"service reports v{model.get('model_version')}"
                )
        return ToolOutput(data=data, evidence_source=sources)

    def diagnose_service(self, p: ServiceInput) -> ToolOutput:
        ns = self._namespace(p.namespace)
        findings: list[dict[str, Any]] = []
        sources: list[str] = []
        svc = self._call(self.core.read_namespaced_service, p.service_name, ns)
        sources.append(f"GET services/{p.service_name}")
        selector = svc.get("spec", {}).get("selector") or {}
        sel = ",".join(f"{k}={v}" for k, v in selector.items())
        eps = self._endpoints(ns, p.service_name)
        sources.append("GET endpointslices")
        raw_pods = self._pods(ns, sel) if sel else []
        pods = [pod_health(x) for x in raw_pods]
        sources.append(f"GET pods?labelSelector={sel}")

        port = (svc.get("spec", {}).get("ports") or [{}])[0].get("name") or "80"
        health = self._health_probe(ns, p.service_name, port, "health/ready")
        sources.append("GET services/proxy/health/ready")

        deployments = [
            d
            for d in self._call(self.apps.list_namespaced_deployment, ns).get("items", [])
            if all(
                (d.get("spec", {}).get("selector", {}).get("matchLabels") or {}).get(k) == v
                for k, v in selector.items()
            )
        ]
        dep_summaries = [self._deployment_summary(d) for d in deployments]
        sources.append("GET deployments")

        if not raw_pods:
            findings.append(
                {
                    "layer": "configuration",
                    "issue": "service selector matches no pods",
                    "evidence": [f"selector {sel or '(none)'} matched 0 pods"],
                    "action": "check the Deployment exists and its pod labels match the Service selector",
                }
            )

        warn_events = self._events(ns, warnings_only=True, limit=100)
        sources.append("GET events (warnings)")
        seen_issue: set[str] = set()
        for pod_raw, ph in zip(raw_pods, pods, strict=True):
            if ph["healthy"]:
                continue
            pod_events = [e for e in warn_events if e["object"].endswith("/" + ph["name"])]
            for c in ph["containers"]:
                reason = c.get("state", "").removeprefix("waiting: ")
                if reason in BAD_WAITING:
                    label, why, action = BAD_WAITING[reason]
                    key = f"{label}:{c.get('image')}"
                    if key in seen_issue:
                        continue
                    seen_issue.add(key)
                    ev = [f"pod {ph['name']} container {c['name']}: {reason} - {c.get('state_message', '')}"]
                    ev += [f"event {e['reason']}: {e['message']}" for e in pod_events[:3]]
                    if reason == "CrashLoopBackOff":
                        try:
                            ev += ["previous log: " + ln for ln in self._logs(ns, ph["name"], 15, previous=True)[-6:]]
                            sources.append(f"GET pods/{ph['name']}/log?previous=true")
                        except ToolError:
                            pass
                    findings.append(
                        {
                            "layer": "deployment" if "image" in label else "application",
                            "issue": f"{label}: {why}",
                            "image": c.get("image"),
                            "evidence": ev,
                            "action": action,
                        }
                    )
                last = c.get("last_termination") or {}
                if last.get("reason") == "OOMKilled" and "oom" not in seen_issue:
                    seen_issue.add("oom")
                    limits = (pod_raw.get("spec", {}).get("containers") or [{}])[0].get("resources", {}).get("limits")
                    findings.append(
                        {
                            "layer": "infrastructure",
                            "issue": "container killed for exceeding its memory limit",
                            "evidence": [f"pod {ph['name']} last termination OOMKilled", f"limits: {limits}"],
                            "action": "raise the memory limit or reduce model/batch memory use",
                        }
                    )
                if c.get("state") == "running" and not c.get("ready") and "readiness" not in seen_issue:
                    seen_issue.add("readiness")
                    probe_events = [
                        f"event {e['reason']}: {e['message']}" for e in pod_events if e["reason"] == "Unhealthy"
                    ][:3]
                    log_lines = []
                    try:
                        log_lines = [
                            ln
                            for ln in self._logs(ns, ph["name"], 60)
                            if re.search(r"ERROR|FAILED|Error|Traceback|not found", ln)
                        ][-5:]
                        sources.append(f"GET pods/{ph['name']}/log")
                    except ToolError:
                        pass
                    cm_refs = [
                        e["configMapRef"]["name"]
                        for e in (pod_raw.get("spec", {}).get("containers") or [{}])[0].get("envFrom", []) or []
                        if e.get("configMapRef")
                    ]
                    config = {}
                    for cm in cm_refs:
                        try:
                            config[cm] = self._call(self.core.read_namespaced_config_map, cm, ns).get("data", {})
                            sources.append(f"GET configmaps/{cm}")
                        except ToolError:
                            config[cm] = "unreadable"
                    model_issue = any("MODEL_LOAD_FAILED" in ln for ln in log_lines)
                    findings.append(
                        {
                            "layer": "application" if model_issue else "health check",
                            "issue": (
                                "readiness probe failing because the model failed to load"
                                if model_issue
                                else "readiness probe failing"
                            ),
                            "evidence": probe_events + [f"log: {ln}" for ln in log_lines],
                            "config_in_use": config,
                            "action": (
                                "fix the model path/config (see config_in_use and logs), then restart the deployment"
                                if model_issue
                                else "check the readiness endpoint and application logs"
                            ),
                        }
                    )
            if (
                ph["phase"] == "Pending"
                and any(pr.startswith("unschedulable") for pr in ph["problems"])
                and "sched" not in seen_issue
            ):
                seen_issue.add("sched")
                findings.append(
                    {
                        "layer": "infrastructure",
                        "issue": "pods cannot be scheduled",
                        "evidence": [pr for pr in ph["problems"] if pr.startswith("unschedulable")],
                        "action": "lower resource requests or add cluster capacity",
                    }
                )

        for d in dep_summaries:
            if d["rollout"].startswith("stalled"):
                findings.append(
                    {
                        "layer": "deployment",
                        "issue": f"rollout of {d['name']} {d['rollout']}",
                        "evidence": [f"images {d['images']}, revision {d['revision']}"],
                        "action": "roll back to the previous revision or fix the new pod template",
                    }
                )

        ready_n = len(eps["ready"])
        status = "available" if ready_n and health.get("ok") else "degraded" if ready_n else "unavailable"
        if status == "available" and not findings:
            summary = f"service is healthy: {ready_n} ready endpoint(s), readiness check passed"
        elif status == "available":
            summary = (
                f"service is still serving from {ready_n} ready pod(s), but there are problems that "
                f"need attention (e.g. a stuck rollout)"
            )
        elif status == "degraded":
            summary = "service has ready endpoints but the health check through the API server failed"
        else:
            root = findings[0]["issue"] if findings else "no ready endpoints"
            summary = f"service is UNAVAILABLE: 0 ready endpoints. Most likely cause: {root}"
        return ToolOutput(
            data={
                "namespace": ns,
                "service": p.service_name,
                "status": status,
                "summary": summary,
                "ready_endpoints": eps["ready"],
                "not_ready_endpoints": eps["not_ready"],
                "health_check": health,
                "deployments": dep_summaries,
                "pods": [{k: x[k] for k in ("name", "phase", "ready", "restarts", "problems")} for x in pods],
                "findings": findings,
                "recent_warning_events": warn_events[:8],
            },
            evidence_source=sources,
        )

    # ----- state-changing tools (confirmation enforced by the registry)
    def restart_deployment(self, p: DeploymentInput) -> ToolOutput:
        ns = self._namespace(p.namespace)
        self._check_mutable(ns, p.deployment_name)
        now = datetime.now(UTC).isoformat()
        body = {"spec": {"template": {"metadata": {"annotations": {"kubectl.kubernetes.io/restartedAt": now}}}}}
        self._call(self.apps.patch_namespaced_deployment, p.deployment_name, ns, body)
        return ToolOutput(
            data={
                "namespace": ns,
                "deployment": p.deployment_name,
                "action": "rolling restart",
                "restarted_at": now,
                "note": "pods are replaced one at a time (maxUnavailable=0); check status with list_pods",
            },
            evidence_source=[f"PATCH deployments/{p.deployment_name} (restartedAt annotation)"],
        )

    def scale_deployment(self, p: ScaleInput) -> ToolOutput:
        ns = self._namespace(p.namespace)
        if not 1 <= p.replicas <= self.settings.k8s_max_replicas:
            raise ToolForbiddenError(f"replicas must be between 1 and {self.settings.k8s_max_replicas} for this agent")
        dep = self._check_mutable(ns, p.deployment_name)
        before = dep.get("spec", {}).get("replicas")
        self._call(self.apps.patch_namespaced_deployment, p.deployment_name, ns, {"spec": {"replicas": p.replicas}})
        return ToolOutput(
            data={
                "namespace": ns,
                "deployment": p.deployment_name,
                "replicas_before": before,
                "replicas_after": p.replicas,
            },
            evidence_source=[f"PATCH deployments/{p.deployment_name} spec.replicas"],
        )

    def rollback_deployment(self, p: RollbackInput) -> ToolOutput:
        ns = self._namespace(p.namespace)
        dep = self._check_mutable(ns, p.deployment_name)
        uid = dep["metadata"]["uid"]
        current = int((dep["metadata"].get("annotations") or {}).get("deployment.kubernetes.io/revision", "0"))
        match = dep["spec"]["selector"]["matchLabels"]
        rss = self._call(
            self.apps.list_namespaced_replica_set, ns, label_selector=",".join(f"{k}={v}" for k, v in match.items())
        ).get("items", [])
        owned = {}
        for rs in rss:
            if any(o.get("uid") == uid for o in rs["metadata"].get("ownerReferences") or []):
                rev = int((rs["metadata"].get("annotations") or {}).get("deployment.kubernetes.io/revision", "0"))
                owned[rev] = rs
        target = p.to_revision or max((r for r in owned if r < current), default=None)
        if target is None or target not in owned:
            raise ToolInvalidRequestError(f"no revision to roll back to (current {current}, available {sorted(owned)})")
        if target == current:
            raise ToolInvalidRequestError(f"revision {target} is already the current revision")
        template = owned[target]["spec"]["template"]
        template.get("metadata", {}).get("labels", {}).pop("pod-template-hash", None)
        images = [c.get("image") for c in template["spec"]["containers"]]
        body = {
            "metadata": {
                "annotations": {"kubernetes.io/change-cause": f"rollback to revision {target} by MLOps agent"}
            },
            "spec": {"template": template},
        }
        self._call(self.apps.patch_namespaced_deployment, p.deployment_name, ns, body)
        return ToolOutput(
            data={
                "namespace": ns,
                "deployment": p.deployment_name,
                "from_revision": current,
                "to_revision": target,
                "restored_images": images,
            },
            evidence_source=[f"GET replicasets (owner {p.deployment_name})", f"PATCH deployments/{p.deployment_name}"],
        )


def register_kubernetes_tools(registry: ToolRegistry, tools: KubernetesTools | None = None) -> KubernetesTools:
    t = tools or KubernetesTools()
    specs = [
        (
            "list_pods",
            ListPodsInput,
            t.list_pods,
            False,
            "List application pods with phase, readiness, restart counts and detected problems.",
        ),
        (
            "list_deployments",
            _NsInput,
            t.list_deployments,
            False,
            "List deployments with desired/ready/updated/available replicas, images, revision and rollout state.",
        ),
        (
            "get_pod_status",
            PodInput,
            t.get_pod_status,
            False,
            "Detailed status of one pod: container states, last termination (e.g. OOMKilled), probes, resources, "
            "conditions and recent events.",
        ),
        (
            "get_pod_logs",
            PodLogsInput,
            t.get_pod_logs,
            False,
            "Recent logs of a pod's container; set previous=true for the crashed instance's logs.",
        ),
        (
            "list_services",
            _NsInput,
            t.list_services,
            False,
            "List services with type, ports, selector and ready/not-ready endpoint counts.",
        ),
        (
            "inspect_service",
            ServiceInput,
            t.inspect_service,
            False,
            "Inspect one service: selector, endpoints, matching pods, and a live readiness check through the "
            "API server.",
        ),
        (
            "find_unhealthy_pods",
            _NsInput,
            t.find_unhealthy_pods,
            False,
            "Find not-ready, crash-looping, OOM-killed or image-pull-failing pods, pods with restarts, and "
            "restart/probe-failure events.",
        ),
        (
            "get_events",
            EventsInput,
            t.get_events,
            False,
            "Kubernetes events for the namespace or one object (newest first; optionally warnings only).",
        ),
        (
            "get_deployed_version",
            DeploymentInput,
            t.get_deployed_version,
            False,
            "Report the currently deployed application version: image, version label, model lineage annotations "
            "(model version, dataset version), rollout revision, and the live /version response of the service.",
        ),
        (
            "diagnose_service",
            ServiceInput,
            t.diagnose_service,
            False,
            "Answer 'why is my prediction API unavailable?': correlates service endpoints, pods, probe failures, "
            "events, logs and config into an evidence-based diagnosis with recommended actions.",
        ),
        (
            "restart_deployment",
            DeploymentInput,
            t.restart_deployment,
            True,
            "STATE-CHANGING: rolling restart of a deployment. Requires explicit user confirmation; approved "
            "namespace and agent-managed deployments only.",
        ),
        (
            "scale_deployment",
            ScaleInput,
            t.scale_deployment,
            True,
            "STATE-CHANGING: set a deployment's replica count (bounded). Requires explicit user confirmation.",
        ),
        (
            "rollback_deployment",
            RollbackInput,
            t.rollback_deployment,
            True,
            "STATE-CHANGING: roll a deployment back to a previous revision (default: the previous one). "
            "Requires explicit user confirmation.",
        ),
    ]
    for name, model, func, mutating, desc in specs:
        registry.register(
            Tool(
                name=name,
                module=MODULE,
                description=desc,
                input_model=model,
                func=func,
                mutating=mutating,
                requires_confirmation=mutating,
            )
        )
    return t
