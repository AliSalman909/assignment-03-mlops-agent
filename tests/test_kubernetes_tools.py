"""Tests for the Kubernetes operations module (Task 4) against fake API objects (no cluster needed)."""

from __future__ import annotations

import copy

import pytest

from agent.config import Settings
from agent.schemas import ToolStatus
from agent.tools.base import ToolRegistry
from agent.tools.kubernetes_tools import KubernetesTools, pod_health, register_kubernetes_tools

NS = "mlops"
MANAGED = {"app.kubernetes.io/name": "churn-api", "app.kubernetes.io/part-of": "mlops-agent"}


def pod(name, *, ready=True, waiting=None, restarts=0, last=None, phase="Running"):
    state = (
        {"waiting": {"reason": waiting, "message": f"{waiting} details"}}
        if waiting
        else {"running": {"startedAt": "x"}}
    )
    cs = {"name": "churn-api", "image": "churn-api:1.0.0", "ready": ready, "restartCount": restarts, "state": state}
    if last:
        cs["lastState"] = {"terminated": {"reason": last, "exitCode": 137}}
    return {
        "metadata": {
            "name": name,
            "labels": {"app.kubernetes.io/name": "churn-api", "app.kubernetes.io/version": "1.0.0"},
            "creationTimestamp": "2026-11-15T10:00:00Z",
        },
        "spec": {
            "containers": [
                {
                    "name": "churn-api",
                    "envFrom": [{"configMapRef": {"name": "churn-api-config"}}],
                    "resources": {"limits": {"memory": "512Mi"}},
                }
            ]
        },
        "status": {
            "phase": phase,
            "podIP": "10.0.0.1",
            "containerStatuses": [cs],
            "conditions": [{"type": "Ready", "status": "True" if ready else "False"}],
        },
    }


class ApiErr(Exception):
    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.status = status


class FakeCluster:
    def __init__(self, pods, *, ready_eps=None, health_ok=True, logs="", events=None, labels=MANAGED):
        self.pods = pods
        self.ready_eps = ready_eps if ready_eps is not None else [p["metadata"]["name"] for p in pods]
        self.health_ok = health_ok
        self.logs = logs
        self.events = events or []
        self.patches: list[tuple] = []
        self.deployment = {
            "metadata": {
                "name": "churn-api",
                "uid": "dep-uid",
                "labels": dict(labels),
                "annotations": {"deployment.kubernetes.io/revision": "3", "mlops.io/model-version": "1"},
            },
            "spec": {
                "replicas": 2,
                "selector": {"matchLabels": {"app.kubernetes.io/name": "churn-api"}},
                "template": {"spec": {"containers": [{"name": "churn-api", "image": "churn-api:1.1.0"}]}},
            },
            "status": {
                "readyReplicas": 2,
                "updatedReplicas": 1,
                "availableReplicas": 2,
                "conditions": [{"type": "Available", "status": "True"}],
            },
        }

    # CoreV1Api
    def list_namespaced_pod(self, ns, label_selector=None, **kw):
        return {"items": self.pods}

    def read_namespaced_pod(self, name, ns, **kw):
        return next(p for p in self.pods if p["metadata"]["name"] == name)

    def read_namespaced_pod_log(self, name, ns, **kw):
        return self.logs

    def list_namespaced_event(self, ns, **kw):
        return {"items": self.events}

    def read_namespaced_service(self, name, ns, **kw):
        if name != "churn-api":
            raise ApiErr(404)
        return {
            "metadata": {"name": name},
            "spec": {
                "type": "ClusterIP",
                "clusterIP": "10.96.0.1",
                "selector": {"app.kubernetes.io/name": "churn-api"},
                "ports": [{"name": "http", "port": 80, "targetPort": "http"}],
            },
        }

    def list_namespaced_service(self, ns, **kw):
        return {"items": [self.read_namespaced_service("churn-api", ns)]}

    def read_namespaced_config_map(self, name, ns, **kw):
        return {"data": {"MODEL_DIR": "/app/models/churn-v2", "LOG_LEVEL": "INFO"}}

    def connect_get_namespaced_service_proxy_with_path(self, name, ns, path, **kw):
        if not self.health_ok:
            raise ApiErr(503)
        return (
            '{"status": "ready"}' if "health" in path else '{"app_version": "1.0.0", "model": {"model_version": "1"}}'
        )

    # DiscoveryV1Api
    def list_namespaced_endpoint_slice(self, ns, **kw):
        eps = [{"targetRef": {"name": n}, "conditions": {"ready": True}} for n in self.ready_eps]
        eps += [
            {"targetRef": {"name": p["metadata"]["name"]}, "conditions": {"ready": False}}
            for p in self.pods
            if p["metadata"]["name"] not in self.ready_eps
        ]
        return {"items": [{"endpoints": eps}]}

    # AppsV1Api
    def list_namespaced_deployment(self, ns, **kw):
        return {"items": [self.deployment]}

    def read_namespaced_deployment(self, name, ns, **kw):
        if name != "churn-api":
            raise ApiErr(404)
        return copy.deepcopy(self.deployment)

    def list_namespaced_replica_set(self, ns, **kw):
        def rs(rev, image):
            return {
                "metadata": {
                    "annotations": {"deployment.kubernetes.io/revision": str(rev)},
                    "ownerReferences": [{"uid": "dep-uid"}],
                },
                "spec": {
                    "template": {
                        "metadata": {"labels": {"app.kubernetes.io/name": "churn-api", "pod-template-hash": "abc"}},
                        "spec": {"containers": [{"name": "churn-api", "image": image}]},
                    }
                },
            }

        return {"items": [rs(1, "churn-api:0.9.0"), rs(2, "churn-api:1.0.0"), rs(3, "churn-api:1.1.0")]}

    def patch_namespaced_deployment(self, name, ns, body, **kw):
        self.patches.append((name, ns, body))
        return {}


def make(cluster: FakeCluster, **settings_kw) -> ToolRegistry:
    s = Settings(k8s_approved_namespaces=(NS,), k8s_default_namespace=NS, **settings_kw)
    reg = ToolRegistry()
    register_kubernetes_tools(reg, KubernetesTools(settings=s, core=cluster, apps=cluster, discovery=cluster))
    return reg


# ------------------------------------------------------------------ health detection
@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"ready": False, "waiting": "CrashLoopBackOff", "restarts": 4}, "CrashLoopBackOff"),
        ({"ready": False, "waiting": "ImagePullBackOff", "phase": "Pending"}, "ImagePullBackOff"),
        ({"restarts": 1, "last": "OOMKilled"}, "OOMKilled"),
        ({"ready": False}, "readiness probe failing"),
    ],
)
def test_pod_health_detects_problems(kwargs, expected):
    h = pod_health(pod("p", **kwargs))
    assert not h["healthy"]
    assert any(expected in pr for pr in h["problems"])


def test_find_unhealthy_pods_and_restarts():
    c = FakeCluster(
        [pod("good"), pod("bad", ready=False, waiting="CrashLoopBackOff", restarts=5)],
        events=[
            {
                "type": "Warning",
                "reason": "BackOff",
                "message": "Back-off restarting failed container",
                "involvedObject": {"kind": "Pod", "name": "bad"},
                "lastTimestamp": "2026-11-15T10:01:00Z",
            }
        ],
    )
    res = make(c).execute("find_unhealthy_pods", {})
    assert res.ok, res.error
    assert [p["name"] for p in res.data["unhealthy_pods"]] == ["bad"]
    assert res.data["pods_with_restarts"] == [{"name": "bad", "restarts": 5}]
    assert res.data["restart_and_probe_events"][0]["reason"] == "BackOff"


# ------------------------------------------------------------------ diagnosis
def test_diagnose_unavailable_api_from_model_load_failure():
    c = FakeCluster(
        [pod("a", ready=False), pod("b", ready=False)],
        ready_eps=[],
        health_ok=False,
        logs='{"level":"ERROR","msg":"MODEL_LOAD_FAILED FileNotFoundError: model directory not found"}',
        events=[
            {
                "type": "Warning",
                "reason": "Unhealthy",
                "message": "Readiness probe failed: HTTP probe failed with statuscode: 503",
                "involvedObject": {"kind": "Pod", "name": "a"},
                "lastTimestamp": "2026-11-15T10:01:00Z",
            }
        ],
    )
    res = make(c).execute("diagnose_service", {"service_name": "churn-api"})
    assert res.ok, res.error
    assert res.data["status"] == "unavailable"
    f = res.data["findings"][0]
    assert f["layer"] == "application" and "model failed to load" in f["issue"]
    assert any("MODEL_LOAD_FAILED" in e for e in f["evidence"])
    assert f["config_in_use"]["churn-api-config"]["MODEL_DIR"] == "/app/models/churn-v2"


def test_diagnose_bad_image_rollout_while_still_serving():
    c = FakeCluster(
        [pod("old1"), pod("old2"), pod("new", ready=False, waiting="ImagePullBackOff", phase="Pending")],
        ready_eps=["old1", "old2"],
    )
    res = make(c).execute("diagnose_service", {"service_name": "churn-api"})
    assert res.data["status"] == "available"
    assert any("image pull failure" in f["issue"] for f in res.data["findings"])


def test_deployed_version_reports_image_and_lineage():
    res = make(FakeCluster([pod("a")])).execute("get_deployed_version", {})
    assert res.ok, res.error
    assert res.data["images"] == ["churn-api:1.1.0"] and res.data["model_lineage"]["model-version"] == "1"
    assert res.data["live_version_endpoint"]["ok"] is True


# ------------------------------------------------------------------ safety
def test_unauthorized_modifications_are_rejected_and_nothing_is_patched():
    c = FakeCluster([pod("a")])
    reg = make(c)
    assert reg.execute("restart_deployment", {}).status == ToolStatus.CONFIRMATION_REQUIRED
    other_ns = reg.execute(
        "restart_deployment", {"namespace": "kube-system", "deployment_name": "coredns"}, confirmed=True
    )
    assert other_ns.status == ToolStatus.FORBIDDEN
    assert reg.execute("scale_deployment", {"replicas": 40}, confirmed=True).status == ToolStatus.FORBIDDEN
    assert reg.execute("list_pods", {"namespace": "kube-system"}).status == ToolStatus.FORBIDDEN
    assert c.patches == []


def test_unmanaged_deployment_cannot_be_modified():
    c = FakeCluster([pod("a")], labels={"app.kubernetes.io/name": "churn-api"})  # no part-of label
    res = make(c).execute("restart_deployment", {}, confirmed=True)
    assert res.status == ToolStatus.FORBIDDEN and c.patches == []


def test_confirmed_restart_and_rollback_patch_the_deployment():
    c = FakeCluster([pod("a")])
    reg = make(c)
    assert reg.execute("restart_deployment", {}, confirmed=True).ok
    assert "kubectl.kubernetes.io/restartedAt" in str(c.patches[-1][2])
    rb = reg.execute("rollback_deployment", {}, confirmed=True)
    assert rb.ok, rb.error
    assert rb.data["from_revision"] == 3 and rb.data["to_revision"] == 2
    template = c.patches[-1][2]["spec"]["template"]
    assert template["spec"]["containers"][0]["image"] == "churn-api:1.0.0"
    assert "pod-template-hash" not in template["metadata"]["labels"]


# ------------------------------------------------------------------ failures
@pytest.mark.parametrize(
    "exc, status",
    [
        (ApiErr(403), ToolStatus.FORBIDDEN),
        (ConnectionError("MaxRetryError: Connection refused"), ToolStatus.UNAVAILABLE),
        (TimeoutError("ReadTimeoutError: timed out"), ToolStatus.TIMEOUT),
    ],
)
def test_api_failures_are_typed_and_carry_no_data(exc, status):
    class Broken(FakeCluster):
        def list_namespaced_pod(self, ns, **kw):
            raise exc

    res = make(Broken([])).execute("list_pods", {})
    assert res.status == status and res.data is None


def test_missing_service_is_reported_not_invented():
    res = make(FakeCluster([pod("a")])).execute("inspect_service", {"service_name": "no-such-svc"})
    assert res.status == ToolStatus.INVALID_REQUEST and res.data is None
