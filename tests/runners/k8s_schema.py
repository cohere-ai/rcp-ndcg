"""A kubectl-free structural check of rendered Kubernetes objects.

Each object is checked against the part of the Kubernetes API the runner writes: every key must be a field of its
object at that path (a typo such as ``volumeMount`` or ``startUpProbe`` fails), required fields must be present,
and leaves must have the API's types. The field names follow the Kubernetes API reference (batch/v1 Job, apps/v1
StatefulSet, v1 Service, core/v1 PodSpec and Container).
"""

from __future__ import annotations

from typing import Any

_STR, _INT, _BOOL = str, int, bool
_QUANTITY = (str, int, float)
_STR_MAP = {str: str}

_PROBE = {
    "httpGet": {"path": _STR, "port": (int, str)},
    "periodSeconds": _INT,
    "failureThreshold": _INT,
    "initialDelaySeconds": _INT,
    "timeoutSeconds": _INT,
}
_RESOURCES = {"requests": {str: _QUANTITY}, "limits": {str: _QUANTITY}}
_CONTAINER = {
    "name": _STR,
    "image": _STR,
    "command": [_STR],
    "args": [_STR],
    "env": [{"name": _STR, "value": _STR}],
    "envFrom": [{"secretRef": {"name": _STR}}],
    "ports": [{"containerPort": _INT, "name": _STR}],
    "resources": _RESOURCES,
    "volumeMounts": [{"name": _STR, "mountPath": _STR}],
    "startupProbe": _PROBE,
    "readinessProbe": _PROBE,
    "livenessProbe": _PROBE,
}
_POD = {
    "restartPolicy": _STR,
    "containers": [_CONTAINER],
    "volumes": [{"name": _STR, "emptyDir": {"medium": _STR}}],
    "serviceAccountName": _STR,
    "nodeSelector": _STR_MAP,
}
_OWNER = {"apiVersion": _STR, "kind": _STR, "name": _STR, "uid": _STR, "blockOwnerDeletion": _BOOL}
_METADATA = {"name": _STR, "namespace": _STR, "labels": _STR_MAP, "ownerReferences": [_OWNER]}
_TEMPLATE = {"metadata": {"labels": _STR_MAP}, "spec": _POD}

SCHEMAS: dict[tuple[str, str], dict[str, Any]] = {
    ("batch/v1", "Job"): {
        "apiVersion": _STR,
        "kind": _STR,
        "metadata": _METADATA,
        "spec": {
            "backoffLimit": _INT,
            "activeDeadlineSeconds": _INT,
            "ttlSecondsAfterFinished": _INT,
            "template": _TEMPLATE,
        },
    },
    ("apps/v1", "StatefulSet"): {
        "apiVersion": _STR,
        "kind": _STR,
        "metadata": _METADATA,
        "spec": {
            "serviceName": _STR,
            "replicas": _INT,
            "podManagementPolicy": _STR,
            "selector": {"matchLabels": _STR_MAP},
            "template": _TEMPLATE,
        },
    },
    ("v1", "Service"): {
        "apiVersion": _STR,
        "kind": _STR,
        "metadata": _METADATA,
        "spec": {
            "clusterIP": _STR,
            "publishNotReadyAddresses": _BOOL,
            "selector": _STR_MAP,
            "ports": [{"name": _STR, "port": _INT, "targetPort": (int, str)}],
        },
    },
}

#: Fields each object, pod and container must have.
_REQUIRED = {
    "object": ("apiVersion", "kind", "metadata", "spec"),
    "metadata": ("name",),
    "container": ("name", "image"),
}


def _check(value: Any, schema: Any, path: str) -> None:
    if isinstance(schema, list):
        assert isinstance(value, list), f"{path}: expected a list, got {type(value).__name__}"
        for index, item in enumerate(value):
            _check(item, schema[0], f"{path}[{index}]")
        return
    if isinstance(schema, dict) and set(schema) == {str}:
        assert isinstance(value, dict), f"{path}: expected a mapping"
        for key, item in value.items():
            assert isinstance(key, str), f"{path}: key {key!r} is not a string"
            _check(item, schema[str], f"{path}.{key}")
        return
    if isinstance(schema, dict):
        assert isinstance(value, dict), f"{path}: expected an object, got {type(value).__name__}"
        unknown = sorted(set(value) - set(schema))
        assert not unknown, f"{path}: unknown field(s) {unknown}"
        if "image" in schema and "name" in schema:
            missing = [key for key in _REQUIRED["container"] if key not in value]
            assert not missing, f"{path}: a container needs {missing}"
        for key, item in value.items():
            _check(item, schema[key], f"{path}.{key}")
        return
    types = schema if isinstance(schema, tuple) else (schema,)
    ok = isinstance(value, types) and not (isinstance(value, bool) and bool not in types)
    assert ok, f"{path}: expected {'/'.join(t.__name__ for t in types)}, got {value!r}"


def check_objects(objects: list[dict[str, Any]]) -> None:
    """Assert every object is a well-formed Job, StatefulSet or Service as far as the runner writes them."""
    for obj in objects:
        kind = (obj.get("apiVersion"), obj.get("kind"))
        assert kind in SCHEMAS, f"unexpected object {kind}"
        missing = [key for key in _REQUIRED["object"] if key not in obj]
        assert not missing, f"{kind}: missing {missing}"
        assert "name" in obj["metadata"], f"{kind}: metadata.name is required"
        _check(obj, SCHEMAS[kind], obj["kind"])
