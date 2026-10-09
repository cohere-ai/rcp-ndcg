"""The per-family reference environment lock (owner decision 35): one ``reference.lock`` per family.

A family's reference environment is built from the family's ``reference.in`` -- the short, justified list
of what its one ``reference.py`` needs -- resolved to exact, hashed pins by this tool.  The tool
**constrains torch and the CUDA stack to the engine image's freeze and nothing else**: a family whose
reference runs on the image's torch declares a floor (``torch>=2.0``) that the image's build must
satisfy, and the floor never reaches the lock as an install (the image's torch is read through the
reference venv's ``--system-site-packages``).  A family whose card or paper reference genuinely needs
another torch declares the per-family exception ``# own-torch: true`` with its evidence; only then does
the tool keep a torch pin in the lock and the bootstrap build a venv of its own for it.  Every other
family pin (transformers, sentence-transformers, flash-attn, ...) installs **into** the venv and takes
precedence over the image's copy -- that is the decision's whole point: one recipe's transformers pin
(e.g. a checkpoint's remote code needing an older release) can no longer break the wave, and the
environment is per family, not the wave's union.

The lock is a pip requirements file (install with ``--no-deps`` from the staged wheelhouse, then let
:mod:`rcp_ndcg_test.jobs.reference_deps` complete the venv's own dependencies), whose header records the
inputs the environment's identity is made of: the family, the engine image, the SHA-256 of the
``reference.in`` and of the image stack the lock was generated against, the ``own-torch`` declaration,
and the image constraints the family's floors were checked against.  Its SHA-256 **is** the environment
identity (the stored reference outputs key on it).

The tool is committed and run offline against a wheelhouse or online against an index::

    python -m rcp_ndcg_test.jobs.reference_lock --family <family> \\
        --in recipes/<family>/reference.in --out recipes/<family>/reference.lock \\
        --image-freeze rcp-ndcg-vllm/reference-image-v0.31.0.txt \\
        --image vllm/vllm-openai:v0.31.0

``check`` validates a committed lock against its ``reference.in`` and the image stack without any
network: the header hashes must match the committed inputs, every family requirement must appear as an
exact pin (or as an image constraint for the stack), and the own-torch declaration must hold.  That is
the CPU guard that a lock was not hand-edited; regeneration is deliberate.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from rcp_ndcg_test.errors import HarnessError

from .reference_deps import canonical, requirement_name, satisfies

__all__ = [
    "LOCK_SCHEMA",
    "STACK_NAMES",
    "ReferenceIn",
    "build_lock",
    "check_lock",
    "compile_requirements",
    "image_stack",
    "is_stack",
    "main",
    "parse_reference_in",
    "render_lock",
]

LOCK_SCHEMA = "rcp-reference-lock/1"
"""The lock format's version, in the header: a changed header or pin rule is a new schema."""

STACK_NAMES = frozenset({"torch", "torchvision", "torchaudio", "triton"})
"""The torch/CUDA-stack distributions the engine image owns: a family floor on one of these is checked
against the image's freeze and never installed; the ``nvidia-*`` distributions are the stack too
(:func:`is_stack`)."""

WORKSPACE_NAMES = frozenset({"rcp-ndcg", "rcp-ndcg-core", "rcp-ndcg-vllm", "rcp-ndcg-test"})
"""The project's own distributions: their pins stay verbatim (the staged wheelhouse carries the built
wheels, so there is no index to hash them against) and must be exact."""

_DEFAULT_UV = "uv"
_DEFAULT_INDEX = "https://pypi.org/simple"
_DEFAULT_PYTHON_VERSION = "3.12"

_OWN_TORCH_RE = re.compile(r"^#\s*own-torch:\s*(true|false)\s*$", re.IGNORECASE)
_OWN_TORCH_EVIDENCE_RE = re.compile(r"^#\s*own-torch-evidence:\s*(.+)$", re.IGNORECASE)
_HEADER_FIELDS = ("family", "image", "image-freeze-sha256", "reference-in-sha256", "own-torch")


def is_stack(name: str) -> bool:
    """Whether a distribution name belongs to the image's torch/CUDA stack (``torch``, ``torchvision``,
    ``torchaudio``, ``triton`` or an ``nvidia-*`` distribution)."""
    canonical_name = canonical(name)
    return canonical_name in STACK_NAMES or canonical_name.startswith("nvidia-")


@dataclass(frozen=True)
class ReferenceIn:
    """A family's ``reference.in``: its requirement lines and the own-torch declaration.

    Attributes:
        requirements: The requirement strings, in file order, comments stripped.
        own_torch: Whether the family declares its own torch (the exception, evidence required).
        evidence: The justification for ``own_torch`` (a card or reference-code fact with file:line).
    """

    requirements: tuple[str, ...]
    own_torch: bool
    evidence: str | None


def parse_reference_in(text: str, *, family: str) -> ReferenceIn:
    """Parse a family's ``reference.in``.

    Inputs: the file's text and the family name (for error messages).  Output: a :class:`ReferenceIn`.
    ``#`` lines are comments; the directives ``# own-torch: true|false`` (default false) and
    ``# own-torch-evidence: <text>`` declare the exception.  A trailing comment on a requirement line is
    stripped when preceded by whitespace.  Raises :class:`HarnessError` when ``own-torch: true`` has no
    evidence, so the exception can never ship unexplained.  Units: none.
    """
    requirements: list[str] = []
    own_torch = False
    evidence: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            match = _OWN_TORCH_RE.match(line)
            if match:
                own_torch = match.group(1).lower() == "true"
                continue
            match = _OWN_TORCH_EVIDENCE_RE.match(line)
            if match:
                evidence = match.group(1).strip()
                continue
            continue
        line = re.sub(r"\s+#.*$", "", line).strip()
        if line:
            requirements.append(line)
    if own_torch and not evidence:
        raise HarnessError(
            f"family {family}: reference.in declares own-torch: true without an own-torch-evidence line; "
            "the per-family exception needs its evidence (the card or reference-code fact, file:line)"
        )
    return ReferenceIn(requirements=tuple(requirements), own_torch=own_torch, evidence=evidence)


def image_stack(freeze_text: str) -> dict[str, str]:
    """The torch/CUDA-stack pins of an image's ``pip freeze``.

    Inputs: a freeze file's text (``name==version`` per line).  Output: ``{canonical name: version}`` for
    every stack distribution (:func:`is_stack`).  Units: none.
    """
    out: dict[str, str] = {}
    for raw in freeze_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        name, _, version = line.partition("==")
        if is_stack(name):
            out[canonical(name)] = version.split(";", 1)[0].strip()
    return out


def compile_requirements(
    requirements: list[str],
    *,
    find_links: tuple[str, ...] = (),
    index_url: str | None = _DEFAULT_INDEX,
    python_version: str = _DEFAULT_PYTHON_VERSION,
    uv: str = _DEFAULT_UV,
) -> str:
    """Resolve requirement strings to exact, hashed pins with ``uv pip compile --no-deps``.

    Inputs: the requirement strings (already stripped of the stack where the image owns it), an optional
    wheelhouse (``find_links``: ``--no-index`` is then used), the index URL (``None``: no index), the
    Python version the markers resolve for, and the uv executable.  Output: uv's compiled requirements
    text (pins with ``--hash=`` lines).  Raises :class:`HarnessError` when uv fails (its stderr is the
    message).  Units: none.
    """
    if not requirements:
        return ""
    with tempfile.TemporaryDirectory(prefix="rcp-reference-lock.") as work:
        source = Path(work) / "requirements.in"
        output = Path(work) / "requirements.lock"
        source.write_text("\n".join(requirements) + "\n", encoding="utf-8")
        argv = [
            uv,
            "pip",
            "compile",
            "--no-deps",
            "--generate-hashes",
            "--python-version",
            python_version,
            "--python-platform",
            "linux",
            "--output-file",
            str(output),
        ]
        if find_links:
            argv.append("--no-index")
            argv.extend(f"--find-links={path}" for path in find_links)
        if index_url is not None:
            argv.extend(["--index-url", index_url])
        argv.append(str(source))
        completed = subprocess.run(argv, capture_output=True, text=True, timeout=1800)
        if completed.returncode != 0:
            raise HarnessError(
                f"reference_lock: uv pip compile failed ({completed.returncode}): {completed.stderr.strip()[-2000:]}"
            )
        compiled = output.read_text(encoding="utf-8")
    return "\n".join(line for line in compiled.splitlines() if not line.strip().startswith("#")) + "\n"


def _sha256_text(text: str) -> str:
    """The SHA-256 of a text's UTF-8 bytes (the lock's input identities)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def render_lock(
    *,
    family: str,
    image: str,
    image_freeze_sha256: str,
    reference_in_sha256: str,
    reference_in: ReferenceIn,
    constraints: dict[str, tuple[str, str]],
    pins_text: str,
    verbatim: list[str] | None = None,
) -> str:
    """Render a lock file from its inputs.

    Inputs: the family and image names, the SHA-256 of the ``reference.in`` and of the image stack file
    the lock was generated against, the parsed input, the image constraints (``{canonical name:
    (image version, family floor spec)}`` -- recorded, never installed), the compiled pins text and the
    verbatim pins (the project's own distributions: exact pins the staged wheelhouse carries, so no
    index can hash them).  Output: the lock's text: the header (schema, family, image, the two input
    hashes, the own-torch declaration and evidence, the image constraints) followed by the install
    pins.  Units: none.
    """
    lines = [
        f"# rcp-reference-lock: {LOCK_SCHEMA}",
        f"# family: {family}",
        f"# image: {image}",
        f"# image-freeze-sha256: {image_freeze_sha256}",
        f"# reference-in-sha256: {reference_in_sha256}",
        f"# own-torch: {'true' if reference_in.own_torch else 'false'}",
    ]
    if reference_in.own_torch:
        lines.append(f"# own-torch-evidence: {reference_in.evidence}")
    for name in sorted(constraints):
        version, floor = constraints[name]
        lines.append(f"# image-constraint: {name}=={version} (family floor: {floor})")
    lines.append("#")
    lines.append(
        "# The pins below install into the family's reference venv with `pip install --no-deps --no-index "
        "--find-links <wheelhouse> -r <this file>`; the image's torch/CUDA stack is read through "
        "--system-site-packages and never replaced."
    )
    body = pins_text.strip()
    parts: list[str] = list(verbatim or [])
    if body:
        parts.append(body)
    return "\n".join(lines) + "\n" + ("\n".join(parts) + "\n" if parts else "")


def build_lock(
    reference_in_text: str,
    image_freeze_text: str,
    *,
    family: str,
    image: str,
    find_links: tuple[str, ...] = (),
    index_url: str | None = _DEFAULT_INDEX,
    python_version: str = _DEFAULT_PYTHON_VERSION,
    uv: str = _DEFAULT_UV,
) -> str:
    """Generate a family's lock from its ``reference.in`` and the engine image's freeze.

    Inputs: the ``reference.in`` text, the image's ``pip freeze`` text, the family and image names, and
    the resolver options (:func:`compile_requirements`).  Output: the lock's text.  Raises
    :class:`HarnessError` when a family floor is not satisfied by the image's stack, when a stack
    requirement is not exact under ``own-torch``, when a workspace pin is not exact, or when uv fails.
    Units: none.
    """
    reference_in = parse_reference_in(reference_in_text, family=family)
    stack = image_stack(image_freeze_text)
    if not stack:
        raise HarnessError(
            f"family {family}: the image freeze names no torch/CUDA stack (torch, torchvision, torchaudio, "
            "triton, nvidia-*); the reference install cannot be constrained to the image"
        )
    install: list[str] = []
    verbatim: list[str] = []
    constraints: dict[str, tuple[str, str]] = {}
    for spec in reference_in.requirements:
        name = requirement_name(spec)
        if is_stack(name):
            if reference_in.own_torch:
                if "==" not in spec:
                    raise HarnessError(
                        f"family {family}: own-torch pins the stack itself, so {spec!r} must be exact "
                        "(name==version); an unpinned stack cannot identify the exception's environment"
                    )
                install.append(spec)
                continue
            version = stack.get(name)
            if version is None:
                raise HarnessError(
                    f"family {family}: {spec!r} names the image stack but the image freeze carries no "
                    f"{name}; a family that needs its own stack declares own-torch with evidence"
                )
            if not satisfies(spec, version):
                raise HarnessError(
                    f"family {family}: the image's {name}=={version} does not satisfy the family floor "
                    f"{spec!r}; pin the version the reference needs and declare own-torch with its evidence"
                )
            constraints[name] = (version, spec)
            continue
        if canonical(name) in WORKSPACE_NAMES:
            if "==" not in spec:
                raise HarnessError(
                    f"family {family}: the workspace pin {spec!r} must be exact (name==version): the staged "
                    "wheelhouse carries the built wheel, and a range would drift"
                )
            verbatim.append(spec)
            continue
        install.append(spec)
    pins_text = compile_requirements(
        install,
        find_links=find_links,
        index_url=index_url,
        python_version=python_version,
        uv=uv,
    )
    return render_lock(
        family=family,
        image=image,
        image_freeze_sha256=_sha256_text(image_freeze_text),
        reference_in_sha256=_sha256_text(reference_in_text),
        reference_in=reference_in,
        constraints=constraints,
        pins_text=pins_text,
        verbatim=verbatim,
    )


def _lock_header(lock_text: str) -> dict[str, str]:
    """A lock's scalar header fields as ``{name: value}`` (the first occurrence wins; the repeated
    ``image-constraint`` lines are read by :func:`_image_constraints`)."""
    header: dict[str, str] = {}
    for raw in lock_text.splitlines():
        line = raw.strip()
        if not line.startswith("#"):
            continue
        body = line.lstrip("#").strip()
        name, _, value = body.partition(":")
        if value and name.strip() not in header:
            header[name.strip()] = value.strip()
    return header


def _image_constraints(lock_text: str) -> dict[str, tuple[str, str]]:
    """The lock's ``image-constraint`` header lines as ``{canonical name: (version, family floor)}``."""
    out: dict[str, tuple[str, str]] = {}
    for raw in lock_text.splitlines():
        line = raw.strip()
        if not line.startswith("# image-constraint:"):
            continue
        body = line.removeprefix("# image-constraint:").strip()
        pin, _, floor = body.partition(" (family floor: ")
        name, _, version = pin.partition("==")
        out[canonical(name.strip())] = (version.strip(), floor.rstrip(")").strip())
    return out


def _pinned_versions(lock_text: str) -> dict[str, str]:
    """The lock's install pins as ``{canonical name: version}`` (the ``name==version`` lines)."""
    pins: dict[str, str] = {}
    for raw in lock_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name, _, version = line.partition("==")
        if _:
            pins[canonical(name)] = version.split("\\", 1)[0].strip()
    return pins


def check_lock(
    lock_text: str,
    reference_in_text: str,
    image_freeze_text: str,
    *,
    family: str,
    image: str | None = None,
) -> list[str]:
    """Validate a committed lock against its ``reference.in`` and the image stack (offline).

    Inputs: the lock's text, the family's ``reference.in`` text, the image stack file's text, the family
    name and (optionally) the expected image name.  Output: the problems, each a line (``[]`` when the
    lock is consistent): header hash mismatches, a family requirement missing from the pins or the image
    constraints, a non-exact pin, an own-torch declaration that does not match, or an image constraint
    that the image stack no longer satisfies.  Units: none.
    """
    problems: list[str] = []
    header = _lock_header(lock_text)
    if header.get("rcp-reference-lock") != LOCK_SCHEMA:
        problems.append(f"schema is {header.get('rcp-reference-lock')!r}, expected {LOCK_SCHEMA!r}")
    if header.get("family") != family:
        problems.append(f"header family is {header.get('family')!r}, expected {family!r}")
    if image is not None and header.get("image") != image:
        problems.append(f"header image is {header.get('image')!r}, expected {image!r}")
    if header.get("reference-in-sha256") != _sha256_text(reference_in_text):
        problems.append("reference-in-sha256 does not match the committed reference.in")
    if header.get("image-freeze-sha256") != _sha256_text(image_freeze_text):
        problems.append("image-freeze-sha256 does not match the committed image stack")
    reference_in = parse_reference_in(reference_in_text, family=family)
    declared = header.get("own-torch", "false").lower() == "true"
    if declared != reference_in.own_torch:
        problems.append(f"own-torch is {declared}, reference.in declares {reference_in.own_torch}")
    stack = image_stack(image_freeze_text)
    pins = _pinned_versions(lock_text)
    constraints = _image_constraints(lock_text)
    for spec in reference_in.requirements:
        name = canonical(requirement_name(spec))
        if is_stack(name) and not reference_in.own_torch:
            if name not in constraints:
                problems.append(f"{spec!r} has no image-constraint in the header")
                continue
            version, _floor = constraints[name]
            if name in stack and not satisfies(spec, stack[name]):
                problems.append(f"the image's {name}=={stack[name]} no longer satisfies {spec!r}")
            if version != stack.get(name):
                problems.append(f"the image-constraint for {name} records {version}, the stack has {stack.get(name)}")
            continue
        if name not in pins:
            problems.append(f"{spec!r} is not pinned in the lock")
            continue
        if not satisfies(spec, pins[name]):
            problems.append(f"the pin {name}=={pins[name]} does not satisfy {spec!r}")
    for name, version in pins.items():
        if not version or any(char in version for char in "<>=!~ "):
            problems.append(f"pin {name}=={version} is not exact")
    return problems


def _read(path: str) -> str:
    """Read a UTF-8 text file, raising :class:`HarnessError` with the path when it is unreadable."""
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError as error:
        raise HarnessError(f"reference_lock: {path} cannot be read: {error}") from error


def main(argv: list[str] | None = None) -> int:
    """The CLI: ``build`` (default) generates a lock; ``check`` validates a committed one."""
    parser = argparse.ArgumentParser(prog="python -m rcp_ndcg_test.jobs.reference_lock", description=__doc__)
    sub = parser.add_subparsers(dest="command")
    build = sub.add_parser("build", help="generate a family's reference.lock")
    check = sub.add_parser("check", help="validate a committed lock offline")
    for command in (build, check):
        command.add_argument("--family", required=True)
        command.add_argument("--in", dest="reference_in", required=True)
        command.add_argument("--image-freeze", required=True)
        command.add_argument("--image", default=None, help="the engine image (required for build)")
    build.add_argument("--out", required=True)
    build.add_argument("--find-links", action="append", default=[], help="a wheelhouse (repeatable; --no-index)")
    build.add_argument("--index-url", default=_DEFAULT_INDEX)
    build.add_argument("--python-version", default=_DEFAULT_PYTHON_VERSION)
    build.add_argument("--uv", default=_DEFAULT_UV)
    check.add_argument("--lock", required=True)
    arguments = parser.parse_args(argv)
    if arguments.command is None:
        parser.error("a command is required: build or check")
    reference_in_text = _read(arguments.reference_in)
    freeze_text = _read(arguments.image_freeze)
    try:
        if arguments.command == "check":
            problems = check_lock(
                _read(arguments.lock),
                reference_in_text,
                freeze_text,
                family=arguments.family,
                image=arguments.image,
            )
            for problem in problems:
                print(f"reference_lock: {arguments.family}: {problem}")
            return 1 if problems else 0
        if not arguments.image:
            parser.error("build needs --image")
        lock_text = build_lock(
            reference_in_text,
            freeze_text,
            family=arguments.family,
            image=arguments.image,
            find_links=tuple(arguments.find_links),
            index_url=arguments.index_url,
            python_version=arguments.python_version,
            uv=arguments.uv,
        )
        Path(arguments.out).write_text(lock_text, encoding="utf-8")
    except HarnessError as error:
        print(f"reference_lock: {error}", file=sys.stderr)
        return 2
    print(f"reference_lock: wrote {arguments.out} ({len(lock_text)} bytes)")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main() with argv
    raise SystemExit(main())
