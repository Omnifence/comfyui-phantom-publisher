from __future__ import annotations

import ast
import asyncio
import gzip
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.parse
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import aiohttp
from aiohttp import web

import folder_paths
from server import PromptServer

PUBLISHER_VERSION = "0.5.2"
CONFIG_FILENAME = ".phantom-publisher.json"
_jobs: dict[str, dict[str, Any]] = {}
PUBLISH_LOG_LIMIT = 200


def _job_log(
    job: dict[str, Any],
    message: str,
    level: str = "info",
    dependency_id: str | None = None,
) -> None:
    logs = job.setdefault("logs", [])
    sequence = int(logs[-1]["sequence"]) + 1 if logs else 1
    logs.append(
        {
            "sequence": sequence,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "level": level,
            "phase": job.get("status", "queued"),
            "message": message,
            **({"dependency_id": dependency_id} if dependency_id else {}),
        }
    )
    if len(logs) > PUBLISH_LOG_LIMIT:
        del logs[: len(logs) - PUBLISH_LOG_LIMIT]


def _comfyui_core_version():
    """
    The ComfyUI core release this machine runs, or None when it cannot be read.

    ComfyUI publishes its version as `comfyui_version.__version__` — a
    generated module at the repo root — NOT as `comfy.__version__`. The `comfy`
    package has no `__init__.py` at all, so the previous `getattr(comfy,
    "__version__", None)` returned None on every publish that has ever run.

    That silence is not cosmetic. Phantom compares this against the ComfyUI core
    its base images bundle, to decide whether a build has to install a newer
    core than the image froze. With None it cannot decide, so it ships the
    image's own core — and a graph that uses a node input a newer core added
    then builds clean, deploys clean, and fails inside ComfyUI's validation on
    the caller's job.
    """
    try:
        import comfyui_version
    except Exception:
        return None
    version = getattr(comfyui_version, "__version__", None)
    if isinstance(version, str) and version.strip():
        return version.strip()
    return None


def _log_server_warnings(job: dict[str, Any], version: Any) -> None:
    """
    Surface advisory `warnings` Phantom attaches to a version response — e.g.
    "this machine's ComfyUI core is newer than every base image the platform
    knows". The field is additive: an older server sends none, and anything
    that is not a list of strings is ignored rather than failing a publish
    that already carries real artifacts.
    """
    warnings = version.get("warnings") if isinstance(version, dict) else None
    if not isinstance(warnings, list):
        return
    for warning in warnings:
        if isinstance(warning, str) and warning:
            _job_log(job, warning, level="warning")


def _job_step(job: dict[str, Any], message: str, **fields: Any) -> None:
    """
    Advance the job and log the message it advanced with.

    The two are ordered: `_job_log` stamps its entry with `job["status"]`, so a
    caller that logged before updating would file the line under the previous
    phase. Pairing them here is what keeps that ordering out of six call sites.
    """
    job.update(message=message, **fields)
    _job_log(job, message, dependency_id=fields.get("current_dependency_id"))


def _config_path() -> Path:
    get_user_directory = getattr(folder_paths, "get_user_directory", None)
    root = Path(get_user_directory()) if callable(get_user_directory) else Path(folder_paths.base_path) / "user"
    root.mkdir(parents=True, exist_ok=True)
    return root / CONFIG_FILENAME


def _read_config() -> dict[str, Any]:
    path = _config_path()
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_config(value: dict[str, Any]) -> None:
    """
    Write the config, which holds the publisher token, owner-readable only.

    Creating the file already restricted matters: writing first and chmod-ing
    after leaves the token world-readable for the width of that window. O_EXCL
    on a freshly removed path also means a pre-existing `.tmp` — stale from a
    crashed write, or planted as a symlink — cannot be written through.
    """
    path = _config_path()
    temporary = path.with_suffix(".tmp")
    temporary.unlink(missing_ok=True)
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
    temporary.replace(path)


def _safe_url(value: str | None) -> str | None:
    if not value:
        return None
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return None
    host = parsed.hostname
    if parsed.port:
        host = f"{host}:{parsed.port}"
    return urllib.parse.urlunsplit((parsed.scheme, host, parsed.path, "", ""))


def _default_console_origin(api_origin: str) -> str:
    parsed = urllib.parse.urlsplit(api_origin)
    hostname = parsed.hostname or ""
    port = parsed.port

    if hostname in {"localhost", "127.0.0.1"} and port == 3060:
        return urllib.parse.urlunsplit((parsed.scheme, f"{hostname}:3062", "", "", ""))
    if hostname == "api.phantomrouter.ai":
        return urllib.parse.urlunsplit((parsed.scheme, "app.phantomrouter.ai", "", "", ""))
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))


def _sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _git_metadata(directory: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", "-C", str(directory), *args],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=5,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            return None

    # `git -C` walks up until it finds A repository, not THIS package's. A
    # hand-copied folder inside a ComfyUI checkout therefore reported ComfyUI's
    # own URL, commit and dirty flag — provenance for the wrong code. Only a
    # package that is its own checkout has Git provenance; anything else is a
    # local copy, and `dirty` stays true because its content matches no commit.
    toplevel = run("rev-parse", "--show-toplevel")
    if not toplevel or Path(toplevel).resolve() != directory.resolve():
        return {"git_commit": None, "repository_url": None, "dirty": True}

    commit = run("rev-parse", "HEAD")
    remote = _safe_url(run("remote", "get-url", "origin"))
    status = run("status", "--porcelain", "--untracked-files=normal")
    return {"git_commit": commit, "repository_url": remote, "dirty": bool(status)}


def _node_properties(ui_workflow: dict[str, Any]) -> dict[str, dict[str, Any]]:
    nodes = ui_workflow.get("nodes", [])
    if not isinstance(nodes, list):
        return {}
    return {
        str(node.get("id")): node
        for node in nodes
        if isinstance(node, dict) and node.get("id") is not None
    }


# Discovery probes every string input of every node against every model
# directory, so it also sees prompts, seeds and free text. A prompt is not a
# filename, and the filesystem does not answer "no" politely: joining an
# over-long string to a model root and calling is_file() raises
# OSError(ENAMETOOLONG) instead of returning False, which failed the whole
# publish. POSIX limits one path component to 255 bytes and a full path to
# PATH_MAX, so anything longer cannot name a file that exists.
_MAX_NAME_BYTES = 255
_MAX_PATH_BYTES = 4096


def _is_plausible_filename(value: str) -> bool:
    if not value or "\n" in value or "\x00" in value:
        return False
    if len(value.encode("utf-8", "surrogatepass")) > _MAX_PATH_BYTES:
        return False
    components = [part for part in value.replace("\\", "/").split("/") if part]
    if not components:
        return False
    return all(
        len(part.encode("utf-8", "surrogatepass")) <= _MAX_NAME_BYTES for part in components
    )


def _resolve_model(filename: str, model_type: str) -> tuple[Path, str] | None:
    if not _is_plausible_filename(filename):
        return None
    aliases = {
        "checkpoint": "checkpoints",
        "lora": "loras",
        "controlnet": "controlnet",
        "embedding": "embeddings",
        "upscaler": "upscale_models",
    }
    model_type = aliases.get(model_type, model_type)
    try:
        roots = folder_paths.get_folder_paths(model_type)
    except (KeyError, TypeError):
        roots = []
    target = filename.replace("\\", "/").lstrip("/")
    for raw_root in roots:
        # A model root can be missing, unreadable or a broken symlink on
        # someone else's install. One bad root must not fail the publish.
        try:
            root = Path(raw_root).resolve()
            direct = (root / target).resolve()
            if direct.is_file() and root in direct.parents:
                try:
                    relative_parent = direct.parent.relative_to(
                        Path(folder_paths.base_path).resolve()
                    )
                    return direct, str(relative_parent).replace("\\", "/")
                except ValueError:
                    return direct, f"models/{model_type}"
            for candidate in root.rglob(Path(target).name):
                resolved = candidate.resolve()
                if resolved.is_file() and root in resolved.parents:
                    try:
                        relative_parent = resolved.parent.relative_to(
                            Path(folder_paths.base_path).resolve()
                        )
                        return resolved, str(relative_parent).replace("\\", "/")
                    except ValueError:
                        return (
                            resolved,
                            f"models/{model_type}/{resolved.parent.relative_to(root)}".rstrip("/"),
                        )
        except OSError:
            continue
    return None


def _discover_models(api_workflow: dict[str, Any], ui_workflow: dict[str, Any]) -> list[dict[str, Any]]:
    ui_nodes = _node_properties(ui_workflow)
    # The destination path is part of the workflow contract. Two filenames may
    # intentionally contain the same bytes (aliases, hard links, or copied
    # checkpoints), and deduplicating those by digest drops one path from the
    # image even though its node still references it.
    discovered: dict[tuple[str, str], dict[str, Any]] = {}
    for node_id, raw_node in api_workflow.items():
        if not isinstance(raw_node, dict):
            continue
        ui_node = ui_nodes.get(str(node_id), {})
        properties = ui_node.get("properties", {}) if isinstance(ui_node, dict) else {}
        declared = properties.get("models", []) if isinstance(properties, dict) else []
        candidates: list[tuple[str, str, list[str]]] = []
        if isinstance(declared, list):
            for model in declared:
                if not isinstance(model, dict):
                    continue
                name = model.get("name") or model.get("filename")
                model_type = model.get("type") or model.get("model_type") or "checkpoints"
                urls = model.get("url") or model.get("source_url")
                if isinstance(name, str):
                    candidates.append((name, str(model_type), [urls] if isinstance(urls, str) else []))
        inputs = raw_node.get("inputs", {})
        if isinstance(inputs, dict):
            for value in inputs.values():
                if not isinstance(value, str):
                    continue
                for model_type in getattr(folder_paths, "folder_names_and_paths", {}).keys():
                    if _resolve_model(value, model_type):
                        candidates.append((value, model_type, []))
                        break
        for filename, model_type, urls in candidates:
            resolved = _resolve_model(filename, model_type)
            if not resolved:
                continue
            path, comfy_path = resolved
            digest, byte_size = _sha256(path)
            destination_filename = Path(filename.replace("\\", "/")).name
            destination_key = (comfy_path, destination_filename)
            safe_urls = [safe for url in urls if (safe := _safe_url(url))]
            if destination_key in discovered:
                existing_urls = discovered[destination_key]["source_urls"]
                discovered[destination_key]["source_urls"] = list(
                    dict.fromkeys([*existing_urls, *safe_urls])
                )
                continue
            discovered[destination_key] = {
                "filename": destination_filename,
                "model_type": model_type,
                "comfyui_path": comfy_path,
                "sha256": digest,
                "byte_size": byte_size,
                "source_urls": safe_urls,
                "_local_path": str(path),
            }
    return list(discovered.values())


def _normalized_package_name(value: str) -> str:
    return value.lower().replace("-", "_")


def _label_matches_directory(cnr_id: str, directory: Path) -> bool:
    return _normalized_package_name(cnr_id) in _normalized_package_name(directory.name)


def _comfyui_provided_distributions() -> set[str]:
    """
    Distributions any ComfyUI base image already provides.

    ComfyUI's own requirements are guaranteed present in the build image, and
    re-pinning them from a developer machine is how you turn a working image
    into a broken one (a mismatched `torch` above all). Excluding them keeps
    the captured set to what the node package actually adds.
    """
    provided = {_normalize_distribution(name) for name in _TORCH_STACK}
    requirements = Path(folder_paths.base_path) / "requirements.txt"
    if not requirements.exists():
        return provided
    for line in requirements.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        name = re.split(r"[<>=!~\[; ]", line, maxsplit=1)[0].strip()
        if name:
            provided.add(_normalize_distribution(name))
    return provided


_TORCH_STACK = ("torch", "torchvision", "torchaudio", "torchsde", "xformers", "numpy")


def _normalize_distribution(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _imported_top_level_modules(directory: Path) -> set[str]:
    """Every top-level module name the package's Python sources import."""
    modules: set[str] = set()
    for source in directory.rglob("*.py"):
        if "__pycache__" in source.parts:
            continue
        try:
            tree = ast.parse(source.read_text(encoding="utf-8", errors="ignore"))
        except (OSError, SyntaxError):
            continue
        for statement in ast.walk(tree):
            if isinstance(statement, ast.Import):
                for alias in statement.names:
                    modules.add(alias.name.split(".", 1)[0])
            elif isinstance(statement, ast.ImportFrom):
                # level > 0 is a relative import — the package's own modules.
                if statement.level == 0 and statement.module:
                    modules.add(statement.module.split(".", 1)[0])
    return modules


def _local_module_names(directory: Path) -> set[str]:
    """Names that resolve inside the package itself, not to a distribution."""
    names = {directory.name}
    for entry in directory.iterdir() if directory.is_dir() else []:
        if entry.is_dir() and (entry / "__init__.py").exists():
            names.add(entry.name)
        elif entry.suffix == ".py":
            names.add(entry.stem)
    return names


def _declared_requirements(directory: Path) -> set[str]:
    """Distribution names the package's own requirements.txt already declares."""
    requirements = directory / "requirements.txt"
    if not requirements.exists():
        return set()
    declared: set[str] = set()
    for line in requirements.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        name = re.split(r"[<>=!~\[; ]", line, maxsplit=1)[0].strip()
        if name:
            declared.add(_normalize_distribution(name))
    return declared


def _pip_dependencies(directory: Path) -> list[str]:
    """
    The installed distributions this node package imports, version-pinned.

    The publisher runs inside a ComfyUI that works, so the installed set is the
    ground truth for what the built image needs. A package that imports a module
    it never declares in requirements.txt — the common case — otherwise installs
    cleanly, fails to import at ComfyUI startup, and silently registers none of
    its node classes. Every render against that image then fails with a ComfyUI
    400 "custom node may not be installed".
    """
    stdlib = set(sys.builtin_module_names) | set(sys.stdlib_module_names)
    ignored = stdlib | _local_module_names(directory) | {"comfy", "comfy_extras", "folder_paths", "nodes", "server", "app", "utils", "execution", "latent_preview", "comfy_api", "comfy_api_nodes"}
    provided = _comfyui_provided_distributions() | _declared_requirements(directory)

    try:
        module_to_distributions = importlib.metadata.packages_distributions()
    except Exception:
        return []

    pinned: dict[str, str] = {}
    for module in sorted(_imported_top_level_modules(directory) - ignored):
        for distribution in module_to_distributions.get(module, []):
            normalized = _normalize_distribution(distribution)
            if normalized in provided or normalized in pinned:
                continue
            try:
                version = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                continue
            pinned[normalized] = f"{distribution}=={version}"
    return sorted(pinned.values())


def _archive_package(directory: Path) -> tuple[Path, str, int]:
    temporary = Path(tempfile.mkdtemp(prefix="phantom-publisher-")) / f"{directory.name}.tar.gz"
    with temporary.open("wb") as compressed:
        with gzip.GzipFile(filename="", mode="wb", fileobj=compressed, mtime=0) as gzip_file:
            with tarfile.open(fileobj=gzip_file, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for source in sorted(directory.rglob("*")):
                    if ".git" in source.parts or "__pycache__" in source.parts:
                        continue
                    info = archive.gettarinfo(
                        str(source), arcname=f"{directory.name}/{source.relative_to(directory)}"
                    )
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    if source.is_file():
                        with source.open("rb") as handle:
                            archive.addfile(info, handle)
                    else:
                        archive.addfile(info)
    digest, size = _sha256(temporary)
    return temporary, digest, size


_HUGGINGFACE_REPO = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")

# Where each documented loader takes its repository id: (keyword, positional
# index). The positional fallback matters — `snapshot_download("org/model")`
# and `pipeline("image-classification", "org/model")` are the common literal
# forms, and reading keywords alone left those models out of the archive. The
# image then downloads mutable upstream content at run time, or fails outright
# when the build host has no network.
_HUGGINGFACE_REPO_ARGUMENTS: dict[str, tuple[str | None, int]] = {
    "pipeline": ("model", 1),
    "snapshot_download": ("repo_id", 0),
    "hf_hub_download": ("repo_id", 0),
    "from_pretrained": ("pretrained_model_name_or_path", 0),
}


def _literal_huggingface_repositories(directory: Path) -> set[str]:
    """Find statically declared Hugging Face model IDs in a used package."""
    repositories: set[str] = set()
    sources = [directory] if directory.is_file() else directory.rglob("*.py")
    for source in sources:
        if "__pycache__" in source.parts:
            continue
        try:
            tree = ast.parse(source.read_text(encoding="utf-8", errors="ignore"))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = (
                node.func.attr
                if isinstance(node.func, ast.Attribute)
                else (node.func.id if isinstance(node.func, ast.Name) else "")
            )
            value: Any = None
            argument = _HUGGINGFACE_REPO_ARGUMENTS.get(function)
            if argument:
                keyword_name, position = argument
                if keyword_name:
                    value = next(
                        (item.value for item in node.keywords if item.arg == keyword_name), None
                    )
                if value is None and len(node.args) > position:
                    value = node.args[position]
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                if _HUGGINGFACE_REPO.fullmatch(value.value):
                    repositories.add(value.value)
    return repositories


# A Hugging Face id is `org/name` whatever it names, so the scanner cannot tell
# a model from a Space or a dataset. snapshot_download assumes "model", and the
# hub answers a wrong type with a 401 rather than a 404 — it will not confirm
# that a repository is absent — which failed a whole publish on one Space.
_HUGGINGFACE_REPO_TYPES = ("model", "space", "dataset")

# Only a model sits at the root of huggingface.co. The other two are namespaced.
_HUGGINGFACE_URL_PREFIXES = {"model": "", "space": "spaces/", "dataset": "datasets/"}


def _huggingface_repo_type(repo_id: str) -> str | None:
    """
    Resolve which kind of repository an id names, or None if it names none.

    Probing costs one extra request for a model and two for a Space, which is
    cheap next to the download it precedes.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        # Let _huggingface_snapshot raise the actionable "not installed" error
        # rather than reporting this as an unresolvable repository.
        return "model"
    repo_exists = getattr(HfApi(), "repo_exists", None)
    if not callable(repo_exists):
        # huggingface_hub before 0.20 has no repo_exists. Assume a model, which
        # is what this did before the probe existed.
        return "model"
    for repo_type in _HUGGINGFACE_REPO_TYPES:
        try:
            if repo_exists(repo_id, repo_type=repo_type):
                return repo_type
        except Exception:
            continue
    return None


def _huggingface_url(repo_id: str, repo_type: str, revision: str) -> str:
    prefix = _HUGGINGFACE_URL_PREFIXES.get(repo_type, "")
    return f"https://huggingface.co/{prefix}{repo_id}/tree/{revision}"


def _huggingface_snapshot(repo_id: str, repo_type: str = "model") -> tuple[Path, str]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError(
            f'Used node declares Hugging Face model "{repo_id}", but huggingface_hub is not installed'
        ) from error
    try:
        snapshot = Path(snapshot_download(repo_id=repo_id, repo_type=repo_type)).resolve()
    except Exception as error:
        raise RuntimeError(f'Could not snapshot Hugging Face {repo_type} "{repo_id}": {error}') from error
    # snapshot_download returns .../<type>s--org--repo/snapshots/<commit>. The
    # whole repository cache includes refs and content-addressed blobs needed by
    # Transformers when the upstream repository is no longer reachable, and the
    # cache directory name carries the repo type, so restoring the archive is
    # enough for an offline snapshot_download of any of the three types.
    if snapshot.parent.name != "snapshots" or not snapshot.parent.parent.is_dir():
        raise RuntimeError(f'Unexpected Hugging Face cache path for "{repo_id}": {snapshot}')
    return snapshot.parent.parent, snapshot.name


def _discover_huggingface_models(
    packages: list[dict[str, Any]],
    on_repository: Callable[[str, int, int], None] | None = None,
    on_skipped: Callable[[str, str], None] | None = None,
) -> list[dict[str, Any]]:
    discovered: list[dict[str, Any]] = []
    repo_ids: set[str] = set()
    for package in packages:
        for source in package.get("_source_files", []):
            repo_ids.update(_literal_huggingface_repositories(Path(source)))
    try:
        sorted_repo_ids = sorted(repo_ids)
        for index, repo_id in enumerate(sorted_repo_ids):
            if on_repository:
                on_repository(repo_id, index, len(sorted_repo_ids))
            repo_type = _huggingface_repo_type(repo_id)
            if repo_type is None:
                # `org/name` is also the shape of a local relative path, so the
                # scanner produces false positives. One of those must not fail
                # a publish that has already archived real dependencies.
                if on_skipped:
                    on_skipped(repo_id, "no model, Space or dataset repository has that id")
                continue
            cache_directory, revision = _huggingface_snapshot(repo_id, repo_type)
            archive, digest, size = _archive_package(cache_directory)
            discovered.append(
                {
                    "filename": archive.name,
                    "model_type": "other",
                    "comfyui_path": ".cache/huggingface/hub",
                    "sha256": digest,
                    "byte_size": size,
                    "source_urls": [_huggingface_url(repo_id, repo_type, revision)],
                    "archive_format": "tar.gz",
                    "install_path": "opt/phantom/huggingface/hub",
                    "external_repository": repo_id,
                    "repo_type": repo_type,
                    "revision": revision,
                    "_local_path": str(archive),
                }
            )
    except Exception:
        for item in discovered:
            shutil.rmtree(Path(item["_local_path"]).parent, ignore_errors=True)
        raise
    return discovered


def _class_source_file(nodes_module: Any, class_type: str) -> tuple[Path | None, str | None]:
    """The file that defines a registered node class, via its class object."""
    node_class = nodes_module.NODE_CLASS_MAPPINGS.get(class_type)
    if node_class is None:
        return None, None
    module_name = getattr(node_class, "__module__", None)
    module = sys.modules.get(module_name) if module_name else None
    if module is None and module_name:
        try:
            module = __import__(module_name, fromlist=["__file__"])
        except ImportError:
            module = None
    module_file = getattr(module, "__file__", None) if module else None
    return (Path(module_file).resolve() if module_file else None), module_name


def _discover_packages(api_workflow: dict[str, Any], ui_workflow: dict[str, Any]) -> list[dict[str, Any]]:
    import nodes

    # The archived directory always comes from the CLASS OBJECT, never from the
    # frontend's `properties.cnr_id` label. The label is whatever ComfyUI's
    # registry said at save time, and one broken third-party package can rewrite
    # that registry for every node loaded before it (`from nodes import *` in a
    # package __init__ re-exports ComfyUI's global NODE_CLASS_MAPPINGS, and the
    # loader then re-attributes every class to that package). A class's
    # `__module__` is stamped at definition and survives any re-registration,
    # so the file it names is the ground truth for what bytes to upload.
    # Grouping by resolved file also IS the pre-upload invariant: every class a
    # package claims resolves to a file inside that package's directory.
    ui_nodes = _node_properties(ui_workflow)
    custom_roots = [Path(root).resolve() for root in folder_paths.get_folder_paths("custom_nodes")]
    comfy_root = Path(folder_paths.base_path).resolve()

    grouped: dict[str, dict[str, Any]] = {}
    for node_id, raw_node in api_workflow.items():
        if not isinstance(raw_node, dict) or not isinstance(raw_node.get("class_type"), str):
            continue
        class_type = raw_node["class_type"]
        ui_node = ui_nodes.get(str(node_id), {})
        properties = ui_node.get("properties", {}) if isinstance(ui_node, dict) else {}
        cnr_id = properties.get("cnr_id") if isinstance(properties, dict) else None
        version = properties.get("ver") if isinstance(properties, dict) else None

        module_file, module_name = _class_source_file(nodes, class_type)
        if module_file is None:
            # Publishing anyway would promise code that is never uploaded, and
            # the lie only surfaces after a full image build.
            raise RuntimeError(
                f'Node class "{class_type}" cannot be resolved to a source file '
                f'(module: {module_name or "not registered in this ComfyUI"}). '
                "The package that defines it is absent or failed to import, so "
                "this workflow cannot be published from this machine."
            )

        directory: Path | None = None
        for root in custom_roots:
            if root in module_file.parents:
                directory = root / module_file.relative_to(root).parts[0]
                break
        if directory is None:
            # Core-ness is also derived from the resolved path, not from a
            # `comfy-core` label — the label lies exactly when it matters.
            if comfy_root == module_file or comfy_root in module_file.parents:
                continue
            raise RuntimeError(
                f'Node class "{class_type}" is defined in "{module_file}", outside both '
                "custom_nodes and the ComfyUI installation, so it cannot be packaged."
            )

        entry = grouped.setdefault(
            str(directory), {"classes": set(), "labels": [], "mismatches": [], "files": set()}
        )
        entry["classes"].add(class_type)
        entry["files"].add(str(module_file))
        if isinstance(cnr_id, str) and cnr_id:
            if cnr_id != "comfy-core" and _label_matches_directory(cnr_id, directory):
                entry["labels"].append((cnr_id, str(version) if version else None))
            else:
                # Keep the disagreement visible: this is how we find out how
                # often a hijacked registry happens in the wild.
                entry["mismatches"].append({"class_type": class_type, "labeled_cnr_id": cnr_id})

    result: list[dict[str, Any]] = []
    for directory_key in sorted(grouped):
        entry = grouped[directory_key]
        directory = Path(directory_key)
        # Registry coordinates stay as PROVENANCE when the label agrees with the
        # resolved directory; a disagreeing label never names the directory and
        # never rides along as provenance either.
        cnr_id, version = next(iter(entry["labels"]), (None, None))
        package: dict[str, Any] = {
            "class_types": sorted(entry["classes"]),
            "cnr_id": cnr_id,
            "version": version,
            "pip_dependencies": _pip_dependencies(directory),
            **_git_metadata(directory),
            "archive_sha256": None,
            "_package_directory": str(directory),
            "_source_files": sorted(entry["files"]),
        }
        if entry["mismatches"]:
            package["attribution_mismatches"] = sorted(
                entry["mismatches"], key=lambda item: item["class_type"]
            )
        # Snapshot every used custom-node package, including clean Registry and
        # Git installs. Registry/Git coordinates remain useful provenance, but
        # they are not an immutable source of truth: releases, repositories, and
        # commits can be removed. The normalized archive is content-addressed by
        # Phantom and is therefore what a workflow-version build consumes.
        archive, digest, size = _archive_package(directory)
        package.update(
            {
                "archive_sha256": digest,
                "_archive_path": str(archive),
                "_archive_size": size,
            }
        )
        result.append(package)
    return result


def _parse_json_body(raw: str) -> Any | None:
    """
    Parse a response body, or return None when it is not JSON.

    Error responses are not guaranteed to be JSON: a load balancer or reverse
    proxy in front of the API answers 502/503 with its own HTML page, and the
    caller still has to classify the status and decide whether to retry.
    """
    if not raw:
        return None
    try:
        return json.loads(raw)
    except ValueError:
        return None


# aiohttp caps every request at 5 minutes unless told otherwise, and finalizing
# a multi-gigabyte artifact takes longer than that: Phantom hashes the whole
# object before it answers. That default failed a 28 GB publish after every
# byte had already landed. Cap the phases that genuinely indicate a dead
# connection instead of the total, so a slow answer is waited out and a stalled
# socket still fails fast.
#
# `connect` must stay bounded. It is the only field that covers DNS resolution:
# aiohttp applies `sock_connect` to the TCP and TLS handshake alone, and wraps
# name resolution in `connect`. With `connect=None` a resolver that stops
# answering — a laptop changing networks, a container losing its nameserver —
# blocks the publish for as long as the OS takes to give up. That cost a
# 28 GB publish 20 minutes of silence before it reported a DNS failure.
_CONNECT_TIMEOUT_SECONDS = 60
_SOCKET_CONNECT_TIMEOUT_SECONDS = 30
_READ_TIMEOUT_SECONDS = 15 * 60

# A publish runs for hours and carries tens of gigabytes, so a network blip
# that lasts seconds must not throw the whole run away. Five retries spend
# 1 + 2 + 4 + 8 + 16 seconds waiting, which outlasts a resolver restart or a
# reconnecting VPN without leaving a genuinely dead host hanging for minutes.
_TRANSIENT_RETRIES = 5
_MAX_BACKOFF_SECONDS = 30.0

# DNS, TCP, TLS and mid-body failures never produce a status code, so the
# status-based retry below never saw them: they escaped the loop as exceptions
# and failed the publish outright. `ClientConnectionError` covers the connector
# errors (`ClientConnectorDNSError`, `ServerDisconnectedError`, `ClientOSError`
# among them); `ClientPayloadError` covers a response body that stops
# mid-stream.
_TRANSIENT_NETWORK_ERRORS: tuple[type[BaseException], ...] = (
    aiohttp.ClientConnectionError,
    aiohttp.ClientPayloadError,
    asyncio.TimeoutError,
)


def _client_timeout() -> Any:
    return aiohttp.ClientTimeout(
        total=None,
        connect=_CONNECT_TIMEOUT_SECONDS,
        sock_connect=_SOCKET_CONNECT_TIMEOUT_SECONDS,
        sock_read=_READ_TIMEOUT_SECONDS,
    )


def _backoff_seconds(attempt: int) -> float:
    """Exponential backoff, capped so a long retry sequence stays bounded."""
    return min(float(2**attempt), _MAX_BACKOFF_SECONDS)


async def _phantom_request(
    method: str,
    path: str,
    config: dict[str, Any],
    body: Any = None,
    transient_retries: int = 0,
    on_retry: Callable[[int, int, float], None] | None = None,
) -> Any:
    headers = {"Authorization": f"Bearer {config['token']}"}
    request_options: dict[str, Any] = {}
    if body is not None or method.upper() not in {"GET", "HEAD", "OPTIONS"}:
        headers["Content-Type"] = "application/json"
        request_options["json"] = body if body is not None else {}
    total_attempts = transient_retries + 1
    # One session for the whole retry sequence: a retry that reopens the
    # connection pays another TCP + TLS handshake before it can even try.
    async with aiohttp.ClientSession(headers=headers, timeout=_client_timeout()) as session:
        for attempt in range(total_attempts):
            last_attempt = attempt == total_attempts - 1
            try:
                # aiohttp 3.13 sends `Content-Type: application/octet-stream` for an
                # otherwise bodyless POST, even when no `data`/`json` argument is
                # supplied. Fastify rejects that before finalize and multipart-part
                # presign handlers can run, so mutation requests carry `{}` explicitly.
                async with session.request(
                    method,
                    f"{config['origin'].rstrip('/')}/api/v1/phantom/comfyui-publisher{path}",
                    **request_options,
                ) as response:
                    # Read the body as TEXT and parse it ourselves. `response.json()`
                    # raises ContentTypeError on the HTML or plain-text 502/503 a
                    # reverse proxy returns, and that exception escapes the loop —
                    # so the one class of failure these retries exist for was the
                    # one class that never retried.
                    raw = await response.text()
                    payload = _parse_json_body(raw)
                    if response.status < 400:
                        if payload is None:
                            raise RuntimeError(
                                f"Phantom {method.upper()} {path} returned HTTP "
                                f"{response.status} with a non-JSON body: {raw.strip()[:200]}"
                            )
                        return payload
                    message = (
                        (payload.get("message") or payload.get("error"))
                        if isinstance(payload, dict)
                        else None
                    ) or (raw.strip()[:200] or "Request failed")
                    retryable = response.status == 429 or response.status >= 500
                    if not retryable or last_attempt:
                        raise RuntimeError(
                            f"Phantom {method.upper()} {path} returned HTTP "
                            f"{response.status}: {message}"
                        )
            except _TRANSIENT_NETWORK_ERRORS as error:
                if last_attempt:
                    raise RuntimeError(
                        f"Phantom {method.upper()} {path} could not reach "
                        f"{config['origin']}: {error}"
                    ) from error
            delay = _backoff_seconds(attempt)
            if on_retry:
                on_retry(attempt + 2, total_attempts, delay)
            await asyncio.sleep(delay)

    raise RuntimeError(f"Phantom {method.upper()} {path} failed after {total_attempts} attempts")


def _dependency_progress(
    models: list[dict[str, Any]], packages: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[tuple[str, Path, int, dict[str, Any]]]]:
    dependencies: list[dict[str, Any]] = []
    uploads: list[tuple[str, Path, int, dict[str, Any]]] = []

    for index, model in enumerate(models):
        size = int(model["byte_size"])
        external_repository = model.get("external_repository")
        name = external_repository or model.get("filename") or f"Model {index + 1}"
        destination = "/".join(
            part for part in (model.get("comfyui_path"), model.get("filename")) if part
        )
        dependency = {
            "id": f"model-{index}",
            "kind": "external_model" if external_repository else "model",
            "name": name,
            "detail": (
                f"Hugging Face snapshot · {destination}"
                if external_repository
                else f"{model.get('model_type') or 'model'} · {destination}"
            ),
            "byte_size": size,
            "uploaded_bytes": 0,
            "progress": 0,
            "status": "pending",
            "upload_required": True,
        }
        dependencies.append(dependency)
        uploads.append((model["sha256"], Path(model["_local_path"]), size, dependency))

    for index, package in enumerate(packages):
        package_directory = package.get("_package_directory")
        name = (
            package.get("cnr_id")
            or (Path(package_directory).name if package_directory else None)
            or ", ".join(package.get("class_types", [])[:2])
            or f"Custom node package {index + 1}"
        )
        version = package.get("version") or package.get("git_commit")
        version_detail = f" · {str(version)[:12]}" if version else ""
        has_upload = bool(package.get("_archive_path"))
        size = int(package.get("_archive_size") or 0)
        dependency = {
            "id": f"custom-node-{index}",
            "kind": "custom_node",
            "name": name,
            "detail": f"Custom node package{version_detail}",
            "byte_size": size,
            "uploaded_bytes": 0 if has_upload else size,
            "progress": 0 if has_upload else 100,
            "status": "pending" if has_upload else "not_required",
            "upload_required": has_upload,
        }
        dependencies.append(dependency)
        if has_upload:
            uploads.append(
                (
                    package["archive_sha256"],
                    Path(package["_archive_path"]),
                    size,
                    dependency,
                )
            )

    return dependencies, uploads


def _part_bytes(part_number: int, part_size: int, total_size: int) -> int:
    """How many bytes one part carries — the last part is a short one."""
    return min(part_size, max(0, total_size - ((part_number - 1) * part_size)))


async def _put_part(
    version_id: str,
    digest: str,
    part_number: int,
    chunk: bytes,
    config: dict[str, Any],
    on_retry: Callable[[int, int, float], None] | None = None,
) -> dict[str, Any]:
    """
    Send one part to the object store, and sign a fresh URL for every attempt.

    A part carries the same bytes to the same slot each time, so a repeat is
    safe. The presigned URL expires, though, and an attempt that waited out a
    backoff can outlive it — so each attempt asks Phantom for a new URL rather
    than replaying one that may already be dead.
    """
    total_attempts = _TRANSIENT_RETRIES + 1
    for attempt in range(total_attempts):
        last_attempt = attempt == total_attempts - 1
        signed = await _phantom_request(
            "POST",
            f"/versions/{version_id}/artifacts/{digest}/uploads/parts/{part_number}",
            config,
            transient_retries=_TRANSIENT_RETRIES,
            on_retry=on_retry,
        )
        try:
            async with aiohttp.ClientSession(timeout=_client_timeout()) as session:
                async with session.put(signed["upload_url"], data=chunk) as response:
                    if response.status < 400:
                        return {
                            "PartNumber": part_number,
                            "ETag": response.headers.get("ETag", ""),
                        }
                    response_detail = (await response.text()).strip()
                    suffix = f": {response_detail[:500]}" if response_detail else ""
                    retryable = response.status == 429 or response.status >= 500
                    if not retryable or last_attempt:
                        raise RuntimeError(
                            f"Artifact part {part_number} upload failed with HTTP "
                            f"{response.status}{suffix}"
                        )
        except _TRANSIENT_NETWORK_ERRORS as error:
            if last_attempt:
                raise RuntimeError(
                    f"Artifact part {part_number} upload could not reach the "
                    f"object store: {error}"
                ) from error
        delay = _backoff_seconds(attempt)
        if on_retry:
            on_retry(attempt + 2, total_attempts, delay)
        await asyncio.sleep(delay)

    raise RuntimeError(
        f"Artifact part {part_number} upload failed after {total_attempts} attempts"
    )


async def _upload(
    version_id: str,
    digest: str,
    path: Path,
    config: dict[str, Any],
    size: int,
    on_progress: Callable[[int, int, bool], None] | None = None,
    on_retry: Callable[[int, int, float], None] | None = None,
) -> bool:
    started = await _phantom_request(
        "POST",
        f"/versions/{version_id}/artifacts/{digest}/uploads",
        config,
        {"byte_size": size},
        transient_retries=_TRANSIENT_RETRIES,
        on_retry=on_retry,
    )
    if started.get("reused"):
        if on_progress:
            on_progress(size, size, True)
        return True
    part_size = int(started["part_size"])
    completed = {int(part["PartNumber"]): part for part in started.get("uploaded_parts", [])}
    # Carried forward rather than re-summed per part: a 50 GB model is ~10 000
    # parts, and re-adding every completed one on each is quadratic work spent
    # to render a progress number.
    uploaded_bytes = sum(
        _part_bytes(part_number, part_size, size) for part_number in completed
    )
    if on_progress:
        on_progress(uploaded_bytes, size, False)
    with path.open("rb") as handle:
        part_number = 1
        while chunk := handle.read(part_size):
            if part_number not in completed:
                completed[part_number] = await _put_part(
                    version_id,
                    digest,
                    part_number,
                    chunk,
                    config,
                    on_retry=on_retry,
                )
                uploaded_bytes += len(chunk)
            if on_progress:
                on_progress(uploaded_bytes, size, False)
            part_number += 1
    # Finalizing is the one request that can be retried for free: every part is
    # already in the object store, so a repeat sends no bytes.
    await _phantom_request(
        "POST",
        f"/versions/{version_id}/artifacts/{digest}/uploads/complete",
        config,
        {"parts": [completed[key] for key in sorted(completed)]},
        transient_retries=_TRANSIENT_RETRIES,
        on_retry=on_retry,
    )
    if on_progress:
        on_progress(size, size, False)
    return False


async def _run_publish(job_id: str, body: dict[str, Any]) -> None:
    job = _jobs[job_id]
    temporary_archives: list[Path] = []
    try:
        config = _read_config()
        _job_step(
            job,
            "Inspecting workflow nodes, models, and custom node packages…",
            status="discovering",
            progress=10,
        )
        api_workflow = body["api_workflow"]
        ui_workflow = body["ui_workflow"]
        models = await asyncio.to_thread(_discover_models, api_workflow, ui_workflow)
        packages = await asyncio.to_thread(_discover_packages, api_workflow, ui_workflow)
        temporary_archives = [
            Path(item["_archive_path"])
            for item in packages
            if item.get("_archive_path")
        ]

        def report_external_repository(repo_id: str, index: int, total: int) -> None:
            _job_step(
                job,
                f"Snapshotting external model {index + 1} of {total}: {repo_id}",
                status="snapshotting_external_models",
                progress=20 + int(9 * (index / max(1, total))),
            )

        def report_skipped_repository(repo_id: str, reason: str) -> None:
            _job_log(job, f'Skipped external model "{repo_id}": {reason}.', level="warning")

        _job_step(
            job,
            "Checking custom nodes for external model repositories…",
            status="snapshotting_external_models",
            progress=20,
        )
        models += await asyncio.to_thread(
            _discover_huggingface_models,
            packages,
            report_external_repository,
            report_skipped_repository,
        )
        temporary_archives += [
            Path(item["_local_path"])
            for item in models
            if item.get("archive_format") == "tar.gz"
        ]
        dependencies, uploads = _dependency_progress(models, packages)
        total_upload_bytes = sum(size for _, _, size, _ in uploads)
        job.update(
            dependencies=dependencies,
            bytes_uploaded=0,
            bytes_total=total_upload_bytes,
            dependency_count=len(dependencies),
        )
        _job_log(
            job,
            f"Discovered {len(dependencies)} dependencies; "
            f"{total_upload_bytes} bytes require upload or reuse checks.",
        )
        manifest = {
            "schema_version": 1,
            "idempotency_key": body.get("idempotency_key") or str(uuid.uuid4()),
            "comfyui": {
                "core_version": _comfyui_core_version(),
                "frontend_version": None,
                "publisher_version": PUBLISHER_VERSION,
            },
            "workflow": {"api": api_workflow, "ui": ui_workflow},
            "node_packages": [
                {key: value for key, value in item.items() if not key.startswith("_")}
                for item in packages
            ],
            "models": [
                {key: value for key, value in item.items() if not key.startswith("_")}
                for item in models
            ],
            **(
                {"interface_mapping": body["interface_mapping"]}
                if body.get("interface_mapping")
                else {}
            ),
        }
        _job_step(
            job,
            "Creating an immutable workflow version in Phantom…",
            status="staging",
            progress=30,
        )
        version = await _phantom_request(
            "POST",
            "/versions",
            config,
            {"workflow_id": body["workflow_id"], "manifest": manifest},
        )
        version_id = version["workflow_version_id"]
        _job_log(job, f"Workflow version v{version['version']} staged as {version_id}.")
        _log_server_warnings(job, version)
        completed_upload_bytes = 0
        for index, (digest, path, size, dependency) in enumerate(uploads):
            dependency.update(status="uploading", progress=0, uploaded_bytes=0)
            _job_step(
                job,
                (
                    f"Uploading dependency {index + 1} of {len(uploads)}: "
                    f"{dependency['name']}"
                ),
                status="uploading",
                progress=35 + int(55 * (completed_upload_bytes / max(1, total_upload_bytes))),
                current_dependency_id=dependency["id"],
            )
            bytes_before_dependency = completed_upload_bytes

            def report_upload(uploaded: int, total: int, _reused: bool) -> None:
                bounded_uploaded = min(total, max(0, uploaded))
                dependency.update(
                    uploaded_bytes=bounded_uploaded,
                    progress=int(100 * (bounded_uploaded / max(1, total))),
                )
                aggregate_uploaded = bytes_before_dependency + bounded_uploaded
                job.update(
                    bytes_uploaded=aggregate_uploaded,
                    progress=35
                    + int(55 * (aggregate_uploaded / max(1, total_upload_bytes))),
                )

            def report_retry(next_attempt: int, total_attempts: int, delay: float) -> None:
                message = (
                    f"The connection to Phantom failed. Retrying {dependency['name']} "
                    f"in {delay:g}s (attempt {next_attempt} of {total_attempts})…"
                )
                job.update(message=message)
                _job_log(job, message, level="warning", dependency_id=dependency["id"])

            reused = await _upload(
                version_id,
                digest,
                path,
                config,
                size,
                on_progress=report_upload,
                on_retry=report_retry,
            )
            dependency.update(
                status="reused" if reused else "uploaded",
                progress=100,
                uploaded_bytes=size,
            )
            _job_log(
                job,
                (
                    f"Reused {dependency['name']}; the artifact is already in Phantom."
                    if reused
                    else f"Uploaded {dependency['name']} ({size} bytes)."
                ),
                dependency_id=dependency["id"],
            )
            completed_upload_bytes += size
            job.update(bytes_uploaded=completed_upload_bytes)

        _job_step(
            job,
            "Verifying dependencies and finalizing the workflow version…",
            status="finalizing",
            progress=92,
            current_dependency_id=None,
        )
        version = await _phantom_request("POST", f"/versions/{version_id}/finalize", config)
        _log_server_warnings(job, version)
        _job_step(
            job,
            f"Version v{version['version']} is ready for review.",
            status="completed",
            progress=100,
            bytes_uploaded=total_upload_bytes,
            version=version,
        )
    except Exception as error:  # surfaced verbatim only to the local authenticated browser session
        for dependency in job.get("dependencies", []):
            if dependency.get("status") == "uploading":
                dependency.update(status="failed", error=str(error))
        job.update(status="failed", message="Publishing failed", error=str(error))
        _job_log(job, f"{type(error).__name__}: {error}", level="error")
    finally:
        for archive in temporary_archives:
            shutil.rmtree(archive.parent, ignore_errors=True)


def register_routes() -> None:
    routes = PromptServer.instance.routes

    @routes.get("/phantom-publisher/config")
    async def get_config(_: web.Request) -> web.Response:
        config = _read_config()
        origin = _safe_url(config.get("origin"))
        console_origin = _safe_url(config.get("console_origin"))
        return web.json_response(
            {
                "origin": origin,
                "console_origin": console_origin or (_default_console_origin(origin) if origin else None),
                "configured": bool(config.get("token")),
            }
        )

    @routes.put("/phantom-publisher/config")
    async def put_config(request: web.Request) -> web.Response:
        body = await request.json()
        origin = _safe_url(body.get("origin"))
        console_origin = _safe_url(body.get("console_origin"))
        token = body.get("token")
        if (
            not origin
            or not console_origin
            or not isinstance(token, str)
            or not token.startswith("php_")
        ):
            raise web.HTTPBadRequest(
                text="Valid Phantom API and console origins and a publisher token are required"
            )
        _write_config(
            {
                "origin": origin.rstrip("/"),
                "console_origin": console_origin.rstrip("/"),
                "token": token,
            }
        )
        return web.json_response(
            {
                "configured": True,
                "origin": origin.rstrip("/"),
                "console_origin": console_origin.rstrip("/"),
            }
        )

    @routes.get("/phantom-publisher/targets")
    async def targets(_: web.Request) -> web.Response:
        return web.json_response(await _phantom_request("GET", "/targets", _read_config()))

    @routes.post("/phantom-publisher/targets")
    async def create_target(request: web.Request) -> web.Response:
        return web.json_response(await _phantom_request("POST", "/targets", _read_config(), await request.json()))

    @routes.post("/phantom-publisher/publish")
    async def publish(request: web.Request) -> web.Response:
        body = await request.json()
        job_id = str(uuid.uuid4())
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0,
            "message": "Waiting to start…",
            "dependencies": [],
            "bytes_uploaded": 0,
            "bytes_total": 0,
            "dependency_count": 0,
            "current_dependency_id": None,
            "logs": [],
        }
        _job_log(_jobs[job_id], "Publish job queued.")
        asyncio.create_task(_run_publish(job_id, body))
        return web.json_response(_jobs[job_id], status=202)

    @routes.get("/phantom-publisher/jobs/{job_id}")
    async def publish_status(request: web.Request) -> web.Response:
        job = _jobs.get(request.match_info["job_id"])
        if not job:
            raise web.HTTPNotFound()
        return web.json_response(job)
