"""Immutable installed-runtime capture. No installers run in the author's environment.

The archive is an installation artifact, not a machine backup: only distribution
RECORD files and ELF dependencies enter it. Models, home directories, environment
variables, driver libraries and caches are never swept into the archive.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
import platform
import re
import shutil
import site
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlsplit

DRIVER = re.compile(
    r"^(?:lib(?:cuda(?:debugger)?\.so|nvidia-|nvcuvid\.|nvoptix\.|"
    r"(?:GLX|EGL|GLESv1_CM|GLESv2|glxserver|vdpau)_nvidia\.)|nvidia_drv\.so)"
)
# These belong to the base OS ABI, never transplant a loader or libc.
BASE_ABI = re.compile(
    r"^(?:ld-linux|libpython|lib(?:c|m|dl|pthread|rt|resolv|util|anl)\.so)"
)
CUDA_RUNTIME = re.compile(r"^libcudart\.so\.(\d+)")


def driver(path: Path) -> bool:
    """Driver files belong to the destination host, including symlink aliases."""
    return bool(
        DRIVER.match(path.name)
        or (path.is_symlink() and DRIVER.match(path.resolve().name))
    )


def validate_source_root(root: Path) -> None:
    for path in {root.absolute(), root.resolve()}:
        if (
            path in {Path(p) for p in ("/", "/home", "/root", "/opt", "/tmp", "/var")}
            or path in {Path.home().resolve(), Path(sys.prefix).resolve()}
            or re.match(
                r"^/(?:etc|bin|sbin|lib|lib64|usr|dev|proc|sys|boot|run|opt/venv|phantom|opt/phantom-tools)(?:/|$)",
                str(path),
            )
        ):
            raise RuntimeError(
                f"Editable source must be a project directory, not {root}"
            )


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def elf(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(4) == b"\x7fELF"


def os_identity() -> dict[str, str]:
    values = {}
    for line in Path("/etc/os-release").read_text().splitlines():
        key, _, value = line.partition("=")
        values[key] = value.strip('"')
    return {"id": values.get("ID", ""), "version": values.get("VERSION_ID", "")}


def inspect_linkage(
    paths: list[Path], check_cancelled
) -> tuple[dict[str, Path], list[dict]]:
    """Inspect in batches; retain unresolved optional backends as findings."""
    paths = sorted(set(paths))
    libraries: dict[str, Path] = {}
    missing: list[dict] = []
    for start in range(0, len(paths), 100):
        check_cancelled()
        batch = paths[start : start + 100]
        result = subprocess.run(
            ["ldd", *map(str, batch)],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        current = str(batch[0])
        for line in result.stdout.splitlines():
            if line.endswith(":") and not line.startswith((" ", "\t")):
                current = line[:-1]
            match = re.match(r"\s*(\S+) => (\S+)", line)
            if not match:
                continue
            name, target = match.groups()
            if DRIVER.match(name):
                continue
            if target == "not":
                missing.append({"object": current, "library": name})
            elif target.startswith("/") and not driver(Path(target)):
                libraries[target] = Path(target)
        if result.returncode and not result.stdout.strip():
            raise RuntimeError(
                f"Cannot inspect native runtime: {result.stderr.strip()}"
            )
    return libraries, missing


def capture(
    distributions: dict[str, str],
    package_roots: list[Path],
    core_root: Path,
    check_cancelled,
    editable_sources=None,
) -> dict:
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError(
            "Runtime capture requires Linux x86_64; publish from the Linux GPU environment that runs this graph"
        )
    if platform.python_implementation() != "CPython" or getattr(sys, "abiflags", ""):
        raise RuntimeError(
            "Runtime capture requires a standard CPython build (not PyPy, debug or free-threaded Python)"
        )
    operating_system = os_identity()
    if operating_system not in (
        {"id": "ubuntu", "version": "22.04"},
        {"id": "ubuntu", "version": "24.04"},
    ):
        raise RuntimeError(
            f"Runtime capture does not yet support {operating_system}; use Ubuntu 22.04 or 24.04"
        )
    sites = {
        p
        for key in ("purelib", "platlib")
        for p in (
            Path(sysconfig.get_path(key)).absolute(),
            Path(sysconfig.get_path(key)).resolve(),
        )
    }
    active_sites = {Path(p or os.getcwd()).resolve() for p in sys.path} | {
        p.resolve() for p in sites
    }
    user_site = Path(site.getusersitepackages()).resolve()
    user_prefix = Path(site.getuserbase()).resolve()
    if site.ENABLE_USER_SITE and user_site in active_sites:
        sites.update({user_site, Path(site.getusersitepackages()).absolute()})
    files: dict[str, Path] = {}
    import_owners: dict[str, tuple[Path, bool]] = {}
    checked_imports: set[tuple[Path, str, bool]] = set()

    def add_file(destination: str, source: Path) -> None:
        if driver(source):
            return
        previous = files.get(destination)
        if previous is not None and previous.resolve() != source.resolve():
            raise RuntimeError(
                f"Installed files collide after relocation: {previous} and {source}; consolidate the Python installation before publishing"
            )
        files[destination] = source

    def check_import_precedence(relative: Path, installation: Path) -> None:
        # Separate namespace-package portions can be merged, but regular packages
        # or modules shadow one another instead of merging on Python's sys.path.
        parts = relative.parts
        for index, part in enumerate(parts):
            directory = index < len(parts) - 1
            key_part = part if directory else part.split(".", 1)[0]
            if not key_part.isidentifier():
                break
            if not directory and relative.suffix not in {".py", ".pyc", ".so"}:
                break
            key = ".".join((*parts[:index], key_part))
            signature = (installation, key, directory)
            if signature in checked_imports:
                continue
            checked_imports.add(signature)
            folder = installation.joinpath(*parts[: index + 1])
            regular = not directory or any(folder.glob("__init__.*"))
            previous = import_owners.get(key)
            if previous and previous[0] != installation and (previous[1] or regular):
                raise RuntimeError(
                    f"Import precedence for {key} cannot be preserved across {previous[0]} and {installation}; consolidate the Python installation before publishing"
                )
            import_owners[key] = (
                installation,
                regular or bool(previous and previous[1]),
            )

    installed: dict[str, str] = {}
    native_objects: list[Path] = []
    editable_sources = editable_sources or {}
    source_roots = [os.path.abspath(item[0]) for item in editable_sources.values()]
    for root, entries in editable_sources.values():
        validate_source_root(root)
        root = root.resolve()
        for source in entries:
            check_cancelled()
            if not source.resolve().is_relative_to(root):
                raise RuntimeError(
                    f"Editable source symlink escapes its project: {source}"
                )
            add_file("source/" + source.absolute().as_posix().lstrip("/"), source)
            if ".so" in source.name and not driver(source) and elf(source):
                native_objects.append(source)
    for dist in importlib.metadata.distributions():
        check_cancelled()
        try:
            metadata = dist.metadata
            raw_name = metadata.get("Name", "") if metadata is not None else ""
            if not isinstance(raw_name, str):
                continue
            name = re.sub(r"[-_.]+", "-", raw_name).lower()
        except Exception:  # noqa: BLE001, S112 - match lock's metadata isolation
            # The lock skips unreadable, unrelated metadata as well. If this
            # entry was required, the absent-distribution check below still fails.
            continue
        if name not in distributions or name in ("pip", "uv"):
            continue
        if name in installed:
            continue  # Match importlib.metadata's first, import-visible installation.
        installation = Path(dist.locate_file("")).resolve()
        if installation not in active_sites:
            raise RuntimeError(
                f"{name}'s installation is not on the active Python path: {installation}"
            )
        sites.update({installation, Path(dist.locate_file("")).absolute()})
        if dist.version != distributions[name]:
            raise RuntimeError(
                f"{name} changed during capture; stop package installation and publish again"
            )
        direct = json.loads(dist.read_text("direct_url.json") or "{}")
        if direct.get("dir_info", {}).get("editable"):
            source = editable_sources.get(name)
            if (
                not source
                or Path(unquote(urlsplit(direct.get("url", "")).path)).resolve()
                != source[0].resolve()
            ):
                raise RuntimeError(
                    f"{name}'s editable source is missing; repair its installation before publishing"
                )
        if dist.files is None:
            raise RuntimeError(
                f"{name} has no installed file inventory (RECORD); reinstall it before publishing"
            )
        installed[name] = dist.version
        for entry in dist.files:
            source = Path(dist.locate_file(entry))
            if driver(source):
                continue
            if source.suffix == ".pyc":
                try:
                    original = Path(importlib.util.source_from_cache(str(source)))
                except ValueError:
                    original = source.with_suffix(".py")
                # These caches regenerate after archive extraction changes source
                # mtimes. Keep bytecode only when it is the sole implementation.
                if original.is_file():
                    continue
            if not source.is_file():
                raise RuntimeError(
                    f"{name}'s installed file is missing: {entry}; repair the source environment before publishing"
                )
            # Keep wheel-installed scripts/data as well as importable modules.
            absolute = Path(os.path.abspath(source))
            site_root = next(
                (
                    p
                    for p in sorted(sites, key=lambda p: len(p.parts), reverse=True)
                    if absolute.is_relative_to(p)
                ),
                None,
            )
            if site_root:
                relative = absolute.relative_to(site_root)
                check_import_precedence(relative, site_root.resolve())
                destination = "site/" + relative.as_posix()
            elif any(absolute.is_relative_to(Path(p)) for p in source_roots):
                destination = "source/" + absolute.as_posix().lstrip("/")
            elif absolute.is_relative_to(Path(sys.prefix)):
                relative = absolute.relative_to(Path(sys.prefix))
                if ".." in relative.parts:
                    absolute = absolute.resolve()
                    if not absolute.is_relative_to(Path(sys.prefix).resolve()):
                        raise RuntimeError(
                            f"{name} installs outside its Python environment: {entry}"
                        )
                    relative = absolute.relative_to(Path(sys.prefix).resolve())
                destination = "prefix/" + relative.as_posix()
            elif (
                site.ENABLE_USER_SITE
                and user_site in active_sites
                and absolute.resolve().is_relative_to(user_prefix)
                and absolute.resolve().relative_to(user_prefix).parts[0]
                in {"bin", "share", "include", "lib"}
            ):
                destination = (
                    "prefix/" + absolute.resolve().relative_to(user_prefix).as_posix()
                )
            else:
                raise RuntimeError(
                    f"Cannot reproduce {name}'s file outside its Python environment: {entry}"
                )
            if source.suffix == ".pth":
                for line in source.read_text().splitlines():
                    line = line.strip()
                    if line and not line.startswith(("#", "import ", "import\t")):
                        target = source.parent / line
                        preserved_source = any(
                            target.resolve().is_relative_to(Path(p).resolve())
                            for p in source_roots
                        )
                        if Path(line).is_absolute() and not preserved_source:
                            raise RuntimeError(
                                f"{name} uses an absolute Python path {line} that cannot be relocated; install it as a wheel before publishing"
                            )
                        if (
                            not Path(line).is_absolute()
                            and site_root
                            and not target.resolve().is_relative_to(site_root.resolve())
                        ):
                            raise RuntimeError(
                                f"{name} uses a relative Python path {line} that escapes its installation and cannot be relocated; install it as a wheel before publishing"
                            )
                        if not any(
                            target.resolve().is_relative_to(p.resolve())
                            for p in [*sites, *map(Path, source_roots)]
                        ):
                            raise RuntimeError(
                                f"{name} uses an external Python path {line}; install it as a wheel before publishing"
                            )
            add_file(destination, source)
            if ".so" in source.name and elf(source):
                native_objects.append(source)
    absent = set(distributions) - set(installed) - {"pip", "uv"}
    if absent:
        raise RuntimeError(
            f"Distributions disappeared during capture: {', '.join(sorted(absent))}"
        )
    # ComfyUI's version string does not identify locally patched core code.
    # Capture tracked files, including modifications, without models or user data.
    tracked = subprocess.run(
        ["git", "-C", str(core_root), "ls-files", "-z"], capture_output=True, check=True
    ).stdout
    for name in tracked.decode().split("\0"):
        if not name or Path(name).parts[0] in {
            "models",
            "input",
            "output",
            "temp",
            "user",
            "custom_nodes",
        }:
            continue
        source = core_root / name
        if source.is_file():
            add_file("core/" + name, source)
    if "core/main.py" not in files or "core/nodes.py" not in files:
        raise RuntimeError(
            "Cannot capture ComfyUI core; publish from a git checkout containing main.py and nodes.py"
        )
    for root in package_roots:
        for source in root.rglob("*.so*"):
            check_cancelled()
            if source.is_file() and not driver(source) and elf(source):
                native_objects.append(source)
    # ComfyUI video/audio nodes commonly launch these outside Python. Preserve
    # the installed binaries as well as their codec/linker dependencies.
    for name in ("ffmpeg", "ffprobe"):
        executable = shutil.which(name)
        if executable:
            source = Path(executable)
            if not elf(source):
                raise RuntimeError(
                    f"{name} is a wrapper, not an ELF binary; install a native {name} before publishing"
                )
            add_file("executable/" + source.as_posix().lstrip("/"), source)
            native_objects.append(source)
    libraries, missing = inspect_linkage(native_objects, check_cancelled)
    # Include dlopen'd libraries visible in the live ComfyUI process, too.
    loaded = []
    for line in Path("/proc/self/maps").read_text().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) == 6 and parts[5].startswith("/"):
            path = Path(parts[5])
            if ".so" in path.name and path.is_file() and not driver(path):
                loaded.append(path)
                libraries[str(path)] = path
    if loaded:
        extra, _ = inspect_linkage(loaded, check_cancelled)
        libraries.update(extra)
    # ldd is a fresh process: a library already preloaded in ComfyUI can
    # satisfy a SONAME there even when ldd cannot see its directory.
    available = {p.name for p in libraries.values()}
    unresolved = [m for m in missing if m["library"] not in available]
    cuda_family = re.compile(
        r"^lib(?:cudnn|cublas|cublasLt|cudart|cufft|cufftw|curand|cusolver|cusparse|cusparseLt|nvrtc|nvJitLink|nccl|nvToolsExt|cufile|nvshmem|cupti)[._-]"
    )
    missing_cuda = sorted(
        {m["library"] for m in unresolved if cuda_family.match(m["library"])}
    )
    if missing_cuda:
        raise RuntimeError(
            "The source environment has unresolved CUDA dependencies: "
            + ", ".join(missing_cuda)
            + ". Run the graph with its GPU providers and repair its runtime before publishing; Phantom will not guess or change package versions."
        )
    native_dirs = set()
    cuda_majors = {
        int(match[1])
        for name in distributions
        if (match := re.search(r"-cu(\d+)$", name)) and name.startswith("nvidia-")
    }
    cudnn_copies = {}
    for path in libraries.values():
        check_cancelled()
        if driver(path):
            continue
        match = CUDA_RUNTIME.match(path.name)
        if match:
            cuda_majors.add(int(match[1]))
        # CUDA generations of cuDNN can share a SONAME; copying both and
        # sorting their directories is not a safe compatibility policy.
        if path.name.startswith("libcudnn"):
            sha = digest(path)
            prior = cudnn_copies.setdefault(path.name, (sha, path))
            if prior[0] != sha:
                raise RuntimeError(
                    f"Conflicting cuDNN libraries share {path.name}: {prior[1]} and {path}; use one compatible cuDNN stack before publishing"
                )
        if BASE_ABI.match(path.name) or any(
            path.resolve().is_relative_to(p) for p in sites
        ):
            continue
        # Reproduce absolute paths: RPATH and relative dlopen targets keep working.
        add_file("native/" + path.as_posix().lstrip("/"), path)
        native_dirs.add(str(path.parent))
    inventory = []
    for path in [*native_dirs, *source_roots]:
        if not re.fullmatch(r"/(?:[A-Za-z0-9_+.-]+/)*[A-Za-z0-9_+.-]+", path):
            raise RuntimeError(
                f"Runtime library/source path is not portable: {path}; use a path without spaces or special characters"
            )
    required_space = sum(path.stat().st_size for path in files.values())
    if shutil.disk_usage(tempfile.gettempdir()).free < required_space:
        raise RuntimeError(
            f"Runtime capture needs up to {required_space / 1024**3:.1f} GiB temporary space; free disk space or set TMPDIR"
        )
    # Publisher cleans up archive.parent, so every archive MUST own a private
    # directory. Never return an archive directly under the shared TMPDIR.
    temporary = Path(tempfile.mkdtemp(prefix="phantom-runtime-"))
    archive = temporary / "runtime.tar.gz"
    try:
        with (
            archive.open("wb") as raw,
            gzip.GzipFile(
                fileobj=raw, mode="wb", compresslevel=1, mtime=0, filename=""
            ) as zipped,
            tarfile.open(fileobj=zipped, mode="w|") as tar,
        ):
            for destination, source in sorted(files.items()):
                check_cancelled()
                before = source.stat()
                sha = digest(source)
                info = tarfile.TarInfo(destination)
                info.size = before.st_size
                info.mode = 0o755 if before.st_mode & 0o111 else 0o644
                with source.open("rb") as handle:
                    tar.addfile(info, handle)
                after = source.stat()
                if (before.st_size, before.st_mtime_ns) != (
                    after.st_size,
                    after.st_mtime_ns,
                ):
                    raise RuntimeError(
                        f"{source} changed during capture; retry after installation finishes"
                    )
                inventory.append({"path": destination, "sha256": sha})
            data = json.dumps(
                {
                    "files": inventory,
                    "distributions": installed,
                    "source_prefix": sys.prefix,
                },
                sort_keys=True,
            ).encode()
            info = tarfile.TarInfo("inventory.json")
            info.size = len(data)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))
        return {
            "schema_version": 1,
            "archive_sha256": digest(archive),
            "byte_size": archive.stat().st_size,
            "python": platform.python_version(),
            "os": operating_system,
            "machine": platform.machine(),
            "native_dirs": sorted(native_dirs),
            "source_roots": sorted(source_roots),
            "cuda_runtime_majors": sorted(cuda_majors),
            "unresolved": unresolved,
            "file_count": len(inventory),
            "_archive_path": str(archive),
        }
    except BaseException:
        shutil.rmtree(temporary)
        raise
