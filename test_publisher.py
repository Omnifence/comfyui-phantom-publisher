from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any


class _Routes:
    def get(self, _path: str):
        return lambda handler: handler

    def put(self, _path: str):
        return lambda handler: handler

    def post(self, _path: str):
        return lambda handler: handler


def _load_publisher():
    aiohttp = types.ModuleType("aiohttp")
    aiohttp.web = types.ModuleType("aiohttp.web")
    aiohttp.ClientTimeout = lambda **fields: types.SimpleNamespace(**fields)

    # The real hierarchy, only as deep as the publisher relies on it:
    # ClientConnectorDNSError — the failure this module retries — is a
    # ClientConnectionError, and both connection and payload errors are
    # ClientErrors.
    class ClientError(Exception):
        pass

    class ClientConnectionError(ClientError):
        pass

    class ClientConnectorDNSError(ClientConnectionError):
        pass

    class ClientPayloadError(ClientError):
        pass

    aiohttp.ClientError = ClientError
    aiohttp.ClientConnectionError = ClientConnectionError
    aiohttp.ClientConnectorDNSError = ClientConnectorDNSError
    aiohttp.ClientPayloadError = ClientPayloadError
    sys.modules["aiohttp"] = aiohttp
    sys.modules["aiohttp.web"] = aiohttp.web

    folder_paths = types.ModuleType("folder_paths")
    folder_paths.base_path = tempfile.gettempdir()
    folder_paths.get_folder_paths = lambda _kind: []
    sys.modules["folder_paths"] = folder_paths

    server = types.ModuleType("server")
    server.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(routes=_Routes()))
    sys.modules["server"] = server

    path = Path(__file__).with_name("publisher.py")
    spec = importlib.util.spec_from_file_location("phantom_publisher_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


publisher = _load_publisher()


class PublisherDiscoveryTests(unittest.TestCase):
    def test_safe_url_removes_credentials_and_signing_parameters(self):
        self.assertEqual(
            publisher._safe_url("https://user:pass@example.com/model?token=secret#fragment"),
            "https://example.com/model",
        )
        self.assertIsNone(publisher._safe_url("file:///tmp/model"))

    def test_normalized_package_archive_is_reproducible(self):
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary) / "custom-package"
            package.mkdir()
            source = package / "node.py"
            source.write_text("NODE_CLASS_MAPPINGS = {}\n", encoding="utf-8")
            first_path, first_digest, first_size = publisher._archive_package(package)
            os.utime(source, (2_000_000_000, 2_000_000_000))
            second_path, second_digest, second_size = publisher._archive_package(package)
            try:
                self.assertEqual(first_digest, second_digest)
                self.assertEqual(first_size, second_size)
            finally:
                first_path.unlink(missing_ok=True)
                second_path.unlink(missing_ok=True)

    def test_clean_registry_package_is_archived_for_phantom_source_of_truth(self):
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary) / "clean-registry-package"
            package.mkdir()
            (package / "node.py").write_text("NODE_CLASS_MAPPINGS = {}\n", encoding="utf-8")
            original_package_directory = publisher._package_directory
            original_git_metadata = publisher._git_metadata
            original_pip_dependencies = publisher._pip_dependencies
            publisher._package_directory = lambda _cnr_id, _classes: package
            publisher._git_metadata = lambda _directory: {
                "git_commit": "a" * 40,
                "repository_url": "https://github.com/example/package",
                "dirty": False,
            }
            publisher._pip_dependencies = lambda _directory: []
            sys.modules["nodes"] = types.SimpleNamespace(NODE_CLASS_MAPPINGS={})
            try:
                packages = publisher._discover_packages(
                    {"1": {"class_type": "ExampleNode", "inputs": {}}},
                    {
                        "nodes": [
                            {
                                "id": 1,
                                "properties": {"cnr_id": "example-package", "ver": "1.2.3"},
                            }
                        ]
                    },
                )
            finally:
                publisher._package_directory = original_package_directory
                publisher._git_metadata = original_git_metadata
                publisher._pip_dependencies = original_pip_dependencies

            self.assertEqual(len(packages), 1)
            discovered = packages[0]
            self.assertFalse(discovered["dirty"])
            self.assertRegex(discovered["archive_sha256"], r"^[a-f0-9]{64}$")
            self.assertGreater(discovered["_archive_size"], 0)
            Path(discovered["_archive_path"]).unlink(missing_ok=True)

    def test_detects_literal_huggingface_model_used_inside_a_node_package(self):
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary)
            (package / "node.py").write_text(
                'classifier = pipeline("image-classification", '
                'model="Falconsai/nsfw_image_detection")\n'
                'dynamic = AutoModel.from_pretrained(model_id)\n',
                encoding="utf-8",
            )
            self.assertEqual(
                publisher._literal_huggingface_repositories(package),
                {"Falconsai/nsfw_image_detection"},
            )

    def test_detects_positional_huggingface_repository_arguments(self):
        # Positional calls are the common literal form. Reading keywords alone
        # left these models out of the archive, so the image downloaded mutable
        # upstream content at run time or failed with no network at all.
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary)
            (package / "node.py").write_text(
                'snapshot = snapshot_download("org/snapshot-model")\n'
                'weights = hf_hub_download("org/hub-model", "model.safetensors")\n'
                'clf = pipeline("image-classification", "org/pipeline-model")\n'
                'enc = AutoModel.from_pretrained("org/pretrained-model")\n',
                encoding="utf-8",
            )
            self.assertEqual(
                publisher._literal_huggingface_repositories(package),
                {
                    "org/snapshot-model",
                    "org/hub-model",
                    "org/pipeline-model",
                    "org/pretrained-model",
                },
            )

    def test_ignores_a_positional_task_that_is_not_a_repository(self):
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary)
            # The single-argument form passes a TASK, not a repository.
            (package / "node.py").write_text(
                'clf = pipeline("image-classification")\n'
                'dynamic = snapshot_download(repo_id_variable)\n',
                encoding="utf-8",
            )
            self.assertEqual(publisher._literal_huggingface_repositories(package), set())

    def test_preserves_distinct_model_filenames_with_identical_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoints = root / "models" / "checkpoints"
            checkpoints.mkdir(parents=True)
            (checkpoints / "primary.safetensors").write_bytes(b"same model")
            (checkpoints / "alias.safetensors").write_bytes(b"same model")
            original_base = publisher.folder_paths.base_path
            original_get_paths = publisher.folder_paths.get_folder_paths
            original_names = getattr(publisher.folder_paths, "folder_names_and_paths", None)
            publisher.folder_paths.base_path = str(root)
            publisher.folder_paths.get_folder_paths = lambda kind: (
                [str(checkpoints)] if kind == "checkpoints" else []
            )
            publisher.folder_paths.folder_names_and_paths = {"checkpoints": object()}
            try:
                models = publisher._discover_models(
                    {
                        "1": {"inputs": {"ckpt_name": "primary.safetensors"}},
                        "2": {"inputs": {"ckpt_name": "alias.safetensors"}},
                    },
                    {},
                )
            finally:
                publisher.folder_paths.base_path = original_base
                publisher.folder_paths.get_folder_paths = original_get_paths
                if original_names is None:
                    del publisher.folder_paths.folder_names_and_paths
                else:
                    publisher.folder_paths.folder_names_and_paths = original_names

            self.assertEqual({model["filename"] for model in models}, {
                "primary.safetensors",
                "alias.safetensors",
            })
            self.assertEqual(len({model["sha256"] for model in models}), 1)

    def test_rejects_strings_that_cannot_name_a_file(self):
        self.assertTrue(publisher._is_plausible_filename("primary.safetensors"))
        self.assertTrue(publisher._is_plausible_filename("sdxl/refiner.safetensors"))
        self.assertFalse(publisher._is_plausible_filename(""))
        self.assertFalse(publisher._is_plausible_filename("a prompt\nwith a newline"))
        self.assertFalse(publisher._is_plausible_filename("x" * 256))
        self.assertFalse(publisher._is_plausible_filename("/".join(["x" * 200] * 40)))

    def test_resolve_model_rejects_prompt_text_before_touching_the_filesystem(self):
        # The failure is platform-dependent — Linux raises ENAMETOOLONG from
        # is_file(), newer macOS builds of CPython swallow it — so the contract
        # under test is that an implausible name never reaches the filesystem
        # at all, on any platform.
        looked_up: list[str] = []
        original_get_paths = publisher.folder_paths.get_folder_paths
        publisher.folder_paths.get_folder_paths = lambda kind: looked_up.append(kind) or []
        try:
            self.assertIsNone(
                publisher._resolve_model("side angle. 25 year old female, " + "x" * 600, "checkpoints")
            )
            self.assertEqual(looked_up, [])
            self.assertIsNone(publisher._resolve_model("primary.safetensors", "checkpoints"))
            self.assertEqual(looked_up, ["checkpoints"])
        finally:
            publisher.folder_paths.get_folder_paths = original_get_paths

    def test_prompt_text_does_not_fail_discovery(self):
        # Discovery probes every string input against every model directory, so
        # it also sees prompts. Joining a prompt to a model root and calling
        # is_file() raises OSError(ENAMETOOLONG) rather than returning False,
        # which failed the whole publish at 10%.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoints = root / "models" / "checkpoints"
            checkpoints.mkdir(parents=True)
            (checkpoints / "primary.safetensors").write_bytes(b"a model")
            original_base = publisher.folder_paths.base_path
            original_get_paths = publisher.folder_paths.get_folder_paths
            original_names = getattr(publisher.folder_paths, "folder_names_and_paths", None)
            publisher.folder_paths.base_path = str(root)
            publisher.folder_paths.get_folder_paths = lambda kind: (
                [str(checkpoints)] if kind == "checkpoints" else []
            )
            publisher.folder_paths.folder_names_and_paths = {"checkpoints": object()}
            try:
                models = publisher._discover_models(
                    {
                        "1": {"inputs": {"ckpt_name": "primary.safetensors"}},
                        "2": {"inputs": {"text": "side angle. 25 year old female, " + "x" * 600}},
                    },
                    {},
                )
            finally:
                publisher.folder_paths.base_path = original_base
                publisher.folder_paths.get_folder_paths = original_get_paths
                if original_names is None:
                    del publisher.folder_paths.folder_names_and_paths
                else:
                    publisher.folder_paths.folder_names_and_paths = original_names

            self.assertEqual([model["filename"] for model in models], ["primary.safetensors"])


class HuggingFaceRepositoryTypeTests(unittest.TestCase):
    """
    A Hugging Face id is `org/name` whether it names a model, a Space or a
    dataset. snapshot_download assumes a model, and the hub answers a wrong
    type with a 401 rather than a 404, so one Space id failed a whole publish.
    """

    def _install_hub(self, exists=None, api=None):
        hub = types.ModuleType("huggingface_hub")
        if api is None:
            probed: list[tuple[str, str]] = []

            class _Api:
                def repo_exists(self, repo_id: str, repo_type: str | None = None) -> bool:
                    probed.append((repo_id, repo_type or "model"))
                    return exists(repo_id, repo_type or "model")

            api = _Api
            self.probed = probed
        hub.HfApi = api
        sys.modules["huggingface_hub"] = hub
        self.addCleanup(sys.modules.pop, "huggingface_hub", None)

    def test_resolves_a_space_id_that_is_not_a_model(self):
        self._install_hub(exists=lambda _repo_id, repo_type: repo_type == "space")
        self.assertEqual(
            publisher._huggingface_repo_type("xxparthparekhxx/NudeNet-FastAPI"), "space"
        )
        # A model still costs a single probe; a Space costs two.
        self.assertEqual(
            self.probed,
            [
                ("xxparthparekhxx/NudeNet-FastAPI", "model"),
                ("xxparthparekhxx/NudeNet-FastAPI", "space"),
            ],
        )

    def test_reports_an_id_that_names_no_repository(self):
        self._install_hub(exists=lambda _repo_id, _repo_type: False)
        self.assertIsNone(publisher._huggingface_repo_type("some/local-path"))

    def test_treats_a_probe_error_as_a_miss_and_keeps_probing(self):
        def exists(_repo_id: str, repo_type: str) -> bool:
            if repo_type == "model":
                raise RuntimeError("gateway timeout")
            return repo_type == "dataset"

        self._install_hub(exists=exists)
        self.assertEqual(publisher._huggingface_repo_type("org/corpus"), "dataset")

    def test_assumes_a_model_when_huggingface_hub_predates_repo_exists(self):
        class _OldApi:
            pass

        self._install_hub(api=_OldApi)
        self.assertEqual(publisher._huggingface_repo_type("org/model"), "model")

    def test_namespaces_the_recorded_url_by_repository_type(self):
        self.assertEqual(
            publisher._huggingface_url("org/model", "model", "abc"),
            "https://huggingface.co/org/model/tree/abc",
        )
        self.assertEqual(
            publisher._huggingface_url("org/space", "space", "abc"),
            "https://huggingface.co/spaces/org/space/tree/abc",
        )
        self.assertEqual(
            publisher._huggingface_url("org/corpus", "dataset", "abc"),
            "https://huggingface.co/datasets/org/corpus/tree/abc",
        )


class HuggingFaceDiscoveryTests(unittest.TestCase):
    def _discover(self, repo_types: dict[str, str | None]):
        skipped: list[tuple[str, str]] = []
        original_type = publisher._huggingface_repo_type
        original_snapshot = publisher._huggingface_snapshot
        original_repositories = publisher._literal_huggingface_repositories
        cache = tempfile.TemporaryDirectory()
        self.addCleanup(cache.cleanup)
        (Path(cache.name) / "config.json").write_text("{}", encoding="utf-8")
        publisher._huggingface_repo_type = lambda repo_id: repo_types[repo_id]
        publisher._huggingface_snapshot = lambda repo_id, repo_type: (Path(cache.name), "abc123")
        publisher._literal_huggingface_repositories = lambda _source: set(repo_types)
        try:
            models = publisher._discover_huggingface_models(
                [{"_source_files": ["node.py"]}],
                None,
                lambda repo_id, reason: skipped.append((repo_id, reason)),
            )
        finally:
            publisher._huggingface_repo_type = original_type
            publisher._huggingface_snapshot = original_snapshot
            publisher._literal_huggingface_repositories = original_repositories
        for model in models:
            self.addCleanup(
                shutil.rmtree, str(Path(model["_local_path"]).parent), ignore_errors=True
            )
        return models, skipped

    def test_archives_a_space_and_records_its_type_and_url(self):
        models, skipped = self._discover({"xxparthparekhxx/NudeNet-FastAPI": "space"})
        self.assertEqual(skipped, [])
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]["repo_type"], "space")
        self.assertEqual(
            models[0]["source_urls"],
            ["https://huggingface.co/spaces/xxparthparekhxx/NudeNet-FastAPI/tree/abc123"],
        )
        self.assertEqual(models[0]["install_path"], "opt/phantom/huggingface/hub")

    def test_skips_an_unresolvable_id_without_failing_the_publish(self):
        # `org/name` is also the shape of a local relative path, so the source
        # scan produces false positives. One must not fail a publish that has
        # already archived real dependencies.
        models, skipped = self._discover({"org/real-model": "model", "some/local-path": None})
        self.assertEqual([model["external_repository"] for model in models], ["org/real-model"])
        self.assertEqual(
            skipped, [("some/local-path", "no model, Space or dataset repository has that id")]
        )


class ClientTimeoutTests(unittest.TestCase):
    def test_waits_out_a_slow_answer_but_not_a_dead_socket(self):
        # aiohttp caps every request at 5 minutes by default. Phantom hashes a
        # whole artifact before it answers the finalize call, so that default
        # failed a 28 GB publish after every byte had already landed.
        timeout = publisher._client_timeout()
        self.assertIsNone(timeout.total)
        self.assertEqual(timeout.sock_connect, 30)
        self.assertEqual(timeout.sock_read, 15 * 60)

    def test_bounds_name_resolution(self):
        # `connect` is the only field that covers DNS resolution — aiohttp
        # applies `sock_connect` to the handshake alone. Leaving it None let a
        # stalled resolver hold a 28 GB publish for 20 minutes before it
        # reported "Name or service not known".
        timeout = publisher._client_timeout()
        self.assertEqual(timeout.connect, 60)

    def test_backoff_is_exponential_and_capped(self):
        self.assertEqual(
            [publisher._backoff_seconds(attempt) for attempt in range(6)],
            [1.0, 2.0, 4.0, 8.0, 16.0, 30.0],
        )


class PipDependencyCaptureTests(unittest.TestCase):
    """
    A node package that imports a module it never declares in requirements.txt
    installs cleanly, then fails to import at ComfyUI startup. ComfyUI logs the
    error and carries on, so the package registers none of its node classes and
    every render fails with a 400 "custom node may not be installed". The
    publisher runs inside a working ComfyUI, so the installed set is the only
    place that undeclared dependency can be recovered from.
    """

    def _package(self, temporary: str, source: str, requirements: str | None = None) -> Path:
        package = Path(temporary) / "DiffusionWave_PickResolution"
        package.mkdir()
        (package / "nodes.py").write_text(source, encoding="utf-8")
        if requirements is not None:
            (package / "requirements.txt").write_text(requirements, encoding="utf-8")
        return package

    def test_captures_an_undeclared_import_pinned_to_the_installed_version(self):
        with tempfile.TemporaryDirectory() as temporary:
            package = self._package(temporary, "import svgwrite\nimport os\n")
            with _installed({"svgwrite": ("svgwrite", "1.4.3")}):
                self.assertEqual(publisher._pip_dependencies(package), ["svgwrite==1.4.3"])

    def test_skips_what_the_package_already_declares(self):
        # Declared dependencies install from requirements.txt; repeating them
        # here would pin a version the package deliberately left open.
        with tempfile.TemporaryDirectory() as temporary:
            package = self._package(temporary, "import svgwrite\n", requirements="svgwrite>=1.4\n")
            with _installed({"svgwrite": ("svgwrite", "1.4.3")}):
                self.assertEqual(publisher._pip_dependencies(package), [])

    def test_skips_the_stdlib_comfyui_internals_and_the_packages_own_modules(self):
        with tempfile.TemporaryDirectory() as temporary:
            package = self._package(
                temporary,
                "import os\nimport json\nimport folder_paths\nimport comfy.utils\nimport nodes\n"
                "from .helpers import thing\nimport helpers\n",
            )
            (package / "helpers.py").write_text("thing = 1\n", encoding="utf-8")
            with _installed({"helpers": ("helpers", "9.9.9")}):
                self.assertEqual(publisher._pip_dependencies(package), [])

    def test_never_pins_the_torch_stack_the_base_image_provides(self):
        # Re-pinning torch from a developer machine is how a working image
        # becomes a broken one.
        with tempfile.TemporaryDirectory() as temporary:
            package = self._package(temporary, "import torch\nimport numpy\nimport svgwrite\n")
            with _installed(
                {
                    "torch": ("torch", "2.4.0"),
                    "numpy": ("numpy", "1.26.4"),
                    "svgwrite": ("svgwrite", "1.4.3"),
                }
            ):
                self.assertEqual(publisher._pip_dependencies(package), ["svgwrite==1.4.3"])

    def test_ignores_an_import_with_no_installed_distribution(self):
        with tempfile.TemporaryDirectory() as temporary:
            package = self._package(temporary, "import not_installed_anywhere\n")
            with _installed({}):
                self.assertEqual(publisher._pip_dependencies(package), [])


class _installed:
    """Pin importlib.metadata to a fixed installed set for the duration."""

    def __init__(self, mapping: dict[str, tuple[str, str]]):
        self._mapping = mapping

    def __enter__(self):
        self._packages_distributions = publisher.importlib.metadata.packages_distributions
        self._version = publisher.importlib.metadata.version
        versions = {distribution: version for distribution, version in self._mapping.values()}

        def version(name: str) -> str:
            if name not in versions:
                raise publisher.importlib.metadata.PackageNotFoundError(name)
            return versions[name]

        publisher.importlib.metadata.packages_distributions = lambda: {
            module: [distribution] for module, (distribution, _) in self._mapping.items()
        }
        publisher.importlib.metadata.version = version
        return self

    def __exit__(self, *_exc: object) -> None:
        publisher.importlib.metadata.packages_distributions = self._packages_distributions
        publisher.importlib.metadata.version = self._version


class ConfigPermissionTests(unittest.TestCase):
    """
    The config holds the publisher token. Writing it and chmod-ing afterwards
    left it world-readable for the width of that window.
    """

    def _isolated_config(self, root: Path):
        publisher.folder_paths.base_path = str(root)
        (root / "user").mkdir(parents=True, exist_ok=True)

    def test_console_origin_uses_the_admin_frontend_in_local_development(self):
        self.assertEqual(
            publisher._default_console_origin("http://localhost:3060"),
            "http://localhost:3062",
        )
        self.assertEqual(
            publisher._default_console_origin("http://127.0.0.1:3060"),
            "http://127.0.0.1:3062",
        )

    def test_console_origin_uses_the_production_admin_frontend(self):
        self.assertEqual(
            publisher._default_console_origin("https://api.phantomrouter.ai"),
            "https://app.phantomrouter.ai",
        )

    def test_console_origin_leaves_a_custom_same_origin_deployment_alone(self):
        self.assertEqual(
            publisher._default_console_origin("https://phantom.example.test/api"),
            "https://phantom.example.test",
        )

    def test_config_is_owner_only_the_moment_it_exists(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._isolated_config(root)
            publisher._write_config({"token": "secret-token", "origin": "https://example.test"})
            config = root / "user" / publisher.CONFIG_FILENAME
            self.assertEqual(config.stat().st_mode & 0o777, 0o600)
            self.assertEqual(publisher._read_config()["token"], "secret-token")

    def test_never_writes_the_token_through_a_planted_temp_symlink(self):
        # The previous writer used write_text, which follows a symlink: a `.tmp`
        # planted at the config path sent the token wherever it pointed. This is
        # the behaviour that actually distinguishes the fix — the final file mode
        # does not, since chmod-after-write ended at 0600 too. The race window
        # itself cannot be asserted deterministically.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._isolated_config(root)
            victim = root / "victim.txt"
            victim.write_text("original", encoding="utf-8")
            (root / "user" / publisher.CONFIG_FILENAME).with_suffix(".tmp").symlink_to(victim)

            publisher._write_config({"token": "secret-token"})

            self.assertEqual(victim.read_text(encoding="utf-8"), "original")
            config = root / "user" / publisher.CONFIG_FILENAME
            self.assertEqual(config.stat().st_mode & 0o777, 0o600)
            self.assertFalse(config.is_symlink())


class PublisherProgressTests(unittest.IsolatedAsyncioTestCase):
    def test_publish_log_records_context_and_keeps_a_bounded_tail(self):
        job = {"status": "uploading", "logs": []}
        publisher._job_log(
            job,
            "Retrying portrait.safetensors",
            level="warning",
            dependency_id="model-1",
        )

        first = job["logs"][0]
        self.assertEqual(first["sequence"], 1)
        self.assertEqual(first["phase"], "uploading")
        self.assertEqual(first["level"], "warning")
        self.assertEqual(first["dependency_id"], "model-1")
        self.assertIn("Retrying portrait.safetensors", first["message"])
        self.assertIn("+00:00", first["timestamp"])

        for index in range(publisher.PUBLISH_LOG_LIMIT + 5):
            publisher._job_log(job, f"entry {index}")
        self.assertEqual(len(job["logs"]), publisher.PUBLISH_LOG_LIMIT)
        self.assertEqual(job["logs"][-1]["message"], "entry 204")

    def test_dependency_progress_lists_every_dependency_without_local_paths(self):
        models = [
            {
                "filename": "portrait.safetensors",
                "model_type": "checkpoints",
                "comfyui_path": "models/checkpoints",
                "sha256": "a" * 64,
                "byte_size": 1024,
                "_local_path": "/private/models/portrait.safetensors",
            }
        ]
        packages = [
            {
                "cnr_id": "comfyui-example",
                "version": "1.2.3",
                "class_types": ["ExampleNode"],
                "archive_sha256": "b" * 64,
                "_archive_size": 512,
                "_archive_path": "/private/packages/example.tar.gz",
            },
            {
                "cnr_id": "registry-only",
                "version": "2.0.0",
                "class_types": ["RegistryNode"],
            },
        ]

        dependencies, uploads = publisher._dependency_progress(models, packages)

        self.assertEqual([item["name"] for item in dependencies], [
            "portrait.safetensors",
            "comfyui-example",
            "registry-only",
        ])
        self.assertEqual(len(uploads), 2)
        self.assertEqual(dependencies[2]["status"], "not_required")
        self.assertEqual(dependencies[2]["progress"], 100)
        self.assertNotIn("/private/", repr(dependencies))

    def test_multipart_progress_counts_resumed_and_final_short_parts(self):
        # Parts 1 and 3 of a 25-byte object at a 10-byte part size: a full part
        # plus the short final one.
        self.assertEqual(publisher._part_bytes(1, 10, 25), 10)
        self.assertEqual(publisher._part_bytes(3, 10, 25), 5)
        self.assertEqual(publisher._part_bytes(4, 10, 25), 0)

    async def test_reused_upload_reports_complete_without_reading_the_local_file(self):
        progress: list[tuple[int, int, bool]] = []
        original_request = publisher._phantom_request

        async def reused(*_args, **_kwargs):
            return {"reused": True}

        publisher._phantom_request = reused
        try:
            was_reused = await publisher._upload(
                "version-id",
                "a" * 64,
                Path("/path/that/does/not/exist"),
                {"origin": "https://example.test", "token": "php_secret"},
                2048,
                on_progress=lambda uploaded, total, reused: progress.append(
                    (uploaded, total, reused)
                ),
            )
        finally:
            publisher._phantom_request = original_request

        self.assertTrue(was_reused)
        self.assertEqual(progress, [(2048, 2048, True)])

    async def test_multipart_upload_reports_resumed_and_new_parts(self):
        progress: list[tuple[int, int, bool]] = []
        requested_paths: list[str] = []
        uploaded_chunks: list[bytes] = []
        part_timeouts: list[Any] = []
        original_request = publisher._phantom_request
        original_session = getattr(publisher.aiohttp, "ClientSession", None)

        request_options: dict[str, dict[str, Any]] = {}

        async def request(_method, path, _config, _body=None, **options):
            requested_paths.append(path)
            request_options[path] = options
            if path.endswith("/uploads"):
                return {
                    "reused": False,
                    "part_size": 4,
                    "uploaded_parts": [{"PartNumber": 1, "ETag": "existing"}],
                }
            if "/parts/" in path:
                return {"upload_url": f"https://uploads.test{path}"}
            return {"ok": True}

        class Response:
            status = 200
            headers = {"ETag": "uploaded"}

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_exc):
                return None

        class Session:
            def __init__(self, *, timeout=None, **_options):
                part_timeouts.append(timeout)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_exc):
                return None

            def put(self, _url, *, data):
                uploaded_chunks.append(data)
                return Response()

        publisher._phantom_request = request
        publisher.aiohttp.ClientSession = Session
        try:
            with tempfile.TemporaryDirectory() as temporary:
                artifact = Path(temporary) / "artifact.bin"
                artifact.write_bytes(b"0123456789")
                was_reused = await publisher._upload(
                    "version-id",
                    "a" * 64,
                    artifact,
                    {"origin": "https://example.test", "token": "php_secret"},
                    10,
                    on_progress=lambda uploaded, total, reused: progress.append(
                        (uploaded, total, reused)
                    ),
                )
        finally:
            publisher._phantom_request = original_request
            if original_session is None:
                del publisher.aiohttp.ClientSession
            else:
                publisher.aiohttp.ClientSession = original_session

        self.assertFalse(was_reused)
        self.assertEqual(uploaded_chunks, [b"4567", b"89"])
        self.assertEqual(progress[0], (4, 10, False))
        self.assertIn((8, 10, False), progress)
        self.assertEqual(progress[-1], (10, 10, False))
        self.assertEqual(sum("/parts/" in path for path in requested_paths), 2)
        # Every part PUT carries the explicit policy, not aiohttp's 5-minute
        # default total.
        self.assertTrue(part_timeouts)
        self.assertTrue(all(timeout.total is None for timeout in part_timeouts))
        self.assertTrue(all(timeout.sock_read == 15 * 60 for timeout in part_timeouts))
        # Finalizing is free to retry: every part is already in the object
        # store, so a repeat sends no bytes.
        complete = next(path for path in requested_paths if path.endswith("/complete"))
        self.assertEqual(
            request_options[complete]["transient_retries"], publisher._TRANSIENT_RETRIES
        )


class PhantomRequestTests(unittest.IsolatedAsyncioTestCase):
    async def test_bodyless_post_sends_an_empty_json_object(self):
        captured: dict[str, object] = {}

        class Response:
            status = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_exc: object) -> None:
                return None

            async def text(self):
                return '{"ok": true}'

        class Session:
            def __init__(self, *, headers, timeout=None):
                captured["headers"] = headers

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_exc: object) -> None:
                return None

            def request(self, method, url, **options):
                captured.update(method=method, url=url, options=options)
                return Response()

        original = getattr(publisher.aiohttp, "ClientSession", None)
        publisher.aiohttp.ClientSession = Session
        try:
            result = await publisher._phantom_request(
                "POST",
                "/versions/version-id/finalize",
                {"origin": "https://api.example.test", "token": "php_secret"},
            )
        finally:
            if original is None:
                del publisher.aiohttp.ClientSession
            else:
                publisher.aiohttp.ClientSession = original

        self.assertEqual(result, {"ok": True})
        self.assertEqual(captured["options"], {"json": {}})
        self.assertEqual(
            captured["headers"],
            {
                "Authorization": "Bearer php_secret",
                "Content-Type": "application/json",
            },
        )

    async def test_retries_a_transient_server_error_when_the_caller_opts_in(self):
        statuses = [500, 200]
        attempts: list[int] = []
        retries: list[tuple[int, int, float]] = []
        sleeps: list[float] = []

        class Response:
            def __init__(self, status):
                self.status = status

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_exc):
                return None

            async def text(self):
                return '{"ok": true}' if self.status == 200 else '{"message": "temporary"}'

        class Session:
            def __init__(self, *, headers, timeout=None):
                self.headers = headers

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_exc):
                return None

            def request(self, _method, _url, **_options):
                attempts.append(1)
                return Response(statuses.pop(0))

        async def sleep(delay):
            sleeps.append(delay)

        original_session = getattr(publisher.aiohttp, "ClientSession", None)
        original_sleep = publisher.asyncio.sleep
        publisher.aiohttp.ClientSession = Session
        publisher.asyncio.sleep = sleep
        try:
            result = await publisher._phantom_request(
                "POST",
                "/versions/version-id/artifacts/digest/uploads",
                {"origin": "https://api.example.test", "token": "php_secret"},
                {"byte_size": 42},
                transient_retries=2,
                on_retry=lambda attempt, total, delay: retries.append((attempt, total, delay)),
            )
        finally:
            publisher.asyncio.sleep = original_sleep
            if original_session is None:
                del publisher.aiohttp.ClientSession
            else:
                publisher.aiohttp.ClientSession = original_session

        self.assertEqual(result, {"ok": True})
        self.assertEqual(len(attempts), 2)
        self.assertEqual(retries, [(2, 3, 1.0)])
        self.assertEqual(sleeps, [1.0])

    async def test_retries_a_transient_error_whose_body_is_not_json(self):
        # A reverse proxy answers 502/503 with its own HTML page. Calling
        # response.json() on that raises ContentTypeError, which escaped the
        # retry loop entirely — so the one class of failure these retries exist
        # for was the one class that never retried.
        responses = [
            (503, "<html><body>503 Service Unavailable</body></html>"),
            (200, '{"ok": true}'),
        ]
        attempts: list[int] = []
        sleeps: list[float] = []

        class Response:
            def __init__(self, status, body):
                self.status = status
                self._body = body

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_exc):
                return None

            async def text(self):
                return self._body

            async def json(self):
                raise AssertionError("must not parse the body as JSON before classifying it")

        class Session:
            def __init__(self, *, headers, timeout=None):
                self.headers = headers

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_exc):
                return None

            def request(self, _method, _url, **_options):
                attempts.append(1)
                return Response(*responses.pop(0))

        async def sleep(delay):
            sleeps.append(delay)

        original_session = getattr(publisher.aiohttp, "ClientSession", None)
        original_sleep = publisher.asyncio.sleep
        publisher.aiohttp.ClientSession = Session
        publisher.asyncio.sleep = sleep
        try:
            result = await publisher._phantom_request(
                "POST",
                "/versions/version-id/artifacts/digest/uploads",
                {"origin": "https://api.example.test", "token": "php_secret"},
                {"byte_size": 42},
                transient_retries=2,
            )
        finally:
            publisher.asyncio.sleep = original_sleep
            if original_session is None:
                del publisher.aiohttp.ClientSession
            else:
                publisher.aiohttp.ClientSession = original_session

        self.assertEqual(result, {"ok": True})
        self.assertEqual(len(attempts), 2)
        self.assertEqual(sleeps, [1.0])

    async def test_reports_a_non_json_error_body_after_the_last_attempt(self):
        class Response:
            status = 400

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_exc):
                return None

            async def text(self):
                return "Bad Request"

        class Session:
            def __init__(self, *, headers, timeout=None):
                self.headers = headers

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_exc):
                return None

            def request(self, _method, _url, **_options):
                return Response()

        original_session = getattr(publisher.aiohttp, "ClientSession", None)
        publisher.aiohttp.ClientSession = Session
        try:
            with self.assertRaises(RuntimeError) as caught:
                await publisher._phantom_request(
                    "POST",
                    "/versions/version-id/finalize",
                    {"origin": "https://api.example.test", "token": "php_secret"},
                )
        finally:
            if original_session is None:
                del publisher.aiohttp.ClientSession
            else:
                publisher.aiohttp.ClientSession = original_session

        # The proxy's own text, not a generic "Request failed".
        self.assertIn("Bad Request", str(caught.exception))


class TransientNetworkFailureTests(unittest.IsolatedAsyncioTestCase):
    """
    A DNS, TCP or TLS failure never produces a status code, so the status-based
    retry never saw it: the exception escaped the loop and failed the publish
    outright. A 28 GB run died that way after 20 minutes and 7 of 12 uploaded
    dependencies, on a resolver that was answering again seconds later.
    """

    def setUp(self):
        self._original_session = getattr(publisher.aiohttp, "ClientSession", None)
        self._original_sleep = publisher.asyncio.sleep
        self.sleeps: list[float] = []

        async def sleep(delay):
            self.sleeps.append(delay)

        publisher.asyncio.sleep = sleep

    def tearDown(self):
        publisher.asyncio.sleep = self._original_sleep
        if self._original_session is None:
            publisher.aiohttp.ClientSession = None
            del publisher.aiohttp.ClientSession
        else:
            publisher.aiohttp.ClientSession = self._original_session

    def _dns_error(self):
        return publisher.aiohttp.ClientConnectorDNSError(
            "Cannot connect to host api.phantomrouter.ai:443 ssl:default "
            "[Name or service not known]"
        )

    async def test_retries_a_dns_failure_and_succeeds(self):
        attempts: list[int] = []
        error = self._dns_error()

        class Response:
            status = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_exc):
                return None

            async def text(self):
                return '{"ok": true}'

        class Session:
            def __init__(self, *, headers, timeout=None):
                self.headers = headers

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_exc):
                return None

            def request(self, _method, _url, **_options):
                attempts.append(1)
                if len(attempts) == 1:
                    raise error
                return Response()

        publisher.aiohttp.ClientSession = Session
        result = await publisher._phantom_request(
            "POST",
            "/versions/version-id/artifacts/digest/uploads",
            {"origin": "https://api.phantomrouter.ai", "token": "php_secret"},
            {"byte_size": 42},
            transient_retries=publisher._TRANSIENT_RETRIES,
        )

        self.assertEqual(result, {"ok": True})
        self.assertEqual(len(attempts), 2)
        self.assertEqual(self.sleeps, [1.0])

    async def test_reports_the_host_after_every_attempt_fails(self):
        attempts: list[int] = []
        error = self._dns_error()

        class Session:
            def __init__(self, *, headers, timeout=None):
                self.headers = headers

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_exc):
                return None

            def request(self, _method, _url, **_options):
                attempts.append(1)
                raise error

        publisher.aiohttp.ClientSession = Session
        with self.assertRaises(RuntimeError) as caught:
            await publisher._phantom_request(
                "POST",
                "/versions/version-id/artifacts/digest/uploads",
                {"origin": "https://api.phantomrouter.ai", "token": "php_secret"},
                {"byte_size": 42},
                transient_retries=2,
            )

        self.assertEqual(len(attempts), 3)
        self.assertIn("could not reach https://api.phantomrouter.ai", str(caught.exception))
        self.assertIn("Name or service not known", str(caught.exception))

    async def test_a_dropped_part_upload_is_re_signed_and_resent(self):
        # The presigned URL is short-lived, so an attempt that waited out a
        # backoff can outlive it. Replaying the dead URL would turn one dropped
        # packet into a failed multi-gigabyte upload.
        signed_urls = [
            "https://uploads.test/part-1?sig=first",
            "https://uploads.test/part-1?sig=second",
        ]
        requested: list[str] = []
        put_urls: list[str] = []
        error = publisher.aiohttp.ClientConnectionError("Server disconnected")

        async def request(_method, path, _config, _body=None, **_options):
            requested.append(path)
            return {"upload_url": signed_urls[len(requested) - 1]}

        class Response:
            status = 200
            headers = {"ETag": "uploaded"}

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_exc):
                return None

        class Session:
            def __init__(self, *, timeout=None, **_options):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_exc):
                return None

            def put(self, url, *, data):
                put_urls.append(url)
                if len(put_urls) == 1:
                    raise error
                return Response()

        original_request = publisher._phantom_request
        publisher._phantom_request = request
        publisher.aiohttp.ClientSession = Session
        try:
            part = await publisher._put_part(
                "version-id",
                "a" * 64,
                1,
                b"0123",
                {"origin": "https://api.phantomrouter.ai", "token": "php_secret"},
            )
        finally:
            publisher._phantom_request = original_request

        self.assertEqual(part, {"PartNumber": 1, "ETag": "uploaded"})
        self.assertEqual(put_urls, signed_urls)
        self.assertEqual(len(requested), 2)
        self.assertEqual(self.sleeps, [1.0])

    async def test_a_rejected_part_upload_fails_without_retrying(self):
        # 403 means the request itself is wrong. Repeating it five times only
        # delays the report.
        attempts: list[int] = []

        async def request(_method, _path, _config, _body=None, **_options):
            return {"upload_url": "https://uploads.test/part-1"}

        class Response:
            status = 403
            headers: dict[str, str] = {}

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_exc):
                return None

            async def text(self):
                return "SignatureDoesNotMatch"

        class Session:
            def __init__(self, *, timeout=None, **_options):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_exc):
                return None

            def put(self, _url, *, data):
                attempts.append(1)
                return Response()

        original_request = publisher._phantom_request
        publisher._phantom_request = request
        publisher.aiohttp.ClientSession = Session
        try:
            with self.assertRaises(RuntimeError) as caught:
                await publisher._put_part(
                    "version-id",
                    "a" * 64,
                    1,
                    b"0123",
                    {"origin": "https://api.phantomrouter.ai", "token": "php_secret"},
                )
        finally:
            publisher._phantom_request = original_request

        self.assertEqual(len(attempts), 1)
        self.assertEqual(self.sleeps, [])
        self.assertIn("SignatureDoesNotMatch", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
