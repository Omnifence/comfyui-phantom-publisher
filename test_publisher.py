from __future__ import annotations

import asyncio
import base64
import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch
import json
import tarfile
from pathlib import Path
from typing import Any


class _Routes:
    """
    `register_routes` registers through this, so it doubles as the way a test
    reaches a route handler: the decorator records it under its method and path
    and hands it back unchanged.
    """

    def __init__(self) -> None:
        self.handlers: dict[tuple[str, str], Any] = {}

    def _record(self, method: str, path: str):
        def decorate(handler):
            self.handlers[(method, path)] = handler
            return handler

        return decorate

    def get(self, path: str):
        return self._record("GET", path)

    def put(self, path: str):
        return self._record("PUT", path)

    def post(self, path: str):
        return self._record("POST", path)

    def delete(self, path: str):
        return self._record("DELETE", path)


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

    class HTTPBadRequest(Exception):
        def __init__(self, *, text: str = "") -> None:
            super().__init__(text)
            self.text = text

    class HTTPNotFound(Exception):
        pass

    aiohttp.web.HTTPBadRequest = HTTPBadRequest
    aiohttp.web.HTTPNotFound = HTTPNotFound
    aiohttp.web.json_response = lambda data, status=200: types.SimpleNamespace(
        body=data, status=status
    )

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


class _CustomNodesEnvironment:
    """
    A fake ComfyUI install for package discovery: a base path, one custom_nodes
    root, and a `nodes.NODE_CLASS_MAPPINGS` whose class objects resolve to real
    files — the same chain `_discover_packages` walks in production.

    `split_source_root` models a ComfyUI Desktop install, which passes
    `--base-directory`: base_path is then the user data directory and ComfyUI's
    own code sits somewhere else entirely. A classic install keeps both in the
    same place, which is the default here.
    """

    def __init__(self, testcase: unittest.TestCase, *, split_source_root: bool = False):
        temporary = tempfile.TemporaryDirectory()
        testcase.addCleanup(temporary.cleanup)
        self.testcase = testcase
        self.root = Path(temporary.name)
        self.source_root = self.root / "comfyui-source" if split_source_root else self.root
        self.source_root.mkdir(exist_ok=True)
        (self.source_root / "nodes.py").touch()
        self.custom_nodes = self.root / "custom_nodes"
        self.custom_nodes.mkdir()
        self.mappings: dict[str, type] = {}
        self._original_base = publisher.folder_paths.base_path
        self._original_get_paths = publisher.folder_paths.get_folder_paths
        publisher.folder_paths.base_path = str(self.root)
        publisher.folder_paths.get_folder_paths = lambda kind: (
            [str(self.custom_nodes)] if kind == "custom_nodes" else []
        )
        testcase.addCleanup(self._restore)
        sys.modules["nodes"] = types.SimpleNamespace(
            NODE_CLASS_MAPPINGS=self.mappings,
            __file__=str(self.source_root / "nodes.py"),
        )
        testcase.addCleanup(sys.modules.pop, "nodes", None)

    def _restore(self) -> None:
        publisher.folder_paths.base_path = self._original_base
        publisher.folder_paths.get_folder_paths = self._original_get_paths

    def register_class(self, class_type: str, source: Path) -> None:
        module_name = f"phantom_test_module_{class_type.lower()}"
        module = types.ModuleType(module_name)
        module.__file__ = str(source)
        sys.modules[module_name] = module
        self.testcase.addCleanup(sys.modules.pop, module_name, None)
        self.mappings[class_type] = type(class_type, (), {"__module__": module_name})

    def package(self, name: str, class_types: list[str]) -> Path:
        directory = self.custom_nodes / name
        directory.mkdir()
        source = directory / "nodes_impl.py"
        source.write_text(
            "".join(f"class {class_type}:\n    pass\n" for class_type in class_types),
            encoding="utf-8",
        )
        for class_type in class_types:
            self.register_class(class_type, source)
        return directory

    def core_class(self, class_type: str) -> None:
        source = self.source_root / "comfy_extras" / "nodes_core.py"
        source.parent.mkdir(exist_ok=True)
        source.touch()
        self.register_class(class_type, source)

    def core_root_class(self, class_type: str) -> None:
        """A class defined in ComfyUI's own `nodes.py`, like `SaveImage`."""
        self.register_class(class_type, self.source_root / "nodes.py")


def _discard(mapping: dict[str, Any], key: str) -> None:
    """Drop a key and return nothing — see `_running_task` for why that matters."""
    mapping.pop(key, None)


def _workflow(nodes_spec: list[tuple[str, dict[str, Any]]]):
    api: dict[str, Any] = {}
    ui: dict[str, Any] = {"nodes": []}
    for index, (class_type, properties) in enumerate(nodes_spec, start=1):
        api[str(index)] = {"class_type": class_type, "inputs": {}}
        ui["nodes"].append({"id": index, "properties": properties})
    return api, ui


class PackageAttributionTests(unittest.TestCase):
    """
    The frontend's `properties.cnr_id` label is whatever ComfyUI's registry said
    at save time, and one broken package (`from nodes import *` in its
    __init__.py) rewrites that registry for every node loaded before it. The
    archived directory must therefore come from the class object, whose
    `__module__` is stamped at definition and survives the hijack — trusting the
    label shipped an archive without the classes it promised, and the lie only
    surfaced after a full image build.
    """

    def _discover(self, env, nodes_spec):
        packages = publisher._discover_packages(*_workflow(nodes_spec))
        for package in packages:
            if package.get("_archive_path"):
                self.addCleanup(
                    shutil.rmtree, str(Path(package["_archive_path"]).parent), ignore_errors=True
                )
        return packages

    def test_archives_the_directory_that_defines_the_class_not_the_label(self):
        env = _CustomNodesEnvironment(self)
        env.package("pack_a", ["A_Node"])
        pack_b = env.package("pack_b", ["B_Node"])
        # A hijacked registry labeled BOTH nodes as pack_a's.
        packages = self._discover(
            env,
            [
                ("A_Node", {"cnr_id": "pack_a", "ver": "1.0.0"}),
                ("B_Node", {"cnr_id": "pack_a", "ver": "1.0.0"}),
            ],
        )

        by_directory = {Path(item["_package_directory"]).name: item for item in packages}
        self.assertEqual(set(by_directory), {"pack_a", "pack_b"})
        misattributed = by_directory["pack_b"]
        self.assertEqual(misattributed["class_types"], ["B_Node"])
        self.assertEqual(Path(misattributed["_package_directory"]).resolve(), pack_b.resolve())
        self.assertRegex(misattributed["archive_sha256"], r"^[a-f0-9]{64}$")
        # The wrong label never selects the directory and never rides along as
        # provenance; it is recorded so hijacked registries are visible.
        self.assertIsNone(misattributed["cnr_id"])
        self.assertEqual(
            misattributed["attribution_mismatches"],
            [{"class_type": "B_Node", "labeled_cnr_id": "pack_a"}],
        )
        self.assertEqual(by_directory["pack_a"]["cnr_id"], "pack_a")
        self.assertNotIn("attribution_mismatches", by_directory["pack_a"])

    def test_registry_label_survives_as_provenance_when_it_agrees(self):
        env = _CustomNodesEnvironment(self)
        env.package("comfyui-example", ["ExampleNode"])
        packages = self._discover(
            env, [("ExampleNode", {"cnr_id": "comfyui-example", "ver": "1.2.3"})]
        )

        self.assertEqual(len(packages), 1)
        self.assertEqual(packages[0]["cnr_id"], "comfyui-example")
        self.assertEqual(packages[0]["version"], "1.2.3")
        self.assertEqual(packages[0]["class_types"], ["ExampleNode"])
        self.assertRegex(packages[0]["archive_sha256"], r"^[a-f0-9]{64}$")
        self.assertGreater(packages[0]["_archive_size"], 0)
        self.assertNotIn("attribution_mismatches", packages[0])

    def test_core_classes_are_skipped_by_resolved_location_not_by_label(self):
        env = _CustomNodesEnvironment(self)
        env.core_class("KSampler")
        # Even a hijacked label on a core class packages nothing: core-ness is
        # derived from where the class resolves, not from what the label says.
        packages = self._discover(
            env,
            [
                ("KSampler", {"cnr_id": "comfy-core", "ver": "0.3.0"}),
            ],
        )
        self.assertEqual(packages, [])
        env.core_class("CLIPTextEncode")
        packages = self._discover(env, [("CLIPTextEncode", {"cnr_id": "pack_a"})])
        self.assertEqual(packages, [])

    def test_fails_the_publish_when_a_class_is_not_registered(self):
        env = _CustomNodesEnvironment(self)
        env.package("pack_a", ["A_Node"])
        with self.assertRaises(RuntimeError) as caught:
            self._discover(
                env,
                [
                    ("A_Node", {"cnr_id": "pack_a"}),
                    ("MissingNode", {"cnr_id": "pack_a"}),
                ],
            )
        self.assertIn("MissingNode", str(caught.exception))
        self.assertIn("not registered", str(caught.exception))

    def test_fails_the_publish_when_a_class_resolves_outside_comfyui(self):
        env = _CustomNodesEnvironment(self)
        with tempfile.TemporaryDirectory() as elsewhere:
            stray = Path(elsewhere) / "stray.py"
            stray.write_text("class StrayNode:\n    pass\n", encoding="utf-8")
            env.register_class("StrayNode", stray)
            with self.assertRaises(RuntimeError) as caught:
                self._discover(env, [("StrayNode", {})])
        self.assertIn("StrayNode", str(caught.exception))

class DesktopInstallTests(unittest.TestCase):
    """
    ComfyUI Desktop starts the server with `--base-directory`, so
    `folder_paths.base_path` is the user data directory and ComfyUI's own code
    lives somewhere else. Reading core-ness off base_path called every core
    class — `SaveImage` first — "outside the ComfyUI installation" and refused
    every publish from such a machine at 10%.
    """

    def _discover(self, env, nodes_spec):
        packages = publisher._discover_packages(*_workflow(nodes_spec))
        for package in packages:
            if package.get("_archive_path"):
                self.addCleanup(
                    shutil.rmtree, str(Path(package["_archive_path"]).parent), ignore_errors=True
                )
        return packages

    def test_core_classes_are_skipped_when_base_path_is_not_the_source_root(self):
        env = _CustomNodesEnvironment(self, split_source_root=True)
        env.core_root_class("SaveImage")
        env.core_class("KSampler")
        self.assertNotEqual(env.source_root, env.root)
        self.assertEqual(
            self._discover(env, [("SaveImage", {"cnr_id": "comfy-core"}), ("KSampler", {})]),
            [],
        )

    def test_custom_packages_still_group_under_a_split_base_directory(self):
        env = _CustomNodesEnvironment(self, split_source_root=True)
        package = env.package("comfyui-example", ["ExampleNode"])
        env.core_root_class("SaveImage")
        packages = self._discover(
            env,
            [("ExampleNode", {"cnr_id": "comfyui-example", "ver": "1.2.3"}), ("SaveImage", {})],
        )
        self.assertEqual(len(packages), 1)
        self.assertEqual(packages[0]["class_types"], ["ExampleNode"])
        self.assertEqual(Path(packages[0]["_package_directory"]).resolve(), package.resolve())

    def test_a_stray_class_still_fails_and_names_every_searched_root(self):
        env = _CustomNodesEnvironment(self, split_source_root=True)
        with tempfile.TemporaryDirectory() as elsewhere:
            stray = Path(elsewhere) / "stray.py"
            stray.write_text("class StrayNode:\n    pass\n", encoding="utf-8")
            env.register_class("StrayNode", stray)
            with self.assertRaises(RuntimeError) as caught:
                self._discover(env, [("StrayNode", {})])
        message = str(caught.exception)
        self.assertIn("StrayNode", message)
        self.assertIn(str(env.source_root), message)
        self.assertIn(str(env.custom_nodes), message)

    def test_source_root_prefers_the_nodes_module_over_the_base_path(self):
        env = _CustomNodesEnvironment(self, split_source_root=True)
        self.assertEqual(publisher._comfy_source_root(), env.source_root.resolve())

    def test_source_root_falls_back_to_the_base_path_without_a_nodes_module(self):
        env = _CustomNodesEnvironment(self)
        sys.modules.pop("nodes", None)
        self.assertEqual(publisher._comfy_source_root(), env.root.resolve())


class GitMetadataTests(unittest.TestCase):
    """
    `git -C` walks up until it finds A repository, not THIS package's: a
    hand-copied folder inside a ComfyUI checkout reported ComfyUI's own URL and
    dirty flag — provenance for the wrong code.
    """

    def _git(self, directory: Path, *args: str) -> None:
        import subprocess

        subprocess.check_output(["git", "-C", str(directory), *args], stderr=subprocess.DEVNULL)

    @unittest.skipUnless(shutil.which("git"), "git is required")
    def test_reports_no_repository_for_a_folder_inside_someone_elses_checkout(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary) / "ComfyUI"
            package = checkout / "custom_nodes" / "hand_copied_pack"
            package.mkdir(parents=True)
            self._git(checkout, "init", "-q")
            self.assertEqual(
                publisher._git_metadata(package),
                {"git_commit": None, "repository_url": None, "dirty": True},
            )

    @unittest.skipUnless(shutil.which("git"), "git is required")
    def test_reports_the_packages_own_checkout(self):
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary) / "cloned_pack"
            package.mkdir()
            (package / "node.py").write_text("NODE_CLASS_MAPPINGS = {}\n", encoding="utf-8")
            self._git(package, "init", "-q")
            self._git(package, "config", "user.email", "test@example.test")
            self._git(package, "config", "user.name", "Test")
            self._git(package, "add", "node.py")
            self._git(package, "commit", "-q", "-m", "initial")
            self._git(package, "remote", "add", "origin", "https://github.com/example/pack.git")

            metadata = publisher._git_metadata(package)

            self.assertRegex(metadata["git_commit"], r"^[a-f0-9]{40}$")
            self.assertEqual(metadata["repository_url"], "https://github.com/example/pack.git")
            self.assertFalse(metadata["dirty"])


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


    def test_pins_a_module_only_the_runtime_trace_saw(self):
        # The DiffusionWave packs import `imageio` from inside a compiled
        # `_dw_core` extension. No `.py` source names it, so static capture
        # missed it, the pack installed clean, failed to import at startup,
        # and the build refused the image for registering no classes.
        with tempfile.TemporaryDirectory() as temporary:
            package = self._package(temporary, "import numpy\n")
            with _installed(
                {"numpy": ("numpy", "1.26.4"), "imageio": ("imageio", "2.37.0")}
            ):
                self.assertEqual(publisher._pip_dependencies(package), [])
                self.assertEqual(
                    publisher._pip_dependencies(package, {"imageio", "numpy"}),
                    ["imageio==2.37.0"],
                )

    def test_a_hoisted_module_map_and_provided_set_are_honoured(self):
        with tempfile.TemporaryDirectory() as temporary:
            package = self._package(temporary, "import svgwrite\nimport imageio\n")
            with _installed(
                {"svgwrite": ("svgwrite", "1.4.3"), "imageio": ("imageio", "2.37.0")}
            ):
                pinned = publisher._pip_dependencies(
                    package,
                    None,
                    module_to_distributions={"svgwrite": ["svgwrite"], "imageio": ["imageio"]},
                    provided={"imageio"},
                )
        self.assertEqual(pinned, ["svgwrite==1.4.3"])


class _installed:
    """Pin importlib.metadata to a fixed installed set for the duration."""

    def __init__(self, mapping: dict[str, tuple[str, str]], direct_urls=None):
        self._direct_urls = direct_urls or {}
        self._mapping = mapping

    def __enter__(self):
        self._packages_distributions = publisher.importlib.metadata.packages_distributions
        self._version = publisher.importlib.metadata.version
        self._distributions = publisher.importlib.metadata.distributions
        versions = {distribution: version for distribution, version in self._mapping.values()}

        def version(name: str) -> str:
            if name not in versions:
                raise publisher.importlib.metadata.PackageNotFoundError(name)
            return versions[name]

        publisher.importlib.metadata.packages_distributions = lambda: {
            module: [distribution] for module, (distribution, _) in self._mapping.items()
        }
        publisher.importlib.metadata.version = version
        publisher.importlib.metadata.distributions = lambda: [
            types.SimpleNamespace(
                metadata={"Name": name},
                version=release,
                read_text=lambda _, name=name: (
                    json.dumps(self._direct_urls[name]) if name in self._direct_urls else None
                ),
            )
            for name, release in versions.items()
        ]
        return self

    def __exit__(self, *_exc: object) -> None:
        publisher.importlib.metadata.packages_distributions = self._packages_distributions
        publisher.importlib.metadata.version = self._version
        publisher.importlib.metadata.distributions = self._distributions


class _RecordEntry(str):
    """One RECORD line as importlib.metadata hands it out: a path, its hash, its size, `locate()`."""

    def __new__(cls, path: str, root: Path, content: bytes, *, size: int | None = None):
        self = super().__new__(cls, path)
        digest = hashlib.sha256(content).digest()
        self.hash = types.SimpleNamespace(
            mode="sha256", value=base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
        )
        self.size = len(content) if size is None else size
        self._root = root
        return self

    def locate(self) -> Path:
        return self._root / str(self)


def _distribution_with_files(name: str, version: str, files: list[_RecordEntry]):
    return types.SimpleNamespace(
        metadata={"Name": name}, version=version, files=files, read_text=lambda _: None
    )


class ShadowedDistributionTests(unittest.TestCase):
    """
    Two wheels that unpack into one package directory: the lock must name the
    one whose files are on disk, or the build reproduces pip's install order
    rather than the venv.
    """

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "onnxruntime" / "capi").mkdir(parents=True)

    def _write(self, path: str, content: bytes) -> None:
        (self.root / path).write_bytes(content)

    def _venv(self, cpu_so: bytes, gpu_so: bytes, *, on_disk: bytes):
        self._write("onnxruntime/__init__.py", b"shared")
        self._write("onnxruntime/capi/_pybind_state.so", on_disk)
        cpu = _distribution_with_files(
            "onnxruntime",
            "1.29.0",
            [
                _RecordEntry("onnxruntime/__init__.py", self.root, b"shared"),
                _RecordEntry("onnxruntime/capi/_pybind_state.so", self.root, cpu_so),
            ],
        )
        gpu = _distribution_with_files(
            "onnxruntime-gpu",
            "1.20.2",
            [
                _RecordEntry("onnxruntime/__init__.py", self.root, b"shared"),
                _RecordEntry("onnxruntime/capi/_pybind_state.so", self.root, gpu_so),
            ],
        )
        return [cpu, gpu]

    def test_the_distribution_whose_files_are_on_disk_wins(self):
        installed = self._venv(b"cpu build", b"gpu build, much larger", on_disk=b"gpu build, much larger")
        self.assertEqual(publisher._shadowed_distributions(installed), {"onnxruntime"})
        installed = self._venv(b"cpu build", b"gpu build, much larger", on_disk=b"cpu build")
        self.assertEqual(publisher._shadowed_distributions(installed), {"onnxruntime-gpu"})

    def test_equal_sizes_are_settled_by_hash(self):
        installed = self._venv(b"cpu build", b"gpu build", on_disk=b"gpu build")
        self.assertEqual(publisher._shadowed_distributions(installed), {"onnxruntime"})

    def test_a_shared_identical_file_does_not_make_either_the_owner(self):
        # Only `__init__.py` is shared and both RECORDs match it: nothing is
        # decided, both stay. Same when the contested file is missing entirely.
        installed = self._venv(b"cpu build", b"gpu build", on_disk=b"neither")
        self.assertEqual(publisher._shadowed_distributions(installed), set())
        (self.root / "onnxruntime" / "capi" / "_pybind_state.so").unlink()
        self.assertEqual(publisher._shadowed_distributions(installed), set())

    def test_a_distribution_that_owns_some_and_loses_some_is_kept(self):
        installed = self._venv(b"cpu build", b"gpu build", on_disk=b"gpu build")
        self._write("onnxruntime/cpu_only.py", b"cpu owns this")
        installed[0].files.append(_RecordEntry("onnxruntime/cpu_only.py", self.root, b"cpu owns this"))
        installed[1].files.append(_RecordEntry("onnxruntime/cpu_only.py", self.root, b"gpu lost this"))
        self.assertEqual(publisher._shadowed_distributions(installed), set())

    def test_distributions_without_files_or_hashes_are_ignored(self):
        bare = types.SimpleNamespace(metadata={"Name": "plain"}, version="1", files=None)
        unhashed = _RecordEntry("onnxruntime/RECORD", self.root, b"")
        unhashed.hash = None
        installed = self._venv(b"cpu build", b"gpu build", on_disk=b"gpu build")
        installed[0].files.append(unhashed)
        installed[1].files.append(unhashed)
        broken = types.SimpleNamespace(metadata=None, version="1")
        self.assertEqual(
            publisher._shadowed_distributions([bare, broken, *installed]), {"onnxruntime"}
        )

    def test_the_lock_leaves_a_shadowed_distribution_out_and_names_it(self):
        installed = self._venv(b"cpu build", b"gpu build", on_disk=b"gpu build")
        with _installed({"onnxruntime": ("onnxruntime", "1.29.0"), "timm": ("timm", "1.0.28")}):
            publisher.importlib.metadata.distributions = lambda: [
                *installed,
                _distribution_with_files("timm", "1.0.28", []),
            ]
            publisher.importlib.metadata.packages_distributions = lambda: {
                "onnxruntime": ["onnxruntime", "onnxruntime-gpu"],
                "timm": ["timm"],
            }
            lock = publisher._environment_lock()
        self.assertEqual(lock["distributions"], {"onnxruntime-gpu": "1.20.2", "timm": "1.0.28"})
        self.assertEqual(lock["modules"], {"onnxruntime": ["onnxruntime-gpu"], "timm": ["timm"]})
        self.assertEqual(lock["shadowed"], {"onnxruntime": "1.29.0"})

    def test_a_failing_shadow_check_keeps_every_distribution(self):
        with _installed({"cv2": ("opencv-python", "4.10.0.84")}):
            with patch.object(publisher, "_shadowed_distributions", side_effect=RuntimeError("x")):
                lock = publisher._environment_lock()
        self.assertEqual(lock["distributions"], {"opencv-python": "4.10.0.84"})
        self.assertEqual(lock["shadowed"], {})

    def test_a_shared_console_script_buries_nothing(self):
        # Two unrelated distributions each ship `bin/foo`; the last installed
        # owns it. Their modules are untouched, so neither is shadowed — only
        # files inside site-packages can bury a distribution.
        site = self.root / "site-packages"
        site.mkdir()
        (self.root / "bin").mkdir()
        self._write("bin/foo", b"#!python\nfrom two import main")
        self._write("site-packages/one.py", b"one")
        self._write("site-packages/two.py", b"two")
        one = _distribution_with_files(
            "one",
            "1.0",
            [
                _RecordEntry("one.py", site, b"one"),
                _RecordEntry("../bin/foo", site, b"#!python\nfrom one import main"),
            ],
        )
        two = _distribution_with_files(
            "two",
            "1.0",
            [
                _RecordEntry("two.py", site, b"two"),
                _RecordEntry("../bin/foo", site, b"#!python\nfrom two import main"),
            ],
        )
        self.assertTrue(publisher._record_entry_is_live(two.files[1], {}))
        self.assertEqual(publisher._shadowed_distributions([one, two]), set())

    def test_a_cancelled_publish_stops_the_shadow_check(self):
        # The check runs in a discovery worker; a set event must surface as
        # `_DiscoveryCancelled` rather than be swallowed by the guards that
        # keep one unreadable RECORD from emptying the lock.
        installed = self._venv(b"cpu build", b"gpu build", on_disk=b"gpu build")
        cancellation = threading.Event()
        cancellation.set()
        with self.assertRaises(publisher._DiscoveryCancelled):
            publisher._shadowed_distributions(installed, cancellation)
        cache: dict = {}
        with self.assertRaises(publisher._DiscoveryCancelled):
            publisher._record_entry_is_live(installed[1].files[1], cache, cancellation)
        with _installed({"onnxruntime": ("onnxruntime", "1.29.0")}):
            publisher.importlib.metadata.distributions = lambda: installed
            with self.assertRaises(publisher._DiscoveryCancelled):
                publisher._environment_lock(cancellation)


class EnvironmentLockTests(unittest.TestCase):
    """
    The manifest carries the publishing venv so the build can resolve a module
    the capture missed — an import a node makes only while a workflow runs —
    against the versions that are known to work, instead of guessing.
    """

    def test_the_two_maps_are_normalised_and_consistent(self):
        with _installed(
            {
                "cv2": ("opencv-python", "4.10.0.84"),
                "imageio": ("ImageIO", "2.37.0"),
                "PIL": ("Pillow", "11.0.0"),
            }
        ):
            lock = publisher._environment_lock()
        self.assertEqual(
            lock["distributions"],
            {"imageio": "2.37.0", "opencv-python": "4.10.0.84", "pillow": "11.0.0"},
        )
        self.assertEqual(
            lock["modules"],
            {"PIL": ["pillow"], "cv2": ["opencv-python"], "imageio": ["imageio"]},
        )
        for names in lock["modules"].values():
            for name in names:
                self.assertIn(name, lock["distributions"])

    def test_a_distribution_with_unreadable_metadata_is_skipped_not_fatal(self):
        class _Broken:
            @property
            def metadata(self):
                raise KeyError("half-written .dist-info")

            version = "0"

        with _installed({"cv2": ("opencv-python", "4.10.0.84")}):
            good = list(publisher.importlib.metadata.distributions())
            publisher.importlib.metadata.distributions = lambda: [_Broken(), *good]
            lock = publisher._environment_lock()
        self.assertEqual(lock["distributions"], {"opencv-python": "4.10.0.84"})
        self.assertEqual(lock["modules"], {"cv2": ["opencv-python"]})

    def test_a_module_whose_distribution_has_no_version_is_left_out(self):
        with _installed({"cv2": ("opencv-python", "4.10.0.84")}):
            publisher.importlib.metadata.packages_distributions = lambda: {
                "cv2": ["opencv-python"],
                "ghost": ["not-installed"],
            }
            lock = publisher._environment_lock()
        self.assertNotIn("ghost", lock["modules"])
        self.assertEqual(lock["modules"], {"cv2": ["opencv-python"]})


class _FakeComfyRoot:
    """
    Enough of a ComfyUI source tree for the trace script: `comfy.cli_args`,
    `utils.extra_config`, `nodes`, a `stubdist` module standing in for an
    installed distribution, and one custom_nodes directory.
    """

    def __init__(self, testcase: unittest.TestCase, *, nodes_source: str = "") -> None:
        temporary = tempfile.TemporaryDirectory()
        testcase.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "comfy").mkdir()
        (self.root / "comfy" / "__init__.py").touch()
        (self.root / "comfy" / "cli_args.py").write_text(
            "import types\nargs = types.SimpleNamespace(cpu=False)\n", encoding="utf-8"
        )
        (self.root / "utils").mkdir()
        (self.root / "utils" / "__init__.py").touch()
        (self.root / "utils" / "extra_config.py").touch()
        (self.root / "nodes.py").write_text(nodes_source, encoding="utf-8")
        # ComfyUI's server: `PromptServer.instance` exists only once the
        # constructor has run, which packs that register routes rely on.
        (self.root / "server.py").write_text(
            "class _Routes:\n"
            "    def get(self, path):\n"
            "        return lambda handler: handler\n"
            "\n"
            "class PromptServer:\n"
            "    instance = None\n"
            "    def __init__(self, loop):\n"
            "        PromptServer.instance = self\n"
            "        self.loop = loop\n"
            "        self.routes = _Routes()\n",
            encoding="utf-8",
        )
        (self.root / "stubdist.py").write_text("VALUE = 1\n", encoding="utf-8")
        self.custom_nodes = self.root / "custom_nodes"
        self.custom_nodes.mkdir()

    def package(self, name: str, init_source: str) -> Path:
        directory = self.custom_nodes / name
        directory.mkdir()
        (directory / "__init__.py").write_text(init_source, encoding="utf-8")
        return directory


class ImportTraceTests(unittest.TestCase):
    """
    The trace script runs under the real interpreter against a fake ComfyUI,
    because what it has to get right — the `sys.modules` diff and the two
    import hooks — cannot be asserted from a string.
    """

    def test_attributes_a_static_import_the_baseline_did_not_load(self):
        comfy = _FakeComfyRoot(self)
        package = comfy.package("pack_a", "import stubdist\n")
        result = publisher._trace_imports(package, comfy.root)
        self.assertIsNone(result.error)
        self.assertIn("stubdist", result.modules)

    def test_attributes_a_cached_dynamic_import_through_the_hook(self):
        # The baseline (ComfyUI's own `nodes`) already imported the module, so
        # the `sys.modules` diff is empty for it, and `importlib.import_module`
        # returns the cached module without going through `builtins.__import__`.
        comfy = _FakeComfyRoot(self, nodes_source="import stubdist\n")
        package = comfy.package(
            "pack_a", "import importlib\n_dep = importlib.import_module('stubdist')\n"
        )
        result = publisher._trace_imports(package, comfy.root)
        self.assertIsNone(result.error)
        self.assertIn("stubdist", result.modules)

    def test_attributes_a_cached_static_import_through_the_hook(self):
        comfy = _FakeComfyRoot(self, nodes_source="import stubdist\n")
        package = comfy.package("pack_a", "import stubdist\n")
        result = publisher._trace_imports(package, comfy.root)
        self.assertIsNone(result.error)
        self.assertIn("stubdist", result.modules)

    def test_a_package_that_raises_reports_the_error_and_no_modules(self):
        comfy = _FakeComfyRoot(self)
        package = comfy.package("pack_a", "raise RuntimeError('boom at import')\n")
        result = publisher._trace_imports(package, comfy.root)
        self.assertIsNotNone(result.error)
        self.assertIn("boom at import", result.error)
        self.assertEqual(result.modules, set())

    def test_a_package_that_floods_stdout_still_yields_a_result(self):
        comfy = _FakeComfyRoot(self)
        package = comfy.package(
            "pack_a", "import sys\nsys.stdout.write('x' * (1 << 20))\nimport stubdist\n"
        )
        result = publisher._trace_imports(package, comfy.root)
        self.assertIsNone(result.error)
        self.assertIn("stubdist", result.modules)

    def test_the_package_own_path_derived_module_name_is_not_a_module(self):
        comfy = _FakeComfyRoot(self)
        package = comfy.package("pack.with.dots", "import stubdist\n")
        result = publisher._trace_imports(package, comfy.root)
        self.assertIsNone(result.error)
        self.assertNotIn(str(package).replace(".", "_x_"), result.modules)

    def test_a_package_that_registers_routes_at_import_is_traced(self):
        # ComfyUI constructs PromptServer before it loads custom nodes, so a
        # pack may reach `PromptServer.instance.routes` at import time.
        comfy = _FakeComfyRoot(self)
        package = comfy.package(
            "pack_a",
            "from server import PromptServer\n"
            "import stubdist\n"
            "@PromptServer.instance.routes.get('/pack_a')\n"
            "async def handler(request):\n"
            "    return None\n",
        )
        result = publisher._trace_imports(package, comfy.root)
        self.assertIsNone(result.error)
        self.assertIn("stubdist", result.modules)
        # `server` is recorded like any import; the pin rule ignores it.
        self.assertIn("server", publisher._COMFYUI_MODULES)

    def test_a_package_that_leaves_a_thread_running_still_yields_its_result(self):
        # Interpreter shutdown would wait for a non-daemon thread; the trace
        # must not, or the parent times out on a result already written.
        self.addCleanup(
            setattr, publisher, "_TRACE_TIMEOUT_SECONDS", publisher._TRACE_TIMEOUT_SECONDS
        )
        publisher._TRACE_TIMEOUT_SECONDS = 20
        comfy = _FakeComfyRoot(self)
        package = comfy.package(
            "pack_a",
            "import threading\nimport time\nimport stubdist\n"
            "threading.Thread(target=time.sleep, args=(60,)).start()\n",
        )
        started = time.monotonic()
        result = publisher._trace_imports(package, comfy.root)
        self.assertLess(time.monotonic() - started, 15)
        self.assertIsNone(result.error)
        self.assertIn("stubdist", result.modules)


class _StubInterpreter:
    """Replace `sys.executable` with a shell script for the duration."""

    def __init__(self, testcase: unittest.TestCase, body: str) -> None:
        temporary = tempfile.TemporaryDirectory()
        testcase.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "python"
        self.path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
        self.path.chmod(0o755)
        original = publisher.sys.executable
        publisher.sys.executable = str(self.path)
        testcase.addCleanup(setattr, publisher.sys, "executable", original)


class ImportTraceProcessTests(unittest.TestCase):
    """The parent's half: result parsing, timeout, cancel and fallback."""

    def setUp(self) -> None:
        self.comfy = _FakeComfyRoot(self)
        self.package = self.comfy.package("pack_a", "")
        self._timeout = publisher._TRACE_TIMEOUT_SECONDS
        self.addCleanup(setattr, publisher, "_TRACE_TIMEOUT_SECONDS", self._timeout)

    def test_parses_the_result_file_the_child_writes(self):
        _StubInterpreter(
            self,
            'printf \'{"modules": ["imageio", "numpy"], "error": null}\' > "$2"\n',
        )
        # $1 is "-" (the script comes on stdin); $2 is the result path.
        result = publisher._trace_imports(self.package, self.comfy.root)
        self.assertIsNone(result.error)
        self.assertEqual(result.modules, {"imageio", "numpy"})

    def test_a_timeout_kills_the_child_and_falls_back(self):
        publisher._TRACE_TIMEOUT_SECONDS = 0.5
        _StubInterpreter(self, 'echo "still loading"\nsleep 30\n')
        started = time.monotonic()
        result = publisher._trace_imports(self.package, self.comfy.root)
        self.assertLess(time.monotonic() - started, 10)
        self.assertEqual(result.modules, set())
        self.assertIn("timed out", result.error)
        self.assertIn("still loading", result.error)

    def test_a_timeout_kills_what_the_child_spawned(self):
        publisher._TRACE_TIMEOUT_SECONDS = 0.5
        # $2 is the result path: a grandchild records its pid beside it.
        _StubInterpreter(self, 'sleep 60 &\necho $! > "$2.grandchild"\nwait\n')
        captured: list[str] = []
        original = publisher._kill

        def kill(process):
            result_dir = Path(process.args[2]).parent
            captured.append((result_dir / "result.json.grandchild").read_text().strip())
            original(process)

        publisher._kill = kill
        self.addCleanup(setattr, publisher, "_kill", original)
        result = publisher._trace_imports(self.package, self.comfy.root)
        self.assertIn("timed out", result.error)
        grandchild = int(captured[0])
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                os.kill(grandchild, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        else:
            os.kill(grandchild, 9)
            self.fail(f"grandchild {grandchild} survived the timeout kill")

    def test_a_cancel_kills_the_child_and_raises(self):
        _StubInterpreter(self, "sleep 30\n")
        cancellation = threading.Event()
        threading.Timer(0.3, cancellation.set).start()
        started = time.monotonic()
        with self.assertRaises(publisher._DiscoveryCancelled):
            publisher._trace_imports(self.package, self.comfy.root, cancellation)
        self.assertLess(time.monotonic() - started, 10)

    def test_a_child_that_writes_no_result_falls_back_with_the_log(self):
        _StubInterpreter(self, 'echo "nothing to report"\nexit 0\n')
        result = publisher._trace_imports(self.package, self.comfy.root)
        self.assertEqual(result.modules, set())
        self.assertIn("no readable result", result.error)
        self.assertIn("nothing to report", result.error)

    def test_a_non_zero_exit_falls_back_with_the_log(self):
        _StubInterpreter(self, 'echo "segfault-ish"\nexit 3\n')
        result = publisher._trace_imports(self.package, self.comfy.root)
        self.assertEqual(result.modules, set())
        self.assertIn("exited with 3", result.error)
        self.assertIn("segfault-ish", result.error)


class DependencyCaptureManifestTests(unittest.TestCase):
    """Every archived package says HOW its dependencies were captured."""

    def _discover(self, env, nodes_spec):
        packages = publisher._discover_packages(*_workflow(nodes_spec))
        for package in packages:
            if package.get("_archive_path"):
                self.addCleanup(
                    shutil.rmtree, str(Path(package["_archive_path"]).parent), ignore_errors=True
                )
        return packages

    def test_every_package_carries_a_dependency_capture(self):
        env = _CustomNodesEnvironment(self)
        env.package("pack_a", ["A_Node"])
        env.package("pack_b", ["B_Node"])
        packages = self._discover(env, [("A_Node", {}), ("B_Node", {})])
        self.assertEqual(len(packages), 2)
        for package in packages:
            capture = package["dependency_capture"]
            self.assertIn(capture["method"], {"runtime", "static"})
            if capture["method"] == "static":
                self.assertIsInstance(capture["error"], str)
            else:
                self.assertIsNone(capture["error"])

    def test_the_log_line_names_the_method_and_the_reason(self):
        message, level = publisher._trace_log_line(
            {
                "_package_directory": "/x/custom_nodes/pack_a",
                "dependency_capture": {"method": "runtime", "error": None},
                "_trace_seconds": 12.44,
                "_trace_module_count": 3,
            }
        )
        self.assertEqual((message, level), ("Traced imports of pack_a (3 modules, 12.4 s)", "info"))
        message, level = publisher._trace_log_line(
            {
                "_package_directory": "/x/custom_nodes/pack_a",
                "dependency_capture": {
                    "method": "static",
                    "error": "Traceback...\nModuleNotFoundError: No module named 'comfy'\n",
                },
            }
        )
        self.assertEqual(level, "warning")
        self.assertIn("captured statically", message)
        self.assertIn("No module named 'comfy'", message)


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


class ServerWarningTests(unittest.TestCase):
    """
    Phantom attaches advisory `warnings` to a version response — e.g. "this
    machine's ComfyUI core is newer than every base image the platform knows".
    They must reach the publish log, and a malformed field must never fail a
    publish that already carries real artifacts.
    """

    def test_logs_each_server_warning_as_a_warning_line(self):
        job = {"status": "staging", "logs": []}
        publisher._log_server_warnings(
            job,
            {"workflow_version_id": "v-1", "warnings": ["core 0.99.0 outruns every base image"]},
        )
        self.assertEqual(len(job["logs"]), 1)
        self.assertEqual(job["logs"][0]["level"], "warning")
        self.assertIn("outruns every base image", job["logs"][0]["message"])

    def test_tolerates_older_servers_and_malformed_fields(self):
        job = {"status": "staging", "logs": []}
        publisher._log_server_warnings(job, {"workflow_version_id": "v-1"})
        publisher._log_server_warnings(job, {"warnings": "not-a-list"})
        publisher._log_server_warnings(job, {"warnings": [7, None, ""]})
        publisher._log_server_warnings(job, None)
        self.assertEqual(job["logs"], [])


class ComfyuiCoreVersionTests(unittest.TestCase):
    """
    Phantom decides whether a build must install a newer ComfyUI core than its
    base image froze by comparing this value. Reading it off the wrong module
    reported None on every publish, so that decision could never be made and a
    graph authored on a newer core shipped against an older one.
    """

    def setUp(self):
        self._saved = sys.modules.get("comfyui_version")

    def tearDown(self):
        if self._saved is None:
            sys.modules.pop("comfyui_version", None)
        else:
            sys.modules["comfyui_version"] = self._saved

    def test_reads_the_generated_comfyui_version_module(self):
        module = types.ModuleType("comfyui_version")
        module.__version__ = "0.26.0"
        sys.modules["comfyui_version"] = module
        self.assertEqual(publisher._comfyui_core_version(), "0.26.0")

    def test_returns_none_when_the_module_is_absent(self):
        sys.modules.pop("comfyui_version", None)
        # Nothing on sys.path provides it in the test environment, so the import
        # fails exactly as it would outside a ComfyUI checkout.
        self.assertIsNone(publisher._comfyui_core_version())

    def test_treats_a_blank_or_non_string_version_as_unknown(self):
        for value in ("", "   ", None, 26):
            module = types.ModuleType("comfyui_version")
            module.__version__ = value
            sys.modules["comfyui_version"] = module
            self.assertIsNone(publisher._comfyui_core_version())

    def test_does_not_read_it_off_the_comfy_package(self):
        # The old source. `comfy` has no __init__.py, so even a stub carrying
        # __version__ must not be what this returns.
        comfy = types.ModuleType("comfy")
        comfy.__version__ = "wrong-source"
        sys.modules["comfy"] = comfy
        try:
            module = types.ModuleType("comfyui_version")
            module.__version__ = "0.26.0"
            sys.modules["comfyui_version"] = module
            self.assertEqual(publisher._comfyui_core_version(), "0.26.0")
        finally:
            sys.modules.pop("comfy", None)


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


class VariationGraphTests(unittest.TestCase):
    """
    A variation graph joins the target workflow's current version beside its
    primary graph. The label says when Phantom should use it, and the publish
    is the only moment the author is sure to know that — so it is required
    before a byte is read, and it travels with the manifest verbatim.
    """

    def test_sanitizes_the_label_and_the_optional_description(self):
        self.assertEqual(
            publisher._variation_request(
                {
                    "variation": {
                        "label": "  Caller sends a reference image ",
                        "description": " IP-Adapter branch ",
                    }
                }
            ),
            {"label": "Caller sends a reference image", "description": "IP-Adapter branch"},
        )
        self.assertEqual(
            publisher._variation_request({"variation": {"label": "x", "description": ""}}),
            {"label": "x"},
        )
        self.assertIsNone(publisher._variation_request({"workflow_id": "wf-1"}))

    def test_an_update_names_the_variation_by_id_and_may_leave_the_label_alone(self):
        # The id says WHICH variation of the current version this graph
        # replaces; the label is then optional and travels only when given.
        self.assertEqual(
            publisher._variation_request({"variation": {"variation_id": " var-1 "}}),
            {"variation_id": "var-1"},
        )
        self.assertEqual(
            publisher._variation_request(
                {"variation": {"variation_id": "var-1", "label": " Renamed ", "description": "d"}}
            ),
            {"variation_id": "var-1", "label": "Renamed", "description": "d"},
        )
        # A blank id is no id: the label is required again.
        with self.assertRaises(ValueError):
            publisher._variation_request({"variation": {"variation_id": "  ", "label": ""}})

    def test_versions_body_carries_the_sanitized_block_when_present(self):
        self.assertEqual(
            publisher._versions_request_body(
                "wf-1",
                {"schema_version": 1},
                {"label": "Caller sends a reference image"},
            ),
            {
                "workflow_id": "wf-1",
                "manifest": {"schema_version": 1},
                "variation": {"label": "Caller sends a reference image"},
            },
        )

    def test_versions_body_omits_the_block_for_a_primary_publish(self):
        self.assertEqual(
            publisher._versions_request_body("wf-1", {"schema_version": 1}, None),
            {"workflow_id": "wf-1", "manifest": {"schema_version": 1}},
        )

    def test_refuses_a_blank_or_malformed_label(self):
        with self.assertRaises(ValueError):
            publisher._variation_request({"variation": {"label": "   "}})
        with self.assertRaises(ValueError):
            publisher._variation_request({"variation": "a label"})


class PublishRouteTests(unittest.IsolatedAsyncioTestCase):
    """
    Discovery hashes every model and archives every custom node package before
    the version call, so a variation that can never be accepted has to be
    refused by the route — not on the way out of discovery.
    """

    @staticmethod
    def _publish_handler():
        publisher.register_routes()
        return publisher.PromptServer.instance.routes.handlers[
            ("POST", "/phantom-publisher/publish")
        ]

    async def test_refuses_a_blank_label_before_a_job_exists(self):
        jobs_before = dict(publisher._jobs)
        started: list[Any] = []
        original_create_task = publisher.asyncio.create_task
        publisher.asyncio.create_task = started.append
        try:
            with self.assertRaises(publisher.web.HTTPBadRequest) as caught:
                await self._publish_handler()(
                    _StubRequest({"workflow_id": "wf-1", "variation": {"label": "  "}})
                )
        finally:
            publisher.asyncio.create_task = original_create_task
        self.assertIn("label", caught.exception.text)
        self.assertEqual(publisher._jobs, jobs_before)
        self.assertEqual(started, [])

    async def test_queues_the_job_with_the_sanitized_variation(self):
        original_create_task = publisher.asyncio.create_task
        # The route never awaits the job itself, so the task is closed here
        # rather than scheduled; `record` has already captured the arguments.
        publisher.asyncio.create_task = lambda coroutine: coroutine.close()
        original_run_publish = publisher._run_publish
        calls: list[tuple[Any, ...]] = []

        async def _finished() -> None:
            return None

        def record(job_id, body, variation=None):
            calls.append((job_id, body, variation))
            return _finished()

        publisher._run_publish = record
        try:
            response = await self._publish_handler()(
                _StubRequest(
                    {"workflow_id": "wf-1", "variation": {"label": "  Reference image  "}}
                )
            )
        finally:
            publisher.asyncio.create_task = original_create_task
            publisher._run_publish = original_run_publish
        self.assertEqual(response.status, 202)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][2], {"label": "Reference image"})
        publisher._jobs.pop(response.body["job_id"], None)
        publisher._job_tasks.pop(response.body["job_id"], None)


class CancelPublishTests(unittest.IsolatedAsyncioTestCase):
    """
    Closing the panel stops nothing: the publish is a task of the ComfyUI
    server. Cancel is what ends the transfer — and tells Phantom to abandon
    the parts already written, so nothing half-uploaded lingers.
    """

    @staticmethod
    def _cancel_handler():
        publisher.register_routes()
        return publisher.PromptServer.instance.routes.handlers[
            ("DELETE", "/phantom-publisher/jobs/{job_id}")
        ]

    def _job(self, job_id: str, **fields: Any) -> dict[str, Any]:
        job = {
            "job_id": job_id,
            "status": "uploading",
            "progress": 50,
            "message": "Uploading dependency 1 of 1",
            "dependencies": [],
            "logs": [],
            **fields,
        }
        publisher._jobs[job_id] = job
        self.addCleanup(publisher._jobs.pop, job_id, None)
        return job

    async def test_cancel_ends_the_task_and_abandons_the_uploads_in_flight(self):
        job = self._job(
            "job-cancel",
            version_id="version-9",
            dependencies=[
                {"id": "model-0", "name": "base.safetensors", "status": "uploading", "sha256": "ab" * 32},
                {"id": "model-1", "name": "done.safetensors", "status": "uploaded", "sha256": "cd" * 32},
            ],
        )
        requests: list[tuple[str, str]] = []

        async def fake_request(method, path, _config, *_args, **_kwargs):
            requests.append((method, path))
            return {"aborted": True}

        started = asyncio.Event()

        async def run_publish() -> None:
            try:
                started.set()
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                await publisher._abandon_uploads(job, {"origin": "https://phantom.test", "token": "t"})
                job.update(status="cancelled", message="Publish cancelled", error=None)
            finally:
                publisher._job_tasks.pop("job-cancel", None)

        original_request = publisher._phantom_request
        original_config = publisher._read_config
        publisher._phantom_request = fake_request
        publisher._read_config = lambda: {"origin": "https://phantom.test", "token": "t"}
        publisher._job_tasks["job-cancel"] = asyncio.create_task(run_publish())
        try:
            await started.wait()
            response = await self._cancel_handler()(_StubMatchRequest("job-cancel"))
        finally:
            publisher._phantom_request = original_request
            publisher._read_config = original_config
        self.assertEqual(response.body["status"], "cancelled")
        self.assertEqual(
            requests, [("DELETE", f"/versions/version-9/artifacts/{'ab' * 32}/uploads")]
        )
        self.assertEqual(job["dependencies"][0]["status"], "cancelled")
        self.assertEqual(job["dependencies"][1]["status"], "uploaded")
        self.assertNotIn("job-cancel", publisher._job_tasks)
        self.assertTrue(any("Cancel requested" in entry["message"] for entry in job["logs"]))

    async def test_cancel_of_a_finished_job_changes_nothing(self):
        job = self._job("job-done", status="completed", progress=100)
        response = await self._cancel_handler()(_StubMatchRequest("job-done"))
        self.assertEqual(response.body["status"], "completed")
        self.assertEqual(job["logs"], [])

    async def test_cancel_of_an_unknown_job_is_not_found(self):
        with self.assertRaises(publisher.web.HTTPNotFound):
            await self._cancel_handler()(_StubMatchRequest("job-missing"))

    async def test_a_phantom_failure_does_not_stop_the_cancel(self):
        job = self._job(
            "job-stubborn",
            version_id="version-9",
            dependencies=[{"id": "model-0", "name": "big", "status": "uploading", "sha256": "ef" * 32}],
        )

        async def failing_request(*_args, **_kwargs):
            raise RuntimeError("Phantom unreachable")

        original_request = publisher._phantom_request
        publisher._phantom_request = failing_request
        try:
            await publisher._abandon_uploads(job, {"origin": "https://phantom.test", "token": "t"})
        finally:
            publisher._phantom_request = original_request
        self.assertEqual(job["dependencies"][0]["status"], "cancelled")
        self.assertTrue(any(entry["level"] == "warning" for entry in job["logs"]))

    async def test_abandon_addresses_the_phantom_the_publish_started_against(self):
        """
        Another tab can repoint the connection mid-publish. The staged version
        exists only in the Phantom that made it, and the same id in the new one
        names something else — so the DELETE must carry the captured config.
        """
        job = self._job(
            "job-switched",
            version_id="version-9",
            dependencies=[{"id": "model-0", "name": "big", "status": "uploading", "sha256": "ab" * 32}],
        )
        used: list[dict[str, Any]] = []

        async def fake_request(_method, _path, config, *_args, **_kwargs):
            used.append(config)
            return {"aborted": True}

        def refuse_config():
            raise AssertionError("cancellation must not re-read the connection")

        original_request = publisher._phantom_request
        original_config = publisher._read_config
        publisher._phantom_request = fake_request
        publisher._read_config = refuse_config
        started_with = {"origin": "https://old.phantom.test", "token": "old"}
        try:
            await publisher._abandon_uploads(job, started_with)
        finally:
            publisher._phantom_request = original_request
            publisher._read_config = original_config
        self.assertEqual(used, [started_with])

    async def test_a_hanging_abandon_is_stopped_at_the_timeout(self):
        """
        A shielded `wait_for` cancelled only the wrapper, leaving the DELETE in
        flight for its full connect timeout. It could then land after a retry
        had resumed the same staged version, and abort THAT upload instead.
        """
        job = self._job(
            "job-hanging",
            version_id="version-9",
            dependencies=[{"id": "model-0", "name": "big", "status": "uploading", "sha256": "ab" * 32}],
        )
        stopped = asyncio.Event()

        async def hanging_request(*_args, **_kwargs):
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                stopped.set()
                raise

        original_request = publisher._phantom_request
        original_timeout = publisher._ABANDON_TIMEOUT_SECONDS
        publisher._phantom_request = hanging_request
        publisher._ABANDON_TIMEOUT_SECONDS = 0.01
        try:
            await publisher._abandon_uploads(job, {"origin": "https://phantom.test", "token": "t"})
        finally:
            publisher._phantom_request = original_request
            publisher._ABANDON_TIMEOUT_SECONDS = original_timeout
        self.assertTrue(stopped.is_set())
        self.assertEqual(job["dependencies"][0]["status"], "cancelled")
        self.assertTrue(any(entry["level"] == "warning" for entry in job["logs"]))

    async def test_a_job_that_never_connected_sends_no_delete(self):
        job = self._job(
            "job-unconnected",
            version_id="version-9",
            dependencies=[{"id": "model-0", "name": "big", "status": "uploading", "sha256": "ab" * 32}],
        )

        async def fake_request(*_args, **_kwargs):
            raise AssertionError("a job with no connection must send nothing")

        original_request = publisher._phantom_request
        publisher._phantom_request = fake_request
        try:
            await publisher._abandon_uploads(job, {})
        finally:
            publisher._phantom_request = original_request
        self.assertEqual(job["dependencies"][0]["status"], "cancelled")


class DiscoveryCancellationTests(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        lock = patch.object(
            publisher,
            "_environment_lock",
            return_value={"distributions": {}, "modules": {}, "sources": {}},
        )
        lock.start()
        self.addCleanup(lock.stop)

    """
    Cancelling the publish task ends the await, not the worker thread. Model
    hashing, package archiving and Hugging Face downloads all run in one, so a
    cancel that only stopped the await left them running — and the archives
    they wrote, registered only once the call returned, were never deleted.
    """

    def test_model_discovery_stops_once_the_publish_is_cancelled(self):
        cancellation = threading.Event()
        cancellation.set()
        with self.assertRaises(publisher._DiscoveryCancelled):
            publisher._discover_models({"1": {"inputs": {}}}, {"nodes": []}, cancellation)

    def test_a_cancelled_package_scan_deletes_the_archives_it_already_wrote(self):
        env = _CustomNodesEnvironment(self)
        env.package("pack_a", ["A_Node"])
        env.package("pack_b", ["B_Node"])
        cancellation = threading.Event()
        original_archive = publisher._archive_package
        written: list[Path] = []

        def archive_then_cancel(directory: Path, cancellation_event=None):
            archive, digest, size = original_archive(directory, cancellation_event)
            written.append(archive)
            # The author pressed Cancel while the next package was still queued.
            cancellation.set()
            return archive, digest, size

        publisher._archive_package = archive_then_cancel
        self.addCleanup(setattr, publisher, "_archive_package", original_archive)
        api, ui = _workflow([("A_Node", {}), ("B_Node", {})])
        with self.assertRaises(publisher._DiscoveryCancelled):
            publisher._discover_packages(api, ui, cancellation)
        # Only this frame ever knew where that archive was: the caller registers
        # archives off the returned list, and a raise returns no list.
        self.assertEqual(len(written), 1)
        self.assertFalse(written[0].parent.exists())

    async def test_cancel_reaps_the_discovery_worker_and_deletes_what_it_wrote(self):
        job_id = "job-discovering"
        job = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0,
            "message": "Waiting to start…",
            "dependencies": [],
            "logs": [],
        }
        publisher._jobs[job_id] = job
        self.addCleanup(publisher._jobs.pop, job_id, None)
        archive_root = Path(tempfile.mkdtemp(prefix="phantom-publisher-test-"))
        self.addCleanup(shutil.rmtree, str(archive_root), True)
        archive = archive_root / "pack_a.tar.gz"
        archive.write_bytes(b"archived")
        entered = threading.Event()
        reaped = threading.Event()

        def slow_packages(_api, _ui, cancellation=None):
            entered.set()
            while cancellation is None or not cancellation.is_set():
                time.sleep(0.01)
            # The worker had already written this archive when it noticed.
            reaped.set()
            return [{"_archive_path": str(archive)}]

        originals = {
            "_read_config": publisher._read_config,
            "_discover_models": publisher._discover_models,
            "_discover_packages": publisher._discover_packages,
        }
        publisher._read_config = lambda: {"origin": "https://phantom.test", "token": "t"}
        publisher._discover_models = lambda *_args: []
        publisher._discover_packages = slow_packages
        try:
            task = asyncio.ensure_future(publisher._run_publish(job_id, _PUBLISH_BODY))
            publisher._job_tasks[job_id] = task
            self.addCleanup(_discard, publisher._job_tasks, job_id)
            await asyncio.to_thread(entered.wait, 5)
            task.cancel()
            # `_run_publish` handles the cancellation itself, so the task ends
            # normally — exactly as the DELETE route awaits it.
            await task
        finally:
            for name, value in originals.items():
                setattr(publisher, name, value)
        # The worker ran to completion rather than being abandoned mid-archive,
        # and what it produced was deleted before the job reported a state.
        self.assertTrue(reaped.is_set())
        self.assertEqual(job["status"], "cancelled")
        self.assertFalse(archive_root.exists())


class _CancelAfter(threading.Event):
    """A cancellation that trips after a set number of checks."""

    def __init__(self, checks: int) -> None:
        super().__init__()
        self._remaining = checks

    def is_set(self) -> bool:
        if self._remaining <= 0:
            return True
        self._remaining -= 1
        return False


class LongRunningStepCancellationTests(unittest.TestCase):
    """
    A cancel waits for the discovery worker rather than abandoning it, so every
    step that runs for minutes has to be interruptible from the inside. Hashing
    a checkpoint and archiving a package are the two that read whole trees.
    """

    def _file_of(self, chunks: int) -> Path:
        directory = Path(tempfile.mkdtemp(prefix="phantom-publisher-test-"))
        self.addCleanup(shutil.rmtree, str(directory), True)
        target = directory / "checkpoint.safetensors"
        # One byte past the chunk boundary, so `chunks` reads are needed.
        target.write_bytes(b"\0" * (8 * 1024 * 1024 * (chunks - 1) + 1))
        return target

    def test_hashing_stops_between_chunks_not_between_files(self):
        # The same cancellation completes a one-chunk file and stops a
        # three-chunk one: the check is inside the read loop, not around it.
        self.assertEqual(len(publisher._sha256(self._file_of(1), _CancelAfter(1))[0]), 64)
        with self.assertRaises(publisher._DiscoveryCancelled):
            publisher._sha256(self._file_of(3), _CancelAfter(1))

    def test_a_cancelled_archive_deletes_its_half_written_temp_directory(self):
        package = Path(tempfile.mkdtemp(prefix="phantom-publisher-test-"))
        self.addCleanup(shutil.rmtree, str(package), True)
        for index in range(5):
            (package / f"node_{index}.py").write_text("x = 1\n", encoding="utf-8")
        created: list[Path] = []
        original_mkdtemp = tempfile.mkdtemp

        def record(*args: Any, **kwargs: Any) -> str:
            directory = original_mkdtemp(*args, **kwargs)
            created.append(Path(directory))
            return directory

        publisher.tempfile.mkdtemp = record
        self.addCleanup(setattr, publisher.tempfile, "mkdtemp", original_mkdtemp)
        with self.assertRaises(publisher._DiscoveryCancelled):
            publisher._archive_package(package, _CancelAfter(2))
        # The caller registers an archive off the tuple this never returned, so
        # nothing else could ever have deleted it.
        self.assertEqual(len(created), 1)
        self.assertFalse(created[0].exists())


class ConcurrentPublishTests(unittest.IsolatedAsyncioTestCase):
    """
    Two ComfyUI tabs publishing the same graph read the same pending key out of
    the browser's shared storage. They therefore name the same staged version
    and the same multipart uploads inside it, so a second local job would upload
    beside the first — and cancelling either would abort the other's uploads.
    """

    def _running_task(self, job_id: str):
        """
        Register a publish that is still in flight, and unregister it after.

        Two details, both about the cleanup rather than the test:

        `_discard` exists because `IsolatedAsyncioTestCase` AWAITS whatever a
        cleanup returns, and `dict.pop` returns the task it removed. Awaiting
        the task this method just cancelled raises CancelledError out of the
        cleanup and errors a test whose assertions all passed — on Python 3.10,
        where the loop runner surfaces it.

        A bare Future stands in for the task because the handler only ever asks
        whether it is done. A `create_task(asyncio.sleep(...))` would answer the
        same question while leaving a real coroutine for loop teardown to chase.
        """
        running = asyncio.get_running_loop().create_future()
        publisher._job_tasks[job_id] = running
        self.addCleanup(_discard, publisher._job_tasks, job_id)
        self.addCleanup(running.cancel)
        return running

    @staticmethod
    def _publish_handler():
        publisher.register_routes()
        return publisher.PromptServer.instance.routes.handlers[
            ("POST", "/phantom-publisher/publish")
        ]

    async def test_a_second_publish_of_the_same_payload_joins_the_running_job(self):
        job = {
            "job_id": "job-first",
            "idempotency_key": "key-1",
            "status": "uploading",
            "progress": 40,
            "message": "Uploading dependency 1 of 2",
            "dependencies": [],
            "logs": [],
        }
        publisher._jobs["job-first"] = job
        self.addCleanup(publisher._jobs.pop, "job-first", None)
        self._running_task("job-first")
        started: list[Any] = []
        original_create_task = publisher.asyncio.create_task
        publisher.asyncio.create_task = started.append
        try:
            response = await self._publish_handler()(
                _StubRequest({**_PUBLISH_BODY, "idempotency_key": "key-1"})
            )
        finally:
            publisher.asyncio.create_task = original_create_task
        self.assertEqual(response.status, 202)
        self.assertEqual(response.body["job_id"], "job-first")
        self.assertEqual(started, [])
        self.assertTrue(any("joined the job" in entry["message"] for entry in job["logs"]))

    async def test_a_finished_job_never_swallows_the_next_publish(self):
        publisher._jobs["job-done"] = {
            "job_id": "job-done",
            "idempotency_key": "key-2",
            "status": "completed",
            "logs": [],
        }
        self.addCleanup(publisher._jobs.pop, "job-done", None)
        original_create_task = publisher.asyncio.create_task
        publisher.asyncio.create_task = lambda coroutine: coroutine.close()
        try:
            response = await self._publish_handler()(
                _StubRequest({**_PUBLISH_BODY, "idempotency_key": "key-2"})
            )
        finally:
            publisher.asyncio.create_task = original_create_task
        self.assertNotEqual(response.body["job_id"], "job-done")
        self.assertEqual(response.body["idempotency_key"], "key-2")
        publisher._jobs.pop(response.body["job_id"], None)
        publisher._job_tasks.pop(response.body["job_id"], None)

    async def test_a_publish_without_a_key_starts_its_own_job(self):
        publisher._jobs["job-keyless"] = {
            "job_id": "job-keyless",
            "idempotency_key": None,
            "status": "uploading",
            "logs": [],
        }
        self.addCleanup(publisher._jobs.pop, "job-keyless", None)
        self._running_task("job-keyless")
        original_create_task = publisher.asyncio.create_task
        publisher.asyncio.create_task = lambda coroutine: coroutine.close()
        try:
            response = await self._publish_handler()(_StubRequest(dict(_PUBLISH_BODY)))
        finally:
            publisher.asyncio.create_task = original_create_task
        self.assertNotEqual(response.body["job_id"], "job-keyless")
        publisher._jobs.pop(response.body["job_id"], None)
        publisher._job_tasks.pop(response.body["job_id"], None)


class CompiledExtensionTests(unittest.TestCase):
    """
    A node pack can ship its logic as a CPython extension — the DiffusionWave
    packs are Nuitka binaries named `_dw_core.cpython-313-x86_64-linux-gnu.so`.
    Such a binary loads on one Python minor and one platform. The publishing
    ComfyUI ran 3.13, Phantom's image runs 3.12, and the pack registered no
    classes there: the build passed and the first request that needed the
    node failed. This is the publish-time check that names it instead.
    """

    def test_reads_the_python_minor_and_platform_off_a_soabi_tag(self):
        self.assertEqual(
            publisher._parse_compiled_extension(
                "DiffusionWave_reactor/_dw_core.cpython-313-x86_64-linux-gnu.so"
            ),
            {
                "path": "DiffusionWave_reactor/_dw_core.cpython-313-x86_64-linux-gnu.so",
                "python": "3.13",
                "abi3": False,
                "system": "linux",
            },
        )
        self.assertEqual(
            publisher._parse_compiled_extension("pack/_core.cp312-win_amd64.pyd")["system"],
            "windows",
        )
        self.assertEqual(
            publisher._parse_compiled_extension("pack/_core.cpython-312-darwin.so")["system"],
            "darwin",
        )
        self.assertEqual(
            publisher._parse_compiled_extension("pack/_core.cpython-314t-x86_64-linux-gnu.so")["python"],
            "3.14",
        )

    def test_marks_stable_abi_and_leaves_an_untagged_library_unknown(self):
        self.assertTrue(publisher._parse_compiled_extension("pack/_core.abi3.so")["abi3"])
        untagged = publisher._parse_compiled_extension("pack/vendor/libonnx.so")
        self.assertEqual((untagged["python"], untagged["abi3"], untagged["system"]), (None, False, None))
        self.assertIsNone(publisher._parse_compiled_extension("pack/nodes.py"))

    def test_lists_compiled_extensions_by_archive_relative_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary) / "DiffusionWave_reactor"
            (package / "vendor").mkdir(parents=True)
            (package / "__pycache__").mkdir()
            (package / "_dw_core.cpython-313-x86_64-linux-gnu.so").write_bytes(b"\x7fELF")
            (package / "vendor" / "libonnx.so").write_bytes(b"\x7fELF")
            (package / "__pycache__" / "junk.cpython-313.so").write_bytes(b"")
            (package / "nodes.py").write_text("", encoding="utf-8")
            self.assertEqual(
                publisher._compiled_extensions(package),
                [
                    "DiffusionWave_reactor/_dw_core.cpython-313-x86_64-linux-gnu.so",
                    "DiffusionWave_reactor/vendor/libonnx.so",
                ],
            )

    def test_package_constraint_is_the_set_of_pythons_its_linux_binaries_cover(self):
        self.assertEqual(
            publisher._package_python_constraint(
                {
                    "cnr_id": None,
                    "_package_directory": "/comfy/custom_nodes/DiffusionWave_reactor",
                    "compiled_extensions": [
                        "DiffusionWave_reactor/_dw_core.cpython-313-x86_64-linux-gnu.so",
                        "DiffusionWave_reactor/_dw_core.cpython-312-x86_64-linux-gnu.so",
                        "DiffusionWave_reactor/_s.abi3.so",
                        "DiffusionWave_reactor/vendor/libonnx.so",
                        "DiffusionWave_reactor/_c.cpython-313-darwin.so",
                    ],
                }
            ),
            {
                "package": "DiffusionWave_reactor",
                "pythons": ["3.12", "3.13"],
                "paths": {
                    "3.13": "DiffusionWave_reactor/_dw_core.cpython-313-x86_64-linux-gnu.so",
                    "3.12": "DiffusionWave_reactor/_dw_core.cpython-312-x86_64-linux-gnu.so",
                },
            },
        )
        self.assertIsNone(publisher._package_python_constraint({"cnr_id": "pure"}))

    def test_a_binary_for_a_newer_python_than_the_base_is_not_a_problem(self):
        # Phantom builds the image for the Python the binary needs.
        runtime = {"python": "3.12", "supported_pythons": ["3.12", "3.13"], "constraints": []}
        packages = [{"cnr_id": "dw", "compiled_extensions": ["DiffusionWave_reactor/_dw_core.cpython-313-x86_64-linux-gnu.so"]}]
        self.assertIsNone(publisher._image_python_problem(packages, runtime))
        self.assertEqual(publisher._foreign_platform_problems(packages, runtime), [])

    def test_a_disagreement_with_the_targets_other_graphs_names_both_sides(self):
        runtime = {
            "python": "3.12",
            "supported_pythons": ["3.12", "3.13"],
            "constraints": [
                {
                    "graph": "the primary graph",
                    "package": "DiffusionWave_PickResolution",
                    "pythons": ["3.12"],
                    "paths": {"3.12": "DiffusionWave_PickResolution/_dw_core.cpython-312-x86_64-linux-gnu.so"},
                }
            ],
        }
        packages = [
            {
                "cnr_id": None,
                "_package_directory": "/comfy/custom_nodes/DiffusionWave_reactor",
                "compiled_extensions": ["DiffusionWave_reactor/_dw_core.cpython-313-x86_64-linux-gnu.so"],
            }
        ]
        self.assertEqual(
            publisher._image_python_problem(packages, runtime),
            'Package "DiffusionWave_PickResolution" in the primary graph is built for Python '
            "3.12 (DiffusionWave_PickResolution/_dw_core.cpython-312-x86_64-linux-gnu.so), but package \"DiffusionWave_reactor\" in this graph is built for "
            "Python 3.13 (DiffusionWave_reactor/_dw_core.cpython-313-x86_64-linux-gnu.so). One image serves every graph of a workflow. Publish both "
            "graphs from the same ComfyUI, install matching builds of the packages, or split "
            "them into separate workflows in Phantom.",
        )

    def test_a_python_torch_has_no_wheels_for_names_what_phantom_can_build(self):
        runtime = {"python": "3.12", "supported_pythons": ["3.12", "3.13"], "constraints": []}
        packages = [
            {
                "cnr_id": "dw",
                "compiled_extensions": ["dw/_dw_core.cpython-314-x86_64-linux-gnu.so"],
            }
        ]
        self.assertEqual(
            publisher._image_python_problem(packages, runtime),
            'Package "dw" in this graph is built for Python 3.14 '
            "(dw/_dw_core.cpython-314-x86_64-linux-gnu.so). Phantom can build images for "
            "Python 3.12 and 3.13; PyTorch ships no wheels for 3.14 yet.",
        )

    def test_a_foreign_platform_binary_is_refused(self):
        problems = publisher._foreign_platform_problems(
            [{"cnr_id": "pack", "compiled_extensions": ["pack/_core.cpython-312-darwin.so"]}],
            {"python": "3.12", "system": "Linux", "machine": "x86_64"},
        )
        self.assertEqual(len(problems), 1)
        self.assertIn("built for macOS", problems[0])
        self.assertIn("Phantom runs workflows on Linux x86_64", problems[0])

    def test_an_older_phantom_without_a_runtime_checks_nothing(self):
        packages = [{"cnr_id": "dw", "compiled_extensions": ["DiffusionWave_reactor/_dw_core.cpython-313-x86_64-linux-gnu.so"]}]
        self.assertIsNone(publisher._image_python_problem(packages, None))
        self.assertEqual(publisher._foreign_platform_problems(packages, None), [])

    def test_runtime_records_this_interpreter(self):
        runtime = publisher._runtime()
        self.assertEqual(
            runtime["python"], ".".join(str(part) for part in sys.version_info[:3])
        )
        self.assertEqual(runtime["python_tag"], f"cp{sys.version_info[0]}{sys.version_info[1]}")
        self.assertIn("system", runtime)
        self.assertIn("machine", runtime)
        # The CUDA stack rides in the same block, every key present even when
        # this interpreter has no torch: Phantom reads the keys, not their presence.
        for key in ("torch", "cuda", "cudnn", "driver", "gpu"):
            self.assertIn(key, runtime)

    def test_gpu_runtime_records_torch_and_the_driver(self):
        # Phantom builds the image FROM this platform, so torch's full version
        # (build tag included), its CUDA and cuDNN, and the driver underneath are
        # what let it start from the same CUDA base and install the same torch.
        cudnn = types.SimpleNamespace(is_available=lambda: True, version=lambda: 91002)
        fake_torch = types.SimpleNamespace(
            __version__="2.10.0+cu128",
            version=types.SimpleNamespace(cuda="12.8"),
            backends=types.SimpleNamespace(cudnn=cudnn),
        )
        smi = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="580.65.06, NVIDIA GeForce RTX 4090\n", stderr=""
        )
        with patch.dict(sys.modules, {"torch": fake_torch}), patch.object(
            publisher.subprocess, "run", return_value=smi
        ):
            facts = publisher._gpu_runtime()
        self.assertEqual(
            facts,
            {
                "torch": "2.10.0+cu128",
                "cuda": "12.8",
                "cudnn": "91002",
                "driver": "580.65.06",
                "gpu": "NVIDIA GeForce RTX 4090",
            },
        )

    def test_gpu_runtime_is_best_effort(self):
        # A CPU torch names no CUDA, and a machine without nvidia-smi records no
        # driver; neither stops the publish.
        fake_torch = types.SimpleNamespace(
            __version__="2.10.0+cpu",
            version=types.SimpleNamespace(cuda=None),
            backends=types.SimpleNamespace(
                cudnn=types.SimpleNamespace(is_available=lambda: False, version=lambda: None)
            ),
        )
        with patch.dict(sys.modules, {"torch": fake_torch}), patch.object(
            publisher.subprocess, "run", side_effect=FileNotFoundError("nvidia-smi")
        ):
            facts = publisher._gpu_runtime()
        self.assertEqual(
            facts, {"torch": "2.10.0+cpu", "cuda": None, "cudnn": None, "driver": None, "gpu": None}
        )


class PublishRuntimeCheckTests(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        lock = patch.object(
            publisher,
            "_environment_lock",
            return_value={"distributions": {}, "modules": {}, "sources": {}},
        )
        lock.start()
        self.addCleanup(lock.stop)

    def _job(self, job_id: str) -> dict[str, Any]:
        job = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0,
            "message": "Waiting to start…",
            "dependencies": [],
            "logs": [],
        }
        publisher._jobs[job_id] = job
        self.addCleanup(publisher._jobs.pop, job_id, None)
        return job

    def _stub(self, packages: list[dict[str, Any]], request: Any) -> None:
        originals = {
            "_read_config": publisher._read_config,
            "_discover_models": publisher._discover_models,
            "_discover_packages": publisher._discover_packages,
            "_discover_huggingface_models": publisher._discover_huggingface_models,
            "_phantom_request": publisher._phantom_request,
        }
        publisher._read_config = lambda: {"origin": "https://phantom.test", "token": "t"}
        publisher._discover_models = lambda *_args: []
        publisher._discover_packages = lambda *_args: packages
        publisher._discover_huggingface_models = lambda *_args: []
        publisher._phantom_request = request
        for name, value in originals.items():
            self.addCleanup(setattr, publisher, name, value)

    async def test_refuses_before_staging_when_the_graphs_would_disagree(self):
        job = self._job("job-runtime-refused")
        calls: list[str] = []

        async def fake_request(method, path, _config, *_args, **_kwargs):
            calls.append(f"{method} {path}")
            if path == "/targets":
                return {
                    "targets": [
                        {
                            "workflow_id": "wf-1",
                            "runtime": {
                                "python": "3.12",
                                "system": "Linux",
                                "machine": "x86_64",
                                "supported_pythons": ["3.12", "3.13"],
                                "constraints": [
                                    {
                                        "graph": "the primary graph",
                                        "package": "DiffusionWave_PickResolution",
                                        "pythons": ["3.12"],
                                        "paths": {
                                            "3.12": "DiffusionWave_PickResolution/_dw_core.cpython-312-x86_64-linux-gnu.so"
                                        },
                                    }
                                ],
                            },
                        }
                    ]
                }
            return {"workflow_version_id": "version-3", "version": 3}

        self._stub(
            [
                {
                    "cnr_id": None,
                    "_package_directory": "/comfy/custom_nodes/DiffusionWave_reactor",
                    "class_types": ["dw_reactor"],
                    "archive_sha256": "c" * 64,
                    "compiled_extensions": [
                        "DiffusionWave_reactor/_dw_core.cpython-313-x86_64-linux-gnu.so"
                    ],
                }
            ],
            fake_request,
        )
        await publisher._run_publish("job-runtime-refused", _PUBLISH_BODY)
        self.assertEqual(job["status"], "failed")
        self.assertIn('Package "DiffusionWave_PickResolution" in the primary graph is built for Python 3.12', job["error"])
        self.assertIn('package "DiffusionWave_reactor" in this graph is built for Python 3.13', job["error"])
        self.assertIn("Publish both graphs from the same ComfyUI", job["error"])
        # Nothing staged, nothing uploaded: the only call was the runtime lookup.
        self.assertEqual(calls, ["GET /targets"])

    async def test_publishes_and_records_the_runtime_when_nothing_disagrees(self):
        # The binary is cp313 and the base runs 3.12: Phantom's image follows
        # the binary, so this publishes.
        job = self._job("job-runtime-ok")
        bodies: list[Any] = []

        async def fake_request(method, path, _config, body=None, *_args, **_kwargs):
            if path == "/targets":
                return {
                    "targets": [
                        {
                            "workflow_id": "wf-1",
                            "runtime": {
                                "python": "3.12",
                                "supported_pythons": ["3.12", "3.13"],
                                "constraints": [],
                            },
                        }
                    ]
                }
            if path == "/versions":
                bodies.append(body)
            return {"workflow_version_id": "version-3", "version": 3}

        self._stub(
            [
                {
                    "cnr_id": "pack",
                    "class_types": ["Node"],
                    "archive_sha256": "c" * 64,
                    "compiled_extensions": ["pack/_core.cpython-313-x86_64-linux-gnu.so"],
                }
            ],
            fake_request,
        )
        await publisher._run_publish("job-runtime-ok", _PUBLISH_BODY)
        self.assertEqual(job["status"], "completed", job.get("error"))
        manifest = bodies[0]["manifest"]
        self.assertEqual(manifest["comfyui"]["runtime"]["python_tag"], f"cp{sys.version_info[0]}{sys.version_info[1]}")
        self.assertEqual(
            manifest["node_packages"][0]["compiled_extensions"],
            ["pack/_core.cpython-313-x86_64-linux-gnu.so"],
        )

    async def test_checkout_manifest_upload_and_cleanup(self):
        job = self._job("job-python-checkout")
        manifests, uploaded = [], []

        async def request(method, path, config, body=None, **kwargs):
            if path == "/versions":
                manifests.append(body["manifest"])
            return {"workflow_version_id": "version-3", "version": 3}

        async def upload(version_id, digest, path, config, size, **kwargs):
            self.assertTrue(path.is_file())
            uploaded.append(path)
            return True

        self._stub([], request)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sam3"
            path.mkdir()
            (path / "setup.py").write_text("setup()")
            lock = {
                "distributions": {"sam3": "0.1"},
                "modules": {},
                "sources": {"sam3": {"kind": "dir", "editable": True, "_local_path": str(path)}},
            }
            with (
                patch.object(publisher, "_environment_lock", return_value=lock),
                patch.object(publisher, "_upload", side_effect=upload),
            ):
                await publisher._run_publish("job-python-checkout", _PUBLISH_BODY)
        self.assertEqual(job["status"], "completed", job.get("error"))
        source = manifests[0]["comfyui"]["environment"]["sources"]["sam3"]
        self.assertEqual(len(source["archive_sha256"]), 64)
        self.assertNotIn("_local_path", source)
        self.assertEqual(job["dependencies"][0]["status"], "reused")
        self.assertEqual(len(uploaded), 1)
        self.assertFalse(uploaded[0].exists())

    async def test_an_older_phantom_without_a_runtime_still_publishes(self):
        job = self._job("job-runtime-unknown")

        async def fake_request(_method, path, _config, *_args, **_kwargs):
            if path == "/targets":
                return {"targets": [{"workflow_id": "wf-1"}]}
            return {"workflow_version_id": "version-3", "version": 3}

        self._stub(
            [
                {
                    "cnr_id": "pack",
                    "class_types": ["Node"],
                    "archive_sha256": "c" * 64,
                    "compiled_extensions": ["pack/_core.cpython-313-x86_64-linux-gnu.so"],
                }
            ],
            fake_request,
        )
        await publisher._run_publish("job-runtime-unknown", _PUBLISH_BODY)
        self.assertEqual(job["status"], "completed", job.get("error"))

    async def test_the_environment_lock_is_taken_in_a_discovery_worker(self):
        # The shadow check hashes native libraries; done on the event loop it
        # would stall progress polling and the cancel route for the duration.
        job = self._job("job-lock-thread")
        seen: list[tuple[threading.Thread, Any]] = []

        def lock(cancellation=None):
            seen.append((threading.current_thread(), cancellation))
            return {"distributions": {}, "modules": {}, "sources": {}}

        async def fake_request(_method, path, _config, *_args, **_kwargs):
            if path == "/targets":
                return {"targets": []}
            return {"workflow_version_id": "version-3", "version": 3}

        self._stub([], fake_request)
        with patch.object(publisher, "_environment_lock", side_effect=lock):
            await publisher._run_publish("job-lock-thread", _PUBLISH_BODY)
        self.assertEqual(job["status"], "completed", job.get("error"))
        self.assertEqual(len(seen), 1)
        self.assertIsNot(seen[0][0], threading.main_thread())
        self.assertIsInstance(seen[0][1], threading.Event)


class StagedVariationTests(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        lock = patch.object(
            publisher,
            "_environment_lock",
            return_value={"distributions": {}, "modules": {}, "sources": {}},
        )
        lock.start()
        self.addCleanup(lock.stop)

    """
    A new variation learns its id from the publish that created it. The panel
    writes that id into the graph, so the next publish updates that variation
    instead of adding a second one under the same label.
    """

    async def test_the_job_reports_the_variation_id_phantom_assigned(self):
        job_id = "job-staged"
        job = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0,
            "message": "Waiting to start…",
            "dependencies": [],
            "logs": [],
        }
        publisher._jobs[job_id] = job
        self.addCleanup(publisher._jobs.pop, job_id, None)

        async def fake_request(_method, path, _config, *_args, **_kwargs):
            version = {"workflow_version_id": "version-3", "version": 3}
            if path == "/versions":
                return {
                    **version,
                    "variation": {"variation_id": "variation-7", "label": "Reference image"},
                }
            return version

        originals = {
            "_read_config": publisher._read_config,
            "_discover_models": publisher._discover_models,
            "_discover_packages": publisher._discover_packages,
            "_discover_huggingface_models": publisher._discover_huggingface_models,
            "_phantom_request": publisher._phantom_request,
        }
        publisher._read_config = lambda: {"origin": "https://phantom.test", "token": "t"}
        publisher._discover_models = lambda *_args: []
        publisher._discover_packages = lambda *_args: []
        publisher._discover_huggingface_models = lambda *_args: []
        publisher._phantom_request = fake_request
        try:
            await publisher._run_publish(
                job_id, _PUBLISH_BODY, {"label": "Reference image"}
            )
        finally:
            for name, value in originals.items():
                setattr(publisher, name, value)
        self.assertEqual(job["status"], "completed")
        self.assertEqual(
            job["variation"], {"variation_id": "variation-7", "label": "Reference image"}
        )

    async def test_a_primary_publish_reports_no_variation(self):
        job_id = "job-primary"
        job = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0,
            "message": "Waiting to start…",
            "dependencies": [],
            "logs": [],
        }
        publisher._jobs[job_id] = job
        self.addCleanup(publisher._jobs.pop, job_id, None)

        async def fake_request(_method, _path, _config, *_args, **_kwargs):
            return {"workflow_version_id": "version-3", "version": 3}

        originals = {
            "_read_config": publisher._read_config,
            "_discover_models": publisher._discover_models,
            "_discover_packages": publisher._discover_packages,
            "_discover_huggingface_models": publisher._discover_huggingface_models,
            "_phantom_request": publisher._phantom_request,
        }
        publisher._read_config = lambda: {"origin": "https://phantom.test", "token": "t"}
        publisher._discover_models = lambda *_args: []
        publisher._discover_packages = lambda *_args: []
        publisher._discover_huggingface_models = lambda *_args: []
        publisher._phantom_request = fake_request
        try:
            await publisher._run_publish(job_id, _PUBLISH_BODY)
        finally:
            for name, value in originals.items():
                setattr(publisher, name, value)
        self.assertEqual(job["status"], "completed")
        self.assertNotIn("variation", job)


_PUBLISH_BODY: dict[str, Any] = {
    "workflow_id": "wf-1",
    "api_workflow": {},
    "ui_workflow": {"nodes": []},
}


class _StubMatchRequest:
    def __init__(self, job_id: str) -> None:
        self.match_info = {"job_id": job_id}


class _StubRequest:
    def __init__(self, body: dict[str, Any]) -> None:
        self._body = body

    async def json(self) -> dict[str, Any]:
        return self._body


class InstallSourceTests(unittest.TestCase):
    def source(self, value):
        return publisher._install_source(
            types.SimpleNamespace(read_text=lambda _: json.dumps(value))
        )

    def test_sources_and_credentials(self):
        vcs = self.source(
            {
                "url": "https://user:secret@github.com/meta/sam3",
                "vcs_info": {"vcs": "git", "commit_id": "a" * 40},
            }
        )
        self.assertEqual(vcs["url"], "https://github.com/meta/sam3")
        self.assertEqual(vcs["kind"], "vcs")
        directory = self.source({"url": "file:///tmp/my%20lib", "dir_info": {"editable": True}})
        self.assertEqual(directory["_local_path"], "/tmp/my lib")
        self.assertTrue(directory["editable"])
        self.assertIsNone(self.source({"url": "https://example.com/x", "dir_info": {}}))
        for info in [{"hashes": {"sha256": "abc"}}, {"hash": "sha256=abc"}]:
            self.assertEqual(
                self.source({"url": "https://example.com/a.whl", "archive_info": info})["hashes"],
                {"sha256": "abc"},
            )
        self.assertIsNone(self.source({"url": "file:///tmp/a.whl", "archive_info": {}}))
        self.assertIsNone(
            self.source({"url": "https://example.com/repo", "vcs_info": {"vcs": "git"}})
        )

    def test_bad_metadata_is_ignored(self):
        for value in [None, "{bad json"]:
            self.assertIsNone(
                publisher._install_source(types.SimpleNamespace(read_text=lambda _: value))
            )
        self.assertIsNone(
            publisher._install_source(types.SimpleNamespace(read_text=lambda _: 1 / 0))
        )

    def test_lock_keeps_versions_and_normalizes_sources(self):
        with _installed(
            {"sam": ("SAM_3", "0.1"), "plain": ("plain", "1")},
            {
                "SAM_3": {
                    "url": "https://example.com/repo",
                    "vcs_info": {"vcs": "git", "commit_id": "a" * 40},
                }
            },
        ):
            lock = publisher._environment_lock()
            self.assertEqual(list(lock["sources"]), ["sam-3"])
            with patch.object(publisher, "_install_source", side_effect=RuntimeError("broken")):
                self.assertEqual(
                    publisher._environment_lock()["distributions"], lock["distributions"]
                )
        with _installed({"plain": ("plain", "1")}):
            self.assertEqual(publisher._environment_lock()["sources"], {})


class PythonSourceArchiveTests(unittest.TestCase):
    def test_reproducibility_exclusions_progress_and_missing_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lib"
            path.mkdir()
            (path / "setup.py").write_text("setup()")
            for name in ["build", ".venv", "lib.egg-info", ".git", "__pycache__"]:
                (path / name).mkdir()
                (path / name / "junk").write_text("ignored")

            def lock():
                return {
                    "distributions": {"lib": "1"},
                    "sources": {"lib": {"kind": "dir", "editable": True, "_local_path": str(path)}},
                }

            first = lock()
            items = publisher._archive_python_sources(first)
            second = publisher._archive_python_sources(lock())
            for item in items + second:
                self.addCleanup(
                    shutil.rmtree, Path(item["_archive_path"]).parent, ignore_errors=True
                )
            self.assertEqual(items[0]["archive_sha256"], second[0]["archive_sha256"])
            self.assertNotIn("_local_path", first["sources"]["lib"])
            with tarfile.open(items[0]["_archive_path"]) as archive:
                self.assertEqual(archive.getnames(), ["lib/setup.py"])
            dependencies, uploads = publisher._dependency_progress([], [], items)
            self.assertEqual(dependencies[0]["kind"], "python_source")
            self.assertEqual(uploads[0][0], items[0]["archive_sha256"])
            with patch.object(publisher, "_PYTHON_SOURCE_MAX_BYTES", 1):
                with self.assertRaisesRegex(RuntimeError, "lib.*Move data out.*PyPI or git"):
                    publisher._archive_python_sources(lock())
            shutil.rmtree(path)
            missing = lock()
            self.assertEqual(publisher._archive_python_sources(missing), [])
            self.assertEqual(missing["sources"], {})
