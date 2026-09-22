from __future__ import annotations

import importlib.util
import json
import py_compile
import sys
import tarfile
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import runtime_capture as runtime


class RuntimeCaptureTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.site = self.root / "venv/lib/python3.12/site-packages"
        self.site.mkdir(parents=True)
        self.core = self.root / "ComfyUI"
        self.core.mkdir()
        for name in ("main.py", "nodes.py"):
            (self.core / name).write_text("# locally patched core\n")
        (self.site / "provider.so").write_bytes(b"\x7fELFfixture")
        (self.site / "package.py").write_text("value = 42\n")
        self.dist = types.SimpleNamespace(
            metadata={"Name": "onnxruntime-gpu"},
            version="1.30.0",
            files=["provider.so", "package.py"],
            locate_file=lambda name: self.site / name,
            read_text=lambda name: None,
        )
        self.ldd_output = ""

        def run(args, **kwargs):
            if args[0] == "git":
                return types.SimpleNamespace(
                    stdout=b"main.py\0nodes.py\0", returncode=0
                )
            self.assertEqual(args[0], "ldd")
            return types.SimpleNamespace(
                stdout=self.ldd_output, stderr="", returncode=0
            )

        read_text = Path.read_text

        def read(path, *args, **kwargs):
            return (
                ""
                if str(path) == "/proc/self/maps"
                else read_text(path, *args, **kwargs)
            )

        for mock in (
            patch.object(runtime.platform, "system", return_value="Linux"),
            patch.object(runtime.platform, "machine", return_value="x86_64"),
            patch.object(
                runtime.platform, "python_implementation", return_value="CPython"
            ),
            patch.object(
                runtime,
                "os_identity",
                return_value={"id": "ubuntu", "version": "22.04"},
            ),
            patch.object(runtime.sysconfig, "get_path", return_value=str(self.site)),
            patch.object(sys, "prefix", str(self.root / "venv")),
            patch.object(
                runtime.importlib.metadata, "distributions", return_value=[self.dist]
            ),
            patch.object(runtime.subprocess, "run", side_effect=run),
            patch.object(runtime.shutil, "which", return_value=None),
            patch.object(Path, "read_text", read),
        ):
            mock.start()
            self.addCleanup(mock.stop)

    def capture(self, **kwargs):
        result = runtime.capture(
            {"onnxruntime-gpu": "1.30.0"}, [], self.core, lambda: None, **kwargs
        )
        directory = Path(result["_archive_path"]).parent
        self.assertTrue(directory.name.startswith("phantom-runtime-"))
        self.addCleanup(runtime.shutil.rmtree, directory)
        return result

    def test_exact_packages_core_and_external_cuda13_are_captured_deterministically(
        self,
    ):
        library = self.root / "cuda-13/libcudart.so.13"
        library.parent.mkdir()
        library.write_bytes(b"\x7fELFCUDA13")
        self.ldd_output = f"libcudart.so.13 => {library} (0x123)\nlibcuda.so.1 => /driver/libcuda.so.1 (0x456)\n"
        first, second = self.capture(), self.capture()
        self.assertEqual(first["archive_sha256"], second["archive_sha256"])
        self.assertEqual(first["cuda_runtime_majors"], [13])
        with tarfile.open(first["_archive_path"]) as tar:
            inventory = json.load(tar.extractfile("inventory.json"))
            self.assertEqual(inventory["distributions"], {"onnxruntime-gpu": "1.30.0"})
            self.assertIn("native/" + str(library).lstrip("/"), tar.getnames())
            self.assertIn("core/main.py", tar.getnames())
            self.assertFalse(any("libcuda.so" in name for name in tar.getnames()))

    def test_missing_cuda_fails_before_upload_and_does_not_guess_extras(self):
        self.ldd_output = "libcublas.so.13 => not found\nlibcudart.so.13 => not found\n"
        with self.assertRaisesRegex(
            RuntimeError, "unresolved CUDA dependencies.*libcublas.so.13"
        ):
            self.capture()

    def test_optional_tensorrt_is_a_finding_not_a_cuda_repair(self):
        self.ldd_output = "libnvinfer.so.10 => not found\n"
        self.assertEqual(self.capture()["unresolved"][0]["library"], "libnvinfer.so.10")

    def test_missing_record_file_is_not_silently_omitted(self):
        self.dist.files.append("missing.py")
        with self.assertRaisesRegex(RuntimeError, "installed file is missing"):
            self.capture()

    def test_script_record_parent_segments_are_normalized(self):
        script = self.root / "venv/bin/onnx-test"
        script.parent.mkdir()
        script.write_text("#!/original/bin/python\n")
        self.dist.files.append("../../../bin/onnx-test")
        result = self.capture()
        with tarfile.open(result["_archive_path"]) as tar:
            self.assertIn("prefix/bin/onnx-test", tar.getnames())
            self.assertFalse(any(".." in name for name in tar.getnames()))

    def test_editable_source_is_preserved_without_running_an_installer(self):
        source = self.root / "sam3"
        source.mkdir()
        entry = source / "sam3.py"
        entry.write_text("locally_edited = True\n")
        self.dist.read_text = lambda name: json.dumps(
            {"url": source.as_uri(), "dir_info": {"editable": True}}
        )
        with self.assertRaisesRegex(RuntimeError, "editable source is missing"):
            self.capture()
        result = self.capture(editable_sources={"onnxruntime-gpu": (source, [entry])})
        self.assertEqual(result["source_roots"], [str(source.absolute())])
        with tarfile.open(result["_archive_path"]) as tar:
            self.assertIn("source/" + str(entry).lstrip("/"), tar.getnames())

    def test_same_soname_different_cudnn_builds_are_refused(self):
        first, second = (
            self.root / "cu12/libcudnn.so.9",
            self.root / "cu13/libcudnn.so.9",
        )
        for path in (first, second):
            path.parent.mkdir()
            path.write_bytes(b"\x7fELF" + str(path).encode())
        self.ldd_output = (
            f"libcudnn.so.9 => {first} (0x1)\nlibcudnn.so.9 => {second} (0x2)\n"
        )
        with self.assertRaisesRegex(RuntimeError, "Conflicting cuDNN libraries"):
            self.capture()

    def test_private_cudnn_and_platform_cudnn_are_preserved_separately(self):
        names = ("nvvfx/libs/libcudnn.so.9", "nvidia/cudnn/lib/libcudnn.so.9")
        for name in names:
            path = self.site / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"\x7fELF" + name.encode())
            self.dist.files.append(name)
        # Match Torch's non-canonical path from the reported failure.
        (self.site / "torch/lib").mkdir(parents=True)
        torch_path = self.site / "torch/lib/../../nvidia/cudnn/lib/libcudnn.so.9"
        self.ldd_output = (
            f"libcudnn.so.9 => {self.site / names[0]} (0x1)\n"
            f"libcudnn.so.9 => {torch_path} (0x2)\n"
        )
        result = self.capture()
        self.assertEqual(result["native_dirs"], [])
        with tarfile.open(result["_archive_path"]) as tar:
            for name in names:
                self.assertEqual(
                    tar.extractfile("site/" + name).read(),
                    (self.site / name).read_bytes(),
                )
            self.assertFalse(any(name.startswith("native/") for name in tar.getnames()))

    def test_external_cudnn_cannot_conflict_with_globally_registered_wheel(self):
        wheel = self.site / "nvidia/cudnn/lib/libcudnn.so.9"
        external = self.root / "cuda/libcudnn.so.9"
        for path in (wheel, external):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"\x7fELF" + str(path).encode())
        self.dist.files.append(wheel.relative_to(self.site).as_posix())
        self.ldd_output = (
            f"libcudnn.so.9 => {wheel} (0x1)\nlibcudnn.so.9 => {external} (0x2)\n"
        )
        with self.assertRaisesRegex(RuntimeError, "Conflicting cuDNN libraries"):
            self.capture()

    def test_driver_families_and_symlink_aliases_are_not_captured(self):
        names = (
            "libcuda.so.1",
            "libcudadebugger.so.1",
            "libnvoptix.so.1",
            "libGLX_nvidia.so.0",
            "libEGL_nvidia.so.0",
            "libGLESv1_CM_nvidia.so.1",
            "libGLESv2_nvidia.so.2",
            "libglxserver_nvidia.so.1",
            "libvdpau_nvidia.so.1",
            "libnvidia-ml.so.1",
            "libnvcuvid.so.1",
            "nvidia_drv.so",
        )
        external = self.root / "driver"
        external.mkdir()
        for name in names:
            (self.site / name).write_bytes(b"\x7fELFdriver")
            self.dist.files.append(name)
            (external / name).write_bytes(b"\x7fELFdriver")
        alias = self.site / "alias.so"
        alias.symlink_to(external / "libnvoptix.so.1")
        self.dist.files.append(alias.name)
        self.ldd_output = "\n".join(
            f"{name} => {external / name} (0x1)" for name in names
        )
        self.ldd_output += f"\nother.so => {alias} (0x1)"
        for name in ("libcudart.so.13", "libGLdispatch.so.0", "libOpenCL.so.1"):
            (external / name).write_bytes(b"\x7fELFruntime")
            self.ldd_output += f"\n{name} => {external / name} (0x1)"
        result = self.capture()
        with tarfile.open(result["_archive_path"]) as tar:
            archived = {Path(name).name for name in tar.getnames()}
            self.assertFalse(archived.intersection(names))
            self.assertNotIn("alias.so", archived)
            self.assertTrue(
                {"libcudart.so.13", "libGLdispatch.so.0", "libOpenCL.so.1"} <= archived
            )

    def test_process_maps_and_editable_sources_do_not_transplant_driver_files(self):
        project = self.root / "project"
        project.mkdir()
        source = project / "libGLX_nvidia.so.0"
        source.write_bytes(b"\x7fELFdriver")
        original_read = Path.read_text

        def read(path, *args, **kwargs):
            if str(path) == "/proc/self/maps":
                return f"0-1 r-xp 0 00:00 1 {source}\n"
            return original_read(path, *args, **kwargs)

        with patch.object(Path, "read_text", read):
            result = self.capture(editable_sources={"project": (project, [source])})
        with tarfile.open(result["_archive_path"]) as tar:
            self.assertFalse(any("libGLX_nvidia" in name for name in tar.getnames()))
        self.assertNotIn(str(project), result["native_dirs"])

    def test_sourceless_recorded_bytecode_remains_importable(self):
        source = self.site / "only_bytecode.py"
        source.write_text("value = 73\n")
        compiled = self.site / "only_bytecode.pyc"
        py_compile.compile(str(source), cfile=str(compiled), doraise=True)
        source.unlink()
        self.dist.files.append(compiled.name)
        result = self.capture()
        restored = self.root / "restored.pyc"
        with tarfile.open(result["_archive_path"]) as tar:
            restored.write_bytes(tar.extractfile("site/only_bytecode.pyc").read())
        spec = importlib.util.spec_from_file_location("restored", restored)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.value, 73)

    def test_unreadable_unrelated_metadata_does_not_abort_capture(self):
        class Broken:
            @property
            def metadata(self):
                raise OSError("partial metadata")

        with patch.object(
            runtime.importlib.metadata,
            "distributions",
            return_value=[Broken(), self.dist],
        ):
            self.capture()
        with (
            patch.object(
                runtime.importlib.metadata, "distributions", return_value=[Broken()]
            ),
            self.assertRaisesRegex(
                RuntimeError, "Distributions disappeared.*onnxruntime-gpu"
            ),
        ):
            self.capture()

    def test_required_package_version_errors_are_not_ignored(self):
        class BrokenVersion:
            locate_file = staticmethod(self.dist.locate_file)

            @property
            def metadata(self):
                return {"Name": "onnxruntime-gpu"}

            @property
            def version(self):
                raise OSError("broken required version")

        with (
            patch.object(
                runtime.importlib.metadata,
                "distributions",
                return_value=[BrokenVersion()],
            ),
            self.assertRaisesRegex(OSError, "broken required version"),
        ):
            self.capture()

    def test_active_user_site_and_recorded_user_script_are_preserved(self):
        user_prefix = self.root / "user"
        user_site = user_prefix / "lib/python3.12/site-packages"
        user_site.mkdir(parents=True)
        script = user_prefix / "bin/onnx-command"
        script.parent.mkdir()
        script.write_text("#!/original/bin/python\n")
        (user_site / "user_package.py").write_text("value = 1\n")
        self.dist.files = ["user_package.py", "../../../bin/onnx-command"]
        self.dist.locate_file = lambda name: user_site / name
        with (
            patch.object(sys, "path", [str(user_site)]),
            patch.object(
                runtime.site, "getusersitepackages", return_value=str(user_site)
            ),
            patch.object(runtime.site, "getuserbase", return_value=str(user_prefix)),
            patch.object(runtime.site, "ENABLE_USER_SITE", True),
        ):
            result = self.capture()
        with tarfile.open(result["_archive_path"]) as tar:
            self.assertIn("site/user_package.py", tar.getnames())
            self.assertIn("prefix/bin/onnx-command", tar.getnames())

    def test_import_visible_target_installation_is_captured(self):
        target = self.root / "custom-target"
        target.mkdir()
        (target / "custom.py").write_text("value = 1\n")
        self.dist.files = ["custom.py"]
        self.dist.locate_file = lambda name: target / name
        with patch.object(sys, "path", [str(target)]):
            result = self.capture()
        with tarfile.open(result["_archive_path"]) as tar:
            self.assertIn("site/custom.py", tar.getnames())

    def test_distinct_site_roots_cannot_silently_merge_regular_packages(self):
        other = self.root / "other-site"
        other.mkdir()
        for root, module in ((self.site, "first.py"), (other, "second.py")):
            (root / "shared").mkdir()
            (root / "shared/__init__.py").write_text("")
            (root / "shared" / module).write_text("value = 1\n")
        self.dist.files = ["shared/first.py"]
        second = types.SimpleNamespace(
            metadata={"Name": "second"},
            version="1",
            files=["shared/second.py"],
            locate_file=lambda name: other / name,
            read_text=lambda name: None,
        )
        with (
            patch.object(sys, "path", [str(self.site), str(other)]),
            patch.object(
                runtime.importlib.metadata,
                "distributions",
                return_value=[self.dist, second],
            ),
            self.assertRaisesRegex(RuntimeError, "Import precedence for shared"),
        ):
            runtime.capture(
                {"onnxruntime-gpu": "1.30.0", "second": "1"},
                [],
                self.core,
                lambda: None,
            )

    def test_recorded_source_backed_caches_are_not_in_the_immutable_inventory(self):
        source = self.site / "package.py"
        cache = Path(py_compile.compile(str(source), doraise=True))
        legacy = source.with_suffix(".pyc")
        py_compile.compile(str(source), cfile=str(legacy), doraise=True)
        self.dist.files.extend([str(cache.relative_to(self.site)), legacy.name])
        legacy.unlink()  # A cleaned generated cache is not a missing implementation.
        result = self.capture()
        with tarfile.open(result["_archive_path"]) as tar:
            self.assertIn("site/package.py", tar.getnames())
            self.assertFalse(any(name.endswith(".pyc") for name in tar.getnames()))

    def test_namespace_portions_merge_but_data_file_collisions_are_refused(self):
        other = self.root / "other-site"
        other.mkdir()
        for root, module in ((self.site, "first.py"), (other, "second.py")):
            (root / "namespace").mkdir()
            (root / "namespace" / module).write_text("value = 1\n")
        self.dist.files = ["namespace/first.py"]
        second = types.SimpleNamespace(
            metadata={"Name": "second"},
            version="1",
            files=["namespace/second.py"],
            locate_file=lambda name: other / name,
            read_text=lambda name: None,
        )
        with (
            patch.object(sys, "path", [str(self.site), str(other)]),
            patch.object(
                runtime.importlib.metadata,
                "distributions",
                return_value=[self.dist, second],
            ),
        ):
            result = runtime.capture(
                {"onnxruntime-gpu": "1.30.0", "second": "1"},
                [],
                self.core,
                lambda: None,
            )
            self.addCleanup(runtime.shutil.rmtree, Path(result["_archive_path"]).parent)
            with tarfile.open(result["_archive_path"]) as tar:
                self.assertTrue(
                    {"site/namespace/first.py", "site/namespace/second.py"}
                    <= set(tar.getnames())
                )
            for root, dist in ((self.site, self.dist), (other, second)):
                (root / "shared.data").write_text("data")
                dist.files.append("shared.data")
            with self.assertRaisesRegex(
                RuntimeError, "Installed files collide after relocation"
            ):
                runtime.capture(
                    {"onnxruntime-gpu": "1.30.0", "second": "1"},
                    [],
                    self.core,
                    lambda: None,
                )

    def test_editable_source_cannot_escape_project_via_symlink(self):
        project = self.root / "project"
        project.mkdir()
        alias = project / "outside.py"
        alias.symlink_to(self.site / "package.py")
        with self.assertRaisesRegex(RuntimeError, "source symlink escapes"):
            self.capture(editable_sources={"project": (project, [alias])})

    def test_relative_pth_cannot_escape_relocated_site(self):
        target = self.site.parent / "other-site"
        target.mkdir()
        (self.site / "paths.pth").write_text("../other-site\n")
        self.dist.files.append("paths.pth")
        with (
            patch.object(sys, "path", [str(self.site), str(target)]),
            self.assertRaisesRegex(
                RuntimeError, "relative Python path.*cannot be relocated"
            ),
        ):
            self.capture()

    def test_relocated_absolute_site_pth_is_refused(self):
        (self.site / "paths.pth").write_text(str(self.site) + "\n")
        self.dist.files.append("paths.pth")
        with self.assertRaisesRegex(
            RuntimeError, "absolute Python path.*cannot be relocated"
        ):
            self.capture()

    def test_standard_container_project_roots_are_allowed_but_system_roots_are_not(
        self,
    ):
        for root in ("/workspace", "/app", "/projects", str(self.root / "project")):
            runtime.validate_source_root(Path(root))
        for root in (
            "/",
            "/home",
            "/root",
            "/opt",
            "/tmp",
            "/var",
            "/etc/project",
            "/lib64/project",
            "/usr/project",
            "/proc/project",
            "/opt/venv/project",
            "/phantom/project",
            "/opt/phantom-tools/project",
            str(Path.home()),
            sys.prefix,
        ):
            with (
                self.subTest(root=root),
                self.assertRaisesRegex(RuntimeError, "must be a project directory"),
            ):
                runtime.validate_source_root(Path(root))


if __name__ == "__main__":
    unittest.main()
