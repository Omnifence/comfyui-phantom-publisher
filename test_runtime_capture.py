from __future__ import annotations

import json
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


if __name__ == "__main__":
    unittest.main()
