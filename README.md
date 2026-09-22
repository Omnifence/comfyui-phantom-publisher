# Phantom Publisher for ComfyUI

Publish the exact workflow you are running — graph, custom nodes, models and
their versions — from ComfyUI straight into [Phantom Router](https://phantomrouter.ai).
Phantom builds the image, pins every dependency and serves the workflow as a
versioned HTTP endpoint.

## Install

### ComfyUI Manager (recommended)

1. Open **Manager → Custom Nodes Manager**.
2. Search for **Phantom Publisher**.
3. Select **Install**, then restart ComfyUI.

### Manual

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/EvershieldAI/comfyui-phantom-publisher.git
```

Restart ComfyUI.

## Connect

1. Sign in to the Phantom console at https://app.phantomrouter.ai.
2. Open **ComfyUI → Publisher Connections** and create a connection token.
3. In ComfyUI, select **Publish to Phantom** in the toolbar and paste the token.

The token is stored in ComfyUI's user directory as `.phantom-publisher.json`
with mode `0600`. The publisher never writes the token into workflow JSON.
A published workflow remembers only `origin`, `workflow_id` and, for a
variation, its `variation` block (id and label) under top-level `extra.phantom`.

## Variations

A workflow version can carry more than one graph: the primary graph and any
number of variations. Phantom runs a variation instead of the primary when the
conditions set for it in the Phantom console hold (for example, "use this
graph when the caller sends an image"). Every publish lands as a new version
of the workflow; the previous version stays as it was.

Publish the primary graph first. Then, with a graph open in ComfyUI, select
**Publish to Phantom**, pick the workflow, and choose under **Publish as**:

- **New version — replace the primary graph.** The current variations carry
  forward unchanged.
- **Update variation: _label_.** One entry per variation the current version
  has. The graph replaces that variation; its conditions are kept and its
  bindings are read again from the new graph. The label and description can be
  changed here.
- **New variation of the current version.** The dialog asks when Phantom
  should use this graph. That label is required: publishing is the only moment
  the author is sure to know it, and it is what the operator sees when they
  set the conditions in the console.

Dependencies are matched by digest, so models and custom nodes shared between
graphs are uploaded once and one image serves every graph of the version.

## Cancelling a publish

Closing the progress panel does not stop the upload: the publish runs inside
the ComfyUI server, and the panel only watches it. Press **Cancel publish** in
the panel to end it. The publisher stops the transfer and asks Phantom to
abandon the parts it had already uploaded, so nothing half-uploaded is left
behind. The staged version stays in Phantom; a later publish of the same
workflow fills it in.

## What the publisher sends

- The workflow graph, in API format and in UI format.
- Every custom node package the graph uses, with its git commit or package
  version, as a normalized archive.
- Every model the graph references, with its SHA-256 digest and source URL.
- The ComfyUI core version, the publisher version, and the Python this
  ComfyUI runs.
- For each package, the paths of its compiled Python extensions (`.so`,
  `.pyd`, `.dylib`).

Publisher 0.14 also uploads a content-addressed runtime archive: installed Python
distribution files, tracked ComfyUI core files (including local modifications),
resolved native-library dependencies, libraries loaded in the ComfyUI process,
and native FFmpeg/ffprobe executables when present. Modern editable installs retain
their source files at their original paths (up to 200 MB per checkout). The image
restores these bytes instead of resolving wheels or rebuilding Python packages.
Every archived file is hashed; node imports and handler setup must not change the
captured package files. Console-script interpreter paths are explicitly relocated.

This capture currently supports standard CPython on Linux x86_64 with Ubuntu
22.04 or 24.04. It records the exact Python patch version and preserves the source
OS release. Drivers, libc/the loader, models, caches and environment variables are
not copied as a machine backup. Model weights use Phantom's existing model-volume
pipeline. Keep secrets and model data outside source checkouts.

A distribution another distribution has buried is left out of the lock. Two
wheels can unpack into one package directory (`onnxruntime` and
`onnxruntime-gpu`, `opencv-python` and `opencv-python-headless`); the one pip
installed last owns the files on disk, and only that one runs in this ComfyUI.
Only files inside site-packages count: two unrelated distributions that ship a
console script of the same name both stay. The lock names the buried ones under
`shadowed` so the review page can say why they are absent.

Before upload, ComfyUI's own prompt validator checks the graph without queuing it.
Compiled extensions are checked against **this graph's** authoring interpreter;
Primary and variation graphs have separate images and endpoints. Missing CUDA
libraries, conflicting cuDNN binaries, missing installed files and unsupported
source layouts produce an actionable publish error, not a speculative package
upgrade or downgrade.

Release the updated Phantom API and build worker **before** installing Publisher
0.14. A capability check prevents a new publisher sending runtime bytes to an old
server that would ignore them. Existing captures remain buildable via the older
lock-replay path; republish to obtain runtime archives. A failed v16 capture cannot
recover system-library bytes it never recorded simply by retrying its old build.

The endpoint uses the pushed image digest, and its CUDA scheduling floor includes
the captured non-Torch runtime libraries. Build-time checks still validate CUDA
linkage and node registration. Every graph must pass the existing RunPod execution
validation before **Make live**. Static capture cannot prove arbitrary subprocess,
network, JIT or input-dependent behavior: execute representative cases on the final
GPU image. A source that only appeared to work through CPU fallback is not proof
that its GPU provider works.

Uploads are content addressed. Phantom asks for a dependency by digest and the
publisher uploads it only when Phantom does not hold it already, so a second
workflow that shares a checkpoint does not re-upload that checkpoint.

## Development

The tests use only the standard library and stub the ComfyUI runtime modules,
so no ComfyUI installation is necessary:

```bash
python3 -m unittest test_publisher -v
```

`PUBLISHER_VERSION` in `publisher.py` and `version` in `pyproject.toml` must
stay equal. CI fails the build when they drift.

## Release

A push to `main` that changes `version` in `pyproject.toml` publishes a new
version to the [ComfyUI Registry](https://registry.comfy.org), which is the
source ComfyUI Manager installs from.

## License

MIT — see [LICENSE](LICENSE).
