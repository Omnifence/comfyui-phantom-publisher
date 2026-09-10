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

Python dependency locks also capture install sources: Git dependencies keep their exact
commit and archive installs keep their URL. Local checkouts are archived and uploaded
automatically (up to 200 MB each, excluding build outputs, caches and virtual
environments), then installed non-editably in the image. Index installs keep their
exact version pins. This reproduces code whose version number alone is ambiguous.

Before anything uploads, the publisher checks each package's compiled
extensions (`.so`, `.pyd`, `.dylib`). Phantom builds the workflow's image for
whichever Python those binaries need, so a package built for a newer Python
than the image's default is fine. Every graph of a workflow shares that one
image, so what stops a publish is a disagreement: a binary in this graph built
for Python 3.13 while a package in the workflow's primary graph, or in another
variation, is built for Python 3.12. The message names both packages, both
files and both versions. Publish both graphs from the same ComfyUI, install
matching builds of the packages, or split them into separate workflows in
Phantom. A binary built for macOS or Windows, or for a Python that PyTorch
ships no wheels for, is refused the same way.

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
