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
A published workflow remembers only `origin`, `workflow_id` and, for an
alternative graph, its `alternative` label under top-level `extra.phantom`.

## Alternative graphs

A workflow can carry more than one graph. Publish the main graph first. Then,
with the variant open in ComfyUI, select **Publish to Phantom**, pick the same
workflow, and set **Publish as** to *Alternative graph of the current version*.
The dialog asks when Phantom should use this graph; that label is required, and
it is what the operator sees when they set the conditions in the Phantom console
(for example, "use this graph when the caller sends an image"). The alternative
lands on a new version of the same workflow beside the main graph, never as a
new workflow.

## What the publisher sends

- The workflow graph, in API format and in UI format.
- Every custom node package the graph uses, with its git commit or package
  version, as a normalized archive.
- Every model the graph references, with its SHA-256 digest and source URL.
- The ComfyUI core version and the publisher version.

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
