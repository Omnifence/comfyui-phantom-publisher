# CLAUDE.md

This file gives guidance to Claude Code (claude.ai/code) when it works in this
repository.

## What This Is

The **Phantom Publisher** custom node for ComfyUI. It publishes the exact
workflow a user is running — graph, custom node packages, models — into
[Phantom Router](https://phantomrouter.ai), which builds the image and serves
the workflow as a versioned HTTP endpoint.

The repo exists as a standalone public repo for one reason: **ComfyUI Manager
installs from the ComfyUI Registry, and the registry requires a public repo with
the node at its root.** It was split out of the private `phantom-router`
monorepo, where the server side (`apps/companion-api`) still lives.

## Releasing — read this before you touch a version

A release is a `version` bump in `pyproject.toml` landing on `main`. That push
triggers `.github/workflows/publish.yml`, which publishes to the registry.

**`version` in `pyproject.toml` and `PUBLISHER_VERSION` in `publisher.py` must
change in the same commit, to the same value.**

- The registry reads `pyproject.toml`. That is the version users install.
- Every published manifest records `PUBLISHER_VERSION`. That is the version
  Phantom stores against the workflow, and the version support reads when a
  publish fails.
- A drift between them mislabels every workflow published in between, and the
  stored value cannot be corrected after the fact. The `version-match` job in
  `.github/workflows/ci.yml` fails the build to prevent it.

Use semantic versioning. **Registry versions are immutable** — republishing a
version that already shipped fails, so a bad release is fixed by another bump,
never by a re-run.

Two more registry values are immutable after the first publish. Never edit them:

- `name = "comfyui-phantom-publisher"` — it is the node's registry identity.
- `PublisherId = "phantom"` — it is in the node's public URL.

`DisplayName`, `description` and `Icon` can change freely. `Icon` must stay an
SVG, PNG, JPG or GIF at most 800x400 px; the registry rejects anything else,
`.ico` included.

## Layout

| Path                         | Role                                                                                                   |
| ---------------------------- | ------------------------------------------------------------------------------------------------------ |
| `__init__.py`                | ComfyUI entry point. Sets `WEB_DIRECTORY`, calls `register_routes()`.                                  |
| `publisher.py`               | The whole backend: discovery, manifest, upload, job state.                                             |
| `web/phantom-publisher.js`   | Toolbar action and publish dialog.                                                                     |
| `web/phantom-publisher.css`  | Dialog styles.                                                                                         |
| `web/publish-idempotency.js` | Idempotency key handling for the publish call.                                                         |
| `test_publisher.py`          | Python tests for `publisher.py`. Standard library `unittest` only.                                     |
| `*.test.js`                  | Tests for the web assets. `node --test`, no dependencies.                                              |
| `package.json`               | Nothing but `"type": "module"`, so the test runner reads the assets as ES modules. Not a Node package. |

The node registers no ComfyUI nodes — `NODE_CLASS_MAPPINGS` is empty on purpose.
It is a toolbar action, not a graph node.

## Testing

```bash
python3 -m unittest test_publisher -v   # publisher.py
node --test                             # web/
```

Both suites run with **zero third-party dependencies, and it must stay that
way.**

- The Python tests stub `folder_paths`, `server` and `aiohttp` before importing
  `publisher.py`, so no ComfyUI installation is needed and none should be added.
  CI runs them on Python 3.10 through 3.13.
- The web assets ship unbundled to a user's ComfyUI install, so there is no
  build step to test through. `publish-idempotency.test.js` imports the module
  directly. `phantom-publisher-ui.test.js` asserts against the source text of
  `phantom-publisher.js` and `phantom-publisher.css`, because a DOM-building
  script has nothing else to assert on without a browser.

Every code change ships with tests.

## Conventions

- The web assets are formatted by Prettier (`.prettierrc`, single quotes,
  100 columns). Run `npx prettier --write web/` rather than hand-formatting —
  without the config file Prettier rewrites every quote in the repo.
- Python 3.10+, `from __future__ import annotations`, modern type syntax
  (`str | None`, not `Optional[str]`).
- The publisher runs inside someone else's ComfyUI. Import from `folder_paths`
  and `server` defensively — probe with `getattr` and fall back, as
  `_config_path()` does, because those APIs change between ComfyUI versions.
- The connection token is a secret. It is written to
  `<user directory>/.phantom-publisher.json` at mode `0600`, created restricted
  rather than chmod-ed after the fact, and it must never reach workflow JSON, a
  log line or an error message.
- A published workflow carries only `origin`, `workflow_id` and, for a
  variation graph, its `variation` block — the label, the optional description,
  and the `variation_id` Phantom assigned — under top-level `extra.phantom`.
  The id is what lets the next publish update that graph instead of adding a
  second variation beside it, so the panel writes it back when a publish
  completes. A graph saved by 0.6.0 carries the same block under the old
  `alternative` key; it is read as a fallback and rewritten in the new shape.

## The Phantom API contract

`publisher.py` calls `{origin}/api/v1/phantom/comfyui-publisher/*` with a bearer
token. The server side is in the private `phantom-router` repo under
`apps/companion-api`. **That contract is published to installed publishers you
cannot update in lockstep.** An older installed publisher keeps calling the old
shape, so change the manifest or the endpoints additively, and bump
`schema_version` in the manifest when the shape genuinely breaks.

Local ComfyUI development points `origin` at `http://localhost:3060`, and the
console origin then resolves to `http://localhost:3062`. Production is
`https://api.phantomrouter.ai` with the console at `https://app.phantomrouter.ai`.

## Packaging

`comfy node publish` packages every git-tracked file, minus `.comfyignore`
matches. When you add a directory that users do not need at runtime — fixtures,
docs, design assets — add it to `.comfyignore` in the same commit.
