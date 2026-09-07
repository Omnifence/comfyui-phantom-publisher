import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { before, describe, it } from 'node:test';

// The publisher's web assets ship unbundled to a user's ComfyUI install, so
// there is nothing to import — the suites assert against the source text. One
// entry point for both, so a rename of the assets breaks in one place.
const JS = new URL('./web/phantom-publisher.js', import.meta.url);
const CSS = new URL('./web/phantom-publisher.css', import.meta.url);

let js = '';
let css = '';

before(async () => {
  [js, css] = await Promise.all([readFile(JS, 'utf8'), readFile(CSS, 'utf8')]);
});

const contains = (source, needle) =>
  assert.ok(source.includes(needle), `expected the source to contain ${JSON.stringify(needle)}`);

const matches = (source, pattern) =>
  assert.ok(pattern.test(source), `expected the source to match ${pattern}`);

const excludes = (source, needle) =>
  assert.ok(
    !source.includes(needle),
    `expected the source NOT to contain ${JSON.stringify(needle)}`,
  );

describe('Phantom publisher progress UI', () => {
  it('shows detailed overall and per-dependency upload progress', () => {
    contains(js, 'Workflow dependencies');
    contains(js, 'dependency.uploaded_bytes');
    contains(js, 'dependency.byte_size');
    contains(js, 'Already in Phantom');
    contains(js, 'dependencies processed');
    contains(js, 'Publish activity log');
    contains(js, 'updateLogRows(job.logs)');
    contains(
      js,
      'showProgress(job.job_id, config.console_origin, target.slug, idempotencyStorageKey)',
    );
    contains(js, 'Phantom console origin');
    matches(js, /job\.status === 'failed'[\s\S]*?logDetails\.open = true/);
    contains(css, '.phantom-publisher-dependency-progress');
    contains(css, "[data-status='uploading']");
    contains(css, '.phantom-publisher-log-entry');
    contains(css, "[data-level='error']");
    matches(css, /\.phantom-publisher-progress-dialog > \*\s*{\s*min-width: 0/);
    matches(css, /\.phantom-publisher-phase\s*{\s*overflow-wrap: anywhere/);
  });
});

describe('Phantom publisher connection switching', () => {
  it('names the destination and offers a switch before an immutable version lands', () => {
    contains(js, 'Publishing to ${config.origin');
    contains(js, 'Change connection');
    contains(js, 'phantom-publisher-connection');
    contains(js, 'resolve(RECONFIGURE)');
    contains(css, '.phantom-publisher-connection');
    contains(css, '.phantom-publisher-link');
  });

  it('re-enters the publish from the top after a switch, never mid-flight', () => {
    // The target list and the console origin both belong to the old Phantom.
    contains(js, 'if (await configure(config)) return publish();');
  });

  it('prefills the origins but never the token', () => {
    contains(js, "origin.value = current.origin || ''");
    contains(js, "consoleOrigin.value = current.console_origin || ''");
    // The server does not return the saved token, so there is nothing to
    // prefill and switching environments must mean pasting a new one.
    excludes(js, 'token.value = current');
    contains(js, 'Change Phantom connection');
  });

  it('treats a dismissed dialog as a cancel rather than hanging the caller', () => {
    contains(js, 'const dialog = (onDismiss)');
    contains(js, 'onDismiss?.()');
    contains(js, 'dialog(() => resolve(false))');
    contains(js, 'resolve(true)');
  });
});

describe('Phantom publisher target confirmation', () => {
  it('confirms a remembered Phantom workflow before publishing a new version', () => {
    assert.ok(
      !/if \(rememberedTarget\) return rememberedTarget/.test(js),
      'a remembered target must still be confirmed, not returned unchecked',
    );
    matches(js, /select\.value = rememberedTarget\?\.workflow_id \|\| ''/);
    contains(js, 'Workflow in Phantom');
    contains(js, 'Publishing will add a new version to the selected workflow');
    matches(
      js,
      /submit\.textContent = publishingAlternative\s*\?\s*'Publish alternative graph'\s*:\s*publishingNewVersion\s*\?\s*'Publish new version'/,
    );
    contains(js, 'nameField.hidden = publishingNewVersion');
    contains(js, 'slugField.hidden = publishingNewVersion');
    contains(js, 'providerField.hidden = publishingNewVersion');
    matches(css, /\.phantom-publisher-field\[hidden\]\s*{\s*display: none/);
  });

  it('keeps the new-target fields intact by hiding them, not clearing them', () => {
    // No draft save/restore: an <input> inside a `hidden` container keeps its
    // value, so switching back to "new workflow" finds what was typed.
    excludes(js, 'newTargetDraft');
    excludes(js, 'showingNewTarget');
  });
});

describe('Phantom publisher alternative graphs', () => {
  it('offers to publish into an existing workflow as an alternative of its current version', () => {
    contains(js, "new Option('New version of the primary graph', PUBLISH_AS_VERSION)");
    contains(js, "new Option('Alternative graph of the current version', PUBLISH_AS_ALTERNATIVE)");
    contains(js, 'publishAsField.hidden = !publishingNewVersion');
    matches(js, /submit\.textContent = publishingAlternative\s*\?\s*'Publish alternative graph'/);
    contains(
      js,
      "joins the selected workflow's current version as an alternative, not as a new version",
    );
  });

  it('requires the label — it is the only moment the author knows when the graph applies', () => {
    contains(js, 'When should Phantom use this graph?');
    contains(js, 'alternativeLabelField.hidden = !publishingAlternative');
    matches(js, /if \(!label\) \{[\s\S]*?the label is required for an alternative/);
    contains(js, 'alternativeLabel.focus()');
    // The block travels beside the graph, and the server refuses it blank too.
    contains(js, '...(target.alternative ? { alternative: target.alternative } : {})');
  });

  it('remembers the label in the graph so a republish opens on it, and never sends a dead mapping', () => {
    contains(js, 'chooseTarget(phantom.workflow_id, config, phantom.alternative || null)');
    contains(js, "alternativeLabel.value = rememberedAlternative?.label || ''");
    // `graphExtra.phantom` is overwritten just before the prompt is read, so a
    // mapping read off it was always undefined. Nothing reads it now.
    excludes(js, 'interface_mapping');
  });
});
