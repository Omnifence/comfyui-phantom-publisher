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
    matches(
      js,
      /showProgress\(\s*job\.job_id,\s*config\.console_origin,\s*target\.slug,\s*idempotencyStorageKey,\s*\)/,
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

describe('Phantom publisher cancel', () => {
  it('offers a Cancel that ends the server-side job, not just the panel', () => {
    // Closing the overlay only stops the polling; the upload is a task of the
    // ComfyUI server and keeps running. Cancel is the control that ends it.
    contains(js, 'Cancel publish');
    matches(js, /request\(`\/jobs\/\$\{jobId\}`, \{ method: 'DELETE' \}\)/);
    contains(css, '.phantom-publisher-secondary');
  });

  it('shows the cancelled state, stops polling, and offers Close', () => {
    matches(
      js,
      /job\.status === 'cancelled'[\s\S]*?phase\.textContent = 'Publish cancelled'[\s\S]*?finish\(\);\s*return job;/,
    );
    contains(js, "cancelled: 'Cancelled'");
    contains(css, "[data-status='cancelled']");
    // Every terminal state removes Cancel, offers Close, and hands the finished
    // job back — the caller reads the variation Phantom assigned off it.
    matches(js, /job\.status === 'failed'[\s\S]*?finish\(\);\s*return job;/);
    matches(js, /job\.status === 'completed'[\s\S]*?finish\(\);\s*return job;/);
    matches(js, /const finish = \(\) => \{\s*cancel\.remove\(\);/);
    // A closed panel learns nothing; the publish carries on in the server.
    contains(js, 'return null;');
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
    contains(js, 'Publishing replaces the primary graph in a new version of the selected workflow');
    matches(
      js,
      /submit\.textContent = updating\s*\?\s*'Update variation'\s*:\s*publishingVariation\s*\?\s*'Publish new variation'\s*:\s*publishingNewVersion\s*\?\s*'Publish new version'/,
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

describe('Phantom publisher variation graphs', () => {
  it('offers the primary, each current variation by id, and a new variation', () => {
    contains(js, "new Option('New version — replace the primary graph', PUBLISH_AS_VERSION)");
    contains(js, "new Option('New variation of the current version', PUBLISH_AS_VARIATION)");
    // One option per variation the target's current version carries, keyed
    // by id so the server replaces THAT graph rather than matching a label.
    matches(
      js,
      /new Option\(\s*`Update variation: \$\{variation\.label\}`,\s*updateVariationValue\(variation\.variation_id\),?\s*\)/,
    );
    contains(js, "const UPDATE_VARIATION_PREFIX = 'variation:'");
    contains(js, 'publishAsField.hidden = !publishingNewVersion');
    contains(js, 'heading.textContent = updating\n      ? `Update variation "${updating.label}"`');
    contains(
      js,
      "joins the selected workflow's current version as a variation, not as a new version",
    );
    contains(
      js,
      'Its conditions in the console are kept; its bindings are re-read from this graph.',
    );
  });

  it('rebuilds the choices per workflow and prefills an update from the variation itself', () => {
    // Each workflow has its own variations, so the select is rebuilt when the
    // target changes, and an update starts from that variation's own text.
    matches(js, /select\.addEventListener\('change', \(\) => \{\s*rebuildPublishAs\(\);/);
    contains(js, 'publishAs.replaceChildren(...publishAsOptions(target))');
    contains(js, "variationLabel.value = variation ? variation.label : remembers?.label || ''");
    // The update resolves with the id beside the label, so the server can
    // tell a replacement from a new variation.
    contains(js, '...(updating ? { variation_id: updating.variation_id } : {})');
  });

  it('reopens on the remembered variation only while the target still has it', () => {
    contains(js, 'const matched = rememberedMatch(target);');
    matches(
      js,
      /publishAs\.value = matched\s*\? updateVariationValue\(matched\.variation_id\)\s*: rememberedFor\(target\)\s*\? PUBLISH_AS_VARIATION\s*: PUBLISH_AS_VERSION;/,
    );
  });

  it('matches the remembered variation by id, and falls back to its label', () => {
    // A graph carries no id when it was published by 0.6.0, or when the panel
    // was closed before the publish finished. Matching the label is what stops
    // the next publish adding a duplicate of a variation that already exists.
    matches(
      js,
      /remembers\.variation_id &&\s*variations\.find\(\(v\) => v\.variation_id === remembers\.variation_id\)/,
    );
    contains(js, 'variations.find((v) => v.label === remembers.label)');
  });

  it('never carries the remembered variation to a different workflow', () => {
    // Two workflows can label a variation the same way. Matching across them
    // would send the other workflow's variation_id and replace a graph the
    // author never chose, so pointing the graph elsewhere drops the block —
    // for the matching, for the default choice, and for the prefilled label.
    contains(
      js,
      'const rememberedFor = (target) =>\n' +
        '    target && target.workflow_id === remembered ? rememberedVariation : null;',
    );
    contains(js, 'const remembers = rememberedFor(target);');
    excludes(js, 'rememberedVariation?.label');
    excludes(js, 'rememberedVariation?.description');
  });

  it('reads the block 0.6.0 wrote and republishes it in the variation shape', () => {
    // 0.6.0 called it `alternative`. Without the fallback the dialog opens such
    // a graph on the primary, so the next publish replaces the primary graph.
    contains(js, 'phantom.variation || phantom.alternative || null');
    contains(js, 'chooseTarget(phantom.workflow_id, config, rememberedVariation)');
  });

  it('writes the variation id Phantom assigned back into the graph', () => {
    // A new variation only learns its id from the publish that created it.
    // Without writing it back, the next publish of the same file cannot match
    // the graph to that variation and adds a second one instead.
    contains(js, 'finished?.variation?.variation_id && graphExtra.phantom.variation');
    contains(
      js,
      'graphExtra.phantom.variation = { ...graphExtra.phantom.variation, ...finished.variation };',
    );
  });

  it('requires the label — it is the only moment the author knows when the graph applies', () => {
    contains(js, 'When should Phantom use this graph?');
    contains(js, 'variationLabelField.hidden = !publishingVariation');
    matches(js, /if \(!label\) \{[\s\S]*?the label is required for a variation/);
    contains(js, 'variationLabel.focus()');
    // The block travels beside the graph, and the server refuses it blank too.
    contains(js, '...(target.variation ? { variation: target.variation } : {})');
  });

  it('remembers the label in the graph so a republish opens on it, and never sends a dead mapping', () => {
    contains(js, "remembers?.label || ''");
    // `graphExtra.phantom` is overwritten just before the prompt is read, so a
    // mapping read off it was always undefined. Nothing reads it now.
    excludes(js, 'interface_mapping');
  });
});
