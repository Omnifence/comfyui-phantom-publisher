import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import {
  fingerprintPublishPayload,
  selectPendingIdempotencyKey,
} from './web/publish-idempotency.js';

describe('Phantom publisher idempotency', () => {
  it('reuses a pending key only for identical manifest content', async () => {
    const first = await fingerprintPublishPayload({ workflow: { 2: 'b', 1: 'a' } });
    const same = await fingerprintPublishPayload({ workflow: { 1: 'a', 2: 'b' } });
    const changed = await fingerprintPublishPayload({ workflow: { 1: 'edited', 2: 'b' } });
    const stored = JSON.stringify({ idempotencyKey: 'pending-key', manifestFingerprint: first });

    assert.equal(
      selectPendingIdempotencyKey(stored, same, () => 'new-key'),
      'pending-key',
    );
    assert.equal(
      selectPendingIdempotencyKey(stored, changed, () => 'new-key'),
      'new-key',
    );
  });

  it('ignores the canvas viewport, so a scroll between two presses is the same publish', async () => {
    const graph = { nodes: [{ id: 1 }], extra: { ds: { scale: 1, offset: [0, 0] } } };
    const scrolled = { nodes: [{ id: 1 }], extra: { ds: { scale: 0.8, offset: [120, -40] } } };
    const edited = { nodes: [{ id: 2 }], extra: { ds: { scale: 1, offset: [0, 0] } } };
    const base = await fingerprintPublishPayload({ workflow_id: 'w', ui_workflow: graph });
    assert.equal(
      await fingerprintPublishPayload({ workflow_id: 'w', ui_workflow: scrolled }),
      base,
    );
    assert.notEqual(
      await fingerprintPublishPayload({ workflow_id: 'w', ui_workflow: edited }),
      base,
    );
    // Other `extra` keys still count: the publisher's own target block lives there.
    assert.notEqual(
      await fingerprintPublishPayload({
        workflow_id: 'w',
        ui_workflow: { ...graph, extra: { ...graph.extra, phantom: { workflow_id: 'other' } } },
      }),
      base,
    );
  });

  it('rotates legacy raw-key storage because its content identity is unknown', () => {
    assert.equal(
      selectPendingIdempotencyKey('legacy-key', 'fingerprint', () => 'new-key'),
      'new-key',
    );
  });
});
