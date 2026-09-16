const canonicalize = (value) => {
  if (Array.isArray(value)) return value.map(canonicalize);
  if (!value || typeof value !== 'object') return value;
  return Object.fromEntries(
    Object.keys(value)
      .sort()
      .filter((key) => value[key] !== undefined)
      .map((key) => [key, canonicalize(value[key])]),
  );
};

// The graph as ComfyUI serializes it carries the canvas viewport in
// `extra.ds` (pan offset and zoom). It is not part of what gets published,
// and a scroll on the canvas between two presses of Publish must not turn a
// retry into a second staged version beside the first.
export const publishedContent = (payload) => {
  const { extra, ...uiWorkflow } = payload?.ui_workflow ?? {};
  if (!extra || typeof extra !== 'object') return payload;
  const { ds: _viewport, ...rest } = extra;
  return {
    ...payload,
    ui_workflow: { ...uiWorkflow, ...(Object.keys(rest).length ? { extra: rest } : {}) },
  };
};

export const fingerprintPublishPayload = async (payload) => {
  const encoded = new TextEncoder().encode(JSON.stringify(canonicalize(publishedContent(payload))));
  const digest = await crypto.subtle.digest('SHA-256', encoded);
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join('');
};

export const selectPendingIdempotencyKey = (
  stored,
  manifestFingerprint,
  createKey = () => crypto.randomUUID(),
) => {
  try {
    const pending = JSON.parse(stored ?? 'null');
    if (
      pending &&
      typeof pending.idempotencyKey === 'string' &&
      pending.manifestFingerprint === manifestFingerprint
    ) {
      return pending.idempotencyKey;
    }
  } catch {
    // Legacy values stored only the raw key and cannot prove content identity.
  }
  return createKey();
};
