import { app } from '../../scripts/app.js';
import { api } from '../../scripts/api.js';
import { fingerprintPublishPayload, selectPendingIdempotencyKey } from './publish-idempotency.js';

const request = async (path, options = {}) => {
  const response = await api.fetchApi(`/phantom-publisher${path}`, {
    ...options,
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
  });
  const text = await response.text();
  const body = text ? JSON.parse(text) : {};
  if (!response.ok)
    throw new Error(body.message || body.error || text || `HTTP ${response.status}`);
  return body;
};

const field = (label, input) => {
  const wrapper = document.createElement('label');
  wrapper.className = 'phantom-publisher-field';
  const title = document.createElement('span');
  title.textContent = label;
  wrapper.append(title, input);
  return wrapper;
};

const input = (placeholder, type = 'text') => {
  const element = document.createElement('input');
  element.type = type;
  element.placeholder = placeholder;
  return element;
};

const dialog = (onDismiss) => {
  const overlay = document.createElement('div');
  overlay.className = 'phantom-publisher-overlay';
  const panel = document.createElement('div');
  panel.className = 'phantom-publisher-dialog';
  overlay.append(panel);
  document.body.append(overlay);
  const close = () => overlay.remove();
  // Only the click-away path is a dismissal. `close()` is also how a dialog
  // exits after it succeeded, and that must not report itself as cancelled.
  overlay.addEventListener('click', (event) => {
    if (event.target !== overlay) return;
    close();
    onDismiss?.();
  });
  return { panel, close };
};

// `current` is the config the server reports, so reconnecting to a different
// Phantom starts from the origins in use rather than from empty fields. The
// token is deliberately absent: the server never returns it, so switching
// environments always means pasting the token for the environment you are
// switching to.
const configure = (current = {}) =>
  new Promise((resolve) => {
    const reconfiguring = Boolean(current.configured);
    const modal = dialog(() => resolve(false));
    modal.panel.innerHTML = `<h2></h2><p></p>`;
    modal.panel.querySelector('h2').textContent = reconfiguring
      ? 'Change Phantom connection'
      : 'Connect Phantom';
    modal.panel.querySelector('p').textContent = reconfiguring
      ? "Publishing goes to the Phantom you connect here. Switching environments needs that environment's own publisher token — the saved one is never shown again."
      : "The publisher token stays in this ComfyUI server's protected user configuration.";
    const origin = input('https://api.phantomrouter.ai');
    const consoleOrigin = input('https://app.phantomrouter.ai');
    const token = input('php_…', 'password');
    origin.value = current.origin || '';
    consoleOrigin.value = current.console_origin || '';
    const save = document.createElement('button');
    save.textContent = reconfiguring ? 'Save new connection' : 'Save connection';
    save.className = 'phantom-publisher-primary';
    const status = document.createElement('p');
    status.className = 'phantom-publisher-status';
    save.onclick = async () => {
      try {
        await request('/config', {
          method: 'PUT',
          body: JSON.stringify({
            origin: origin.value,
            console_origin: consoleOrigin.value,
            token: token.value,
          }),
        });
        modal.close();
        resolve(true);
      } catch (error) {
        status.textContent = error.message;
      }
    };
    modal.panel.append(
      field('Phantom API origin', origin),
      field('Phantom console origin', consoleOrigin),
      field('Publisher token', token),
      save,
      status,
    );
  });

// Resolved instead of a target when the user asks to reconnect, so the caller
// restarts the publish against whichever Phantom they end up connected to.
const RECONFIGURE = Symbol('reconfigure');

// How a graph joins an existing workflow: as its next primary version, as a
// NEW variation graph of the current version that Phantom runs instead of the
// primary when the caller's inputs meet the conditions set in the console, or
// as the replacement of a variation the current version already has.
const PUBLISH_AS_VERSION = 'version';
const PUBLISH_AS_VARIATION = 'variation';
const UPDATE_VARIATION_PREFIX = 'variation:';
const updateVariationValue = (variationId) => `${UPDATE_VARIATION_PREFIX}${variationId}`;
const updatedVariationId = (value) =>
  value.startsWith(UPDATE_VARIATION_PREFIX) ? value.slice(UPDATE_VARIATION_PREFIX.length) : null;

// The "Publish as" choices for one target: the primary, each of its current
// variations by id, and a new variation. Names are stored text, so options are
// built as elements — see the workflow select below.
const publishAsOptions = (target) => [
  new Option('New version — replace the primary graph', PUBLISH_AS_VERSION),
  ...(target?.variations || []).map(
    (variation) =>
      new Option(
        `Update variation: ${variation.label}`,
        updateVariationValue(variation.variation_id),
      ),
  ),
  new Option('New variation of the current version', PUBLISH_AS_VARIATION),
];

const chooseTarget = async (remembered, config = {}, rememberedVariation = null) => {
  const data = await request('/targets');
  const rememberedTarget = data.targets.find((target) => target.workflow_id === remembered);
  const modal = dialog();
  modal.panel.innerHTML = `<h2>Publish workflow</h2><p class="phantom-publisher-target-help">Select an existing target or create a new immutable workflow history.</p>`;
  const heading = modal.panel.querySelector('h2');
  const help = modal.panel.querySelector('.phantom-publisher-target-help');
  const select = document.createElement('select');
  // Built as elements, not markup: a workflow name is stored text, and
  // interpolating it into innerHTML would run whatever it contains inside the
  // ComfyUI page, which can reach the ComfyUI API. `new Option(text, value)`
  // assigns both as properties, so neither is ever parsed as HTML.
  select.replaceChildren(new Option('Create a new workflow…', ''));
  for (const target of data.targets) {
    select.append(new Option(`${target.name} · ${target.slug}`, target.workflow_id));
  }
  select.value = rememberedTarget?.workflow_id || '';
  const name = input('Portrait generator');
  const slug = input('portrait-generator');
  const provider = document.createElement('select');
  provider.innerHTML = `<option value="runpod">RunPod</option><option value="vast-ai">Vast AI</option>`;
  const publishAs = document.createElement('select');
  // The label is the one thing only the author knows, and only now: it tells
  // whoever configures the conditions in Phantom WHEN this graph should run.
  const variationLabel = input('e.g. Caller sends a reference image');
  variationLabel.maxLength = 120;
  const variationDescription = input('Optional — what this graph does differently');
  const selectedVariation = () => {
    const target = data.targets.find((candidate) => candidate.workflow_id === select.value);
    const id = updatedVariationId(publishAs.value);
    return id ? (target?.variations || []).find((v) => v.variation_id === id) || null : null;
  };
  // The remembered block belongs to ONE workflow. Pointing the graph at a
  // different one drops it: two workflows can label a variation the same way,
  // and matching across them would send the other workflow's variation_id and
  // replace a graph the author never chose.
  const rememberedFor = (target) =>
    target && target.workflow_id === remembered ? rememberedVariation : null;
  // The variation that block names, by id and then by label. The label fallback
  // is what a graph published before the id came back carries — a 0.6.0 graph,
  // or one whose panel was closed before the publish finished — and matching it
  // is what stops the next publish adding a duplicate.
  const rememberedMatch = (target) => {
    const remembers = rememberedFor(target);
    if (!remembers) return null;
    const variations = target?.variations || [];
    return (
      (remembers.variation_id &&
        variations.find((v) => v.variation_id === remembers.variation_id)) ||
      (remembers.label && variations.find((v) => v.label === remembers.label)) ||
      null
    );
  };
  // Rebuilt per target: each workflow has its own variations. The remembered
  // choice is kept when the target still offers it — an update of a variation
  // that has since been removed falls back to publishing a new one under the
  // remembered label.
  const rebuildPublishAs = () => {
    const target = data.targets.find((candidate) => candidate.workflow_id === select.value);
    publishAs.replaceChildren(...publishAsOptions(target));
    const matched = rememberedMatch(target);
    publishAs.value = matched
      ? updateVariationValue(matched.variation_id)
      : rememberedFor(target)
        ? PUBLISH_AS_VARIATION
        : PUBLISH_AS_VERSION;
    prefillVariationFields();
  };
  // An update starts from the variation's own label and description; a new
  // variation from whatever was remembered from the last publish.
  const prefillVariationFields = () => {
    const target = data.targets.find((candidate) => candidate.workflow_id === select.value);
    const remembers = rememberedFor(target);
    const variation = selectedVariation();
    variationLabel.value = variation ? variation.label : remembers?.label || '';
    variationDescription.value = variation
      ? variation.description || ''
      : remembers?.description || '';
  };
  const nameField = field('Name', name);
  const slugField = field('Slug', slug);
  const providerField = field('Provider', provider);
  const publishAsField = field('Publish as', publishAs);
  const variationLabelField = field('When should Phantom use this graph?', variationLabel);
  const variationDescriptionField = field('Description', variationDescription);
  const submit = document.createElement('button');
  submit.className = 'phantom-publisher-primary';
  const status = document.createElement('p');
  status.className = 'phantom-publisher-status';
  const updateTargetConfirmation = () => {
    const selectedTarget = data.targets.find((target) => target.workflow_id === select.value);
    const publishingNewVersion = Boolean(selectedTarget);
    const updating = publishingNewVersion ? selectedVariation() : null;
    const publishingVariation =
      publishingNewVersion && (Boolean(updating) || publishAs.value === PUBLISH_AS_VARIATION);
    // The new-target fields are only HIDDEN below, never cleared, so switching
    // back to "new workflow" finds whatever the user typed still in them.
    heading.textContent = updating
      ? `Update variation "${updating.label}"`
      : publishingVariation
        ? 'Publish new variation'
        : publishingNewVersion
          ? 'Publish new workflow version'
          : 'Publish workflow';
    help.textContent = updating
      ? `This graph replaces the "${updating.label}" variation on the selected workflow's current version. Its conditions in the console are kept; its bindings are re-read from this graph. Every version keeps every variation, so this lands as a new version too.`
      : publishingVariation
        ? "This graph joins the selected workflow's current version as a variation, not as a new version of the workflow. Phantom runs it instead of the primary graph when the conditions set in the console hold — the label below says when."
        : publishingNewVersion
          ? 'Confirm the destination in Phantom. Publishing replaces the primary graph in a new version of the selected workflow; its variations carry forward, and existing versions remain unchanged.'
          : 'Create a new workflow in Phantom and publish its first version.';
    submit.textContent = updating
      ? 'Update variation'
      : publishingVariation
        ? 'Publish new variation'
        : publishingNewVersion
          ? 'Publish new version'
          : 'Create and publish';
    nameField.hidden = publishingNewVersion;
    slugField.hidden = publishingNewVersion;
    providerField.hidden = publishingNewVersion;
    publishAsField.hidden = !publishingNewVersion;
    variationLabelField.hidden = !publishingVariation;
    variationDescriptionField.hidden = !publishingVariation;
  };
  select.addEventListener('change', () => {
    rebuildPublishAs();
    updateTargetConfirmation();
  });
  publishAs.addEventListener('change', () => {
    prefillVariationFields();
    updateTargetConfirmation();
  });
  rebuildPublishAs();
  updateTargetConfirmation();

  return new Promise((resolve, reject) => {
    submit.onclick = async () => {
      try {
        if (select.value) {
          const selected = data.targets.find((target) => target.workflow_id === select.value);
          const updating = selectedVariation();
          if (updating || publishAs.value === PUBLISH_AS_VARIATION) {
            const label = variationLabel.value.trim();
            if (!label) {
              status.textContent =
                'Say when Phantom should use this graph — the label is required for a variation.';
              variationLabel.focus();
              return;
            }
            modal.close();
            resolve({
              ...selected,
              variation: {
                ...(updating ? { variation_id: updating.variation_id } : {}),
                label,
                ...(variationDescription.value.trim()
                  ? { description: variationDescription.value.trim() }
                  : {}),
              },
            });
            return;
          }
          modal.close();
          resolve(selected);
          return;
        }
        const created = await request('/targets', {
          method: 'POST',
          body: JSON.stringify({
            name: name.value,
            slug: slug.value,
            provider: provider.value,
          }),
        });
        modal.close();
        resolve(created);
      } catch (error) {
        status.textContent = error.message;
      }
    };
    // Naming the destination here is the point: the dialog is the last step
    // before an immutable version lands, and dev and prod look identical once
    // the token is saved.
    const connection = document.createElement('div');
    connection.className = 'phantom-publisher-connection';
    const connectedTo = document.createElement('span');
    connectedTo.textContent = `Publishing to ${config.origin || 'Phantom'}`;
    const change = document.createElement('button');
    change.type = 'button';
    change.className = 'phantom-publisher-link';
    change.textContent = 'Change connection';
    change.onclick = () => {
      modal.close();
      resolve(RECONFIGURE);
    };
    connection.append(connectedTo, change);
    modal.panel.append(
      field('Workflow in Phantom', select),
      nameField,
      slugField,
      providerField,
      publishAsField,
      variationLabelField,
      variationDescriptionField,
      submit,
      status,
      connection,
    );
    modal.panel.addEventListener('cancel', () => reject(new Error('Publish cancelled')));
  });
};

const formatBytes = (value) => {
  const bytes = Number(value) || 0;
  if (bytes === 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  const unit = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const amount = bytes / 1024 ** unit;
  return `${amount >= 10 || unit === 0 ? amount.toFixed(0) : amount.toFixed(1)} ${units[unit]}`;
};

const dependencyStatus = {
  pending: 'Waiting',
  uploading: 'Uploading',
  uploaded: 'Uploaded',
  reused: 'Already in Phantom',
  not_required: 'No upload needed',
  failed: 'Failed',
  cancelled: 'Cancelled',
};

const dependencyKind = {
  model: 'Model',
  external_model: 'External model',
  custom_node: 'Custom node',
};

const showProgress = async (jobId, origin, workflowSlug, idempotencyStorageKey) => {
  const modal = dialog();
  modal.panel.classList.add('phantom-publisher-progress-dialog');
  modal.panel.innerHTML = `
    <div class="phantom-publisher-progress-heading">
      <div>
        <h2>Publishing to Phantom</h2>
        <p class="phantom-publisher-phase" aria-live="polite">Preparing manifest…</p>
        <button type="button" class="phantom-publisher-secondary phantom-publisher-cancel">Cancel publish</button>
      </div>
      <strong class="phantom-publisher-progress-value">0%</strong>
    </div>
    <progress class="phantom-publisher-overall-progress" max="100" value="0" aria-label="Overall publish progress"></progress>
    <p class="phantom-publisher-transfer-summary">Inspecting the workflow before upload…</p>
    <section class="phantom-publisher-dependencies" aria-label="Workflow dependencies">
      <div class="phantom-publisher-dependency-list"></div>
      <p class="phantom-publisher-dependency-empty">Dependencies will appear here as they are discovered.</p>
    </section>
    <details class="phantom-publisher-log">
      <summary><span>Publish log</span><span class="phantom-publisher-log-count">0 entries</span></summary>
      <div class="phantom-publisher-log-list" role="log" aria-live="polite" aria-label="Publish activity log"></div>
    </details>`;
  const phase = modal.panel.querySelector('.phantom-publisher-phase');
  const progress = modal.panel.querySelector('.phantom-publisher-overall-progress');
  const progressValue = modal.panel.querySelector('.phantom-publisher-progress-value');
  const summary = modal.panel.querySelector('.phantom-publisher-transfer-summary');
  const list = modal.panel.querySelector('.phantom-publisher-dependency-list');
  const empty = modal.panel.querySelector('.phantom-publisher-dependency-empty');
  const logDetails = modal.panel.querySelector('.phantom-publisher-log');
  const logCount = modal.panel.querySelector('.phantom-publisher-log-count');
  const logList = modal.panel.querySelector('.phantom-publisher-log-list');
  const cancel = modal.panel.querySelector('.phantom-publisher-cancel');
  const dependencyRows = new Map();
  const logRows = new Map();
  let currentDependencyId = null;

  // Closing the panel stops nothing — the publish is a task of the ComfyUI
  // server, and the tab only watches it. Cancel is the one control that ends
  // the transfer, so it stays visible until the job has finished either way.
  cancel.onclick = async () => {
    cancel.disabled = true;
    cancel.textContent = 'Cancelling…';
    try {
      await request(`/jobs/${jobId}`, { method: 'DELETE' });
    } catch (error) {
      cancel.disabled = false;
      cancel.textContent = 'Cancel publish';
      phase.textContent = `Could not cancel: ${error.message}`;
    }
  };

  const finish = () => {
    cancel.remove();
    const close = document.createElement('button');
    close.type = 'button';
    close.className = 'phantom-publisher-secondary';
    close.textContent = 'Close';
    close.onclick = () => modal.close();
    modal.panel.append(close);
  };

  const updateDependencyRows = (dependencies = []) => {
    empty.hidden = dependencies.length > 0;
    const visibleIds = new Set(dependencies.map((dependency) => dependency.id));
    for (const [id, refs] of dependencyRows) {
      if (!visibleIds.has(id)) {
        refs.row.remove();
        dependencyRows.delete(id);
      }
    }

    for (const dependency of dependencies) {
      let refs = dependencyRows.get(dependency.id);
      if (!refs) {
        const row = document.createElement('article');
        row.className = 'phantom-publisher-dependency';
        const header = document.createElement('div');
        header.className = 'phantom-publisher-dependency-header';
        const kind = document.createElement('span');
        kind.className = 'phantom-publisher-dependency-kind';
        const name = document.createElement('strong');
        name.className = 'phantom-publisher-dependency-name';
        const state = document.createElement('span');
        state.className = 'phantom-publisher-dependency-state';
        const detail = document.createElement('p');
        detail.className = 'phantom-publisher-dependency-detail';
        const itemProgress = document.createElement('progress');
        itemProgress.className = 'phantom-publisher-dependency-progress';
        itemProgress.max = 100;
        header.append(kind, name, state);
        row.append(header, detail, itemProgress);
        list.append(row);
        refs = { row, kind, name, state, detail, progress: itemProgress };
        dependencyRows.set(dependency.id, refs);
      }

      const state = dependencyStatus[dependency.status] || dependency.status;
      refs.row.dataset.status = dependency.status;
      refs.kind.textContent = dependencyKind[dependency.kind] || 'Dependency';
      refs.name.textContent = dependency.name;
      refs.state.textContent = state;
      refs.progress.value = dependency.progress || 0;
      refs.progress.setAttribute('aria-label', `${dependency.name}: ${state}`);
      const byteDetail =
        dependency.status === 'reused'
          ? `${formatBytes(dependency.byte_size)} · upload skipped`
          : dependency.upload_required
            ? `${formatBytes(dependency.uploaded_bytes)} of ${formatBytes(dependency.byte_size)}`
            : 'Resolved without a local upload';
      refs.detail.textContent = `${dependency.detail} · ${byteDetail}${dependency.error ? ` · ${dependency.error}` : ''}`;
    }
  };

  const updateLogRows = (logs = []) => {
    logCount.textContent = `${logs.length} ${logs.length === 1 ? 'entry' : 'entries'}`;
    const visibleSequences = new Set(logs.map((entry) => entry.sequence));
    for (const [sequence, row] of logRows) {
      if (!visibleSequences.has(sequence)) {
        row.remove();
        logRows.delete(sequence);
      }
    }

    for (const entry of logs) {
      if (logRows.has(entry.sequence)) continue;
      const row = document.createElement('div');
      row.className = 'phantom-publisher-log-entry';
      row.dataset.level = entry.level;
      const time = document.createElement('time');
      time.dateTime = entry.timestamp;
      const timestamp = new Date(entry.timestamp);
      time.textContent = Number.isNaN(timestamp.getTime())
        ? entry.timestamp
        : timestamp.toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
          });
      const phaseLabel = document.createElement('span');
      phaseLabel.className = 'phantom-publisher-log-phase';
      phaseLabel.textContent = entry.phase.replaceAll('_', ' ');
      const message = document.createElement('span');
      message.className = 'phantom-publisher-log-message';
      message.textContent = entry.message;
      row.append(time, phaseLabel, message);
      logList.append(row);
      logRows.set(entry.sequence, row);
    }
    if (logDetails.open) logList.scrollTop = logList.scrollHeight;
  };

  while (document.body.contains(modal.panel)) {
    const job = await request(`/jobs/${jobId}`);
    phase.textContent = job.message || job.status.replaceAll('_', ' ');
    progress.value = job.progress || 0;
    progressValue.textContent = `${job.progress || 0}%`;
    updateDependencyRows(job.dependencies);
    updateLogRows(job.logs);
    if (job.current_dependency_id && job.current_dependency_id !== currentDependencyId) {
      dependencyRows
        .get(job.current_dependency_id)
        ?.row.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      currentDependencyId = job.current_dependency_id;
    }
    const completedDependencies = (job.dependencies || []).filter((dependency) =>
      ['uploaded', 'reused', 'not_required'].includes(dependency.status),
    ).length;
    summary.textContent = job.bytes_total
      ? `${formatBytes(job.bytes_uploaded)} of ${formatBytes(job.bytes_total)} processed · ${completedDependencies} of ${job.dependency_count} dependencies processed`
      : job.dependency_count
        ? `${completedDependencies} of ${job.dependency_count} dependencies processed`
        : 'Inspecting the workflow before upload…';
    if (job.status === 'failed') {
      phase.textContent = job.error || 'Publish failed';
      phase.classList.add('phantom-publisher-error');
      progressValue.classList.add('phantom-publisher-error');
      logDetails.open = true;
      logList.scrollTop = logList.scrollHeight;
      finish();
      return job;
    }
    if (job.status === 'cancelled') {
      // The idempotency key stays: a retry of this exact payload resumes the
      // version the cancelled publish staged instead of creating another.
      phase.textContent = 'Publish cancelled';
      phase.classList.add('phantom-publisher-muted');
      progressValue.classList.add('phantom-publisher-muted');
      progress.classList.add('phantom-publisher-muted');
      logDetails.open = true;
      logList.scrollTop = logList.scrollHeight;
      finish();
      return job;
    }
    if (job.status === 'completed') {
      localStorage.removeItem(idempotencyStorageKey);
      phase.textContent = job.message || `Version v${job.version.version} is ready for review.`;
      const open = document.createElement('button');
      open.className = 'phantom-publisher-primary';
      open.textContent = 'Open review in Phantom';
      open.onclick = () =>
        window.open(
          `${origin}/admin/gpu-workflows/${encodeURIComponent(workflowSlug)}/versions/${encodeURIComponent(job.version.workflow_version_id)}`,
          '_blank',
          'noopener',
        );
      modal.panel.append(open);
      finish();
      return job;
    }
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  // The panel was closed. The publish carries on in the ComfyUI server, and
  // nothing here learns how it ended.
  return null;
};

const publish = async () => {
  try {
    const config = await request('/config');
    if (!config.configured) {
      await configure(config);
      return;
    }
    const graphExtra = app.graph.extra || (app.graph.extra = {});
    const phantom = graphExtra.phantom || {};
    // 0.6.0 wrote the block as `alternative`, so a graph saved by that version
    // is read here and written back below in the `variation` shape. Without the
    // fallback the dialog would open such a graph on the primary and the next
    // publish would replace the primary graph instead of the variation.
    const rememberedVariation = phantom.variation || phantom.alternative || null;
    const target = await chooseTarget(phantom.workflow_id, config, rememberedVariation);
    if (target === RECONFIGURE) {
      // The target list belongs to the old Phantom, so re-enter from the top
      // rather than reusing anything read before the switch. A dismissed
      // dialog leaves the old connection in place and publishes nothing.
      if (await configure(config)) return publish();
      return;
    }
    // Remembered in the graph so the next publish of this file opens on the
    // same target — and, for a variation, on the same label.
    graphExtra.phantom = {
      origin: config.origin,
      workflow_id: target.workflow_id,
      ...(target.variation ? { variation: target.variation } : {}),
    };
    const refreshed = await app.graphToPrompt();
    const idempotencyStorageKey = `phantom-publisher:${target.workflow_id}:pending`;
    const publishPayload = {
      workflow_id: target.workflow_id,
      api_workflow: refreshed.output,
      ui_workflow: refreshed.workflow,
      ...(target.variation ? { variation: target.variation } : {}),
    };
    const manifestFingerprint = await fingerprintPublishPayload(publishPayload);
    const idempotencyKey = selectPendingIdempotencyKey(
      localStorage.getItem(idempotencyStorageKey),
      manifestFingerprint,
    );
    localStorage.setItem(
      idempotencyStorageKey,
      JSON.stringify({ idempotencyKey, manifestFingerprint }),
    );
    const job = await request('/publish', {
      method: 'POST',
      body: JSON.stringify({
        ...publishPayload,
        idempotency_key: idempotencyKey,
      }),
    });
    const finished = await showProgress(
      job.job_id,
      config.console_origin,
      target.slug,
      idempotencyStorageKey,
    );
    // Phantom assigns the variation id, and a new variation only learns its own
    // here. Written back so the next publish of this file updates that graph
    // instead of adding another variation beside it.
    if (finished?.variation?.variation_id && graphExtra.phantom.variation) {
      graphExtra.phantom.variation = { ...graphExtra.phantom.variation, ...finished.variation };
    }
  } catch (error) {
    const modal = dialog();
    modal.panel.innerHTML = `<h2>Publish failed</h2><p class="phantom-publisher-error"></p>`;
    modal.panel.querySelector('p').textContent = error.message;
  }
};

const PUBLISH_TOOLTIP = 'Publish the executable workflow and exact dependencies to Phantom';

const addToolbarButton = () => {
  // Current ComfyUI frontends render `actionBarButtons` below. Keep this DOM
  // fallback for older frontends, but do not add a duplicate after the
  // official action-bar button appears.
  if (
    document.querySelector('[data-phantom-publisher]') ||
    document.querySelector('.phantom-publisher-action')
  )
    return true;
  const toolbar = document.querySelector(
    '.comfyui-menu-mobile-collapse-primary, .comfy-menu, header',
  );
  if (!toolbar) return false;
  const button = document.createElement('button');
  button.dataset.phantomPublisher = 'true';
  button.className = 'comfyui-button phantom-publisher-toolbar';
  button.title = PUBLISH_TOOLTIP;
  button.innerHTML = `<span aria-hidden="true">↥</span><span>Publish to Phantom</span>`;
  button.onclick = publish;
  toolbar.append(button);
  return true;
};

app.registerExtension({
  name: 'phantom.publisher',
  // ComfyUI frontend 1.41+ renders extension action-bar buttons beside its
  // built-in Extensions and Run controls. This is the supported integration
  // point; direct toolbar DOM insertion is retained below only for older UI
  // versions that do not expose this hook.
  actionBarButtons: [
    {
      icon: 'pi pi-upload',
      label: 'Publish to Phantom',
      tooltip: PUBLISH_TOOLTIP,
      class: 'phantom-publisher-action',
      onClick: publish,
    },
  ],
  async setup() {
    if (!document.querySelector('link[data-phantom-publisher-style]')) {
      const style = document.createElement('link');
      style.rel = 'stylesheet';
      style.href = new URL('./phantom-publisher.css', import.meta.url).href;
      style.dataset.phantomPublisherStyle = 'true';
      document.head.append(style);
    }
    if (!addToolbarButton()) {
      const observer = new MutationObserver(() => addToolbarButton() && observer.disconnect());
      observer.observe(document.body, { childList: true, subtree: true });
    }
  },
});
