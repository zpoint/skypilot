# PR #247 Review Fixes — Dashboard Analytics Improvements

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Address all open review comments from romilbhardwaj on PR #247 (assemble-org/skypilot).

**Architecture:** All changes are in the dashboard frontend (`sky/dashboard/src/`). The core issue is that plugin pages use `history.pushState()` for navigation (they're standalone Vite-built React apps, not Next.js components), so Next.js `routeChangeComplete` events never fire and PostHogProvider misses those pageviews. For autocapture improvements (comments #2-6), PostHog's `autocapture: true` already captures clicks/inputs but labels them generically ("clicked svg", "clicked td"). We improve this by configuring PostHog's `custom_campaign_params` and element attribute settings, plus adding `data-ph-capture-attribute-*` annotations to key interactive elements.

**Tech Stack:** Next.js (Pages Router), PostHog (posthog-js), Jest

---

## Review Comments Map

| # | Comment | Status | Task |
|---|---------|--------|------|
| 1 | Devspace navigation doesn't log pageview | Open | Task 1 |
| 2 | Download button logged as "clicked svg" — need better labels for all buttons | Open | Task 2 |
| 3 | GPU manager remediation button — track by class/title | Open | Task 2 |
| 4 | GPU manager node click logged as "clicked td" — should be "clicked on node" | Open | Task 2 |
| 5 | "typed something into input" — record which input field | Open | Task 2 |
| 6 | Quota page rectangles — "clicked rect" should be "clicked queue" | Open | Task 2 |
| 7 | users.jsx changes are unrelated? | Open | Task 3 |

---

## Task 1: Track pageviews for plugin pushState navigations

### Problem

Plugin frontends (devspaces, kueue, etc.) are standalone React apps loaded at runtime. They navigate via `window.history.pushState()` — not the Next.js router — so the `routeChangeComplete` event in PostHogProvider never fires. Result: no `$pageview` events for plugin sub-navigations (e.g. `/devspaces` → `/devspaces/romil-dev`).

### Approach

`PluginProvider.jsx` already intercepts `history.pushState` and `history.replaceState` (lines 459-523, for URL normalization). We dispatch a custom event from that interception point, and listen for it in `PostHogProvider.jsx` to fire `trackPageView()`.

We do NOT patch `history.pushState` again in PostHogProvider (double-patching is fragile). Instead:
1. In `PluginProvider.jsx` `interceptHistoryApi()`: after each successful `pushState` call, dispatch `window.dispatchEvent(new CustomEvent('skydashboard:url-changed', { detail: { url } }))`.
2. In `PostHogProvider.jsx`: listen for `skydashboard:url-changed` and call `trackPageView(url)` (with dedup guard already in `trackPageView`).

Also add a `/devspaces/[name]` pattern to `ROUTE_PATTERNS` in `analytics.js` so devspace detail paths are normalized.

### Files

- Modify: `sky/dashboard/src/plugins/PluginProvider.jsx` (lines 468-522 — `interceptHistoryApi`)
- Modify: `sky/dashboard/src/components/telemetry/PostHogProvider.jsx` (add `skydashboard:url-changed` listener)
- Modify: `sky/dashboard/src/lib/analytics.js` (add devspaces route pattern)
- Modify: `sky/dashboard/src/lib/analytics.test.js` (add devspaces normalization tests)

### Steps

- [ ] **Step 1: Add devspaces route pattern to analytics.js**

In `ROUTE_PATTERNS` array (after the plugins catch-all), add:

```javascript
// Devspaces: /devspaces/[name] (plugin route, uses pushState not Next.js router)
[/^\/devspaces\/[^/]+$/, '/devspaces/[name]'],
```

- [ ] **Step 2: Add normalization tests for devspaces paths**

In `analytics.test.js`, inside the `normalizePath` describe block, add:

```javascript
test('normalizes devspace detail paths', () => {
  expect(analytics.normalizePath('/devspaces/romil-dev')).toBe('/devspaces/[name]');
  expect(analytics.normalizePath('/devspaces/my-workspace')).toBe('/devspaces/[name]');
});

test('leaves devspaces list path unchanged', () => {
  expect(analytics.normalizePath('/devspaces')).toBe('/devspaces');
});
```

- [ ] **Step 3: Run tests to verify normalization**

```bash
cd sky/dashboard && npx jest src/lib/analytics.test.js --verbose
```

Expected: all tests pass including new devspaces tests.

- [ ] **Step 4: Dispatch custom event from PluginProvider's history interception**

In `PluginProvider.jsx`, in the `interceptHistoryApi()` function, after each successful `originalPushState.call(...)`, dispatch a custom event. Modify the `pushState` override (line 469):

```javascript
// Override pushState
window.history.pushState = function (state, title, url) {
  let normalizedUrl = url;
  if (url && typeof url === 'string') {
    normalizedUrl = normalizeUrlForHistory(url);
  }
  try {
    const result = originalPushState.call(this, state, title, normalizedUrl);
    // Notify analytics listeners (e.g. PostHogProvider) about URL changes
    // from plugin navigations that bypass the Next.js router.
    window.dispatchEvent(
      new CustomEvent('skydashboard:url-changed', {
        detail: { url: normalizedUrl || url },
      })
    );
    return result;
  } catch (error) {
    // If pushState still fails (e.g., due to origin mismatch), try with a relative URL
    if (
      error.name === 'SecurityError' &&
      normalizedUrl &&
      typeof normalizedUrl === 'string'
    ) {
      try {
        const urlObj = new URL(normalizedUrl, window.location.href);
        const relativeUrl = urlObj.pathname + urlObj.search + urlObj.hash;
        const result = originalPushState.call(this, state, title, relativeUrl);
        window.dispatchEvent(
          new CustomEvent('skydashboard:url-changed', {
            detail: { url: relativeUrl },
          })
        );
        return result;
      } catch {
        throw error;
      }
    }
    throw error;
  }
};
```

Do NOT add the event to `replaceState` — replaceState is for URL cleanup (e.g. removing query params), not actual navigations.

- [ ] **Step 5: Listen for the custom event in PostHogProvider**

In `PostHogProvider.jsx`, add a third `useEffect` after the existing `routeChangeComplete` listener:

```javascript
// Track plugin navigations that use history.pushState directly
// (bypasses Next.js router, so routeChangeComplete doesn't fire).
useEffect(() => {
  const handleUrlChanged = (e) => {
    if (ready.current && e.detail?.url) {
      trackPageView(e.detail.url);
    }
  };
  window.addEventListener('skydashboard:url-changed', handleUrlChanged);
  return () => {
    window.removeEventListener('skydashboard:url-changed', handleUrlChanged);
  };
}, []);
```

- [ ] **Step 6: Run all analytics tests**

```bash
cd sky/dashboard && npx jest src/lib/analytics.test.js --verbose
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add sky/dashboard/src/plugins/PluginProvider.jsx \
        sky/dashboard/src/components/telemetry/PostHogProvider.jsx \
        sky/dashboard/src/lib/analytics.js \
        sky/dashboard/src/lib/analytics.test.js
git commit -m "[Dashboard] Track pageviews for plugin pushState navigations

Plugins (devspaces, kueue, etc.) navigate via history.pushState() which
bypasses Next.js router events. Dispatch a custom 'skydashboard:url-changed'
event from PluginProvider's existing history interception and listen for it
in PostHogProvider to fire trackPageView().

Also adds /devspaces/[name] to ROUTE_PATTERNS for path normalization."
```

---

## Task 2: Improve PostHog autocapture element labeling

### Problem

PostHog's `autocapture: true` captures all clicks/inputs but labels them generically based on the HTML tag name: "clicked svg", "clicked td", "clicked rect", "typed something into input". Romil's comments #2-6 all want better labels.

### Approach — PostHog `custom_campaign_params` and element properties

PostHog autocapture already collects `data-attr`, `aria-label`, `title`, `name`, `id`, and `placeholder` attributes from the clicked element and its ancestors. The issue is that many dashboard elements (and especially plugin elements) lack these attributes.

**What we can do in the dashboard repo (this PR):**

1. **Configure PostHog to collect additional element attributes** — add `element_attribute_ignorelist` (to drop noisy classes) and set `custom_campaign_params` in the PostHog init config. Specifically, enable capturing of `data-ph-capture-attribute-*` attributes.

2. **Add `data-ph-capture-attribute-action` and `aria-label` attributes** to key interactive elements in the **dashboard's own components**. This improves tracking for buttons, inputs, and table rows that we control.

**What we CANNOT do here (plugin elements):**
Comments #3, #4, #5, #6 reference plugin-specific elements (GPU manager buttons, quota page rectangles, kueue node rows). These live in the `feature-plugin` repo, not this one. We should:
- Reply to those comments explaining this is a plugin-side fix
- Provide guidance for plugin authors on which attributes to add

### Files

- Modify: `sky/dashboard/src/lib/analytics.js` (PostHog init config — add `properties_string_max_length`, custom attribute allowlist)

### Steps

- [ ] **Step 1: Update PostHog init config to improve autocapture quality**

In `analytics.js`, update the `posthog.init()` call to include better autocapture config:

```javascript
posthog.init(POSTHOG_API_KEY, {
  api_host: POSTHOG_HOST,
  autocapture: {
    dom_event_allowlist: ['click', 'change', 'submit'],
    element_allowlist: ['a', 'button', 'input', 'select', 'textarea', 'label'],
    css_selector_allowlist: ['[data-ph-capture-attribute-action]'],
  },
  capture_pageview: false,
  capture_pageleave: true,
  persistence: 'localStorage',
  disable_session_recording: false,
  properties_string_max_length: 300,
});
```

Key changes:
- **`element_allowlist`**: Only autocapture on meaningful interactive elements (buttons, inputs, links, selects) instead of every SVG/rect/td. This directly fixes comments #2 (clicked svg), #4 (clicked td), and #6 (clicked rect) — those generic element clicks won't fire at all.
- **`css_selector_allowlist`**: Also capture any element explicitly annotated with `data-ph-capture-attribute-action`, even if it's not in the element allowlist (e.g. a `<tr>` row with `data-ph-capture-attribute-action="view_node"`).
- **`dom_event_allowlist`**: Only capture click, change, submit (skip touch events etc.)

- [ ] **Step 2: Update the analytics test for new autocapture config**

In `analytics.test.js`, update the `initPostHog` test:

```javascript
test('initializes posthog with correct config', () => {
  analytics.initPostHog();

  expect(posthog.init).toHaveBeenCalledTimes(1);
  expect(posthog.init).toHaveBeenCalledWith(
    expect.any(String),
    expect.objectContaining({
      api_host: 'https://usage-v3.skypilot.co',
      autocapture: expect.objectContaining({
        dom_event_allowlist: ['click', 'change', 'submit'],
        element_allowlist: expect.arrayContaining(['button', 'input']),
      }),
      capture_pageview: false,
      capture_pageleave: true,
      persistence: 'localStorage',
    })
  );
});
```

- [ ] **Step 3: Run tests**

```bash
cd sky/dashboard && npx jest src/lib/analytics.test.js --verbose
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add sky/dashboard/src/lib/analytics.js \
        sky/dashboard/src/lib/analytics.test.js
git commit -m "[Dashboard] Restrict autocapture to interactive elements

Configure PostHog autocapture element_allowlist to only capture clicks on
buttons, inputs, links, selects, and elements annotated with
data-ph-capture-attribute-action. This stops generic 'clicked svg',
'clicked td', 'clicked rect' events from firing."
```

---

## Task 3: Reply to romilbhardwaj's review comments

This task is about composing review replies, not writing code. Below are the recommended replies for each open comment.

### Comment #7: users.jsx unrelated changes (line 64)

> **Reply:** The users.jsx changes add analytics tracking (trackUserAction, trackFilterUsed) to the users management page — same instrumentation pattern applied to clusters.jsx, jobs.jsx, and other pages. They're part of this PR's scope.

### Comments #3, #4, #5, #6: Plugin element tracking (GPU manager, quota page)

> **Reply (single thread):** These elements (GPU manager buttons/rows, quota rectangles, input fields) come from plugin frontends in the `feature-plugin` repo, not the dashboard. To fix the labeling, plugin authors need to add `data-ph-capture-attribute-action="<descriptive_name>"` attributes to their interactive elements.
>
> In this PR, I've restricted PostHog autocapture to only fire on interactive HTML elements (buttons, inputs, links, selects) and elements explicitly annotated with `data-ph-capture-attribute-action`. This means:
> - Generic "clicked svg", "clicked td", "clicked rect" events will stop firing
> - Plugin elements that add `data-ph-capture-attribute-action` will be captured with descriptive labels
>
> I'll file a follow-up issue for the plugin repo to add these attributes to GPU manager, quota, and devspaces components.

### Comment #2: Download button "clicked svg"

> **Reply:** Fixed by restricting autocapture to `element_allowlist: ['a', 'button', 'input', 'select', 'textarea', 'label']`. The download button click will now be captured as "clicked button" with the button's text/aria-label instead of "clicked svg" (the inner SVG icon). If we want richer detail, we can add `data-ph-capture-attribute-action="download_logs"` to the button element.

---

## Summary of changes by file

| File | Changes |
|------|---------|
| `analytics.js` | Add devspaces ROUTE_PATTERN; change `autocapture: true` to object config with element_allowlist |
| `analytics.test.js` | Add devspaces normalization tests; update init config assertion |
| `PostHogProvider.jsx` | Add `skydashboard:url-changed` event listener |
| `PluginProvider.jsx` | Dispatch `skydashboard:url-changed` after pushState calls |
