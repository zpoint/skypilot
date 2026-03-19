/**
 * Product analytics utilities for SkyPilot Dashboard.
 *
 * Thin wrapper around posthog-js. The PostHogProvider is responsible for
 * calling optOut() when the server reports that usage collection is disabled.
 */
import posthog from 'posthog-js';
import { BASE_PATH } from '@/data/connectors/constants';

const POSTHOG_API_KEY = 'phc_QHyDrOac26TKUEYArlailR2EDe8xOCg2Vb4kjzdoADi';
const POSTHOG_HOST = 'https://usage-v3.skypilot.co';

let _initialized = false;
let _optedOut = false;

// ── Autocapture enrichment ──────────────────────────────────────────────────

/**
 * Standard interactive HTML tags — matched first in the walk-up.
 */
const INTERACTIVE_TAGS = new Set([
  'a',
  'button',
  'input',
  'select',
  'textarea',
  'label',
]);

/**
 * CSS class patterns that signal an element is intentionally interactive.
 * Matched in the second pass when no standard interactive tag is found.
 */
const INTERACTIVE_CLASS_PATTERN =
  /\b(clickable|btn|button|toggle|pill|tab|switch|chip)\b/i;

/**
 * Parse a single component from the $elements_chain string into a structured
 * object with tag_name and attributes.
 *
 * Elements_chain format (semicolon-separated, innermost first):
 *   tag.class1.class2:attr__key="value"attr__key2="value2"nth-child="N"text="T"
 */
function parseChainComponent(component) {
  if (!component) return null;
  // Tag name is everything before the first dot, colon, or bracket
  const tagMatch = component.match(/^([a-z][a-z0-9]*)/i);
  const tag_name = tagMatch ? tagMatch[1] : '';

  // Extract attr__* values
  const attrs = {};
  const attrRe = /attr__([a-z_-]+)="([^"]*)"/gi;
  let m;
  while ((m = attrRe.exec(component)) !== null) {
    attrs['attr__' + m[1]] = m[2];
  }

  // Extract text="..." (element text content)
  const textMatch = component.match(/(?:^|[;:"])text="([^"]*)"/);
  const text = textMatch ? textMatch[1] : '';

  return { tag_name, text, attrs };
}

/**
 * Check whether a parsed chain element has signals of interactivity.
 */
function hasInteractivitySignal(el) {
  if (
    el.attrs.attr__title ||
    el.attrs.attr__aria_label ||
    el.attrs['attr__aria-label'] ||
    el.attrs.attr__role
  )
    return true;
  const cls = el.attrs.attr__class || '';
  return INTERACTIVE_CLASS_PATTERN.test(cls);
}

/**
 * Extract a human-readable label from a parsed chain element.
 */
function extractLabel(el) {
  return (
    el.text ||
    el.attrs.attr__title ||
    el.attrs.attr__aria_label ||
    el.attrs['attr__aria-label'] ||
    el.attrs.attr__placeholder ||
    el.attrs.attr__name ||
    el.attrs.attr__id ||
    extractInteractiveClassName(el) ||
    ''
  );
}

/**
 * Extract a human-readable label from an element's CSS class when it
 * matches an interactive pattern (e.g. "clickable-node-row" → "node row").
 */
function extractInteractiveClassName(el) {
  const cls = el.attrs.attr__class || '';
  const match = cls.match(
    /\b(?:clickable|btn|button|toggle|pill|tab|switch|chip)[-_]?([\w-]*)/i
  );
  if (match) {
    // Return the full matched class segment, cleaned up
    const full = match[0].replace(/[-_]/g, ' ').trim();
    return full || '';
  }
  return '';
}

/**
 * Try to derive a label from an icon SVG's class name.
 * Lucide icons use classes like "lucide lucide-download" — extract "download".
 * Also handles "icon-foo", "fa-bar", etc.
 */
function extractIconLabel(el) {
  const cls = el.attrs.attr__class || '';
  // lucide-<name>
  const lucide = cls.match(/\blucide-([a-z][-a-z0-9]*)/i);
  if (lucide) return lucide[1].replace(/-/g, ' ');
  // icon-<name> or fa-<name>
  const icon = cls.match(/\b(?:icon|fa)-([a-z][-a-z0-9]*)/i);
  if (icon) return icon[1].replace(/-/g, ' ');
  return '';
}

/**
 * Search child elements (earlier in chain) for a label when the interactive
 * element itself has none. Checks children's text, title, aria-label, and
 * icon class names.
 */
function extractLabelFromChildren(components, interactiveIdx) {
  for (let i = 0; i < interactiveIdx; i++) {
    const child = components[i];
    const label = extractLabel(child);
    if (label) return label;
    const iconLabel = extractIconLabel(child);
    if (iconLabel) return iconLabel;
  }
  return '';
}

/**
 * Enrich an autocapture event with human-readable action context.
 *
 * PostHog autocapture fires on the innermost DOM element (e.g. an <svg>
 * icon inside a <button>), producing vague labels like "clicked svg".
 * This function parses the $elements_chain string and adds:
 *
 *   action_element — tag of the best interactive ancestor (or target)
 *   action_label   — human-readable label from text/title/aria-label/
 *                    placeholder
 *
 * NOTE: We parse $elements_chain (string) because $elements (array) is
 * deprecated and not available at before_send time in posthog-js v1.356+.
 *
 * Two-pass strategy:
 *   1. Find nearest standard interactive tag (button, a, input, etc.)
 *   2. If none, find nearest element with interactivity signals
 *      (title, aria-label, role, or class like "clickable"/"btn"/"pill")
 *   3. If none, fall back to $el_text if available
 */
export function enrichAutocaptureEvent(event) {
  if (!event || event.event !== '$autocapture') return event;
  const chain = event.properties?.$elements_chain;
  if (typeof chain !== 'string' || chain.length === 0) return event;

  // Split chain into components (semicolon-separated, innermost first)
  const components = chain.split(';').map(parseChainComponent).filter(Boolean);
  if (components.length === 0) return event;

  // Pass 1: standard interactive tag
  for (let i = 0; i < components.length; i++) {
    const el = components[i];
    if (INTERACTIVE_TAGS.has(el.tag_name)) {
      event.properties.action_element = el.tag_name;
      event.properties.action_label =
        extractLabel(el) ||
        event.properties.$el_text ||
        extractLabelFromChildren(components, i) ||
        '';
      return event;
    }
  }

  // Pass 2: element with interactivity signals
  for (let i = 0; i < components.length; i++) {
    const el = components[i];
    if (hasInteractivitySignal(el)) {
      event.properties.action_element = el.tag_name;
      event.properties.action_label =
        extractLabel(el) ||
        event.properties.$el_text ||
        extractLabelFromChildren(components, i) ||
        '';
      return event;
    }
  }

  // Pass 3: fall back to the target element (first in chain)
  const target = components[0];
  event.properties.action_element = target.tag_name;
  event.properties.action_label =
    extractLabel(target) || event.properties.$el_text || '';
  return event;
}

/**
 * Initialize PostHog. Safe to call multiple times – only the first call has
 * any effect. Call optOut() after init to disable collection at runtime.
 */
export function initPostHog() {
  if (_initialized) return;
  _initialized = true;

  if (typeof window === 'undefined') return;

  posthog.init(POSTHOG_API_KEY, {
    api_host: POSTHOG_HOST,
    autocapture: true,
    // Disable automatic pageview capture; we fire standard '$pageview'
    // events manually via trackPageView() from PostHogProvider on both
    // initial load and Next.js routeChangeComplete events.  See the
    // detailed comment in PostHogProvider.jsx for why we can't use the
    // built-in 'history_change' mode (timing conflict with async init).
    capture_pageview: false,
    capture_pageleave: true,
    persistence: 'localStorage',
    disable_session_recording: false,
    // Enrich autocapture events with human-readable action context.
    // Wrapping in try-catch because PostHog does NOT catch before_send
    // errors — an uncaught throw would kill event delivery.
    before_send: (event) => {
      try {
        return enrichAutocaptureEvent(event) ?? event;
      } catch {
        return event;
      }
    },
  });
}

/**
 * Opt out of all analytics collection. Called by PostHogProvider when the
 * server reports SKYPILOT_DISABLE_USAGE_COLLECTION=1.
 */
export function optOut() {
  _optedOut = true;
  if (_initialized && typeof window !== 'undefined') {
    posthog.opt_out_capturing();
  }
}

/** Returns true when analytics collection is active (initialized and not opted out). */
export function isEnabled() {
  return _initialized && !_optedOut && typeof window !== 'undefined';
}

// ── Identification ──────────────────────────────────────────────────────────

/**
 * Identify the current user and register "super properties" that are attached
 * to every subsequent event.
 */
export function identifyUser(userHash, username, extraProperties = {}) {
  if (!isEnabled()) return;
  posthog.identify(userHash, {
    username,
    source: 'dashboard',
    ...extraProperties,
  });
}

/**
 * Register deployment-level super properties (version, auth mode, etc.).
 * These are sent with every event automatically.
 */
export function registerDeployment(properties) {
  if (!isEnabled()) return;
  posthog.register({
    source: 'dashboard',
    ...properties,
  });
}

// ── Path Normalization ──────────────────────────────────────────────────────

// ⚠️  IMPORTANT: Keep this list in sync with dashboard routes!
// When adding a new route with dynamic segments to sky/dashboard/src/pages/,
// add a corresponding pattern here so analytics paths are normalized correctly.
// Order matters — more specific patterns must come before less specific ones.
const ROUTE_PATTERNS = [
  // Jobs: /jobs/pools/[pool] (must be before /jobs/[job]/[task])
  [/^\/jobs\/pools\/[^/]+$/, '/jobs/pools/[pool]'],
  // Jobs: /jobs/[job]/[task] (must be before /jobs/[job])
  [/^\/jobs\/[^/]+\/[^/]+$/, '/jobs/[job]/[task]'],
  // Jobs: /jobs/[job] - must not match /jobs/pools (static route)
  [
    /^\/jobs\/[^/]+$/,
    (path) => (path === '/jobs/pools' ? path : '/jobs/[job]'),
  ],
  // Clusters: /clusters/[cluster]/[job] (must be before /clusters/[cluster])
  [/^\/clusters\/[^/]+\/[^/]+$/, '/clusters/[cluster]/[job]'],
  // Clusters: /clusters/[cluster]
  [/^\/clusters\/[^/]+$/, '/clusters/[cluster]'],
  // Recipes: /recipes/[recipe]
  [/^\/recipes\/[^/]+$/, '/recipes/[recipe]'],
  // Workspaces: /workspaces/[name]
  [/^\/workspaces\/[^/]+$/, '/workspaces/[name]'],
  // Infra: /infra/[context]
  [/^\/infra\/[^/]+$/, '/infra/[context]'],
  // Devspaces: /devspaces/[name] (plugin route, uses pushState not Next.js router)
  [/^\/devspaces\/[^/]+$/, '/devspaces/[name]'],
  // Plugins: catch-all /plugins/[...slug]
  [/^\/plugins\/.*$/, '/plugins/[...slug]'],
];

/**
 * Normalize a path by replacing dynamic segments with parameter names.
 * Static routes pass through unchanged.
 * @param {string} path - The raw path to normalize
 * @returns {string} The normalized path
 */
export function normalizePath(path) {
  // Strip the basePath prefix (e.g. "/dashboard") before matching.
  // router.asPath omits it, but routeChangeComplete includes it.
  const stripped =
    BASE_PATH && path.startsWith(BASE_PATH)
      ? path.slice(BASE_PATH.length) || '/'
      : path;
  for (const [pattern, normalized] of ROUTE_PATTERNS) {
    if (pattern.test(stripped)) {
      return typeof normalized === 'function'
        ? normalized(stripped)
        : normalized;
    }
  }
  return stripped;
}

// ── Pageviews ───────────────────────────────────────────────────────────────

let _lastPageviewPath = null;
let _lastPageviewTime = 0;

/**
 * Track a page view using the standard PostHog '$pageview' event.
 *
 * We fire this manually (instead of using posthog-js's built-in
 * capture_pageview: 'history_change') because our async init flow
 * (health check → init → identify) must complete before any events are
 * sent.  See PostHogProvider.jsx for a detailed explanation.
 *
 * Includes a 1-second dedup guard to prevent double-firing on initial load
 * (e.g. from concurrent init + routeChangeComplete sources).
 */
export function trackPageView(path, properties = {}) {
  if (!isEnabled()) return;
  const normalized = normalizePath(path);
  const now = Date.now();
  if (normalized === _lastPageviewPath && now - _lastPageviewTime < 1000)
    return;
  _lastPageviewPath = normalized;
  _lastPageviewTime = now;
  posthog.capture('$pageview', {
    $current_url: window.location.href,
    $pathname: normalized,
    path: normalized,
    raw_path: path,
    ...properties,
  });
}

// ── Generic event helper ────────────────────────────────────────────────────

export function trackEvent(eventName, properties = {}) {
  if (!isEnabled()) return;
  posthog.capture(eventName, { source: 'dashboard', ...properties });
}

// ── Domain-specific helpers ─────────────────────────────────────────────────

export function trackClusterAction(action, properties = {}) {
  trackEvent('cluster_action', { action, ...properties });
}

export function trackJobAction(action, properties = {}) {
  trackEvent('job_action', { action, ...properties });
}

export function trackWorkspaceAction(action, properties = {}) {
  trackEvent('workspace_action', { action, ...properties });
}

export function trackRecipeAction(action, properties = {}) {
  trackEvent('recipe_action', { action, ...properties });
}

export function trackInfraAction(action, properties = {}) {
  trackEvent('infra_action', { action, ...properties });
}

export function trackVolumeAction(action, properties = {}) {
  trackEvent('volume_action', { action, ...properties });
}

export function trackUserAction(action, properties = {}) {
  trackEvent('user_action', { action, ...properties });
}

export function trackSettingsAction(action, properties = {}) {
  trackEvent('settings_action', { action, ...properties });
}

export function trackFilterUsed(filterType, properties = {}) {
  trackEvent('filter_used', { filter_type: filterType, ...properties });
}

export function trackPluginPageView(pluginName, pagePath) {
  trackEvent('plugin_page_view', { plugin: pluginName, path: pagePath });
}
