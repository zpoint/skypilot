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
