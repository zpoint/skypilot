/**
 * Analytics stub for SkyPilot Dashboard.
 *
 * All exported functions are silent no-ops by default. A plugin (e.g. the
 * UsageEnforcementPlugin in feature-plugin) calls registerAnalyticsProvider()
 * to inject the real PostHog-backed implementation. Without the plugin, every
 * track*() call is a no-op — zero runtime cost, zero network requests.
 */

let _provider = null;

/**
 * Register the real analytics implementation. Called by a plugin
 * after the analytics backend (e.g. PostHog) is initialized.
 * Pass null to unregister.
 */
export function registerAnalyticsProvider(provider) {
  _provider = provider;
}

/** Returns the current provider, or null if none registered. */
export function getAnalyticsProvider() {
  return _provider;
}

// ── Tracking functions (no-op without provider) ─────────────────────────────

export function trackEvent(eventName, properties = {}) {
  _provider?.trackEvent?.(eventName, properties);
}

export function trackPageView(path, properties = {}) {
  _provider?.trackPageView?.(path, properties);
}

export function trackClusterAction(action, properties = {}) {
  _provider?.trackClusterAction?.(action, properties);
}

export function trackJobAction(action, properties = {}) {
  _provider?.trackJobAction?.(action, properties);
}

export function trackWorkspaceAction(action, properties = {}) {
  _provider?.trackWorkspaceAction?.(action, properties);
}

export function trackRecipeAction(action, properties = {}) {
  _provider?.trackRecipeAction?.(action, properties);
}

export function trackInfraAction(action, properties = {}) {
  _provider?.trackInfraAction?.(action, properties);
}

export function trackVolumeAction(action, properties = {}) {
  _provider?.trackVolumeAction?.(action, properties);
}

export function trackUserAction(action, properties = {}) {
  _provider?.trackUserAction?.(action, properties);
}

export function trackSettingsAction(action, properties = {}) {
  _provider?.trackSettingsAction?.(action, properties);
}

export function trackFilterUsed(filterType, properties = {}) {
  _provider?.trackFilterUsed?.(filterType, properties);
}

export function trackPluginPageView(pluginName, pagePath) {
  _provider?.trackPluginPageView?.(pluginName, pagePath);
}
