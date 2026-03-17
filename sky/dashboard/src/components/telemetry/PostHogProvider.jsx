'use client';

import { useEffect, useRef } from 'react';
import { useRouter } from 'next/router';
import {
  initPostHog,
  optOut,
  identifyUser,
  registerDeployment,
  trackPageView,
} from '@/lib/analytics';
import { ENDPOINT } from '@/data/connectors/constants';
import { getCurrentUserInfo } from '@/data/connectors/client';

/**
 * PostHogProvider initializes analytics on mount, identifies the user,
 * registers deployment metadata, and tracks page views.
 *
 * ## Why we track pageviews manually instead of using PostHog's built-in mode
 *
 * PostHog's official SPA approach is `capture_pageview: 'history_change'`,
 * which auto-patches `history.pushState` to fire `$pageview` events on every
 * client-side navigation — zero per-page code required.
 *
 * We can't use that mode because of a timing conflict with our async
 * telemetry opt-out check.  Before calling `initPostHog()` we must fetch
 * `/api/health?verbose=1` to check whether the server has set
 * `SKYPILOT_DISABLE_USAGE_COLLECTION=1` — otherwise we'd leak events when
 * telemetry is disabled.  This means PostHog init is deferred until the
 * health response arrives.
 *
 * The `history_change` mode fires `$pageview` on `pushState` BEFORE our
 * async `identify()` call completes.  With `person_profiles: 'identified_only'`
 * (the PostHog project default), those pre-identify events have
 * `$process_person_profile: false` and are invisible in PostHog queries.
 *
 * Our solution: `capture_pageview: false` + manual `trackPageView()` after
 * `identify()` completes.  SPA navigations are tracked via Next.js
 * `routeChangeComplete` events.  This gives us the same zero-per-page-code
 * as auto mode, just centralized in this provider.
 *
 * trackPageView() fires:
 *   1. On initial mount (once PostHog is initialized and the user identified)
 *   2. On every Next.js routeChangeComplete event (SPA navigations)
 */
export default function PostHogProvider({ children }) {
  const router = useRouter();
  const identified = useRef(false);
  const ready = useRef(false);

  // Initialize PostHog only after confirming telemetry is enabled, then
  // identify the user, register deployment metadata, and fire the initial
  // pageview.
  useEffect(() => {
    if (identified.current) return;
    identified.current = true;

    const identify = async () => {
      // Fetch health (verbose) for deployment metadata and telemetry opt-out.
      // We must check this BEFORE calling initPostHog() so that no events
      // are sent when telemetry is disabled.
      let deploymentData = null;
      try {
        const res = await fetch(`${ENDPOINT}/api/health?verbose=1`);
        if (res.ok) {
          const data = await res.json();
          // Use strict === false so older servers (without the field) don't opt out.
          if (data.telemetry_enabled === false) {
            optOut();
            return;
          }
          deploymentData = data;
        }
      } catch {
        // Ignore – analytics should never break the app
      }

      // Telemetry is allowed – now initialize PostHog.
      initPostHog();

      // Fetch active plugins to tag every event with plugin metadata.
      let pluginNames = [];
      try {
        const pluginsRes = await fetch(`${ENDPOINT}/api/plugins`);
        if (pluginsRes.ok) {
          const pluginsData = await pluginsRes.json();
          if (pluginsData && Array.isArray(pluginsData.plugins)) {
            pluginNames = pluginsData.plugins
              .map((p) => p.name)
              .filter(Boolean);
          }
        }
      } catch {
        // Ignore – analytics should never break the app
      }

      if (deploymentData) {
        registerDeployment({
          sky_version: deploymentData.version || 'unknown',
          api_version: deploymentData.api_version || 'unknown',
          active_plugins: pluginNames,
          plugin_count: pluginNames.length,
          has_plugins: pluginNames.length > 0,
        });
      }

      // Reuse the cached /users/role fetch from client.js, which returns
      // the correct user hash and name in both auth and no-auth/local modes.
      // (The health endpoint's `user` field is null in no-auth mode.)
      try {
        const userInfo = await getCurrentUserInfo();
        identifyUser(userInfo.id, userInfo.name);
      } catch {
        // Ignore – analytics should never break the app
      }

      // Fire the initial pageview now that PostHog is ready.
      trackPageView(window.location.pathname);
      ready.current = true;
    };

    identify();
  }, []);

  // Track SPA navigations via Next.js router events.
  useEffect(() => {
    const handleRouteChange = (url) => {
      if (ready.current) {
        trackPageView(url);
      }
    };
    router.events.on('routeChangeComplete', handleRouteChange);
    return () => {
      router.events.off('routeChangeComplete', handleRouteChange);
    };
    // router.events is stable – no need to include router in deps.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return children;
}
