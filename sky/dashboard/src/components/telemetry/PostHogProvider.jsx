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

/**
 * PostHogProvider initializes analytics on mount, identifies the user,
 * registers deployment metadata, and tracks page views on route changes.
 *
 * Wrap this around your app (inside PluginProvider) so that every page
 * transition is captured.
 */
export default function PostHogProvider({ children }) {
  const router = useRouter();
  const identified = useRef(false);

  // Initialize PostHog only after confirming telemetry is enabled, then
  // identify the user and register deployment metadata.
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

      if (deploymentData) {
        registerDeployment({
          sky_version: deploymentData.version || 'unknown',
          api_version: deploymentData.api_version || 'unknown',
        });
      }

      // Fetch user role for identification.
      try {
        const res = await fetch(`${ENDPOINT}/users/role`);
        if (!res.ok) return;
        const data = await res.json();
        const userHash = data.id || 'anonymous';
        const username = data.name || '';
        identifyUser(userHash, username);
      } catch {
        // Ignore
      }
    };

    identify();
  }, []);

  // Track page views on route changes
  useEffect(() => {
    const handleRouteChange = (url) => {
      trackPageView(url);
    };

    // Track the initial page view
    trackPageView(router.asPath);

    router.events.on('routeChangeComplete', handleRouteChange);
    return () => {
      router.events.off('routeChangeComplete', handleRouteChange);
    };
  }, [router]);

  return children;
}
