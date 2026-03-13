'use client';

import { useEffect, useRef, useState } from 'react';
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
  const [ready, setReady] = useState(false);

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
        const user = deploymentData.user;
        const userHash = (user && user.id) || 'anonymous';
        const username = (user && user.name) || '';
        identifyUser(userHash, username);
      }

      // Signal that PostHog is ready so page views can start.
      setReady(true);
    };

    identify();
  }, []);

  // Fire the initial page view exactly once, when PostHog becomes ready.
  const initialTracked = useRef(false);
  useEffect(() => {
    if (!ready || initialTracked.current) return;
    initialTracked.current = true;
    trackPageView(router.asPath);
  }, [ready]); // eslint-disable-line react-hooks/exhaustive-deps

  // Track subsequent page views via routeChangeComplete.
  // router.events is a stable singleton so we only need to bind once.
  useEffect(() => {
    if (!ready) return;
    const handleRouteChange = (url) => {
      trackPageView(url);
    };
    router.events.on('routeChangeComplete', handleRouteChange);
    return () => {
      router.events.off('routeChangeComplete', handleRouteChange);
    };
  }, [ready]); // eslint-disable-line react-hooks/exhaustive-deps

  return children;
}
