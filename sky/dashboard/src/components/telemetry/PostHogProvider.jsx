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

  // Initialize PostHog once on mount
  useEffect(() => {
    initPostHog();
  }, []);

  // Identify user and register deployment metadata
  useEffect(() => {
    if (identified.current) return;
    identified.current = true;

    const identify = async () => {
      // Fetch health for deployment metadata.
      try {
        const res = await fetch(`${ENDPOINT}/api/health`);
        if (res.ok) {
          const data = await res.json();
          registerDeployment({
            sky_version: data.version || 'unknown',
            api_version: data.api_version || 'unknown',
          });
        }
      } catch {
        // Ignore – analytics should never break the app
      }

      // Fetch user role; also carries the telemetry opt-out signal.
      try {
        const res = await fetch(`${ENDPOINT}/users/role`);
        if (!res.ok) return;
        const data = await res.json();
        // Use strict === false so older servers (without the field) don't opt out.
        if (data.telemetry_enabled === false) {
          optOut();
          return;
        }
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
