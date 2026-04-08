/**
 * Tests for analytics stub (analytics.js).
 *
 * The stub provides a registerAnalyticsProvider() / track*() API that
 * is a silent no-op without a provider. When a plugin registers a
 * provider, all track*() calls delegate to it.
 */
import {
  registerAnalyticsProvider,
  getAnalyticsProvider,
  trackEvent,
  trackPageView,
  trackClusterAction,
  trackJobAction,
  trackWorkspaceAction,
  trackRecipeAction,
  trackInfraAction,
  trackVolumeAction,
  trackUserAction,
  trackSettingsAction,
  trackFilterUsed,
  trackPluginPageView,
} from './analytics';

afterEach(() => {
  registerAnalyticsProvider(null);
});

describe('analytics stub', () => {
  test('all track functions are silent no-ops without a provider', () => {
    // Should not throw
    trackEvent('test');
    trackPageView('/');
    trackClusterAction('stop');
    trackJobAction('view_logs');
    trackWorkspaceAction('create');
    trackRecipeAction('view');
    trackInfraAction('refresh');
    trackVolumeAction('delete');
    trackUserAction('create');
    trackSettingsAction('save');
    trackFilterUsed('cluster');
    trackPluginPageView('test', '/path');
  });

  test('getAnalyticsProvider returns null by default', () => {
    expect(getAnalyticsProvider()).toBeNull();
  });

  test('registerAnalyticsProvider sets the provider', () => {
    const provider = { trackEvent: jest.fn() };
    registerAnalyticsProvider(provider);
    expect(getAnalyticsProvider()).toBe(provider);
  });

  test('trackEvent delegates to provider when registered', () => {
    const provider = { trackEvent: jest.fn() };
    registerAnalyticsProvider(provider);
    trackEvent('test_event', { key: 'value' });
    expect(provider.trackEvent).toHaveBeenCalledWith('test_event', {
      key: 'value',
    });
  });

  test('trackClusterAction delegates to provider', () => {
    const provider = { trackClusterAction: jest.fn() };
    registerAnalyticsProvider(provider);
    trackClusterAction('stop', { cluster: 'mycluster' });
    expect(provider.trackClusterAction).toHaveBeenCalledWith('stop', {
      cluster: 'mycluster',
    });
  });

  test('trackPageView delegates to provider', () => {
    const provider = { trackPageView: jest.fn() };
    registerAnalyticsProvider(provider);
    trackPageView('/clusters', { extra: true });
    expect(provider.trackPageView).toHaveBeenCalledWith('/clusters', {
      extra: true,
    });
  });

  test('trackFilterUsed delegates to provider', () => {
    const provider = { trackFilterUsed: jest.fn() };
    registerAnalyticsProvider(provider);
    trackFilterUsed('job', { property: 'status' });
    expect(provider.trackFilterUsed).toHaveBeenCalledWith('job', {
      property: 'status',
    });
  });

  test('trackPluginPageView delegates to provider', () => {
    const provider = { trackPluginPageView: jest.fn() };
    registerAnalyticsProvider(provider);
    trackPluginPageView('gpu_healer', '/health');
    expect(provider.trackPluginPageView).toHaveBeenCalledWith(
      'gpu_healer',
      '/health'
    );
  });

  test('unregister by passing null reverts to no-op', () => {
    const provider = { trackEvent: jest.fn() };
    registerAnalyticsProvider(provider);
    registerAnalyticsProvider(null);
    trackEvent('ignored');
    expect(provider.trackEvent).not.toHaveBeenCalled();
  });

  test('provider missing a method does not throw', () => {
    const provider = { trackEvent: jest.fn() };
    registerAnalyticsProvider(provider);
    // trackClusterAction is not on provider — should not throw
    trackClusterAction('stop');
  });
});
