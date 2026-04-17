/**
 * Tests for analytics stub (analytics.js).
 *
 * The provider contract is just { trackEvent, trackPageView }. Domain
 * helpers (trackClusterAction, etc.) fan in to trackEvent with a
 * hard-coded event name and the caller's properties.
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

  test('trackPageView delegates to provider', () => {
    const provider = { trackPageView: jest.fn() };
    registerAnalyticsProvider(provider);
    trackPageView('/clusters', { extra: true });
    expect(provider.trackPageView).toHaveBeenCalledWith('/clusters', {
      extra: true,
    });
  });

  describe('domain helpers fan in to trackEvent', () => {
    const cases = [
      ['trackClusterAction', trackClusterAction, 'cluster_action', 'stop'],
      ['trackJobAction', trackJobAction, 'job_action', 'view_logs'],
      [
        'trackWorkspaceAction',
        trackWorkspaceAction,
        'workspace_action',
        'create',
      ],
      ['trackRecipeAction', trackRecipeAction, 'recipe_action', 'view'],
      ['trackInfraAction', trackInfraAction, 'infra_action', 'refresh'],
      ['trackVolumeAction', trackVolumeAction, 'volume_action', 'delete'],
      ['trackUserAction', trackUserAction, 'user_action', 'create'],
      ['trackSettingsAction', trackSettingsAction, 'settings_action', 'save'],
    ];

    test.each(cases)(
      '%s → trackEvent(%s)',
      (_name, fn, expectedEvent, action) => {
        const provider = { trackEvent: jest.fn() };
        registerAnalyticsProvider(provider);
        fn(action, { extra: 1 });
        expect(provider.trackEvent).toHaveBeenCalledWith(expectedEvent, {
          action,
          extra: 1,
        });
      }
    );
  });

  test('trackFilterUsed fans in to trackEvent with filter_type', () => {
    const provider = { trackEvent: jest.fn() };
    registerAnalyticsProvider(provider);
    trackFilterUsed('job', { property: 'status' });
    expect(provider.trackEvent).toHaveBeenCalledWith('filter_used', {
      filter_type: 'job',
      property: 'status',
    });
  });

  test('trackPluginPageView fans in to trackEvent', () => {
    const provider = { trackEvent: jest.fn() };
    registerAnalyticsProvider(provider);
    trackPluginPageView('gpu_healer', '/health');
    expect(provider.trackEvent).toHaveBeenCalledWith('plugin_page_view', {
      plugin: 'gpu_healer',
      path: '/health',
    });
  });

  test('unregister by passing null reverts to no-op', () => {
    const provider = { trackEvent: jest.fn() };
    registerAnalyticsProvider(provider);
    registerAnalyticsProvider(null);
    trackEvent('ignored');
    expect(provider.trackEvent).not.toHaveBeenCalled();
  });

  test('provider missing trackEvent does not throw', () => {
    // Provider implementing only trackPageView — domain helpers still safe
    const provider = { trackPageView: jest.fn() };
    registerAnalyticsProvider(provider);
    trackClusterAction('stop');
    trackEvent('x');
  });
});
