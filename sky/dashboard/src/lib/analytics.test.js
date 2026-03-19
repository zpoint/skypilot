/**
 * Tests for PostHog product analytics wrapper (analytics.js).
 */

// Mock posthog-js before importing analytics module.
jest.mock('posthog-js', () => ({
  init: jest.fn(),
  identify: jest.fn(),
  register: jest.fn(),
  capture: jest.fn(),
  opt_out_capturing: jest.fn(),
}));

// Use a fresh module for each test so _initialized resets.
let analytics;
let posthog;

beforeEach(() => {
  jest.resetModules();
  jest.resetAllMocks();

  // Re-require after module reset to get fresh _initialized state.
  posthog = require('posthog-js');
  analytics = require('./analytics');
});

describe('initPostHog', () => {
  test('initializes posthog with correct config', () => {
    analytics.initPostHog();

    expect(posthog.init).toHaveBeenCalledTimes(1);
    expect(posthog.init).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({
        api_host: 'https://usage-v3.skypilot.co',
        autocapture: true,
        capture_pageview: false,
        capture_pageleave: true,
        persistence: 'localStorage',
      })
    );
  });

  test('only initializes once on multiple calls', () => {
    analytics.initPostHog();
    analytics.initPostHog();
    analytics.initPostHog();

    expect(posthog.init).toHaveBeenCalledTimes(1);
  });
});

describe('isEnabled', () => {
  test('returns false before init', () => {
    expect(analytics.isEnabled()).toBe(false);
  });

  test('returns true after init', () => {
    analytics.initPostHog();
    expect(analytics.isEnabled()).toBe(true);
  });
});

describe('identifyUser', () => {
  test('calls posthog.identify with correct args', () => {
    analytics.initPostHog();
    analytics.identifyUser('hash123', 'alice');

    expect(posthog.identify).toHaveBeenCalledWith('hash123', {
      username: 'alice',
      source: 'dashboard',
    });
  });

  test('passes extra properties', () => {
    analytics.initPostHog();
    analytics.identifyUser('hash123', 'alice', { role: 'admin' });

    expect(posthog.identify).toHaveBeenCalledWith('hash123', {
      username: 'alice',
      source: 'dashboard',
      role: 'admin',
    });
  });

  test('is a no-op before init', () => {
    analytics.identifyUser('hash123', 'alice');
    expect(posthog.identify).not.toHaveBeenCalled();
  });
});

describe('registerDeployment', () => {
  test('calls posthog.register with source', () => {
    analytics.initPostHog();
    analytics.registerDeployment({ sky_version: '1.0' });

    expect(posthog.register).toHaveBeenCalledWith({
      source: 'dashboard',
      sky_version: '1.0',
    });
  });

  test('registers plugin super properties when plugins active', () => {
    analytics.initPostHog();
    analytics.registerDeployment({
      sky_version: '1.0',
      api_version: '2',
      active_plugins: ['sidebar', 'cron'],
      plugin_count: 2,
      has_plugins: true,
    });

    expect(posthog.register).toHaveBeenCalledWith({
      source: 'dashboard',
      sky_version: '1.0',
      api_version: '2',
      active_plugins: ['sidebar', 'cron'],
      plugin_count: 2,
      has_plugins: true,
    });
  });

  test('registers empty plugin properties when no plugins', () => {
    analytics.initPostHog();
    analytics.registerDeployment({
      sky_version: '1.0',
      api_version: '2',
      active_plugins: [],
      plugin_count: 0,
      has_plugins: false,
    });

    expect(posthog.register).toHaveBeenCalledWith({
      source: 'dashboard',
      sky_version: '1.0',
      api_version: '2',
      active_plugins: [],
      plugin_count: 0,
      has_plugins: false,
    });
  });
});

describe('trackPageView', () => {
  test('captures $pageview event with path', () => {
    analytics.initPostHog();
    analytics.trackPageView('/clusters');

    expect(posthog.capture).toHaveBeenCalledWith(
      '$pageview',
      expect.objectContaining({
        path: '/clusters',
      })
    );
  });

  test('normalizes dynamic paths', () => {
    analytics.initPostHog();
    analytics.trackPageView('/clusters/my-gpu-vm');

    expect(posthog.capture).toHaveBeenCalledWith(
      '$pageview',
      expect.objectContaining({
        path: '/clusters/[cluster]',
        raw_path: '/clusters/my-gpu-vm',
      })
    );
  });

  test('deduplicates same path within 1 second', () => {
    analytics.initPostHog();
    analytics.trackPageView('/clusters');
    analytics.trackPageView('/clusters');

    expect(posthog.capture).toHaveBeenCalledTimes(1);
  });

  test('allows same path after navigating elsewhere', () => {
    analytics.initPostHog();
    analytics.trackPageView('/clusters');
    analytics.trackPageView('/jobs');
    analytics.trackPageView('/clusters');

    expect(posthog.capture).toHaveBeenCalledTimes(3);
  });
});

describe('normalizePath', () => {
  test('normalizes cluster detail paths', () => {
    expect(analytics.normalizePath('/clusters/my-gpu-vm')).toBe(
      '/clusters/[cluster]'
    );
    expect(analytics.normalizePath('/clusters/training-cluster')).toBe(
      '/clusters/[cluster]'
    );
  });

  test('normalizes cluster job paths', () => {
    expect(analytics.normalizePath('/clusters/my-cluster/job-123')).toBe(
      '/clusters/[cluster]/[job]'
    );
  });

  test('normalizes job detail paths', () => {
    expect(analytics.normalizePath('/jobs/42')).toBe('/jobs/[job]');
    expect(analytics.normalizePath('/jobs/job-abc-123')).toBe('/jobs/[job]');
  });

  test('normalizes job task paths', () => {
    expect(analytics.normalizePath('/jobs/42/3')).toBe('/jobs/[job]/[task]');
    expect(analytics.normalizePath('/jobs/job-123/task-456')).toBe(
      '/jobs/[job]/[task]'
    );
  });

  test('normalizes job pool paths', () => {
    expect(analytics.normalizePath('/jobs/pools/gpu-pool')).toBe(
      '/jobs/pools/[pool]'
    );
    expect(analytics.normalizePath('/jobs/pools/default')).toBe(
      '/jobs/pools/[pool]'
    );
  });

  test('normalizes recipe detail paths', () => {
    expect(analytics.normalizePath('/recipes/llama-serve')).toBe(
      '/recipes/[recipe]'
    );
    expect(analytics.normalizePath('/recipes/stable-diffusion')).toBe(
      '/recipes/[recipe]'
    );
  });

  test('normalizes workspace detail paths', () => {
    expect(analytics.normalizePath('/workspaces/default')).toBe(
      '/workspaces/[name]'
    );
    expect(analytics.normalizePath('/workspaces/team-a')).toBe(
      '/workspaces/[name]'
    );
  });

  test('normalizes infra context paths', () => {
    expect(analytics.normalizePath('/infra/k8s-prod')).toBe('/infra/[context]');
    expect(analytics.normalizePath('/infra/aws-us-east-1')).toBe(
      '/infra/[context]'
    );
  });

  test('normalizes plugin paths', () => {
    expect(analytics.normalizePath('/plugins/gpu_healer')).toBe(
      '/plugins/[...slug]'
    );
    expect(analytics.normalizePath('/plugins/gpu_healer/health')).toBe(
      '/plugins/[...slug]'
    );
    expect(analytics.normalizePath('/plugins/monitor/dashboard/stats')).toBe(
      '/plugins/[...slug]'
    );
  });

  test('normalizes devspace detail paths', () => {
    expect(analytics.normalizePath('/devspaces/romil-dev')).toBe(
      '/devspaces/[name]'
    );
    expect(analytics.normalizePath('/devspaces/my-workspace')).toBe(
      '/devspaces/[name]'
    );
  });

  test('leaves devspaces list path unchanged', () => {
    expect(analytics.normalizePath('/devspaces')).toBe('/devspaces');
  });

  test('leaves static routes unchanged', () => {
    expect(analytics.normalizePath('/clusters')).toBe('/clusters');
    expect(analytics.normalizePath('/jobs')).toBe('/jobs');
    expect(analytics.normalizePath('/jobs/pools')).toBe('/jobs/pools');
    expect(analytics.normalizePath('/recipes')).toBe('/recipes');
    expect(analytics.normalizePath('/workspaces')).toBe('/workspaces');
    expect(analytics.normalizePath('/infra')).toBe('/infra');
    expect(analytics.normalizePath('/settings')).toBe('/settings');
    expect(analytics.normalizePath('/settings/config')).toBe(
      '/settings/config'
    );
    expect(analytics.normalizePath('/')).toBe('/');
  });

  test('handles edge cases', () => {
    // Empty string
    expect(analytics.normalizePath('')).toBe('');
    // Root path
    expect(analytics.normalizePath('/')).toBe('/');
    // Unknown dynamic paths pass through
    expect(analytics.normalizePath('/unknown/path')).toBe('/unknown/path');
  });
});

describe('domain-specific tracking', () => {
  beforeEach(() => {
    analytics.initPostHog();
  });

  test('trackClusterAction captures cluster_action', () => {
    analytics.trackClusterAction('ssh', { cluster: 'mycluster' });

    expect(posthog.capture).toHaveBeenCalledWith('cluster_action', {
      source: 'dashboard',
      action: 'ssh',
      cluster: 'mycluster',
    });
  });

  test('trackJobAction captures job_action', () => {
    analytics.trackJobAction('view_logs', { jobId: '42' });

    expect(posthog.capture).toHaveBeenCalledWith('job_action', {
      source: 'dashboard',
      action: 'view_logs',
      jobId: '42',
    });
  });

  test('trackWorkspaceAction captures workspace_action', () => {
    analytics.trackWorkspaceAction('create');

    expect(posthog.capture).toHaveBeenCalledWith('workspace_action', {
      source: 'dashboard',
      action: 'create',
    });
  });

  test('trackRecipeAction captures recipe_action', () => {
    analytics.trackRecipeAction('view', { recipe: 'llama' });

    expect(posthog.capture).toHaveBeenCalledWith('recipe_action', {
      source: 'dashboard',
      action: 'view',
      recipe: 'llama',
    });
  });

  test('trackInfraAction captures infra_action', () => {
    analytics.trackInfraAction('refresh');

    expect(posthog.capture).toHaveBeenCalledWith('infra_action', {
      source: 'dashboard',
      action: 'refresh',
    });
  });

  test('trackVolumeAction captures volume_action', () => {
    analytics.trackVolumeAction('delete', { volume: 'my-vol' });

    expect(posthog.capture).toHaveBeenCalledWith('volume_action', {
      source: 'dashboard',
      action: 'delete',
      volume: 'my-vol',
    });
  });

  test('trackUserAction captures user_action', () => {
    analytics.trackUserAction('create');

    expect(posthog.capture).toHaveBeenCalledWith('user_action', {
      source: 'dashboard',
      action: 'create',
    });
  });

  test('trackSettingsAction captures settings_action', () => {
    analytics.trackSettingsAction('save');

    expect(posthog.capture).toHaveBeenCalledWith('settings_action', {
      source: 'dashboard',
      action: 'save',
    });
  });

  test('trackFilterUsed captures filter_used with filter_type', () => {
    analytics.trackFilterUsed('cluster', { property: 'status' });

    expect(posthog.capture).toHaveBeenCalledWith('filter_used', {
      source: 'dashboard',
      filter_type: 'cluster',
      property: 'status',
    });
  });

  test('trackPluginPageView captures plugin_page_view', () => {
    analytics.trackPluginPageView('gpu_healer', '/health');

    expect(posthog.capture).toHaveBeenCalledWith('plugin_page_view', {
      source: 'dashboard',
      plugin: 'gpu_healer',
      path: '/health',
    });
  });
});

describe('optOut', () => {
  test('disables isEnabled after init', () => {
    analytics.initPostHog();
    expect(analytics.isEnabled()).toBe(true);

    analytics.optOut();
    expect(analytics.isEnabled()).toBe(false);
  });

  test('calls posthog.opt_out_capturing when initialized', () => {
    analytics.initPostHog();
    analytics.optOut();

    expect(posthog.opt_out_capturing).toHaveBeenCalledTimes(1);
  });

  test('tracking functions are no-ops after optOut', () => {
    analytics.initPostHog();
    analytics.optOut();

    analytics.trackPageView('/clusters');
    analytics.trackEvent('test_event');
    analytics.trackClusterAction('ssh');

    expect(posthog.capture).not.toHaveBeenCalled();
  });

  test('identify and register are no-ops after optOut', () => {
    analytics.initPostHog();
    analytics.optOut();

    analytics.identifyUser('hash123', 'alice');
    analytics.registerDeployment({ sky_version: '1.0' });

    expect(posthog.identify).not.toHaveBeenCalled();
    expect(posthog.register).not.toHaveBeenCalled();
  });
});

describe('initPostHog before_send', () => {
  test('passes before_send option to posthog.init', () => {
    analytics.initPostHog();

    expect(posthog.init).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({
        before_send: expect.any(Function),
      })
    );
  });
});

describe('enrichAutocaptureEvent', () => {
  // Pass 1: standard interactive tags
  test('pass 1: finds button ancestor for svg click', () => {
    const event = {
      event: '$autocapture',
      properties: {
        $elements: [
          { tag_name: 'svg', $el_text: '' },
          {
            tag_name: 'button',
            $el_text: 'Download',
            attr__title: 'Download logs',
          },
          { tag_name: 'div', $el_text: '' },
        ],
      },
    };
    const result = analytics.enrichAutocaptureEvent(event);
    expect(result.properties.action_element).toBe('button');
    expect(result.properties.action_label).toBe('Download');
  });

  test('pass 1: uses title when button has no text', () => {
    const event = {
      event: '$autocapture',
      properties: {
        $elements: [
          { tag_name: 'svg', $el_text: '' },
          {
            tag_name: 'button',
            $el_text: '',
            attr__title: 'Request remediation',
          },
        ],
      },
    };
    const result = analytics.enrichAutocaptureEvent(event);
    expect(result.properties.action_element).toBe('button');
    expect(result.properties.action_label).toBe('Request remediation');
  });

  test('pass 1: uses placeholder for input', () => {
    const event = {
      event: '$autocapture',
      properties: {
        $elements: [
          {
            tag_name: 'input',
            $el_text: '',
            attr__placeholder: 'Filter workloads',
          },
          { tag_name: 'div', $el_text: '' },
        ],
      },
    };
    const result = analytics.enrichAutocaptureEvent(event);
    expect(result.properties.action_element).toBe('input');
    expect(result.properties.action_label).toBe('Filter workloads');
  });

  // Pass 2: interactivity signals (class patterns, aria-label, title, role)
  test('pass 2: finds tr with "clickable" class', () => {
    const event = {
      event: '$autocapture',
      properties: {
        $elements: [
          { tag_name: 'td', $el_text: 'gpu-node-01' },
          {
            tag_name: 'tr',
            $el_text: '',
            attr__class: 'clickable-node-row selected',
          },
          { tag_name: 'tbody', $el_text: '' },
        ],
      },
    };
    const result = analytics.enrichAutocaptureEvent(event);
    expect(result.properties.action_element).toBe('tr');
    expect(result.properties.action_label).toBe('clickable-node-row selected');
  });

  test('pass 2: finds span with "pill" class', () => {
    const event = {
      event: '$autocapture',
      properties: {
        $elements: [
          {
            tag_name: 'span',
            $el_text: 'H100',
            attr__class: 'gpu-obs-filter-pill search',
          },
          { tag_name: 'div', $el_text: '' },
        ],
      },
    };
    const result = analytics.enrichAutocaptureEvent(event);
    expect(result.properties.action_element).toBe('span');
    expect(result.properties.action_label).toBe('H100');
  });

  test('pass 2: finds div with "clickable" class', () => {
    const event = {
      event: '$autocapture',
      properties: {
        $elements: [
          { tag_name: 'svg', $el_text: '' },
          {
            tag_name: 'div',
            $el_text: '',
            attr__class: 'gpu-inv-header clickable',
          },
          { tag_name: 'div', $el_text: '' },
        ],
      },
    };
    const result = analytics.enrichAutocaptureEvent(event);
    expect(result.properties.action_element).toBe('div');
    expect(result.properties.action_label).toBe('gpu-inv-header clickable');
  });

  test('pass 2: finds element with aria-label', () => {
    const event = {
      event: '$autocapture',
      properties: {
        $elements: [
          { tag_name: 'svg', $el_text: '' },
          {
            tag_name: 'div',
            $el_text: '',
            attr__aria_label: 'Toggle GPU inventory',
          },
          { tag_name: 'section', $el_text: '' },
        ],
      },
    };
    const result = analytics.enrichAutocaptureEvent(event);
    expect(result.properties.action_element).toBe('div');
    expect(result.properties.action_label).toBe('Toggle GPU inventory');
  });

  test('pass 2: finds element with role attribute', () => {
    const event = {
      event: '$autocapture',
      properties: {
        $elements: [
          { tag_name: 'span', $el_text: 'ON', attr__role: 'switch' },
          { tag_name: 'div', $el_text: '' },
        ],
      },
    };
    const result = analytics.enrichAutocaptureEvent(event);
    expect(result.properties.action_element).toBe('span');
    expect(result.properties.action_label).toBe('ON');
  });

  // Pass 3: fallback to target
  test('pass 3: falls back to target when no signals found', () => {
    const event = {
      event: '$autocapture',
      properties: {
        $elements: [
          { tag_name: 'rect', $el_text: '', attr__class: 'quota-tree-node' },
          { tag_name: 'g', $el_text: '' },
          { tag_name: 'svg', $el_text: '' },
          { tag_name: 'div', $el_text: '' },
        ],
      },
    };
    const result = analytics.enrichAutocaptureEvent(event);
    expect(result.properties.action_element).toBe('rect');
    expect(result.properties.action_label).toBe('quota-tree-node');
  });

  // Priority: pass 1 wins over pass 2
  test('prefers standard interactive tag over class signal', () => {
    const event = {
      event: '$autocapture',
      properties: {
        $elements: [
          { tag_name: 'svg', $el_text: '' },
          { tag_name: 'div', $el_text: '', attr__class: 'clickable wrapper' },
          { tag_name: 'button', $el_text: 'Save' },
        ],
      },
    };
    const result = analytics.enrichAutocaptureEvent(event);
    expect(result.properties.action_element).toBe('button');
    expect(result.properties.action_label).toBe('Save');
  });

  // Edge cases
  test('ignores non-autocapture events', () => {
    const event = {
      event: 'cluster_action',
      properties: { action: 'ssh' },
    };
    const result = analytics.enrichAutocaptureEvent(event);
    expect(result.properties.action_element).toBeUndefined();
  });

  test('handles missing $elements gracefully', () => {
    const event = { event: '$autocapture', properties: {} };
    const result = analytics.enrichAutocaptureEvent(event);
    expect(result.properties.action_element).toBeUndefined();
  });

  test('handles null event', () => {
    expect(analytics.enrichAutocaptureEvent(null)).toBeNull();
  });

  test('handles elements with undefined attributes (not empty string)', () => {
    const event = {
      event: '$autocapture',
      properties: {
        $elements: [
          { tag_name: 'svg' },
          { tag_name: 'button', attr__title: 'Save changes' },
        ],
      },
    };
    const result = analytics.enrichAutocaptureEvent(event);
    expect(result.properties.action_element).toBe('button');
    expect(result.properties.action_label).toBe('Save changes');
  });
});

describe('no-ops before init', () => {
  test('all track functions are no-ops before init', () => {
    analytics.trackPageView('/clusters');
    analytics.trackEvent('test_event');
    analytics.trackClusterAction('ssh');
    analytics.trackJobAction('view_logs');
    analytics.trackWorkspaceAction('create');
    analytics.trackRecipeAction('view');
    analytics.trackInfraAction('refresh');
    analytics.trackVolumeAction('delete');
    analytics.trackUserAction('create');
    analytics.trackSettingsAction('save');
    analytics.trackFilterUsed('cluster');
    analytics.trackPluginPageView('test', '/path');
    analytics.registerDeployment({ version: '1.0' });

    expect(posthog.capture).not.toHaveBeenCalled();
    expect(posthog.register).not.toHaveBeenCalled();
  });
});
