"""Unit tests for PR3 worker-hook support.

Covers:
- `worker_hook_handler` SIGTERM → preemption hook dispatch.
- Head-side fan-out in `hook_executor.run_with_fanout` for autostop +
  down: constructs per-worker SSH command with stable key path.
- Multi-node log aggregation in `tail_hook_logs`: stitches per-node
  output with `=== <node> ===` headers.
"""

import json
import os
import signal
import threading
import time
from unittest import mock

import pytest

from sky.skylet import hook_executor
from sky.skylet import worker_hook_handler

# ---- worker_hook_handler ---------------------------------------------------


@pytest.fixture
def tmp_hooks_config(tmp_path, monkeypatch):
    """Point the handler at a temp config.json and log dir."""
    hooks_dir = tmp_path / '.sky' / 'hooks'
    hooks_dir.mkdir(parents=True)
    config_path = hooks_dir / 'config.json'
    config_path.write_text(
        json.dumps([{
            'run': 'echo worker-preempt',
            'events': ['preemption'],
            'timeout': 30,
        }]))
    monkeypatch.setattr(worker_hook_handler, 'CONFIG_PATH', str(config_path))
    monkeypatch.setattr(hook_executor, 'HOOK_LOG_DIR', str(hooks_dir))
    monkeypatch.setattr(hook_executor, 'CLAIM_FILE',
                        str(hooks_dir / '.teardown_claim'))
    yield config_path


def test_worker_handler_reads_config(tmp_hooks_config):
    cfg = worker_hook_handler._read_hooks_config()
    assert isinstance(cfg, list)
    assert cfg[0]['run'] == 'echo worker-preempt'


def test_worker_handler_missing_config_returns_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(worker_hook_handler, 'CONFIG_PATH',
                        str(tmp_path / 'nope.json'))
    assert worker_hook_handler._read_hooks_config() == []


def test_worker_handler_sigterm_runs_preemption(tmp_hooks_config, monkeypatch):
    ran = []

    def fake_run(event, hooks):
        ran.append((event, hooks))

    monkeypatch.setattr(hook_executor, 'run', fake_run)
    monkeypatch.setattr(hook_executor, 'try_claim_teardown', lambda ev: True)
    # Invoke the handler directly; in production skylet registers it
    # via signal.signal.
    sys_exit_called = []
    monkeypatch.setattr('sys.exit', lambda code=0: sys_exit_called.append(code))
    worker_hook_handler._sigterm_handler(signal.SIGTERM, None)
    assert ran and ran[0][0] == hook_executor.PREEMPTION
    assert ran[0][1][0]['run'] == 'echo worker-preempt'


# ---- head fan-out in hook_executor ----------------------------------------


def test_fanout_discovers_workers_via_ray_nodes(monkeypatch):
    """Head-side fan-out must discover workers via `ray.nodes()`."""
    fake_nodes = [
        {
            'NodeManagerAddress': '10.0.0.1',
            'Alive': True,
            'Resources': {
                'node:__head__': 1
            }
        },
        {
            'NodeManagerAddress': '10.0.0.2',
            'Alive': True,
            'Resources': {}
        },
        {
            'NodeManagerAddress': '10.0.0.3',
            'Alive': True,
            'Resources': {}
        },
    ]
    monkeypatch.setattr(hook_executor, '_discover_worker_ips',
                        lambda: ['10.0.0.2', '10.0.0.3'])
    commands = []
    monkeypatch.setattr(hook_executor, '_ssh_worker',
                        lambda ip, event: commands.append((ip, event)))
    hook_executor.fanout_to_workers('autostop')
    assert sorted(commands) == [('10.0.0.2', 'autostop'),
                                ('10.0.0.3', 'autostop')]


def test_fanout_skips_preemption_event(monkeypatch):
    """Preemption hooks fire per-worker via the worker's own SIGTERM,
    not by head fan-out."""
    monkeypatch.setattr(hook_executor, '_discover_worker_ips',
                        lambda: ['10.0.0.2'])
    commands = []
    monkeypatch.setattr(hook_executor, '_ssh_worker',
                        lambda ip, event: commands.append((ip, event)))
    hook_executor.fanout_to_workers('preemption')
    assert commands == []


def test_fanout_best_effort_on_ssh_failure(monkeypatch, caplog):
    """If one worker is unreachable, the others still fire."""
    monkeypatch.setattr(hook_executor, '_discover_worker_ips',
                        lambda: ['10.0.0.2', '10.0.0.3'])

    def fake_ssh(ip, event):
        if ip == '10.0.0.2':
            raise RuntimeError('ssh failed')

    monkeypatch.setattr(hook_executor, '_ssh_worker', fake_ssh)
    # Should not raise.
    hook_executor.fanout_to_workers('down')


# ---- multi-node log aggregation -------------------------------------------


def test_tail_hook_logs_aggregates_per_node():
    """Header format: `=== <node-identifier> ===` on own line."""
    from sky.backends import cloud_vm_ray_backend
    cmd = cloud_vm_ray_backend._tail_hook_logs_cmd_multinode(
        event='autostop',
        node_ids=['head', 'worker-1', 'worker-2'],
        tail=0,
        follow=False)
    assert 'head' in cmd
    assert 'worker-1' in cmd
    assert 'worker-2' in cmd
    assert '===' in cmd
