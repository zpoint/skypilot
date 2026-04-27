"""Worker-node lifecycle-hook handler.

Runs on every worker node alongside the existing Ray worker
process. Responsibilities:

1. Register a SIGTERM handler that runs any `preemption` hooks —
   used on K8s worker pods (kubelet SIGTERM) and on VM workers
   (poller-generated SIGTERM).
2. Conditionally start the VM `preemption_poller` when the worker
   is on a cloud VM (auto-detected like on the head).

Autostop/down hooks on workers are driven by the head fan-out
(``hook_executor.fanout_to_workers``) — not from this handler — so
this module is intentionally narrow.

Hooks-config layout on workers: `~/.sky/hooks/config.json` is a
JSON list with the same shape as on the head (a plain serialization
of ``autostop_lib.get_hooks()``). It is synced at cluster launch
and on re-launch when `hooks:` changes (see
`cloud_vm_ray_backend._check_existing_cluster`).
"""

import argparse
import json
import logging
import os
import signal
import sys
from typing import Any, Dict, List

from sky.skylet import hook_executor
from sky.skylet import preemption_poller

logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.expanduser('~/.sky/hooks/config.json')
# PID file the K8s preStop hook reads to forward kubelet's SIGTERM to
# this handler (skylet isn't PID 1 in the pod; bash setup is). Must
# match the pgrep / fallback paths in
# `sky/templates/kubernetes-ray.yml.j2`.
HANDLER_PID_FILE = os.path.expanduser('~/.sky/hooks/handler_pid')


def _read_hooks_config() -> List[Dict[str, Any]]:
    """Return the hooks list stored at CONFIG_PATH, or [] if missing."""
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return []
    except FileNotFoundError:
        return []
    except Exception as e:  # pylint: disable=broad-except
        logger.warning(f'worker_hook_handler: failed to read {CONFIG_PATH}: '
                       f'{e}')
        return []


def _sigterm_handler(signum, frame):  # pylint: disable=unused-argument
    """SIGTERM → run preemption hooks via the shared CAS path."""
    logger.info('worker_hook_handler: received SIGTERM, running preemption '
                'hooks.')
    hooks = _read_hooks_config()
    if hook_executor.try_claim_teardown(hook_executor.PREEMPTION):
        try:
            hook_executor.run(hook_executor.PREEMPTION, hooks)
        except Exception as e:  # pylint: disable=broad-except
            logger.warning(f'worker_hook_handler: preemption hook failed: '
                           f'{e}')
    try:
        os.unlink(HANDLER_PID_FILE)
    except FileNotFoundError:
        pass
    except OSError as e:  # pylint: disable=broad-except
        logger.debug(f'worker_hook_handler: failed to remove PID file: {e}')
    sys.exit(0)


def main():
    """Long-lived worker-side hook handler process.

    Registers the SIGTERM handler, optionally starts the cloud
    preemption poller, and then sleeps forever waiting for signals.
    The handler is expected to be started via `nohup` at cluster
    provision time (see `sky.provision.instance_setup
    .start_worker_hook_handler`).
    """
    parser = argparse.ArgumentParser(
        description='Worker-node lifecycle hook handler')
    parser.add_argument('--cloud',
                        default=None,
                        help='Cloud identity for the preemption poller. '
                        'Auto-detected if omitted.')
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _sigterm_handler)
    hook_executor.clear_teardown_claim()

    # Publish our PID so the K8s preStop hook can `kill -TERM` us.
    # Best-effort: a failed write only matters on K8s, where the
    # absence of the file falls back to pgrep in the preStop body.
    try:
        os.makedirs(os.path.dirname(HANDLER_PID_FILE), exist_ok=True)
        with open(HANDLER_PID_FILE, 'w', encoding='utf-8') as f:
            f.write(str(os.getpid()))
    except OSError as e:
        logger.warning(
            f'worker_hook_handler: failed to write PID file '
            f'{HANDLER_PID_FILE}: {e}; preStop will fall back to pgrep.')

    cloud = args.cloud
    if cloud is None:
        # Lazy import to avoid starting the poller when imported as a
        # library for unit tests.
        from sky.skylet import skylet  # pylint: disable=import-outside-toplevel
        cloud = skylet._detect_cloud_for_preemption_poller()  # pylint: disable=protected-access
    if cloud is not None:
        logger.info(
            f'worker_hook_handler: starting VM preemption poller ({cloud})')
        preemption_poller.start(cloud)
    else:
        logger.info(
            'worker_hook_handler: no cloud metadata endpoint; no poller.')

    # Block forever; SIGTERM will wake and exit via the handler.
    signal.pause()


if __name__ == '__main__':
    main()
