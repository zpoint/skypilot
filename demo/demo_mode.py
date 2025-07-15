"""Demo mode for SkyPilot API server with pre-populated fake data.

This module monkey-patches global_user_state functions and server endpoints
to provide a read-only demo experience with realistic fake data.

The demo data is loaded from multiple JSON5 files in the demo/mock_data/ directory:
- mock_data/mock_users.json5: User accounts and roles
- mock_data/mock_clusters.json5: Cluster configurations and YAML configs  
- mock_data/mock_jobs.json5: Managed jobs data
- mock_data/mock_cluster_jobs.json5: Cluster-specific jobs
- mock_data/mock_volumes.json5: Volume and storage data
- mock_data/mock_workspaces.json5: Workspace configurations
- mock_data/mock_infrastructure.json5: Infrastructure data (clouds, GPUs, nodes)

Log files are stored in demo/logs/ directory with realistic content for each job.

All files support JSON5 format with comments for easy editing.

HOT RELOADING: The JSON5 files are loaded fresh on every API query, enabling
real-time editing of mock data without server restarts. Simply edit any
JSON5 file and refresh your browser to see changes immediately!
"""

import copy
import json5
import uuid
import multiprocessing
import os
import time
from typing import Any, Dict, List, Optional, Set

import fastapi

from sky import clouds
from sky import global_user_state
from sky import models
from sky import resources as resources_lib
from sky import sky_logging
from sky.backends.cloud_vm_ray_backend import CloudVmRayResourceHandle
from sky.jobs.state import ManagedJobStatus, ManagedJobScheduleState
from sky.skylet import constants
from sky.skylet import job_lib
from sky.utils import common_utils
from sky.utils import status_lib

logger = sky_logging.init_logger(__name__)

# Demo data timestamps
CURRENT_TIME = int(time.time())

# Path to demo directory containing mock data files
DEMO_DIR = os.path.dirname(__file__)

# Mock data file paths
MOCK_DATA_FILES = {
    'users': os.path.join(DEMO_DIR, 'mock_data', 'mock_users.json5'),
    'clusters': os.path.join(DEMO_DIR, 'mock_data', 'mock_clusters.json5'),
    'jobs': os.path.join(DEMO_DIR, 'mock_data', 'mock_jobs.json5'),
    'cluster_jobs': os.path.join(DEMO_DIR, 'mock_data',
                                 'mock_cluster_jobs.json5'),
    'volumes': os.path.join(DEMO_DIR, 'mock_data', 'mock_volumes.json5'),
    'workspaces': os.path.join(DEMO_DIR, 'mock_data', 'mock_workspaces.json5'),
    'infrastructure': os.path.join(DEMO_DIR, 'mock_data',
                                   'mock_infrastructure.json5'),
}


def load_mock_data_file(data_type: str):
    """Load mock data from a specific JSON5 file with comments support."""
    file_path = MOCK_DATA_FILES.get(data_type)
    if not file_path:
        raise ValueError(f"Unknown data type: {data_type}")

    try:
        with open(file_path, 'r') as f:
            return json5.load(f)
    except FileNotFoundError:
        logger.error(f"Mock data file not found: {file_path}")
        raise
    except json5.JSONError as e:
        logger.error(f"Invalid JSON5 in mock data file {file_path}: {e}")
        raise


def reload_mock_data():
    """Reload mock data from JSON5 files for hot reloading.
    
    This function is now deprecated since data is loaded fresh on every query.
    It's kept for backward compatibility.
    """
    logger.info(
        "Mock data is now loaded fresh on every query - reload_mock_data() is deprecated"
    )

    # Test loading to ensure all JSON5 files are valid
    try:
        users_data = load_mock_data_file('users')
        clusters_data = load_mock_data_file('clusters')
        jobs_data = load_mock_data_file('jobs')
        cluster_jobs_data = load_mock_data_file('cluster_jobs')
        volumes_data = load_mock_data_file('volumes')
        workspaces_data = load_mock_data_file('workspaces')
        infrastructure_data = load_mock_data_file('infrastructure')

        users = _convert_users_from_json(users_data)
        clusters = _convert_clusters_from_json(clusters_data)
        jobs = _convert_jobs_from_json(jobs_data)
        cluster_jobs = _convert_cluster_jobs_from_json(cluster_jobs_data)
        volumes = _convert_volumes_from_json(volumes_data)

        logger.info(
            f"Mock data validated successfully - {len(users)} users, {len(clusters)} clusters, {len(jobs)} jobs, {len(cluster_jobs)} cluster jobs, {len(volumes)} volumes"
        )
    except Exception as e:
        logger.error(f"Failed to reload mock data: {e}")
        raise


def _convert_timestamp_offset(offset):
    """Convert timestamp offset to absolute timestamp."""
    if offset is None:
        return None
    if offset == -1:
        return -1
    return CURRENT_TIME + offset


def _get_cloud_from_string(cloud_str):
    """Convert cloud string to cloud object."""
    if cloud_str.lower() == 'gcp':
        return clouds.GCP()
    elif cloud_str.lower() == 'aws':
        return clouds.AWS()
    elif cloud_str.lower() == 'kubernetes':
        return clouds.Kubernetes()
    else:
        raise ValueError(f"Unknown cloud: {cloud_str}")


def _get_status_from_string(status_str):
    """Convert status string to status enum."""
    if hasattr(status_lib.ClusterStatus, status_str):
        return getattr(status_lib.ClusterStatus, status_str)
    else:
        raise ValueError(f"Unknown cluster status: {status_str}")


def _get_volume_status_from_string(status_str):
    """Convert volume status string to volume status enum."""
    if hasattr(status_lib.VolumeStatus, status_str):
        return getattr(status_lib.VolumeStatus, status_str)
    else:
        raise ValueError(f"Unknown volume status: {status_str}")


def _get_job_status_from_string(status_str):
    """Convert job status string to job status enum."""
    if hasattr(ManagedJobStatus, status_str):
        return getattr(ManagedJobStatus, status_str)
    else:
        raise ValueError(f"Unknown job status: {status_str}")


def _get_cluster_job_status_from_string(status_str):
    """Convert cluster job status string to cluster job status enum."""
    if hasattr(job_lib.JobStatus, status_str):
        return getattr(job_lib.JobStatus, status_str)
    else:
        raise ValueError(f"Unknown cluster job status: {status_str}")


def _get_schedule_state_from_string(state_str):
    """Convert schedule state string to schedule state enum."""
    if hasattr(ManagedJobScheduleState, state_str):
        return getattr(ManagedJobScheduleState, state_str)
    else:
        raise ValueError(f"Unknown schedule state: {state_str}")


def _convert_users_from_json(users_data):
    """Convert users from JSON data to User objects."""
    users = []
    for user_data in users_data['users']:
        user = models.User(id=user_data['id'],
                           name=user_data['name'],
                           password=user_data['password'],
                           created_at=_convert_timestamp_offset(
                               user_data['created_at_offset']))
        users.append(user)
    return users


def _convert_clusters_from_json(clusters_data):
    """Convert clusters from JSON data to cluster objects."""
    clusters = []

    for cluster_data in clusters_data['clusters']:
        # Load the YAML config from the cluster_yamls directory
        yaml_config_name = cluster_data['last_creation_yaml']
        if yaml_config_name is None:
            yaml_config = None
        else:
            yaml_config_path = os.path.join(os.path.dirname(__file__),
                                            'cluster_yamls',
                                            f'{yaml_config_name}.yaml')

            try:
                with open(yaml_config_path, 'r') as f:
                    yaml_config = f.read()
            except FileNotFoundError:
                logger.warning(
                    f"Demo mode: YAML config file not found: {yaml_config_path}"
                )
                yaml_config = f"# YAML config '{yaml_config_name}' not found"
            except Exception as e:
                logger.warning(
                    f"Demo mode: Error reading YAML config file {yaml_config_path}: {e}"
                )
                yaml_config = f"# Error reading YAML config '{yaml_config_name}'"

        cluster = {
            'name': cluster_data['name'],
            'launched_at': _convert_timestamp_offset(
                cluster_data['launched_at_offset']),
            'handle': _create_demo_handle(
                cluster_data['name'],
                _get_cloud_from_string(cluster_data['cloud']),
                cluster_data['region'], cluster_data['nodes'],
                cluster_data.get('resources')),
            'last_use': cluster_data['last_use'],
            'status': _get_status_from_string(cluster_data['status']),
            'autostop': cluster_data['autostop'],
            'to_down': cluster_data['to_down'],
            'owner': cluster_data['owner'],
            'metadata': cluster_data['metadata'],
            'cluster_hash': cluster_data['cluster_hash'],
            'storage_mounts_metadata': cluster_data['storage_mounts_metadata'],
            'cluster_ever_up': cluster_data['cluster_ever_up'],
            'status_updated_at': _convert_timestamp_offset(
                cluster_data['status_updated_at_offset']),
            'user_hash': cluster_data['user_hash'],
            'user_name': cluster_data['user_name'],
            'config_hash': cluster_data['config_hash'],
            'workspace': cluster_data['workspace'],
            'last_creation_yaml': yaml_config,
            'last_creation_command': cluster_data['last_creation_command'],
        }
        clusters.append(cluster)
    return clusters


def _convert_volumes_from_json(volumes_data):
    """Convert volumes from JSON data to volume objects."""
    volumes = []
    for volume_data in volumes_data['volumes']:
        volume = {
            'name': volume_data['name'],
            'type': volume_data['type'],
            'launched_at': _convert_timestamp_offset(
                volume_data['launched_at_offset']),
            'cloud': volume_data['cloud'],
            'region': volume_data['region'],
            'zone': volume_data['zone'],
            'size': volume_data['size'],
            'config': volume_data['config'],
            'name_on_cloud': volume_data['name_on_cloud'],
            'user_hash': volume_data['user_hash'],
            'user_name': volume_data['user_name'],
            'workspace': volume_data['workspace'],
            'last_attached_at': _convert_timestamp_offset(
                volume_data['last_attached_at_offset']),
            'last_use': volume_data['last_use'],
            'status': _get_volume_status_from_string(volume_data['status']),
            'usedby_clusters': volume_data['usedby_clusters'],
            'usedby_pods': volume_data['usedby_pods'],
        }
        volumes.append(volume)
    return volumes


def _convert_jobs_from_json(jobs_data):
    """Convert jobs from JSON data to job objects."""
    jobs = []
    for job_data in jobs_data['jobs']:
        job = {
            'job_id': job_data['job_id'],
            'task_id': job_data['task_id'],
            'job_name': job_data['job_name'],
            'task_name': job_data['task_name'],
            'status': _get_job_status_from_string(job_data['status']),
            'submitted_at': _convert_timestamp_offset(
                job_data['submitted_at_offset']),
            'start_at': _convert_timestamp_offset(
                job_data.get('start_at_offset')) if 'start_at_offset'
                        in job_data else job_data.get('start_at'),
            'end_at': _convert_timestamp_offset(job_data.get('end_at_offset'))
                      if 'end_at_offset' in job_data else
                      job_data.get('end_at'),
            'job_duration': job_data['job_duration'],
            'cluster_resources': job_data['cluster_resources'],
            'cluster_resources_full': job_data['cluster_resources_full'],
            'region': job_data['region'],
            'cloud': job_data['cloud'],
            'zone': job_data['zone'],
            'user_name': job_data['user_name'],
            'user_hash': job_data['user_hash'],
            'workspace': job_data['workspace'],
            'resources': job_data['resources'],
            'recovery_count': job_data['recovery_count'],
            'failure_reason': job_data['failure_reason'],
            'schedule_state': job_data['schedule_state'],
            'specs': job_data['specs'],
            'entrypoint': job_data['entrypoint'],
            'run_timestamp': str(
                _convert_timestamp_offset(job_data['run_timestamp_offset'])),
            'last_recovered_at': _convert_timestamp_offset(
                job_data['last_recovered_at_offset']),
        }
        jobs.append(job)
    jobs.sort(key=lambda x: x['job_id'], reverse=True)
    return jobs


def _convert_cluster_jobs_from_json(cluster_jobs_data):
    """Convert cluster jobs from JSON data to cluster job objects."""
    cluster_jobs = []
    for job_data in cluster_jobs_data['cluster_jobs']:
        job = {
            'job_id': job_data['job_id'],
            'job_name': job_data['job_name'],
            'username': job_data['username'],
            'user_hash': job_data['user_hash'],
            'submitted_at': _convert_timestamp_offset(
                job_data['submitted_at_offset']),
            'start_at': _convert_timestamp_offset(
                job_data.get('start_at_offset')) if 'start_at_offset'
                        in job_data else job_data.get('start_at'),
            'end_at': _convert_timestamp_offset(job_data.get('end_at_offset'))
                      if 'end_at_offset' in job_data else
                      job_data.get('end_at'),
            'resources': job_data['resources'],
            'status': _get_cluster_job_status_from_string(job_data['status']),
            'log_path': job_data['log_path'],
            'cluster_name': job_data['cluster_name'],
        }
        cluster_jobs.append(job)
    cluster_jobs.sort(key=lambda x: x['job_id'], reverse=True)
    return cluster_jobs


# Helper function to create demo cluster handles
def _create_demo_handle(
    cluster_name: str,
    cloud: clouds.Cloud,
    region: str,
    nodes: int = 1,
    resources_config: Optional[Dict[str,
                                    Any]] = None) -> CloudVmRayResourceHandle:
    """Creates a realistic CloudVmRayResourceHandle for demo purposes."""
    # Use resources from cluster configuration if provided
    if resources_config:
        instance_type = resources_config.get('instance_type', 'n1-standard-4')
        accelerators = resources_config.get('accelerators')
        cpus = resources_config.get('cpus')
        memory = resources_config.get('memory')
    else:
        # Fallback to old hard-coded logic if no resources provided
        if 'kubernetes' in str(cloud).lower():
            instance_type = '4CPU--16GB'
            accelerators = None
            cpus = 4
            memory = 16
        elif 'multinode' in cluster_name:
            instance_type = 'n1-standard-8'
            accelerators = {'A100': 8}
            cpus = None
            memory = None
        elif 'inference' in cluster_name:
            instance_type = 'g4dn.xlarge'
            accelerators = {'A100': 1}
            cpus = None
            memory = None
        else:
            instance_type = 'n1-standard-4'
            accelerators = None
            cpus = None
            memory = None

    # Create Resources object
    resources = resources_lib.Resources(cloud=cloud,
                                        region=region,
                                        instance_type=instance_type,
                                        accelerators=accelerators,
                                        cpus=cpus,
                                        memory=memory,
                                        use_spot=False)

    # Generate fake IPs
    base_ip = 10 + hash(
        cluster_name) % 200  # Generate deterministic but varied IPs
    internal_ips = [f'10.0.{base_ip}.{i+1}' for i in range(nodes)]
    external_ips = [f'34.{base_ip}.{base_ip}.{i+1}' for i in range(nodes)]
    stable_internal_external_ips = list(zip(internal_ips, external_ips))
    stable_ssh_ports = [22] * nodes

    return CloudVmRayResourceHandle(
        cluster_name=cluster_name,
        cluster_name_on_cloud=
        f'{cluster_name}-{common_utils.get_user_hash()[:8]}',
        cluster_yaml=None,  # No actual YAML file for demo
        launched_nodes=nodes,
        launched_resources=resources,
        stable_internal_external_ips=stable_internal_external_ips,
        stable_ssh_ports=stable_ssh_ports)


# All demo data is now loaded fresh from JSON on each query for hot reloading


# Mock functions to replace global_user_state functions
def mock_get_clusters() -> List[Dict[str, Any]]:
    """Mock implementation of get_clusters."""
    # Load fresh data from JSON for hot reloading
    clusters_data = load_mock_data_file('clusters')
    result = _convert_clusters_from_json(clusters_data)

    cluster_names = [c['name'] for c in result]
    logger.info(
        f"Demo mode: mock_get_clusters() returning {len(result)} clusters: {cluster_names}"
    )

    # Debug: log the structure of the first cluster for debugging
    if result:
        first_cluster = result[0]
        logger.info(
            f"Demo mode: First cluster structure - name: '{first_cluster.get('name')}', keys: {list(first_cluster.keys())}"
        )

    return result


def mock_get_cluster_from_name(
        cluster_name: Optional[str]) -> Optional[Dict[str, Any]]:
    """Mock implementation of get_cluster_from_name."""
    logger.info(
        f"Demo mode: mock_get_cluster_from_name() called with cluster_name='{cluster_name}'"
    )

    if cluster_name is None:
        return None

    # Load fresh data from JSON for hot reloading
    clusters_data = load_mock_data_file('clusters')
    clusters = _convert_clusters_from_json(clusters_data)

    for cluster in clusters:
        if cluster['name'] == cluster_name:
            logger.info(
                f"Demo mode: Found cluster '{cluster_name}' in demo data")
            return cluster

    logger.warning(
        f"Demo mode: Cluster '{cluster_name}' not found in demo data. Available clusters: {[c['name'] for c in clusters]}"
    )
    return None


def mock_get_all_users() -> List[models.User]:
    """Mock implementation of get_all_users."""
    # Load fresh data from JSON for hot reloading
    users_data = load_mock_data_file('users')
    return _convert_users_from_json(users_data)


def mock_get_user(user_id: str) -> Optional[models.User]:
    """Mock implementation of get_user."""
    # Load fresh data from JSON for hot reloading
    users_data = load_mock_data_file('users')
    users = _convert_users_from_json(users_data)

    for user in users:
        if user.id == user_id:
            return user
    return None


def mock_get_storage() -> List[Dict[str, Any]]:
    """Mock implementation of get_storage."""
    # Load fresh data from JSON for hot reloading
    volumes_data = load_mock_data_file('volumes')
    return volumes_data['storage'].copy()


def mock_get_volumes() -> List[Dict[str, Any]]:
    """Mock implementation of get_volumes."""
    # Load fresh data from JSON for hot reloading
    volumes_data = load_mock_data_file('volumes')
    return _convert_volumes_from_json(volumes_data)


def mock_get_cached_enabled_clouds(cloud_capability, workspace: str) -> List:
    """Mock implementation of get_cached_enabled_clouds."""
    # Load fresh data from JSON for hot reloading
    infrastructure_data = load_mock_data_file('infrastructure')
    enabled_clouds = infrastructure_data['enabled_clouds']

    # Return actual cloud objects for demo
    mock_clouds = []
    if 'gcp' in enabled_clouds:
        mock_clouds.append(clouds.GCP())
    if 'aws' in enabled_clouds:
        mock_clouds.append(clouds.AWS())
    return mock_clouds


def mock_get_cluster_names_start_with(starts_with: str) -> List[str]:
    """Mock implementation of get_cluster_names_start_with."""
    # Load fresh data from JSON for hot reloading
    clusters_data = load_mock_data_file('clusters')
    clusters = _convert_clusters_from_json(clusters_data)

    return [
        cluster['name']
        for cluster in clusters
        if cluster['name'].startswith(starts_with)
    ]


def mock_get_glob_cluster_names(cluster_name: str) -> List[str]:
    """Mock implementation of get_glob_cluster_names."""
    import fnmatch
    logger.info(
        f"Demo mode: mock_get_glob_cluster_names() called with pattern='{cluster_name}'"
    )

    # Load fresh data from JSON for hot reloading
    clusters_data = load_mock_data_file('clusters')
    clusters = _convert_clusters_from_json(clusters_data)

    # Get all cluster names from demo data
    all_cluster_names = [cluster['name'] for cluster in clusters]

    # Apply glob pattern matching
    matching_names = [
        name for name in all_cluster_names
        if fnmatch.fnmatch(name, cluster_name)
    ]

    logger.info(
        f"Demo mode: Pattern '{cluster_name}' matched clusters: {matching_names}"
    )
    return matching_names


def mock_get_storage_names_start_with(starts_with: str) -> List[str]:
    """Mock implementation of get_storage_names_start_with."""
    # Load fresh data from JSON for hot reloading
    volumes_data = load_mock_data_file('volumes')
    storage = volumes_data['storage']

    return [
        storage_item['name']
        for storage_item in storage
        if storage_item['name'].startswith(starts_with)
    ]


def mock_get_volume_names_start_with(starts_with: str) -> List[str]:
    """Mock implementation of get_volume_names_start_with."""
    # Load fresh data from JSON for hot reloading
    volumes_data = load_mock_data_file('volumes')
    volumes = _convert_volumes_from_json(volumes_data)

    return [
        volume['name']
        for volume in volumes
        if volume['name'].startswith(starts_with)
    ]


# Infrastructure-related mock functions
def mock_enabled_clouds(workspace: Optional[str] = None,
                        expand: bool = False) -> List[str]:
    """Mock implementation of core.enabled_clouds."""
    logger.info(
        f"Demo mode: mock_enabled_clouds() called with workspace={workspace}, expand={expand}"
    )

    # Load fresh data from JSON for hot reloading
    infrastructure_data = load_mock_data_file('infrastructure')
    enabled_clouds = infrastructure_data['enabled_clouds']

    if not expand:
        return enabled_clouds.copy()
    else:
        # Return expanded infrastructure names
        # For demo purposes, just return the basic cloud names since we don't need complex infra expansion
        return enabled_clouds.copy()


def mock_get_all_contexts() -> List[str]:
    """Mock implementation of core.get_all_contexts."""
    logger.info("Demo mode: mock_get_all_contexts() called")

    # Load fresh data from JSON for hot reloading
    infrastructure_data = load_mock_data_file('infrastructure')

    # Return the contexts from our demo node info (these are the available Kubernetes contexts)
    contexts = list(infrastructure_data['node_info'].keys())

    logger.info(f"Demo mode: Returning {len(contexts)} contexts: {contexts}")
    return contexts


def mock_realtime_kubernetes_gpu_availability(
        context: Optional[str] = None,
        name_filter: Optional[str] = None,
        quantity_filter: Optional[int] = None,
        is_ssh: Optional[bool] = None):
    """Mock implementation of core.realtime_kubernetes_gpu_availability."""
    logger.info(
        f"Demo mode: mock_realtime_kubernetes_gpu_availability() called with context={context}, name_filter={name_filter}, quantity_filter={quantity_filter}, is_ssh={is_ssh}"
    )

    # Import models here to avoid circular imports
    from sky import models

    # Load fresh data from JSON for hot reloading
    infrastructure_data = load_mock_data_file('infrastructure')
    demo_gpu_data = infrastructure_data['gpu_availability']

    # Build the response as list of tuples: (context_name, list_of_gpu_availability)
    result = []

    contexts_to_process = []
    if context is None:
        # Return all contexts
        contexts_to_process = list(demo_gpu_data.keys())
    else:
        # Return specific context if it exists in our demo data
        if context in demo_gpu_data:
            contexts_to_process = [context]

    for ctx in contexts_to_process:
        gpu_availability_list = []
        for gpu_info in demo_gpu_data[ctx]:
            gpu_name = gpu_info['gpu_name']
            requestable_qtys = gpu_info['requestable_qtys']
            total_capacity = gpu_info['total_capacity']
            available = gpu_info['available']

            # Apply name filter
            if name_filter and name_filter.lower() not in gpu_name.lower():
                continue

            # Apply quantity filter
            if quantity_filter and quantity_filter not in requestable_qtys:
                continue

            # Create RealtimeGpuAvailability object
            gpu_availability = models.RealtimeGpuAvailability(
                gpu=gpu_name,
                counts=requestable_qtys,
                capacity=total_capacity,
                available=available)
            gpu_availability_list.append(gpu_availability)

        if gpu_availability_list:  # Only add context if it has GPUs
            result.append((ctx, gpu_availability_list))

    logger.info(
        f"Demo mode: Returning GPU availability for {len(result)} contexts")
    return result


def mock_kubernetes_node_info(context: Optional[str] = None):
    """Mock implementation of kubernetes_utils.get_kubernetes_node_info."""
    logger.info(
        f"Demo mode: mock_kubernetes_node_info() called with context={context}")

    # Import models here to avoid circular imports
    from sky import models

    # Load fresh data from JSON for hot reloading
    infrastructure_data = load_mock_data_file('infrastructure')
    demo_node_data = infrastructure_data['node_info']

    # Get node data for the specified context, or return empty if context not found
    node_info_dict = {}
    if context and context in demo_node_data:
        for node_name, node_data in demo_node_data[context].items():
            node_info_dict[node_name] = models.KubernetesNodeInfo(
                name=node_data['name'],
                accelerator_type=node_data['accelerator_type'],
                total=node_data['total'],
                free=node_data['free'],
                ip_address=node_data['ip_address'],
            )

    # Create KubernetesNodesInfo response
    result = models.KubernetesNodesInfo(
        node_info_dict=node_info_dict,
        hint=None  # No specific hints for demo
    )

    logger.info(
        f"Demo mode: Returning node info for context '{context}' with {len(node_info_dict)} nodes"
    )
    return result


def mock_volume_list() -> List[Dict[str, Any]]:
    """Mock implementation of volumes.server.core.volume_list."""
    logger.info("Demo mode: mock_volume_list() called")

    # Load fresh data from JSON for hot reloading
    volumes_data = load_mock_data_file('volumes')
    volumes = _convert_volumes_from_json(volumes_data)

    # Convert VolumeStatus enum objects to string values for JSON serialization
    for volume in volumes:
        if 'status' in volume and hasattr(volume['status'], 'value'):
            volume['status'] = volume['status'].value

    logger.info(f"Demo mode: Returning {len(volumes)} demo volumes")
    return volumes


def mock_get_workspaces() -> Dict[str, Any]:
    """Mock implementation of workspaces.core.get_workspaces."""
    logger.info("Demo mode: mock_get_workspaces() called")

    # Load fresh data from JSON for hot reloading
    workspaces_data = load_mock_data_file('workspaces')
    workspaces = copy.deepcopy(workspaces_data['workspaces'])

    logger.info(
        f"Demo mode: Returning {len(workspaces)} demo workspaces: {list(workspaces.keys())}"
    )
    return workspaces


# Read-only operation handlers - these will no-op or return appropriate responses
def mock_readonly_operation(*args, **kwargs):
    """Generic handler for read-only mode - raises 403 Forbidden."""
    raise fastapi.HTTPException(
        status_code=403, detail="Demo mode: Write operations are not allowed")


# Volume delete and apply operations are now handled by the middleware
# that intercepts responses and provides proper request/get flow


def mock_add_or_update_user(user: models.User,
                            allow_duplicate_name: bool = True) -> bool:
    """Mock add_or_update_user - adds user to demo users if not exists."""
    # Load fresh data from JSON for hot reloading
    users_data = load_mock_data_file('users')
    users = _convert_users_from_json(users_data)

    # Check if user already exists
    for existing_user in users:
        if existing_user.id == user.id:
            return False  # User already exists, not newly added

    # In demo mode, we can't actually persist new users, so just return True
    return True  # User was newly added


def mock_set_ssh_keys(*args, **kwargs):
    """Mock set_ssh_keys - no-op."""
    pass


def mock_get_ssh_keys(user_hash: str) -> tuple:
    """Mock get_ssh_keys - returns empty keys."""
    return '', '', False


def mock_get_current_user() -> models.User:
    """Mock get_current_user - returns current demo user."""
    # Load fresh data from JSON for hot reloading
    users_data = load_mock_data_file('users')
    current_user_id = users_data['current_user']
    return mock_get_user(current_user_id)


def mock_get_user_hash() -> str:
    """Mock get_user_hash - returns default demo user hash."""
    # Load fresh data from JSON for hot reloading
    users_data = load_mock_data_file('users')
    users = _convert_users_from_json(users_data)
    return users[0].id


def mock_get_user_roles(user_id: str) -> List[str]:
    """Mock implementation of get_user_roles."""
    # Load fresh data from JSON for hot reloading
    users_data = load_mock_data_file('users')
    user_roles = users_data.get('user_roles', {})
    return user_roles.get(user_id, [])


def mock_override_user_info_in_request_body(request_body: dict,
                                            auth_user: models.User):
    """Mock implementation of user info injection into request body."""
    if not auth_user:
        return request_body

    # Inject user environment variables into the request body
    if 'env_vars' in request_body:
        if isinstance(request_body.get('env_vars'), dict):
            request_body['env_vars'][constants.USER_ID_ENV_VAR] = auth_user.id
            request_body['env_vars'][constants.USER_ENV_VAR] = auth_user.name
        else:
            logger.warning(
                'env_vars in request body is not a dictionary. Skipping user info injection.'
            )
    else:
        request_body['env_vars'] = {
            constants.USER_ID_ENV_VAR: auth_user.id,
            constants.USER_ENV_VAR: auth_user.name
        }

    return request_body


# Jobs-related mock functions
def mock_jobs_queue(*args, **kwargs) -> List[Dict[str, Any]]:
    """Mock implementation of core.queue - returns demo job data."""
    logger.info(
        f"Demo mode: mock_jobs_queue() called with args={args}, kwargs={kwargs}"
    )

    # Load fresh data from JSON for hot reloading
    jobs_data = load_mock_data_file('jobs')
    jobs = _convert_jobs_from_json(jobs_data)

    # Handle skip_finished filter
    if kwargs.get('skip_finished', False):
        jobs = [job for job in jobs if not job['status'].is_terminal()]
        logger.info(
            f"Demo mode: Filtered to non-finished jobs: {len(jobs)} jobs")

    # Handle all_users filter
    if not kwargs.get('all_users', True):
        current_user_hash = mock_get_user_hash()
        jobs = [job for job in jobs if job['user_hash'] == current_user_hash]
        logger.info(
            f"Demo mode: Filtered to current user jobs: {len(jobs)} jobs")

    # Handle job_ids filter
    job_ids = kwargs.get('job_ids')
    if job_ids:
        jobs = [job for job in jobs if job['job_id'] in job_ids]
        logger.info(
            f"Demo mode: Filtered to specific job IDs {job_ids}: {len(jobs)} jobs"
        )

    logger.info(f"Demo mode: Returning {len(jobs)} demo jobs")
    return jobs


def mock_maybe_restart_controller(*args, **kwargs):
    """Mock implementation of _maybe_restart_controller - returns fake handle."""
    logger.info(f"Demo mode: mock_maybe_restart_controller() called")

    # Return a fake controller handle that passes basic checks
    fake_controller_handle = _create_demo_handle('sky-jobs-controller-fake',
                                                 clouds.GCP(), 'us-central1', 1)
    logger.info(f"Demo mode: Returning fake jobs controller handle")
    return fake_controller_handle


def mock_cluster_jobs_queue(cluster_name: str,
                            skip_finished: bool = False,
                            all_users: bool = False) -> List[Dict[str, Any]]:
    """Mock implementation of core.queue for cluster jobs - returns demo cluster job data."""
    logger.info(
        f"Demo mode: mock_cluster_jobs_queue() called with cluster_name='{cluster_name}', skip_finished={skip_finished}, all_users={all_users}"
    )
    print(
        f"[DEBUG] mock_cluster_jobs_queue called with cluster_name='{cluster_name}'",
        flush=True)

    # Load fresh data from JSON for hot reloading
    cluster_jobs_data = load_mock_data_file('cluster_jobs')
    cluster_jobs = _convert_cluster_jobs_from_json(cluster_jobs_data)

    # Filter jobs for the specific cluster
    cluster_jobs = [
        job for job in cluster_jobs if job['cluster_name'] == cluster_name
    ]
    logger.info(
        f"Demo mode: Found {len(cluster_jobs)} jobs for cluster '{cluster_name}'"
    )
    print(
        f"[DEBUG] Found {len(cluster_jobs)} jobs for cluster '{cluster_name}'",
        flush=True)

    # Handle skip_finished filter
    if skip_finished:
        cluster_jobs = [
            job for job in cluster_jobs if not job['status'].is_terminal()
        ]
        logger.info(
            f"Demo mode: After skip_finished filtering: {len(cluster_jobs)} jobs"
        )

    # Handle all_users filter
    if not all_users:
        current_user_hash = mock_get_user_hash()
        cluster_jobs = [
            job for job in cluster_jobs if job['user_hash'] == current_user_hash
        ]
        logger.info(
            f"Demo mode: After user filtering: {len(cluster_jobs)} jobs")

    # Ensure we return the correct format matching core.queue docstring
    result = []
    for job in cluster_jobs:
        job_dict = {
            'job_id': job['job_id'],
            'job_name': job['job_name'],
            'username': job['username'],
            'user_hash': job['user_hash'],
            'submitted_at': job['submitted_at'],
            'start_at': job['start_at'],
            'end_at': job['end_at'],
            'resources': job['resources'],
            'status': job['status'],
            'log_path': job['log_path'],
        }
        result.append(job_dict)

    logger.info(f"Demo mode: Returning {len(result)} demo cluster jobs")
    print(
        f"[DEBUG] Returning {len(result)} demo cluster jobs: {[job['job_name'] for job in result]}",
        flush=True)
    return result


def mock_cluster_tail_logs(cluster_name: str,
                           job_id: Optional[int],
                           follow: bool = True,
                           tail: int = 0) -> int:
    """Mock implementation of core.tail_logs for cluster jobs - reads from actual log files."""
    logger.info(
        f"Demo mode: mock_cluster_tail_logs() called with cluster_name='{cluster_name}', job_id={job_id}, follow={follow}, tail={tail}"
    )

    # Load fresh data from JSON for hot reloading
    cluster_jobs_data = load_mock_data_file('cluster_jobs')
    cluster_jobs = _convert_cluster_jobs_from_json(cluster_jobs_data)

    # Find the specific job or get the latest one
    target_job = None
    if job_id is not None:
        target_job = next((
            job for job in cluster_jobs
            if job['cluster_name'] == cluster_name and job['job_id'] == job_id),
                          None)
    else:
        # Get the latest job for this cluster
        cluster_jobs_filtered = [
            job for job in cluster_jobs if job['cluster_name'] == cluster_name
        ]
        if cluster_jobs_filtered:
            target_job = cluster_jobs_filtered[
                0]  # Already sorted by job_id desc

    if target_job is None:
        logger.warning(
            f"Demo mode: No job found for cluster '{cluster_name}' with job_id={job_id}"
        )
        print(f"No job found for cluster '{cluster_name}'" +
              (f" with job_id={job_id}" if job_id else ""),
              flush=True)
        return 1

    # Get the log file path (relative to demo directory)
    log_path = target_job['log_path']
    full_log_path = os.path.join(DEMO_DIR, log_path)

    # Check if log file exists
    if not os.path.exists(full_log_path):
        logger.warning(f"Demo mode: Log file not found: {full_log_path}")
        print(f"Log file not found: {log_path}", flush=True)
        return 1

    try:
        # Read the log file
        with open(full_log_path, 'r') as f:
            log_lines = f.readlines()

        # Remove newline characters but preserve empty lines
        log_lines = [line.rstrip('\n\r') for line in log_lines]

        # Apply tail filtering if specified
        if tail > 0:
            log_lines = log_lines[-tail:]

        # Print the log content
        for log_line in log_lines:
            print(log_line, flush=True)

        # If following and job is not terminal, simulate some additional ongoing logs
        job_status = target_job['status']
        job_name = target_job['job_name']

        # Return exit code based on job status
        if job_status == job_lib.JobStatus.SUCCEEDED:
            return 0
        elif job_status in [
                job_lib.JobStatus.FAILED, job_lib.JobStatus.FAILED_SETUP,
                job_lib.JobStatus.FAILED_DRIVER
        ]:
            return 100  # Failed exit code
        elif job_status == job_lib.JobStatus.CANCELLED:
            return 103  # Cancelled exit code
        else:
            return 101  # Not finished exit code

    except Exception as e:
        logger.error(f"Demo mode: Error reading log file {full_log_path}: {e}")
        print(f"Error reading log file: {e}", flush=True)
        return 1


# Patch global_user_state functions
def patch_global_user_state():
    """Monkey-patch global_user_state functions with demo data."""
    global_user_state.get_clusters = mock_get_clusters
    global_user_state.get_cluster_from_name = mock_get_cluster_from_name
    global_user_state.get_all_users = mock_get_all_users
    global_user_state.get_user = mock_get_user
    global_user_state.get_storage = mock_get_storage
    global_user_state.get_volumes = mock_get_volumes
    global_user_state.get_cached_enabled_clouds = mock_get_cached_enabled_clouds
    global_user_state.get_cluster_names_start_with = mock_get_cluster_names_start_with
    global_user_state.get_glob_cluster_names = mock_get_glob_cluster_names
    global_user_state.get_storage_names_start_with = mock_get_storage_names_start_with
    global_user_state.get_volume_names_start_with = mock_get_volume_names_start_with

    # Block write operations
    global_user_state.add_or_update_cluster = mock_readonly_operation
    global_user_state.remove_cluster = mock_readonly_operation
    global_user_state.set_cluster_status = mock_readonly_operation
    global_user_state.update_cluster_handle = mock_readonly_operation
    global_user_state.add_or_update_storage = mock_readonly_operation
    global_user_state.remove_storage = mock_readonly_operation
    global_user_state.add_volume = mock_readonly_operation
    global_user_state.delete_volume = mock_readonly_operation
    global_user_state.add_or_update_user = mock_add_or_update_user
    global_user_state.set_ssh_keys = mock_set_ssh_keys
    global_user_state.get_ssh_keys = mock_get_ssh_keys

    # Mock user context functions
    common_utils.get_current_user = mock_get_current_user
    common_utils.get_user_hash = mock_get_user_hash


def mock_load_managed_job_queue(payload: str) -> List[Dict[str, Any]]:
    """Mock implementation of load_managed_job_queue - returns demo job data."""
    logger.info(f"Demo mode: mock_load_managed_job_queue() called")

    # Load fresh data from JSON for hot reloading
    jobs_data = load_mock_data_file('jobs')
    jobs = _convert_jobs_from_json(jobs_data)

    logger.info(
        f"Demo mode: Returning {len(jobs)} demo jobs from load_managed_job_queue"
    )
    return jobs


def mock_jobs_queue_direct(
        refresh: bool,
        skip_finished: bool = False,
        all_users: bool = False,
        job_ids: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    """Mock implementation of core.queue - returns demo job data in API format."""
    logger.info(
        f"Demo mode: mock_jobs_queue_direct() called with refresh={refresh}, skip_finished={skip_finished}, all_users={all_users}, job_ids={job_ids}"
    )

    # Load fresh data from JSON for hot reloading
    jobs_data = load_mock_data_file('jobs')
    demo_jobs = _convert_jobs_from_json(jobs_data)

    # Convert demo data to API format (matching core.queue docstring format exactly)
    api_jobs = []
    for job in demo_jobs:
        api_job = {
            'job_id': job['job_id'],
            'job_name': job['job_name'],
            'task_id': job['task_id'],
            'job_duration': job['job_duration'],
            'task_name': job['task_name'],
            'resources': job['resources'],
            'submitted_at': job['submitted_at'],
            'end_at': job['end_at'],
            'duration': job['job_duration'],  # Rename to match API format
            'recovery_count': job['recovery_count'],
            'status': job['status'],  # Keep as enum object (API format)
            'cluster_resources': job['cluster_resources'],
            'region': job['region'],
            'user_name': job['user_name'],
            'user_hash': job['user_hash'],
            'workspace': job['workspace'],
            'cloud': job['cloud'],
            'region': job['region'],
            'zone': job['zone'],
        }
        api_jobs.append(api_job)
    jobs = demo_jobs.copy()

    # Apply filtering based on parameters (same logic as real core.queue)
    # jobs = api_jobs

    # User filtering
    if not all_users:

        def user_hash_matches_or_missing(job: Dict[str, Any]) -> bool:
            user_hash = job.get('user_hash', None)
            if user_hash is None:
                return True  # For backwards compatibility
            return user_hash == mock_get_user_hash()

        jobs = list(filter(user_hash_matches_or_missing, jobs))
        logger.info(f"Demo mode: After user filtering: {len(jobs)} jobs")

    # Workspace filtering (simplified - all our demo jobs are in 'default')
    # (Note: workspace is not in API format, so this is implicitly handled)

    # Skip finished filtering
    if skip_finished:
        non_finished_tasks = list(
            filter(lambda job: not job['status'].is_terminal(), jobs))
        non_finished_job_ids = {job['job_id'] for job in non_finished_tasks}
        jobs = list(
            filter(lambda job: job['job_id'] in non_finished_job_ids, jobs))
        logger.info(
            f"Demo mode: After skip_finished filtering: {len(jobs)} jobs")

    # Job IDs filtering
    if job_ids:
        jobs = [job for job in jobs if job['job_id'] in job_ids]
        logger.info(f"Demo mode: After job_ids filtering: {len(jobs)} jobs")

    logger.info(f"Demo mode: Returning {len(jobs)} demo jobs")
    return jobs


def mock_managed_job_tail_logs(name: Optional[str],
                               job_id: Optional[int],
                               follow: bool,
                               controller: bool,
                               refresh: bool,
                               tail: Optional[int] = None) -> int:
    """Mock implementation of managed job tail_logs - reads from actual log files."""
    logger.debug(
        f"Demo mode: mock_managed_job_tail_logs() called with name='{name}', job_id={job_id}, follow={follow}, controller={controller}, refresh={refresh}, tail={tail}"
    )

    # Load fresh data from JSON for hot reloading
    jobs_data = load_mock_data_file('jobs')
    jobs = _convert_jobs_from_json(jobs_data)

    # Find the specific job or get the latest one
    target_job = None
    if job_id is not None:
        target_job = next((job for job in jobs if job['job_id'] == job_id),
                          None)
    elif name is not None:
        # Find by name - get the latest job with that name
        name_jobs = [job for job in jobs if job['job_name'] == name]
        if name_jobs:
            target_job = name_jobs[0]  # Already sorted by job_id desc
    else:
        # Get the latest job
        if jobs:
            target_job = jobs[0]  # Already sorted by job_id desc

    if target_job is None:
        logger.warning(
            f"Demo mode: No managed job found with name='{name}', job_id={job_id}"
        )
        print(f"No managed job found" +
              (f" with name '{name}'"
               if name else f" with job_id {job_id}" if job_id else ""),
              flush=True)
        return 1

    # Determine log file path based on controller flag
    if controller:
        log_filename = f'controller-job-{target_job["job_id"]}.log'
        log_path = os.path.join(DEMO_DIR, 'logs', 'managed_jobs', log_filename)
    else:
        # Use the log_path from job data if available, otherwise construct it
        if 'log_path' in target_job and target_job['log_path']:
            log_path = os.path.join(DEMO_DIR, target_job['log_path'])
        else:
            log_filename = f'job-{target_job["job_id"]}.log'
            log_path = os.path.join(DEMO_DIR, 'logs', 'managed_jobs',
                                    log_filename)

    # Check if log file exists
    if not os.path.exists(log_path):
        logger.warning(f"Demo mode: Managed job log file not found: {log_path}")
        print(f"Log file not found: {os.path.basename(log_path)}", flush=True)
        return 1

    try:
        # Read the log file
        with open(log_path, 'r') as f:
            log_lines = f.readlines()

        # Remove newline characters but preserve empty lines
        log_lines = [line.rstrip('\n\r') for line in log_lines]

        # Apply tail filtering if specified
        if tail is not None and tail > 0:
            log_lines = log_lines[-tail:]

        # Print the log content
        for log_line in log_lines:
            print(log_line, flush=True)

        # If following and job is not terminal, simulate some additional ongoing logs
        job_status = target_job['status']
        job_name = target_job['job_name']

        # Return exit code based on job status
        if job_status == ManagedJobStatus.SUCCEEDED:
            return 0
        elif job_status in [
                ManagedJobStatus.FAILED, ManagedJobStatus.FAILED_SETUP,
                ManagedJobStatus.FAILED_CONTROLLER
        ]:
            return 100  # Failed exit code
        elif job_status == ManagedJobStatus.CANCELLED:
            return 103  # Cancelled exit code
        else:
            return 101  # Not finished exit code

    except Exception as e:
        logger.error(
            f"Demo mode: Error reading managed job log file {log_path}: {e}")
        print(f"Error reading log file: {e}", flush=True)
        return 1


def patch_jobs_functions():
    """Patch jobs-related functions."""
    logger.info("Demo mode: Patching jobs functions")
    try:
        from sky.jobs.server import core as jobs_core
        from sky.backends import cloud_vm_ray_backend

        # Mock the main queue function directly (avoid SSH connection issues)
        jobs_core.queue = mock_jobs_queue_direct

        # Mock the managed job tail_logs function
        jobs_core.tail_logs = mock_managed_job_tail_logs

        # Also patch the backend method if it exists
        if hasattr(cloud_vm_ray_backend.CloudVmRayBackend,
                   'tail_managed_job_logs'):

            def mock_backend_tail_managed_job_logs(self,
                                                   handle,
                                                   job_id=None,
                                                   job_name=None,
                                                   controller=False,
                                                   follow=True,
                                                   tail=None):
                """Mock backend tail_managed_job_logs method."""
                return mock_managed_job_tail_logs(job_name, job_id, follow,
                                                  controller, False, tail)

            cloud_vm_ray_backend.CloudVmRayBackend.tail_managed_job_logs = mock_backend_tail_managed_job_logs

        logger.info("Demo mode: Successfully patched jobs functions")
    except ImportError as e:
        logger.warning(f"Demo mode: Could not import jobs core module: {e}")
    except Exception as e:
        logger.warning(f"Demo mode: Error patching jobs functions: {e}")


def patch_core_functions():
    """Patch core functions for cluster jobs."""
    logger.info("Demo mode: Patching core functions")
    try:
        from sky import core

        # Store original functions for debugging
        original_queue = getattr(core, 'queue', None)
        original_tail_logs = getattr(core, 'tail_logs', None)

        # Mock the cluster jobs queue function
        core.queue = mock_cluster_jobs_queue
        logger.info(
            f"Demo mode: Patched core.queue (was: {original_queue}, now: {core.queue})"
        )

        # Mock the cluster jobs tail_logs function
        core.tail_logs = mock_cluster_tail_logs
        logger.info(
            f"Demo mode: Patched core.tail_logs (was: {original_tail_logs}, now: {core.tail_logs})"
        )

        logger.info("Demo mode: Successfully patched core functions")
    except ImportError as e:
        logger.warning(f"Demo mode: Could not import core module: {e}")
    except Exception as e:
        logger.warning(f"Demo mode: Error patching core functions: {e}")


def patch_infrastructure_functions():
    """Patch infrastructure-related functions."""
    logger.info("Demo mode: Patching infrastructure functions")
    try:
        from sky import core
        from sky.provision.kubernetes import utils as kubernetes_utils

        # Patch core infrastructure functions
        core.enabled_clouds = mock_enabled_clouds
        core.get_all_contexts = mock_get_all_contexts
        core.realtime_kubernetes_gpu_availability = mock_realtime_kubernetes_gpu_availability

        # Patch Kubernetes utils function
        kubernetes_utils.get_kubernetes_node_info = mock_kubernetes_node_info

        logger.info("Demo mode: Successfully patched infrastructure functions")
    except ImportError as e:
        logger.warning(
            f"Demo mode: Could not import infrastructure modules: {e}")
    except Exception as e:
        logger.warning(
            f"Demo mode: Error patching infrastructure functions: {e}")


def patch_volumes_functions():
    """Patch volumes-related functions."""
    logger.info("Demo mode: Patching volumes functions")
    try:
        from sky.volumes.server import core as volumes_core

        # Mock the main volume_list function
        volumes_core.volume_list = mock_volume_list

        # Note: volume_delete and volume_apply are now handled by middleware
        # that intercepts responses and provides proper request/get flow

        logger.info("Demo mode: Successfully patched volumes functions")
    except ImportError as e:
        logger.warning(f"Demo mode: Could not import volumes core module: {e}")
    except Exception as e:
        logger.warning(f"Demo mode: Error patching volumes functions: {e}")


def patch_workspaces_functions():
    """Patch workspaces-related functions."""
    logger.info("Demo mode: Patching workspaces functions")
    try:
        from sky.workspaces import core as workspaces_core

        # Mock the main get_workspaces function for API endpoints
        workspaces_core.get_workspaces = mock_get_workspaces

        logger.info("Demo mode: Successfully patched workspaces functions")
    except ImportError as e:
        logger.warning(
            f"Demo mode: Could not import workspaces core module: {e}")
    except Exception as e:
        logger.warning(f"Demo mode: Error patching workspaces functions: {e}")


def patch_permission_functions():
    """Patch permission-related functions."""
    logger.info("Demo mode: Patching permission functions")
    try:
        from sky.users import permission

        # Mock the get_user_roles method on the permission service
        if hasattr(permission, 'permission_service'):
            # Store the original method for potential restoration
            permission.permission_service._original_get_user_roles = permission.permission_service.get_user_roles
            permission.permission_service.get_user_roles = mock_get_user_roles

            # Mock workspace permission check to always return True in demo mode
            def mock_check_workspace_permission(user_id: str,
                                                workspace_name: str) -> bool:
                """Mock workspace permission check - always return True in demo mode."""
                logger.debug(
                    f"Demo mode: Allowing workspace '{workspace_name}' access for user '{user_id}'"
                )
                return True

            permission.permission_service._original_check_workspace_permission = permission.permission_service.check_workspace_permission
            permission.permission_service.check_workspace_permission = mock_check_workspace_permission

        logger.info("Demo mode: Successfully patched permission functions")
    except ImportError as e:
        logger.warning(f"Demo mode: Could not import permission module: {e}")
    except Exception as e:
        logger.warning(f"Demo mode: Error patching permission functions: {e}")


def enable_demo_mode():
    """Enable demo mode by patching functions and endpoints."""
    logger.info("Enabling SkyPilot demo mode with fake data")

    # Patch global user state functions
    patch_global_user_state()

    # Patch jobs functions
    patch_jobs_functions()

    # Patch core functions (for cluster jobs)
    patch_core_functions()

    # Patch infrastructure functions
    patch_infrastructure_functions()

    # Patch volumes functions
    patch_volumes_functions()

    # Patch workspaces functions
    patch_workspaces_functions()

    # Patch permission functions
    patch_permission_functions()

    # CRITICAL: Patch executor to apply demo patches in worker processes
    patch_executor_initializer()

    logger.info("Demo mode enabled successfully")


def enable_demo_mode_worker():
    """Enable essential demo mode patches for worker processes (no server endpoints)."""
    pid = multiprocessing.current_process().pid
    logger.info(f"Demo mode: Enabling worker patches in process {pid}")

    # Apply only the essential patches needed
    logger.info(f"Demo mode: Worker {pid} - Applying permission patches")
    patch_permission_functions()
    logger.info(f"Demo mode: Worker {pid} - Applying core patches")
    patch_core_functions()

    # CRITICAL: Need skypilot_config.get_nested patch for workspace validation in worker processes
    logger.info(f"Demo mode: Worker {pid} - Applying skypilot_config patches")
    try:
        from sky import skypilot_config

        def mock_get_nested_worker(keys, default_value, override_configs=None):
            """Mock get_nested for worker processes - return workspace data from demo files."""
            if keys == ('workspaces',):
                workspaces_data = load_mock_data_file('workspaces')
                return workspaces_data.get('workspaces', {})
            # For other keys, return default value
            return default_value

        # Store original and patch
        if not hasattr(skypilot_config, '_original_get_nested_worker'):
            skypilot_config._original_get_nested_worker = skypilot_config.get_nested
        skypilot_config.get_nested = mock_get_nested_worker
        logger.info(
            f"Demo mode: Worker {pid} - Patched skypilot_config.get_nested")

    except Exception as e:
        logger.error(
            f"Demo mode: Worker {pid} - Error patching skypilot_config: {e}")

    logger.info(
        f"Demo mode: Worker {pid} - Worker patches applied successfully")


# Global variable to store original initializer
_ORIGINAL_EXECUTOR_INITIALIZER = None


def demo_executor_initializer(proc_group: str):
    """Demo mode executor initializer that applies patches in worker processes.
    
    This needs to be at module level (not nested) so it can be pickled by multiprocessing.
    """
    # Call original initializer first
    if _ORIGINAL_EXECUTOR_INITIALIZER is not None:
        _ORIGINAL_EXECUTOR_INITIALIZER(proc_group)

    # Apply demo mode patches in worker process
    enable_demo_mode_worker()
    logger.info(
        f"Demo mode: Patches applied in worker process {multiprocessing.current_process().pid}"
    )


def patch_executor_initializer():
    """Patch the executor initializer to apply demo mode patches in worker processes."""
    logger.info("Demo mode: Patching executor initializer")
    try:
        from sky.server.requests import executor
        global _ORIGINAL_EXECUTOR_INITIALIZER

        # Store the original initializer
        _ORIGINAL_EXECUTOR_INITIALIZER = executor.executor_initializer

        # Replace the executor initializer with our module-level function
        executor.executor_initializer = demo_executor_initializer

        logger.info("Demo mode: Successfully patched executor initializer")
    except ImportError as e:
        logger.warning(f"Demo mode: Could not import executor module: {e}")
    except Exception as e:
        logger.warning(f"Demo mode: Error patching executor initializer: {e}")


def patch_server_endpoints(app):
    """Patch server write endpoints to be read-only. Call this after app is created."""
    import os
    import multiprocessing

    # Add debugging to understand worker process context
    current_process = multiprocessing.current_process()
    worker_pid = os.getpid()

    logger.info(f"Demo mode: Patching server endpoints in process {worker_pid} "
                f"(process name: {current_process.name})")

    # List of write endpoints to make read-only
    # Endpoints that use request/get pattern (return 200 with X-Request-ID, then GET /api/get)
    request_get_endpoints = [
        '/launch',
        '/exec',
        '/stop',
        '/down',
        '/start',
        '/autostop',
        '/cancel',
        '/storage/delete',
        '/local_up',
        '/local_down',
        '/upload',
        '/volumes/delete',
        '/volumes/apply',
        '/workspaces/create',
        '/workspaces/update',
        '/workspaces/delete',
        '/workspaces/config',
    ]

    # Endpoints that expect immediate response (return success/error immediately)
    immediate_response_endpoints = [
        '/users/update', '/users/delete', '/users/create',
        '/users/service-account-tokens/delete',
        '/users/service-account-tokens/update-role',
        '/users/service-account-tokens/rotate'
    ]

    # Fixed request ID for demo mode blocked operations
    DEMO_BLOCKED_REQUEST_ID = 'demo-blocked-operation'

    @app.middleware("http")
    async def demo_readonly_middleware(request: fastapi.Request, call_next):
        """Middleware to block write operations in demo mode."""
        # Add debug logging to track middleware execution
        logger.info(f'Demo middleware: Processing request {request.method} '
                    f'{request.url.path} in process {os.getpid()}')

        # Set current user from demo data for all requests
        auth_user = mock_get_current_user()
        request.state.auth_user = auth_user

        # Override user info in request body if needed
        if request.method in ['POST', 'PUT', 'PATCH'] and auth_user:
            # Get the body and inject user info
            body = await request.body()
            if body:
                try:
                    import json
                    original_json = json.loads(body.decode('utf-8'))
                    # Inject user environment variables
                    modified_json = mock_override_user_info_in_request_body(
                        original_json, auth_user)
                    # Replace the request body
                    request._body = json.dumps(modified_json).encode('utf-8')
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    logger.warning(
                        f'Demo mode: Error parsing request JSON for user injection: {e}'
                    )
                except Exception as e:
                    logger.warning(
                        f'Demo mode: Error injecting user info into request: {e}'
                    )

        # Handle GET /api/get requests for demo blocked operations
        is_get_request = request.method == 'GET'
        is_get_request = (is_get_request and
                          (request.url.path == '/api/get' or
                           request.url.path == '/internal/dashboard/api/get'))
        if is_get_request:
            request_id = request.query_params.get('request_id')
            logger.info(
                f"Demo middleware: GET /api/get request with request_id: {request_id}"
            )
            if request_id and request_id.startswith(DEMO_BLOCKED_REQUEST_ID):
                logger.info(
                    f"Demo middleware: Returning demo error for blocked operation: {request_id}"
                )

                # Return generic demo error message
                error_content = {
                    'detail': {
                        'error':
                            '{"message": "Write operations are not '
                            'allowed in Demo Mode", "type": "DemoModeError"}'
                    }
                }
                return fastapi.responses.JSONResponse(status_code=500,
                                                      content=error_content)

        # Handle immediate response endpoints - return error immediately
        if request.method in ['POST', 'PUT', 'PATCH', 'DELETE']:
            for endpoint in immediate_response_endpoints:
                # Check both with and without /api/v1 prefix
                if request.url.path == endpoint:
                    logger.info(
                        f"Demo middleware: Blocking immediate response endpoint {request.method} {request.url.path}"
                    )
                    return fastapi.responses.JSONResponse(
                        status_code=500,
                        content={
                            'detail': 'Write operations are not allowed in Demo Mode'
                        })

        # Handle request/get pattern endpoints - return fake success response
        if request.method in ['POST', 'PUT', 'PATCH', 'DELETE']:
            for endpoint in request_get_endpoints:
                if request.url.path == endpoint:
                    # Generate the demo request ID
                    random_request_id = str(uuid.uuid4())[:8]
                    demo_request_id = DEMO_BLOCKED_REQUEST_ID + f'-{random_request_id}'

                    # Return fake 200 response with demo request ID
                    response = fastapi.responses.JSONResponse(
                        status_code=200,
                        content={}  # Empty response body like normal success
                    )

                    # Set headers - demo middleware runs before RequestIDMiddleware
                    response.headers['X-Request-ID'] = demo_request_id
                    response.headers['X-Skypilot-Request-ID'] = demo_request_id
                    response.headers['X-Demo-Middleware'] = 'true'

                    return response

        # Process request normally (for non-blocked endpoints)
        response = await call_next(request)
        return response


# Enable demo mode functions (but not server endpoints) when imported
enable_demo_mode()
