"""Demo mode for SkyPilot API server with pre-populated fake data.

This module monkey-patches global_user_state functions and server endpoints
to provide a read-only demo experience with realistic fake data.

The demo data is loaded from demo/mock_data.json file, which contains all the
fake users, clusters, jobs, volumes, and other resources to simulate a 
realistic SkyPilot environment.

HOT RELOADING: The JSON file is loaded fresh on every API query, enabling
real-time editing of mock data without server restarts. Simply edit the
JSON file and refresh your browser to see changes immediately!
"""

import copy
import json
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
from sky.utils import common_utils
from sky.utils import status_lib

logger = sky_logging.init_logger(__name__)

# Demo data timestamps
CURRENT_TIME = int(time.time())

# Path to mock data file
MOCK_DATA_FILE = os.path.join(os.path.dirname(__file__), 'mock_data.json')

def load_mock_data():
    """Load mock data from JSON file."""
    try:
        with open(MOCK_DATA_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Mock data file not found: {MOCK_DATA_FILE}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in mock data file: {e}")
        raise

def reload_mock_data():
    """Reload mock data from JSON file for hot reloading.
    
    This function is now deprecated since data is loaded fresh on every query.
    It's kept for backward compatibility.
    """
    logger.info("Mock data is now loaded fresh on every query - reload_mock_data() is deprecated")
    
    # Test loading to ensure the JSON file is valid
    try:
        mock_data = _get_fresh_mock_data()
        users = _convert_users_from_json(mock_data)
        clusters = _convert_clusters_from_json(mock_data)
        volumes = _convert_volumes_from_json(mock_data)
        jobs = _convert_jobs_from_json(mock_data)
        
        logger.info(f"Mock data validated successfully - {len(users)} users, {len(clusters)} clusters, {len(jobs)} jobs, {len(volumes)} volumes")
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

def _get_schedule_state_from_string(state_str):
    """Convert schedule state string to schedule state enum."""
    if hasattr(ManagedJobScheduleState, state_str):
        return getattr(ManagedJobScheduleState, state_str)
    else:
        raise ValueError(f"Unknown schedule state: {state_str}")

def _get_fresh_mock_data():
    """Load fresh mock data from JSON file for hot reloading."""
    return load_mock_data()

def _convert_users_from_json(mock_data):
    """Convert users from JSON data to User objects."""
    users = []
    for user_data in mock_data['users']:
        user = models.User(
            id=user_data['id'],
            name=user_data['name'],
            password=user_data['password'],
            created_at=_convert_timestamp_offset(user_data['created_at_offset'])
        )
        users.append(user)
    return users

def _convert_clusters_from_json(mock_data):
    """Convert clusters from JSON data to cluster objects."""
    clusters = []
    yaml_configs = mock_data['yaml_configs']
    
    for cluster_data in mock_data['clusters']:
        yaml_config = yaml_configs[cluster_data['last_creation_yaml']]
        cluster = {
            'name': cluster_data['name'],
            'launched_at': _convert_timestamp_offset(cluster_data['launched_at_offset']),
            'handle': _create_demo_handle(
                cluster_data['name'], 
                _get_cloud_from_string(cluster_data['cloud']), 
                cluster_data['region'], 
                cluster_data['nodes']
            ),
            'last_use': cluster_data['last_use'],
            'status': _get_status_from_string(cluster_data['status']),
            'autostop': cluster_data['autostop'],
            'to_down': cluster_data['to_down'],
            'owner': cluster_data['owner'],
            'metadata': cluster_data['metadata'],
            'cluster_hash': cluster_data['cluster_hash'],
            'storage_mounts_metadata': cluster_data['storage_mounts_metadata'],
            'cluster_ever_up': cluster_data['cluster_ever_up'],
            'status_updated_at': _convert_timestamp_offset(cluster_data['status_updated_at_offset']),
            'user_hash': cluster_data['user_hash'],
            'user_name': cluster_data['user_name'],
            'config_hash': cluster_data['config_hash'],
            'workspace': cluster_data['workspace'],
            'last_creation_yaml': yaml_config,
            'last_creation_command': cluster_data['last_creation_command'],
        }
        clusters.append(cluster)
    return clusters

def _convert_volumes_from_json(mock_data):
    """Convert volumes from JSON data to volume objects."""
    volumes = []
    for volume_data in mock_data['volumes']:
        volume = {
            'name': volume_data['name'],
            'type': volume_data['type'],
            'launched_at': _convert_timestamp_offset(volume_data['launched_at_offset']),
            'cloud': volume_data['cloud'],
            'region': volume_data['region'],
            'zone': volume_data['zone'],
            'size': volume_data['size'],
            'config': volume_data['config'],
            'name_on_cloud': volume_data['name_on_cloud'],
            'user_hash': volume_data['user_hash'],
            'user_name': volume_data['user_name'],
            'workspace': volume_data['workspace'],
            'last_attached_at': _convert_timestamp_offset(volume_data['last_attached_at_offset']),
            'last_use': volume_data['last_use'],
            'status': _get_volume_status_from_string(volume_data['status']),
            'usedby_clusters': volume_data['usedby_clusters'],
            'usedby_pods': volume_data['usedby_pods'],
        }
        volumes.append(volume)
    return volumes

def _convert_jobs_from_json(mock_data):
    """Convert jobs from JSON data to job objects."""
    jobs = []
    for job_data in mock_data['jobs']:
        job = {
            'job_id': job_data['job_id'],
            'task_id': job_data['task_id'],
            'job_name': job_data['job_name'],
            'task_name': job_data['task_name'],
            'status': _get_job_status_from_string(job_data['status']),
            'submitted_at': _convert_timestamp_offset(job_data['submitted_at_offset']),
            'start_at': _convert_timestamp_offset(job_data.get('start_at_offset')) if 'start_at_offset' in job_data else job_data.get('start_at'),
            'end_at': _convert_timestamp_offset(job_data.get('end_at_offset')) if 'end_at_offset' in job_data else job_data.get('end_at'),
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
            'schedule_state': _get_schedule_state_from_string(job_data['schedule_state']),
            'specs': job_data['specs'],
            'run_timestamp': str(_convert_timestamp_offset(job_data['run_timestamp_offset'])),
            'last_recovered_at': _convert_timestamp_offset(job_data['last_recovered_at_offset']),
        }
        jobs.append(job)
    return jobs

# Helper function to create demo cluster handles
def _create_demo_handle(cluster_name: str, cloud: clouds.Cloud, region: str,
                       nodes: int = 1) -> CloudVmRayResourceHandle:
    """Creates a realistic CloudVmRayResourceHandle for demo purposes."""
    # Create appropriate resources based on cluster type
    # TODO(romilb): We should include and parse this from the job definition
    if 'kubernetes' in str(cloud).lower():
        instance_type = '4CPU--16GB'
        accelerators = None
    elif 'multinode' in cluster_name:
        instance_type = 'n1-standard-8'
        accelerators = {'A100': 8}
    elif 'inference' in cluster_name:
        instance_type = 'g4dn.xlarge'
        accelerators = {'A100': 1}
    else:
        instance_type = 'n1-standard-4'
        accelerators = None
    
    # Create Resources object
    resources = resources_lib.Resources(
        cloud=cloud,
        region=region,
        instance_type=instance_type,
        accelerators=accelerators,
        cpus=None,
        memory=None,
        use_spot=False
    )
    
    # Generate fake IPs 
    base_ip = 10 + hash(cluster_name) % 200  # Generate deterministic but varied IPs
    internal_ips = [f'10.0.{base_ip}.{i+1}' for i in range(nodes)]
    external_ips = [f'34.{base_ip}.{base_ip}.{i+1}' for i in range(nodes)]
    stable_internal_external_ips = list(zip(internal_ips, external_ips))
    stable_ssh_ports = [22] * nodes
    
    return CloudVmRayResourceHandle(
        cluster_name=cluster_name,
        cluster_name_on_cloud=f'{cluster_name}-{common_utils.get_user_hash()[:8]}',
        cluster_yaml=None,  # No actual YAML file for demo
        launched_nodes=nodes,
        launched_resources=resources,
        stable_internal_external_ips=stable_internal_external_ips,
        stable_ssh_ports=stable_ssh_ports
    )

# All demo data is now loaded fresh from JSON on each query for hot reloading

# Mock functions to replace global_user_state functions
def mock_get_clusters() -> List[Dict[str, Any]]:
    """Mock implementation of get_clusters."""
    # Load fresh data from JSON for hot reloading
    mock_data = _get_fresh_mock_data()
    result = _convert_clusters_from_json(mock_data)
    
    cluster_names = [c['name'] for c in result]
    logger.info(f"Demo mode: mock_get_clusters() returning {len(result)} clusters: {cluster_names}")
    
    # Debug: log the structure of the first cluster for debugging
    if result:
        first_cluster = result[0]
        logger.info(f"Demo mode: First cluster structure - name: '{first_cluster.get('name')}', keys: {list(first_cluster.keys())}")
    
    return result

def mock_get_cluster_from_name(cluster_name: Optional[str]) -> Optional[Dict[str, Any]]:
    """Mock implementation of get_cluster_from_name."""
    logger.info(f"Demo mode: mock_get_cluster_from_name() called with cluster_name='{cluster_name}'")
    
    if cluster_name is None:
        return None
    
    # Load fresh data from JSON for hot reloading
    mock_data = _get_fresh_mock_data()
    clusters = _convert_clusters_from_json(mock_data)
    
    for cluster in clusters:
        if cluster['name'] == cluster_name:
            logger.info(f"Demo mode: Found cluster '{cluster_name}' in demo data")
            return cluster
    
    logger.warning(f"Demo mode: Cluster '{cluster_name}' not found in demo data. Available clusters: {[c['name'] for c in clusters]}")
    return None

def mock_get_all_users() -> List[models.User]:
    """Mock implementation of get_all_users."""
    # Load fresh data from JSON for hot reloading
    mock_data = _get_fresh_mock_data()
    return _convert_users_from_json(mock_data)

def mock_get_user(user_id: str) -> Optional[models.User]:
    """Mock implementation of get_user."""
    # Load fresh data from JSON for hot reloading
    mock_data = _get_fresh_mock_data()
    users = _convert_users_from_json(mock_data)
    
    for user in users:
        if user.id == user_id:
            return user
    return None

def mock_get_storage() -> List[Dict[str, Any]]:
    """Mock implementation of get_storage."""
    # Load fresh data from JSON for hot reloading
    mock_data = _get_fresh_mock_data()
    return mock_data['storage'].copy()

def mock_get_volumes() -> List[Dict[str, Any]]:
    """Mock implementation of get_volumes."""
    # Load fresh data from JSON for hot reloading
    mock_data = _get_fresh_mock_data()
    return _convert_volumes_from_json(mock_data)

def mock_get_cached_enabled_clouds(cloud_capability, workspace: str) -> List:
    """Mock implementation of get_cached_enabled_clouds."""
    # Load fresh data from JSON for hot reloading
    mock_data = _get_fresh_mock_data()
    enabled_clouds = mock_data['enabled_clouds']
    
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
    mock_data = _get_fresh_mock_data()
    clusters = _convert_clusters_from_json(mock_data)
    
    return [cluster['name'] for cluster in clusters 
            if cluster['name'].startswith(starts_with)]

def mock_get_glob_cluster_names(cluster_name: str) -> List[str]:
    """Mock implementation of get_glob_cluster_names."""
    import fnmatch
    logger.info(f"Demo mode: mock_get_glob_cluster_names() called with pattern='{cluster_name}'")
    
    # Load fresh data from JSON for hot reloading
    mock_data = _get_fresh_mock_data()
    clusters = _convert_clusters_from_json(mock_data)
    
    # Get all cluster names from demo data
    all_cluster_names = [cluster['name'] for cluster in clusters]
    
    # Apply glob pattern matching
    matching_names = [name for name in all_cluster_names if fnmatch.fnmatch(name, cluster_name)]
    
    logger.info(f"Demo mode: Pattern '{cluster_name}' matched clusters: {matching_names}")
    return matching_names

def mock_get_storage_names_start_with(starts_with: str) -> List[str]:
    """Mock implementation of get_storage_names_start_with."""
    # Load fresh data from JSON for hot reloading
    mock_data = _get_fresh_mock_data()
    storage = mock_data['storage']
    
    return [storage_item['name'] for storage_item in storage 
            if storage_item['name'].startswith(starts_with)]

def mock_get_volume_names_start_with(starts_with: str) -> List[str]:
    """Mock implementation of get_volume_names_start_with."""
    # Load fresh data from JSON for hot reloading
    mock_data = _get_fresh_mock_data()
    volumes = _convert_volumes_from_json(mock_data)
    
    return [volume['name'] for volume in volumes 
            if volume['name'].startswith(starts_with)]

# Infrastructure-related mock functions
def mock_enabled_clouds(workspace: Optional[str] = None, expand: bool = False) -> List[str]:
    """Mock implementation of core.enabled_clouds."""
    logger.info(f"Demo mode: mock_enabled_clouds() called with workspace={workspace}, expand={expand}")
    
    # Load fresh data from JSON for hot reloading
    mock_data = _get_fresh_mock_data()
    enabled_clouds = mock_data['enabled_clouds']
    
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
    mock_data = _get_fresh_mock_data()
    
    # Return the contexts from our demo node info (these are the available Kubernetes contexts)
    contexts = list(mock_data['node_info'].keys())
    
    logger.info(f"Demo mode: Returning {len(contexts)} contexts: {contexts}")
    return contexts

def mock_realtime_kubernetes_gpu_availability(
    context: Optional[str] = None,
    name_filter: Optional[str] = None,
    quantity_filter: Optional[int] = None,
    is_ssh: Optional[bool] = None
):
    """Mock implementation of core.realtime_kubernetes_gpu_availability."""
    logger.info(f"Demo mode: mock_realtime_kubernetes_gpu_availability() called with context={context}, name_filter={name_filter}, quantity_filter={quantity_filter}, is_ssh={is_ssh}")
    
    # Import models here to avoid circular imports
    from sky import models
    
    # Load fresh data from JSON for hot reloading
    mock_data = _get_fresh_mock_data()
    demo_gpu_data = mock_data['gpu_availability']
    
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
                available=available
            )
            gpu_availability_list.append(gpu_availability)
        
        if gpu_availability_list:  # Only add context if it has GPUs
            result.append((ctx, gpu_availability_list))
    
    logger.info(f"Demo mode: Returning GPU availability for {len(result)} contexts")
    return result

def mock_kubernetes_node_info(context: Optional[str] = None):
    """Mock implementation of kubernetes_utils.get_kubernetes_node_info."""
    logger.info(f"Demo mode: mock_kubernetes_node_info() called with context={context}")
    
    # Import models here to avoid circular imports
    from sky import models
    
    # Load fresh data from JSON for hot reloading
    mock_data = _get_fresh_mock_data()
    demo_node_data = mock_data['node_info']
    
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
    
    logger.info(f"Demo mode: Returning node info for context '{context}' with {len(node_info_dict)} nodes")
    return result

def mock_volume_list() -> List[Dict[str, Any]]:
    """Mock implementation of volumes.server.core.volume_list."""
    logger.info("Demo mode: mock_volume_list() called")
    
    # Load fresh data from JSON for hot reloading
    mock_data = _get_fresh_mock_data()
    volumes = _convert_volumes_from_json(mock_data)
    
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
    mock_data = _get_fresh_mock_data()
    workspaces = copy.deepcopy(mock_data['workspaces'])
    
    logger.info(f"Demo mode: Returning {len(workspaces)} demo workspaces: {list(workspaces.keys())}")
    return workspaces

# Read-only operation handlers - these will no-op or return appropriate responses
def mock_readonly_operation(*args, **kwargs):
    """Generic handler for read-only mode - raises 403 Forbidden."""
    raise fastapi.HTTPException(
        status_code=403, 
        detail="Demo mode: Write operations are not allowed"
    )

def mock_add_or_update_user(user: models.User, allow_duplicate_name: bool = True) -> bool:
    """Mock add_or_update_user - adds user to demo users if not exists."""
    # Load fresh data from JSON for hot reloading
    mock_data = _get_fresh_mock_data()
    users = _convert_users_from_json(mock_data)
    
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
    """Mock get_current_user - returns default demo user."""
    # Load fresh data from JSON for hot reloading
    mock_data = _get_fresh_mock_data()
    users = _convert_users_from_json(mock_data)
    return users[0]  # Return first user as default user

def mock_get_user_hash() -> str:
    """Mock get_user_hash - returns default demo user hash."""
    # Load fresh data from JSON for hot reloading
    mock_data = _get_fresh_mock_data()
    users = _convert_users_from_json(mock_data)
    return users[0].id

# Jobs-related mock functions
def mock_jobs_queue(*args, **kwargs) -> List[Dict[str, Any]]:
    """Mock implementation of core.queue - returns demo job data."""
    logger.info(f"Demo mode: mock_jobs_queue() called with args={args}, kwargs={kwargs}")
    
    # Load fresh data from JSON for hot reloading
    mock_data = _get_fresh_mock_data()
    jobs = _convert_jobs_from_json(mock_data)
    
    # Handle skip_finished filter
    if kwargs.get('skip_finished', False):
        jobs = [job for job in jobs if not job['status'].is_terminal()]
        logger.info(f"Demo mode: Filtered to non-finished jobs: {len(jobs)} jobs")
    
    # Handle all_users filter
    if not kwargs.get('all_users', True):
        current_user_hash = mock_get_user_hash()
        jobs = [job for job in jobs if job['user_hash'] == current_user_hash]
        logger.info(f"Demo mode: Filtered to current user jobs: {len(jobs)} jobs")
    
    # Handle job_ids filter
    job_ids = kwargs.get('job_ids')
    if job_ids:
        jobs = [job for job in jobs if job['job_id'] in job_ids]
        logger.info(f"Demo mode: Filtered to specific job IDs {job_ids}: {len(jobs)} jobs")
    
    logger.info(f"Demo mode: Returning {len(jobs)} demo jobs")
    return jobs

def mock_maybe_restart_controller(*args, **kwargs):
    """Mock implementation of _maybe_restart_controller - returns fake handle."""
    logger.info(f"Demo mode: mock_maybe_restart_controller() called")
    
    # Return a fake controller handle that passes basic checks
    fake_controller_handle = _create_demo_handle('sky-jobs-controller-fake', clouds.GCP(), 'us-central1', 1)
    logger.info(f"Demo mode: Returning fake jobs controller handle")
    return fake_controller_handle

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
    mock_data = _get_fresh_mock_data()
    jobs = _convert_jobs_from_json(mock_data)
    
    logger.info(f"Demo mode: Returning {len(jobs)} demo jobs from load_managed_job_queue")
    return jobs

def mock_jobs_queue_direct(refresh: bool,
          skip_finished: bool = False,
          all_users: bool = False,
          job_ids: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    """Mock implementation of core.queue - returns demo job data in API format."""
    logger.info(f"Demo mode: mock_jobs_queue_direct() called with refresh={refresh}, skip_finished={skip_finished}, all_users={all_users}, job_ids={job_ids}")
    
    # Load fresh data from JSON for hot reloading
    mock_data = _get_fresh_mock_data()
    demo_jobs = _convert_jobs_from_json(mock_data)
    
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
        }
        api_jobs.append(api_job)
    
    # Apply filtering based on parameters (same logic as real core.queue)
    jobs = api_jobs
    
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
        non_finished_tasks = list(filter(lambda job: not job['status'].is_terminal(), jobs))
        non_finished_job_ids = {job['job_id'] for job in non_finished_tasks}
        jobs = list(filter(lambda job: job['job_id'] in non_finished_job_ids, jobs))
        logger.info(f"Demo mode: After skip_finished filtering: {len(jobs)} jobs")
    
    # Job IDs filtering
    if job_ids:
        jobs = [job for job in jobs if job['job_id'] in job_ids]
        logger.info(f"Demo mode: After job_ids filtering: {len(jobs)} jobs")
    
    logger.info(f"Demo mode: Returning {len(jobs)} demo jobs")
    return jobs

def patch_jobs_functions():
    """Patch jobs-related functions."""
    logger.info("Demo mode: Patching jobs functions")
    try:
        from sky.jobs.server import core as jobs_core
        
        # Mock the main queue function directly (avoid SSH connection issues)
        jobs_core.queue = mock_jobs_queue_direct
        
        logger.info("Demo mode: Successfully patched jobs functions")
    except ImportError as e:
        logger.warning(f"Demo mode: Could not import jobs core module: {e}")
    except Exception as e:
        logger.warning(f"Demo mode: Error patching jobs functions: {e}")

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
        logger.warning(f"Demo mode: Could not import infrastructure modules: {e}")
    except Exception as e:
        logger.warning(f"Demo mode: Error patching infrastructure functions: {e}")

def patch_volumes_functions():
    """Patch volumes-related functions."""
    logger.info("Demo mode: Patching volumes functions")
    try:
        from sky.volumes.server import core as volumes_core
        
        # Mock the main volume_list function
        volumes_core.volume_list = mock_volume_list
        
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
        
        # Mock the main get_workspaces function
        workspaces_core.get_workspaces = mock_get_workspaces
        
        logger.info("Demo mode: Successfully patched workspaces functions")
    except ImportError as e:
        logger.warning(f"Demo mode: Could not import workspaces core module: {e}")
    except Exception as e:
        logger.warning(f"Demo mode: Error patching workspaces functions: {e}")



def enable_demo_mode():
    """Enable demo mode by patching functions and endpoints."""
    logger.info("Enabling SkyPilot demo mode with fake data")
    
    # Patch global user state functions
    patch_global_user_state()
    
    # Patch jobs functions
    patch_jobs_functions()
    
    # Patch infrastructure functions
    patch_infrastructure_functions()
    
    # Patch volumes functions
    patch_volumes_functions()
    
    # Patch workspaces functions
    patch_workspaces_functions()
    
    logger.info("Demo mode enabled successfully")

def patch_server_endpoints(app):
    """Patch server write endpoints to be read-only. Call this after app is created."""
    # List of write endpoints to make read-only
    write_endpoints = [
        '/launch', '/exec', '/stop', '/down', '/start', '/autostop', '/cancel',
        '/storage/delete', '/local_up', '/local_down', '/upload',
        '/volumes/delete', '/volumes/apply',
        '/workspaces/create', '/workspaces/update', '/workspaces/delete', '/workspaces/config'
    ]
    
    @app.middleware("http")
    async def demo_readonly_middleware(request: fastapi.Request, call_next):
        """Middleware to block write operations in demo mode."""
        # Block write operations
        if (request.method in ['POST', 'PUT', 'PATCH', 'DELETE'] and 
            any(request.url.path.startswith(f'/api/v1{endpoint}') for endpoint in write_endpoints)):
            return fastapi.responses.JSONResponse(
                status_code=403,
                content={'detail': 'Demo mode: Write operations are not allowed'}
            )
        
        # Allow read operations
        response = await call_next(request)
        return response
    
    logger.info("Demo mode: Server write endpoints patched successfully")

# Enable demo mode functions (but not server endpoints) when imported
enable_demo_mode()
