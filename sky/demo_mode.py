"""Demo mode for SkyPilot API server with pre-populated fake data.

This module monkey-patches global_user_state functions and server endpoints
to provide a read-only demo experience with realistic fake data.
"""

import copy
import json
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
ONE_HOUR_AGO = CURRENT_TIME - 3600
ONE_DAY_AGO = CURRENT_TIME - 86400
TWO_DAYS_AGO = CURRENT_TIME - 172800

# Demo users
DEMO_USERS = [
    models.User(id="alice123", name="alice@skypilot.co", password=None, created_at=TWO_DAYS_AGO),
    models.User(id="bob456", name="bob@skypilot.co", password=None, created_at=TWO_DAYS_AGO),
    models.User(id="mike789", name="mike@skypilot.co", password=None, created_at=ONE_DAY_AGO),
]

# Sample YAML configurations
TRAINING_YAML = """
resources:
  accelerators: A100:8
  cloud: gcp
  region: us-central1

num_nodes: 4

setup: |
  pip install torch transformers datasets

run: |
  torchrun --nproc_per_node=8 --nnodes=4 \\
    train_llama.py \\
    --model_name_or_path meta-llama/Llama-2-70b \\
    --dataset_name alpaca \\
    --output_dir ./checkpoints \\
    --num_train_epochs 3 \\
    --per_device_train_batch_size 1 \\
    --gradient_accumulation_steps 32 \\
    --learning_rate 2e-5 \\
    --save_steps 500 \\
    --logging_steps 10
"""

INFERENCE_YAML = """
resources:
  accelerators: A100:1
  cloud: aws
  region: us-west-2

setup: |
  pip install vllm transformers

run: |
  python -c "
  from vllm import LLM, SamplingParams
  
  llm = LLM(model='meta-llama/Llama-2-13b-chat-hf')
  sampling_params = SamplingParams(temperature=0.7, top_p=0.9)
  
  prompts = ['Tell me about artificial intelligence']
  outputs = llm.generate(prompts, sampling_params)
  
  for output in outputs:
      print(output.outputs[0].text)
  "
"""

DEV_YAML = """
resources:
  cpus: 4
  memory: 16
  cloud: gcp

setup: |
  pip install jupyter numpy pandas matplotlib

run: |
  jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
"""

# Helper function to create demo cluster handles
def _create_demo_handle(cluster_name: str, cloud: clouds.Cloud, 
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

# Demo clusters data
DEMO_CLUSTERS = [
    {
        'name': 'lambda-k8s',
        'launched_at': TWO_DAYS_AGO,
        'handle': _create_demo_handle('lambda-k8s', clouds.Kubernetes(), 1),
        'last_use': 'sky status',
        'status': status_lib.ClusterStatus.UP,
        'autostop': -1,
        'to_down': False,
        'owner': ['alice123'],
        'metadata': {'cloud': 'kubernetes', 'region': 'lambda-cloud'},
        'cluster_hash': 'cluster-hash-lambda-k8s',
        'storage_mounts_metadata': None,
        'cluster_ever_up': True,
        'status_updated_at': ONE_HOUR_AGO,
        'user_hash': 'alice123',
        'user_name': 'alice@skypilot.co',
        'config_hash': 'config-hash-1',
        'workspace': 'research-private',
        'last_creation_yaml': DEV_YAML,
        'last_creation_command': 'sky launch -c lambda-k8s dev.yaml',
    },
    {
        'name': 'nebius-k8s',
        'launched_at': ONE_DAY_AGO,
        'handle': _create_demo_handle('nebius-k8s', clouds.Kubernetes(), 1),
        'last_use': 'sky status',
        'status': status_lib.ClusterStatus.UP,
        'autostop': -1,
        'to_down': False,
        'owner': ['bob456'],
        'metadata': {'cloud': 'kubernetes', 'region': 'nebius'},
        'cluster_hash': 'cluster-hash-nebius-k8s',
        'storage_mounts_metadata': None,
        'cluster_ever_up': True,
        'status_updated_at': ONE_HOUR_AGO,
        'user_hash': 'bob456',
        'user_name': 'bob@skypilot.co',
        'config_hash': 'config-hash-2',
        'workspace': 'default',
        'last_creation_yaml': DEV_YAML,
        'last_creation_command': 'sky launch -c nebius-k8s dev.yaml',
    },
    {
        'name': 'training-multinode-1',
        'launched_at': ONE_DAY_AGO,
        'handle': _create_demo_handle('training-multinode-1', clouds.GCP(), 4),
        'last_use': 'sky exec training-multinode-1 train.yaml',
        'status': status_lib.ClusterStatus.UP,
        'autostop': 60,
        'to_down': False,
        'owner': ['alice123'],
        'metadata': {'cloud': 'gcp', 'region': 'us-central1'},
        'cluster_hash': 'cluster-hash-training-1',
        'storage_mounts_metadata': None,
        'cluster_ever_up': True,
        'status_updated_at': ONE_HOUR_AGO,
        'user_hash': 'alice123',
        'user_name': 'alice@skypilot.co',
        'config_hash': 'config-hash-3',
        'workspace': 'ml-team',
        'last_creation_yaml': TRAINING_YAML,
        'last_creation_command': 'sky launch -c training-multinode-1 train.yaml',
    },
    {
        'name': 'inference-cluster',
        'launched_at': ONE_HOUR_AGO,
        'handle': _create_demo_handle('inference-cluster', clouds.AWS(), 1),
        'last_use': 'sky exec inference-cluster inference.yaml',
        'status': status_lib.ClusterStatus.UP,
        'autostop': 30,
        'to_down': False,
        'owner': ['mike789'],
        'metadata': {'cloud': 'aws', 'region': 'us-west-2'},
        'cluster_hash': 'cluster-hash-inference',
        'storage_mounts_metadata': None,
        'cluster_ever_up': True,
        'status_updated_at': ONE_HOUR_AGO,
        'user_hash': 'mike789',
        'user_name': 'mike@skypilot.co',
        'config_hash': 'config-hash-4',
        'workspace': 'default',
        'last_creation_yaml': INFERENCE_YAML,
        'last_creation_command': 'sky launch -c inference-cluster inference.yaml',
    },
    {
        'name': 'dev-cluster-alice',
        'launched_at': TWO_DAYS_AGO,
        'handle': _create_demo_handle('dev-cluster-alice', clouds.GCP(), 1),
        'last_use': 'sky launch dev-work.yaml',
        'status': status_lib.ClusterStatus.STOPPED,
        'autostop': -1,
        'to_down': False,
        'owner': ['alice123'],
        'metadata': {'cloud': 'gcp', 'region': 'us-west1'},
        'cluster_hash': 'cluster-hash-dev-alice',
        'storage_mounts_metadata': None,
        'cluster_ever_up': True,
        'status_updated_at': ONE_DAY_AGO,
        'user_hash': 'alice123',
        'user_name': 'alice@skypilot.co',
        'config_hash': 'config-hash-5',
        'workspace': 'default',
        'last_creation_yaml': DEV_YAML,
        'last_creation_command': 'sky launch -c dev-cluster-alice dev.yaml',
    },
]

# Demo storage data
DEMO_STORAGE = []

# Demo volumes data  
DEMO_VOLUMES = [
    {
        'name': 'ml-datasets',
        'type': 'k8s-pvc',
        'launched_at': TWO_DAYS_AGO,
        'cloud': 'Kubernetes', 
        'region': 'lambda-cloud',
        'zone': None,
        'size': '500',  # 500GB
        'config': {
            'storage_class_name': 'lambda-standard',
            'access_mode': 'ReadWriteMany'
        },
        'name_on_cloud': 'ml-datasets-pvc-abc123',
        'user_hash': 'alice123',
        'user_name': 'alice@skypilot.co',
        'workspace': 'research-private',
        'last_attached_at': ONE_HOUR_AGO,
        'last_use': 'sky exec lambda-k8s train.yaml',
        'status': status_lib.VolumeStatus.IN_USE,
        'usedby_clusters': ['lambda-k8s'],
        'usedby_pods': [],
    },
    {
        'name': 'model-checkpoints',
        'type': 'k8s-pvc',
        'launched_at': ONE_DAY_AGO,
        'cloud': 'Kubernetes',
        'region': 'nebius', 
        'zone': None,
        'size': '1000',  # 1TB
        'config': {
            'storage_class_name': 'nebius-fast-ssd',
            'access_mode': 'ReadWriteOnce'
        },
        'name_on_cloud': 'model-checkpoints-pvc-def456',
        'user_hash': 'alice123',
        'user_name': 'alice@skypilot.co',
        'workspace': 'ml-team',
        'last_attached_at': ONE_DAY_AGO + 1800,  # 30 minutes after creation
        'last_use': 'sky launch training-job.yaml',
        'status': status_lib.VolumeStatus.READY,
        'usedby_clusters': [],
        'usedby_pods': [],
    },
    {
        'name': 'shared-workspace',
        'type': 'k8s-pvc',
        'launched_at': ONE_DAY_AGO - 3600,  # 1 day and 1 hour ago
        'cloud': 'Kubernetes',
        'region': 'nebius',
        'zone': None,
        'size': '100',  # 100GB
        'config': {
            'storage_class_name': 'nebius-standard',
            'access_mode': 'ReadWriteMany'
        },
        'name_on_cloud': 'shared-workspace-pvc-ghi789',
        'user_hash': 'bob456',
        'user_name': 'bob@skypilot.co',
        'workspace': 'default',
        'last_attached_at': None,  # Never attached
        'last_use': 'sky volume apply shared-workspace.yaml',
        'status': status_lib.VolumeStatus.READY,
        'usedby_clusters': [],
        'usedby_pods': [],
    },
]

# Demo enabled clouds
DEMO_ENABLED_CLOUDS = ['gcp', 'aws']

# Demo workspaces data
DEMO_WORKSPACES = {
    'default': {
        # Default workspace - can use all accessible infrastructure
        # Empty config means no restrictions
    },
    'ml-team': {
        'gcp': {
            'project_id': 'skypilot-ml-team-prod'
        },
        'aws': {
            'disabled': False
        }
    },
    'research-private': {
        'private': True,
        'allowed_users': [
            'alice@skypilot.co',
            'mike@skypilot.co'
        ],
        'gcp': {
            'project_id': 'skypilot-research-private'
        },
        'aws': {
            'disabled': True  # Only GCP for this private workspace
        }
    }
}

# Demo jobs data
THREE_DAYS_AGO = CURRENT_TIME - (3 * 86400)
TWO_HOURS_AGO = CURRENT_TIME - (2 * 3600)

DEMO_JOBS = [
    {
        'job_id': 1,
        'task_id': 0,  # Required field that was missing
        'job_name': 'llama-training',
        'task_name': 'llama-training',
        'status': ManagedJobStatus.RUNNING,
        'submitted_at': ONE_DAY_AGO,
        'start_at': ONE_DAY_AGO + 300,  # Started 5 minutes after submission
        'end_at': None,
        'job_duration': 3600,  # 1 hour (renamed from duration)
        'cluster_resources': '4x(gpus=A100:8, n1-standard-8, ...)',
        'cluster_resources_full': '4x(gpus=A100:8, cpus=8, mem=60, n1-standard-8)',
        'region': 'us-central1',
        'cloud': 'gcp',
        'zone': 'us-central1-a',
        'user_name': 'alice@skypilot.co',
        'user_hash': 'alice123',
        'workspace': 'ml-team',
        'resources': 'A100:8x4',
        'recovery_count': 0,
        'failure_reason': None,
        'schedule_state': ManagedJobScheduleState.ALIVE,  # Job is running
        'specs': None,
        'run_timestamp': str(ONE_DAY_AGO),
        'last_recovered_at': ONE_DAY_AGO + 300,
    },
    {
        'job_id': 2, 
        'task_id': 0,
        'job_name': 'batch-inference',
        'task_name': 'batch-inference',
        'status': ManagedJobStatus.SUCCEEDED,
        'submitted_at': TWO_DAYS_AGO,
        'start_at': TWO_DAYS_AGO + 180,  # Started 3 minutes after submission
        'end_at': ONE_DAY_AGO,
        'job_duration': 7200,  # 2 hours
        'cluster_resources': '1x(gpus=A100:1, g4dn.xlarge, ...)',
        'cluster_resources_full': '1x(gpus=A100:1, cpus=4, mem=16, g4dn.xlarge)',
        'region': 'us-west-2',
        'cloud': 'aws',
        'zone': 'us-west-2a',
        'user_name': 'mike@skypilot.co', 
        'user_hash': 'mike789',
        'workspace': 'default',
        'resources': 'A100:1',
        'recovery_count': 1,
        'failure_reason': None,
        'schedule_state': ManagedJobScheduleState.DONE,
        'specs': None,
        'run_timestamp': str(TWO_DAYS_AGO),
        'last_recovered_at': TWO_DAYS_AGO + 3600,  # Recovered after 1 hour
    },
    {
        'job_id': 3,
        'task_id': 0,
        'job_name': 'data-preprocessing', 
        'task_name': 'data-preprocessing',
        'status': ManagedJobStatus.PENDING,
        'submitted_at': ONE_HOUR_AGO,
        'start_at': None,  # Not started yet
        'end_at': None,
        'job_duration': 0,
        'cluster_resources': '-',
        'cluster_resources_full': '-',
        'region': '-',
        'cloud': '-',
        'zone': '-',
        'user_name': 'bob@skypilot.co',
        'user_hash': 'bob456',
        'workspace': 'default',
        'resources': 'CPUs:8',
        'recovery_count': 0,
        'failure_reason': None,
        'schedule_state': ManagedJobScheduleState.WAITING,
        'specs': None,
        'run_timestamp': str(ONE_HOUR_AGO),
        'last_recovered_at': -1,  # Not started yet
    },
    {
        'job_id': 4,
        'task_id': 0,
        'job_name': 'failed-experiment',
        'task_name': 'failed-experiment',
        'status': ManagedJobStatus.FAILED,
        'submitted_at': THREE_DAYS_AGO,
        'start_at': THREE_DAYS_AGO + 240,  # Started 4 minutes after submission
        'end_at': TWO_DAYS_AGO,
        'job_duration': 1800,  # 30 minutes
        'cluster_resources': '1x(gpus=A100:1, n1-standard-4, ...)',
        'cluster_resources_full': '1x(gpus=A100:1, cpus=4, mem=15, n1-standard-4)',
        'region': 'us-west1',
        'cloud': 'gcp',
        'zone': 'us-west1-b',
        'user_name': 'alice@skypilot.co',
        'user_hash': 'alice123', 
        'workspace': 'research-private',
        'resources': 'A100:1',
        'recovery_count': 2,
        'failure_reason': 'Out of memory error in training script',
        'schedule_state': ManagedJobScheduleState.DONE,
        'specs': None,
        'run_timestamp': str(THREE_DAYS_AGO),
        'last_recovered_at': THREE_DAYS_AGO + 7200,  # Last recovery 2 hours in
    },
    {
        'job_id': 5,
        'task_id': 0,
        'job_name': 'cancelled-run',
        'task_name': 'cancelled-run',
        'status': ManagedJobStatus.CANCELLED,
        'submitted_at': TWO_HOURS_AGO,
        'start_at': TWO_HOURS_AGO + 120,  # Started 2 minutes after submission
        'end_at': ONE_HOUR_AGO,
        'job_duration': 600,  # 10 minutes  
        'cluster_resources': '1x(cpus=4, mem=16, n1-standard-4, ...)',
        'cluster_resources_full': '1x(cpus=4, mem=16, n1-standard-4)',
        'region': 'us-central1',
        'cloud': 'gcp',
        'zone': 'us-central1-c',
        'user_name': 'bob@skypilot.co',
        'user_hash': 'bob456',
        'workspace': 'default', 
        'resources': 'CPUs:4',
        'recovery_count': 0,
        'failure_reason': None,
        'schedule_state': ManagedJobScheduleState.DONE,
        'specs': None,
        'run_timestamp': str(TWO_HOURS_AGO),
        'last_recovered_at': TWO_HOURS_AGO + 120,
    }
]

# Mock functions to replace global_user_state functions
def mock_get_clusters() -> List[Dict[str, Any]]:
    """Mock implementation of get_clusters."""
    # Deep copy to avoid handle mutations and recreate fresh handle objects
    result = []
    for cluster in DEMO_CLUSTERS:
        cluster_copy = copy.deepcopy(cluster)
        # Recreate the handle object to ensure it's not a string or mutated
        cluster_copy['handle'] = _create_demo_handle(
            cluster['name'], 
            cluster['handle'].launched_resources.cloud,
            cluster['handle'].launched_nodes
        )
        result.append(cluster_copy)
    
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
    for cluster in DEMO_CLUSTERS:
        if cluster['name'] == cluster_name:
            cluster_copy = copy.deepcopy(cluster)
            # Recreate the handle object to ensure it's not a string or mutated
            cluster_copy['handle'] = _create_demo_handle(
                cluster['name'], 
                cluster['handle'].launched_resources.cloud,
                cluster['handle'].launched_nodes
            )
            logger.info(f"Demo mode: Found cluster '{cluster_name}' in demo data")
            return cluster_copy
    
    logger.warning(f"Demo mode: Cluster '{cluster_name}' not found in demo data. Available clusters: {[c['name'] for c in DEMO_CLUSTERS]}")
    return None

def mock_get_all_users() -> List[models.User]:
    """Mock implementation of get_all_users."""
    return DEMO_USERS.copy()

def mock_get_user(user_id: str) -> Optional[models.User]:
    """Mock implementation of get_user."""
    for user in DEMO_USERS:
        if user.id == user_id:
            return user
    return None

def mock_get_storage() -> List[Dict[str, Any]]:
    """Mock implementation of get_storage."""
    return DEMO_STORAGE.copy()

def mock_get_volumes() -> List[Dict[str, Any]]:
    """Mock implementation of get_volumes."""
    return DEMO_VOLUMES.copy()

def mock_get_cached_enabled_clouds(cloud_capability, workspace: str) -> List:
    """Mock implementation of get_cached_enabled_clouds."""
    # Return actual cloud objects for demo
    mock_clouds = []
    if 'gcp' in DEMO_ENABLED_CLOUDS:
        mock_clouds.append(clouds.GCP())
    if 'aws' in DEMO_ENABLED_CLOUDS:
        mock_clouds.append(clouds.AWS())
    return mock_clouds

def mock_get_cluster_names_start_with(starts_with: str) -> List[str]:
    """Mock implementation of get_cluster_names_start_with."""
    return [cluster['name'] for cluster in DEMO_CLUSTERS 
            if cluster['name'].startswith(starts_with)]

def mock_get_glob_cluster_names(cluster_name: str) -> List[str]:
    """Mock implementation of get_glob_cluster_names."""
    import fnmatch
    logger.info(f"Demo mode: mock_get_glob_cluster_names() called with pattern='{cluster_name}'")
    
    # Get all cluster names from demo data
    all_cluster_names = [cluster['name'] for cluster in DEMO_CLUSTERS]
    
    # Apply glob pattern matching
    matching_names = [name for name in all_cluster_names if fnmatch.fnmatch(name, cluster_name)]
    
    logger.info(f"Demo mode: Pattern '{cluster_name}' matched clusters: {matching_names}")
    return matching_names

def mock_get_storage_names_start_with(starts_with: str) -> List[str]:
    """Mock implementation of get_storage_names_start_with."""
    return []

def mock_get_volume_names_start_with(starts_with: str) -> List[str]:
    """Mock implementation of get_volume_names_start_with."""
    return [volume['name'] for volume in DEMO_VOLUMES 
            if volume['name'].startswith(starts_with)]

# Infrastructure-related mock functions
def mock_enabled_clouds(workspace: Optional[str] = None, expand: bool = False) -> List[str]:
    """Mock implementation of core.enabled_clouds."""
    logger.info(f"Demo mode: mock_enabled_clouds() called with workspace={workspace}, expand={expand}")
    
    if not expand:
        return DEMO_ENABLED_CLOUDS.copy()
    else:
        # Return expanded infrastructure names
        # For demo purposes, just return the basic cloud names since we don't need complex infra expansion
        return DEMO_ENABLED_CLOUDS.copy()

def mock_get_all_contexts() -> List[str]:
    """Mock implementation of core.get_all_contexts."""
    logger.info("Demo mode: mock_get_all_contexts() called")
    
    # Return the Kubernetes contexts from our demo clusters
    # Based on the demo clusters, we have lambda-k8s and nebius-k8s
    contexts = ['lambda-cloud', 'nebius']
    
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
    
    # Define demo GPU availability for each context
    demo_gpu_data = {
        'lambda-cloud': [
            # Each entry: [gpu_name, [requestable_qty_per_node], total_capacity, available]
            ['A100', [1, 2, 4], 8, 6],  # 8 total A100s, 6 available
            ['H100', [1, 2], 4, 2],     # 4 total H100s, 2 available
        ],
        'nebius': [
            ['A100', [1, 2, 4, 8], 16, 8],  # 16 total A100s, 8 available  
            ['V100', [1, 2], 4, 4],         # 4 total V100s, all available
        ]
    }
    
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
            gpu_name, requestable_qtys, total_capacity, available = gpu_info
            
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
    
    # Define demo node information for each context
    demo_node_data = {
        'lambda-cloud': {
            'lambda-worker-1': {
                'name': 'lambda-worker-1',
                'accelerator_type': 'A100',
                'total': {'accelerator_count': 4},
                'free': {'accelerators_available': 3},
                'ip_address': '10.0.1.10',
            },
            'lambda-worker-2': {
                'name': 'lambda-worker-2', 
                'accelerator_type': 'A100',
                'total': {'accelerator_count': 4},
                'free': {'accelerators_available': 3},
                'ip_address': '10.0.1.11',
            },
            'lambda-gpu-node-1': {
                'name': 'lambda-gpu-node-1',
                'accelerator_type': 'H100', 
                'total': {'accelerator_count': 4},
                'free': {'accelerators_available': 2},
                'ip_address': '10.0.1.12',
            },
        },
        'nebius': {
            'nebius-worker-1': {
                'name': 'nebius-worker-1',
                'accelerator_type': 'A100',
                'total': {'accelerator_count': 8},
                'free': {'accelerators_available': 4},
                'ip_address': '10.0.2.10',
            },
            'nebius-worker-2': {
                'name': 'nebius-worker-2',
                'accelerator_type': 'A100', 
                'total': {'accelerator_count': 8},
                'free': {'accelerators_available': 4},
                'ip_address': '10.0.2.11',
            },
            'nebius-v100-node': {
                'name': 'nebius-v100-node',
                'accelerator_type': 'V100',
                'total': {'accelerator_count': 4},
                'free': {'accelerators_available': 4},
                'ip_address': '10.0.2.12',
            },
        }
    }
    
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
    
    # Return demo volumes data in the format expected by the volumes server
    volumes = copy.deepcopy(DEMO_VOLUMES)
    
    # Convert VolumeStatus enum objects to string values for JSON serialization
    for volume in volumes:
        if 'status' in volume and hasattr(volume['status'], 'value'):
            volume['status'] = volume['status'].value
    
    logger.info(f"Demo mode: Returning {len(volumes)} demo volumes")
    return volumes

def mock_get_workspaces() -> Dict[str, Any]:
    """Mock implementation of workspaces.core.get_workspaces."""
    logger.info("Demo mode: mock_get_workspaces() called")
    
    # Return demo workspaces configuration
    workspaces = copy.deepcopy(DEMO_WORKSPACES)
    
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
    # Check if user already exists
    for existing_user in DEMO_USERS:
        if existing_user.id == user.id:
            return False  # User already exists, not newly added
    
    # Add new user to demo users list
    DEMO_USERS.append(user)
    return True  # User was newly added

def mock_set_ssh_keys(*args, **kwargs):
    """Mock set_ssh_keys - no-op."""
    pass

def mock_get_ssh_keys(user_hash: str) -> tuple:
    """Mock get_ssh_keys - returns empty keys."""
    return '', '', False

def mock_get_current_user() -> models.User:
    """Mock get_current_user - returns default demo user."""
    return DEMO_USERS[0]  # Return Alice as default user

def mock_get_user_hash() -> str:
    """Mock get_user_hash - returns default demo user hash."""
    return DEMO_USERS[0].id

# Jobs-related mock functions
def mock_jobs_queue(*args, **kwargs) -> List[Dict[str, Any]]:
    """Mock implementation of core.queue - returns demo job data."""
    logger.info(f"Demo mode: mock_jobs_queue() called with args={args}, kwargs={kwargs}")
    
    # Apply filtering based on kwargs if needed
    jobs = copy.deepcopy(DEMO_JOBS)
    
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
    fake_controller_handle = _create_demo_handle('sky-jobs-controller-fake', clouds.GCP(), 1)
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
    
    # Return demo jobs data directly (bypass the JSON parsing)
    jobs = copy.deepcopy(DEMO_JOBS)
    logger.info(f"Demo mode: Returning {len(jobs)} demo jobs from load_managed_job_queue")
    return jobs

def mock_jobs_queue_direct(refresh: bool,
          skip_finished: bool = False,
          all_users: bool = False,
          job_ids: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    """Mock implementation of core.queue - returns demo job data in API format."""
    logger.info(f"Demo mode: mock_jobs_queue_direct() called with refresh={refresh}, skip_finished={skip_finished}, all_users={all_users}, job_ids={job_ids}")
    
    # Convert demo data to API format (matching core.queue docstring format exactly)
    api_jobs = []
    for job in DEMO_JOBS:
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

def patch_server_write_endpoints():
    """Patch server write endpoints to be read-only."""
    # Import server module to access the app
    from sky.server import server
    
    # List of write endpoints to make read-only
    write_endpoints = [
        '/launch', '/exec', '/stop', '/down', '/start', '/autostop', '/cancel',
        '/storage/delete', '/local_up', '/local_down', '/upload',
        '/volumes/delete', '/volumes/apply',
        '/workspaces/create', '/workspaces/update', '/workspaces/delete', '/workspaces/config'
    ]
    
    # We'll monkey-patch by replacing the route handlers
    # This is a bit tricky with FastAPI, so we'll use a different approach:
    # Add middleware to intercept write operations
    
    @server.app.middleware("http")
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
    
    # Patch server write endpoints  
    patch_server_write_endpoints()
    
    logger.info("Demo mode enabled successfully")

# Deploy demo mode patches. Executed when imported.
enable_demo_mode()
