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
    if 'kubernetes' in str(cloud).lower():
        instance_type = '4CPU--16GB'  # Valid Kubernetes instance type
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
        'workspace': 'default',
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
        'workspace': 'default',
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
DEMO_VOLUMES = []

# Demo enabled clouds
DEMO_ENABLED_CLOUDS = ['gcp', 'aws']

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
    return result

def mock_get_cluster_from_name(cluster_name: Optional[str]) -> Optional[Dict[str, Any]]:
    """Mock implementation of get_cluster_from_name."""
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
            return cluster_copy
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

def mock_get_storage_names_start_with(starts_with: str) -> List[str]:
    """Mock implementation of get_storage_names_start_with."""
    return []

def mock_get_volume_names_start_with(starts_with: str) -> List[str]:
    """Mock implementation of get_volume_names_start_with."""
    return []

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

def patch_server_write_endpoints():
    """Patch server write endpoints to be read-only."""
    # Import server module to access the app
    from sky.server import server
    
    # List of write endpoints to make read-only
    write_endpoints = [
        '/launch', '/exec', '/stop', '/down', '/start', '/autostop', '/cancel',
        '/storage/delete', '/local_up', '/local_down', '/upload'
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
    
    # Patch server write endpoints  
    patch_server_write_endpoints()
    
    logger.info("Demo mode enabled successfully")

# Auto-enable demo mode
enable_demo_mode() 