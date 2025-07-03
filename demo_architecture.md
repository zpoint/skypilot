# SkyPilot Demo Mode Architecture

## Project Overview

**Goal**: Create a public demo for SkyPilot dashboard at `demo.skypilot.co` with read-only content and pre-populated fake data for sales demonstrations to potential customers (e.g., AI50 list).

**Requirements**:
- Read-only dashboard with realistic fake data
- Auto-updating with SkyPilot nightly releases (applied as patch)
- No demo flags in master branch (self-contained)
- Minimal surface area of changes to avoid merge conflicts

## Implementation Approach

**Solution**: Self-contained demo mode that monkey-patches `global_user_state` functions and server endpoints.

### Key Files
- `sky/demo_mode.py` - Main implementation (new file)
- `sky/server/server.py` - Added import to enable demo mode

### Architecture Pattern
1. **Monkey-patch database layer** - Replace `global_user_state` functions with mock implementations
2. **Block write operations** - FastAPI middleware intercepts and blocks write endpoints  
3. **Auto-enable on import** - No conditional logic needed, always active on this branch

## Implementation Details

### Demo Data Populated
- **Clusters**: 5 clusters (lambda-k8s, nebius-k8s, training-multinode-1, inference-cluster, dev-cluster-alice)
- **Users**: 3 users (alice@skypilot.co, bob@skypilot.co, mike@skypilot.co)
- **Clouds**: GCP and AWS enabled
- **Resources**: Mix of K8s, multinode training (A100:8x4), inference (A100:1), dev clusters
- **Statuses**: Realistic mix of UP/STOPPED with autostop settings

### Read-Only Enforcement
- **FastAPI Middleware**: Blocks POST/PUT/PATCH/DELETE to write endpoints
- **Mock Functions**: Write operations return 403 Forbidden
- **User Context**: CLI commands get proper authentication via mocked user functions

## Engineering Gotchas & Fixes

### 1. Handle Object Requirements
**Problem**: Initial mock classes caused `AttributeError: 'MockClusterHandle' object has no attribute 'cluster_yaml'`  
**Solution**: Must use real `CloudVmRayResourceHandle` objects from `sky.backends.cloud_vm_ray_backend`

### 2. Kubernetes Instance Type Format  
**Problem**: `ValueError: Invalid instance name: 4CPU-16GB`  
**Solution**: Kubernetes uses double hyphens: `4CPU--16GB` (not single hyphen)

### 3. Cloud Objects vs Strings
**Problem**: `AttributeError: 'str' object has no attribute 'expand_infras'`  
**Solution**: `mock_get_cached_enabled_clouds()` must return actual cloud objects (`clouds.GCP()`) not strings

### 4. Handle Object Mutations
**Problem**: `AttributeError: 'str' object has no attribute 'launched_nodes'` on second call  
**Solution**: Handle objects get mutated between calls. Use deep copy + recreate fresh handles each time:

```python
def mock_get_clusters():
    result = []
    for cluster in DEMO_CLUSTERS:
        cluster_copy = copy.deepcopy(cluster)
        cluster_copy['handle'] = _create_demo_handle(...)  # Fresh handle
        result.append(cluster_copy)
    return result
```

### 5. User Context for CLI
**Problem**: `AttributeError: 'NoneType' object has no attribute 'id'`  
**Solution**: Mock user context functions:
```python
common_utils.get_current_user = mock_get_current_user
common_utils.get_user_hash = mock_get_user_hash
```

## Design Practices

### Minimizing Change Surface Area
- **Single new file**: `sky/demo_mode.py` contains all demo logic
- **One import line**: Only change to existing code is importing demo_mode in server.py
- **Self-contained**: No flags, configuration, or conditional logic needed

### Realistic Demo Data
- **Authentic cluster names**: lambda-k8s, nebius-k8s (real partner names)
- **Realistic resources**: Mix of GPU types, node counts, regions that customers would recognize
- **Real YAML configs**: Actual training/inference/dev task examples
- **Proper timestamps**: Recent launch times, autostop settings

### Maintainability
- **Auto-enables**: No manual activation needed
- **Patch-friendly**: Designed to apply cleanly on nightly releases
- **Monkey-patching**: Intercepts at data layer, UI code unchanged

## Testing

To test, you need to stop the API server to reload changes. Next `sky` command will automatically start the API server.
```bash
# Test the demo mode
sky api stop
# Run a command, e.g. `sky status` should show demo clusters
```

## Deployment Notes

- Apply as patch to SkyPilot nightly builds
- No environment-specific configuration needed  
- Demo mode is always active when `sky/demo_mode.py` exists
- Remove the file to disable demo mode 