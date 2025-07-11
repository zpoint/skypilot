# SkyPilot Demo Mode

This directory contains the demo mode implementation and mock data for SkyPilot's demo.

## Overview

The demo mode loads realistic fake data from `mock_data.json` to simulate a working SkyPilot environment without requiring actual cloud resources. This allows users to explore the SkyPilot API and web interface safely.

## Enabling Demo Mode

Demo mode is only enabled when the environment variable `SKYPILOT_INTERNAL_APPLY_DEMO_PATCH` is set to `true`:

```bash
export SKYPILOT_INTERNAL_APPLY_DEMO_PATCH=true
# Then start your SkyPilot server or application
```

When enabled, demo mode will:
- ✅ **Load fake data** from `mock_data.json`
- ✅ **Block write operations** (returns HTTP 403 errors)
- ✅ **Enable hot reloading** of mock data
- ✅ **Patch all SkyPilot functions** to use demo data
- ✅ **Mock user authentication** and role-based permissions
- ✅ **Set current user** from demo data on all requests

## Files

- `demo_mode.py` - The main demo mode implementation that patches SkyPilot functions
- `mock_data.json` - Contains all demo data including users, clusters, jobs, volumes, workspaces, and infrastructure information

## Mock Data Structure

The JSON file contains the following sections:

### Users
Demo users that can access the system:
- `alice@skypilot.co` - Primary user with access to multiple workspaces
- `bob@skypilot.co` - Standard user with default workspace access  
- `mike@skypilot.co` - User with limited workspace access

### Clusters
Sample clusters in different states:
- Kubernetes clusters on Lambda Cloud and Nebius
- GCP and AWS clusters with various configurations
- Different statuses: UP, STOPPED, etc.

### Jobs
Sample managed jobs showing different states:
- Running training jobs
- Completed inference jobs
- Failed experiments
- Pending and cancelled jobs

### Volumes
Sample Kubernetes persistent volumes:
- Different storage classes and sizes
- Various usage states (IN_USE, READY)

### Workspaces
Three demo workspaces:
- `default` - Open access workspace
- `ml-team` - Team workspace with GCP and AWS access
- `research-private` - Private workspace with limited access

### User Roles & Permissions
Demo users with different role assignments:
- `alice@skypilot.co` - Admin with multiple roles: `["admin", "user"]`
- `bob@skypilot.co` - Standard user: `["user"]`
- `mike@skypilot.co` - User with viewer access: `["user", "viewer"]`

The current user is set to `alice@skypilot.co` by default and can be changed by modifying the `current_user` field in the JSON.

## Modifying Mock Data

To customize the demo data:

1. **Edit `mock_data.json`** directly with your preferred editor
2. **Timestamps** - All timestamps use offsets from current time:
   - `-3600` = 1 hour ago
   - `-86400` = 1 day ago
   - `-172800` = 2 days ago
3. **Status values** - Use string values like "UP", "STOPPED", "RUNNING", "FAILED", etc.
4. **Cloud types** - Use "gcp", "aws", "kubernetes"

### Example: Adding a New User

```json
{
  "id": "charlie999",
  "name": "charlie@skypilot.co",
  "password": null,
  "created_at_offset": -86400
}
```

### Example: Adding a New User with Roles

```json
{
  "id": "charlie999",
  "name": "charlie@skypilot.co",
  "password": null,
  "created_at_offset": -86400
}
```

And add their roles to the `user_roles` section:

```json
"user_roles": {
  "alice123": ["admin", "user"],
  "bob456": ["user"],
  "mike789": ["user", "viewer"],
  "charlie999": ["user", "operator"]
}
```

### Example: Adding a New Cluster

```json
{
  "name": "my-cluster",
  "launched_at_offset": -3600,
  "cloud": "aws",
  "region": "us-east-1",
  "nodes": 1,
  "last_use": "sky launch my-app.yaml",
  "status": "UP",
  "autostop": 30,
  "to_down": false,
  "owner": ["charlie999"],
  "metadata": {"cloud": "aws", "region": "us-east-1"},
  "cluster_hash": "cluster-hash-my-cluster",
  "storage_mounts_metadata": null,
  "cluster_ever_up": true,
  "status_updated_at_offset": -1800,
  "user_hash": "charlie999", 
  "user_name": "charlie@skypilot.co",
  "config_hash": "config-hash-new",
  "workspace": "default",
  "last_creation_yaml": "dev",
  "last_creation_command": "sky launch -c my-cluster app.yaml"
}
```

## Hot Reloading

The demo mode now features **automatic hot reloading** - the JSON file is loaded fresh on every API query! This means:

- ✅ **No server restart required** - Changes are reflected immediately
- ✅ **No manual reload needed** - Just edit the JSON file and refresh your browser
- ✅ **Real-time development** - Perfect for testing different data scenarios

### Example Hot Edit Workflow:

1. Edit `mock_data.json` with your changes
2. Save the file
3. Make any API request or refresh the SkyPilot web interface
4. See your changes immediately!
