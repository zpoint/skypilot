# SkyPilot Demo Mode

This directory contains the demo mode implementation for SkyPilot, providing a read-only demonstration experience with realistic fake data.

## Directory Structure

```
demo/
├── mock_data/                     # Mock data files (JSON5 format)
│   ├── mock_users.json5          # User accounts and roles
│   ├── mock_clusters.json5       # Cluster configurations
│   ├── mock_jobs.json5           # Managed jobs data
│   ├── mock_cluster_jobs.json5   # Cluster-specific jobs
│   ├── mock_volumes.json5        # Volume and storage data
│   ├── mock_workspaces.json5     # Workspace configurations
│   └── mock_infrastructure.json5 # Infrastructure data (clouds, GPUs, nodes)
├── cluster_yamls/                 # YAML configurations for clusters
│   ├── dev.yaml                  # Development cluster config
│   ├── inference.yaml            # Inference cluster config
│   └── training.yaml             # Training cluster config
├── logs/                         # Realistic log files for cluster jobs
│   ├── dev-alice/                # Logs for dev-alice cluster
│   ├── training-multinode/       # Logs for training-multinode cluster
│   ├── inference-cluster/        # Logs for inference-cluster cluster
│   └── dev-cluster-alice/        # Logs for dev-cluster-alice cluster
├── demo_mode.py                  # Main demo mode implementation
└── README.md                     # This file
```

## Features

- **Hot Reloading**: JSON5 files are loaded fresh on every API query - edit any file and refresh to see changes
- **Real Log Files**: Authentic log content for different job types stored in actual files
- **Cluster Job Support**: Realistic cluster-specific job data with proper status enums
- **User Management**: Multiple demo users with different roles and permissions
- **Infrastructure Data**: Mock GPU availability, cloud configurations, and node information

## Starting the Demo

### Option 1: Using the API Server

1. **Start the API Server with Demo Mode**:
   ```bash
   cd /path/to/skypilot
   SKYPILOT_INTERNAL_APPLY_DEMO_PATCH=true python -m sky.server.server
   ```

2. **Access the Dashboard**:
   - Open your browser and go to `http://localhost:8000`
   - You'll see the SkyPilot dashboard with demo data

### Option 2: Using the Development Server

1. **Start the Development Server**:
   ```bash
   cd /path/to/skypilot
   SKYPILOT_INTERNAL_APPLY_DEMO_PATCH=true python -m sky.server.server --dev
   ```

2. **Access the Dashboard**:
   - The development server typically runs on `http://localhost:8000`

## Demo Data

The demo includes:

- **4 Clusters**: `dev-alice`, `training-multinode`, `inference-cluster`, `dev-cluster-alice`
- **7 Cluster Jobs**: Various jobs running on different clusters with different statuses
- **3 Users**: Demo users with different roles and permissions
- **Multiple Workspaces**: Different workspace configurations
- **Infrastructure Data**: Mock GPU availability and cloud information

## Customizing Demo Data

### Editing Mock Data

All mock data files are in JSON5 format (JSON with comments). You can edit them directly:

```bash
# Edit cluster data
vim demo/mock_data/mock_clusters.json5

# Edit cluster jobs
vim demo/mock_data/mock_cluster_jobs.json5

# Edit user data
vim demo/mock_data/mock_users.json5
```

### Editing Cluster YAML Configurations

Cluster YAML files are stored in `demo/cluster_yamls/`:

```bash
# Edit development cluster config
vim demo/cluster_yamls/dev.yaml

# Edit training cluster config
vim demo/cluster_yamls/training.yaml

# Edit inference cluster config
vim demo/cluster_yamls/inference.yaml
```

### Editing Log Files

Log files are stored in `demo/logs/` and are read by the demo mode:

```bash
# Edit logs for dev-alice cluster
vim demo/logs/dev-alice/job-1.log

# Edit logs for training cluster
vim demo/logs/training-multinode/job-1.log
```

## Important Notes

1. **Read-Only Mode**: Demo mode blocks all write operations to prevent modifications
2. **Environment Variable**: You must set `SKYPILOT_INTERNAL_APPLY_DEMO_PATCH=true` to enable demo mode
3. **Hot Reloading**: Changes to JSON5 files are reflected immediately without server restart
4. **Realistic Data**: All data is designed to be realistic for demonstration purposes

## Troubleshooting

### Cluster Jobs Not Showing

If cluster jobs aren't appearing in the dashboard:

1. **Check Demo Mode**: Ensure `SKYPILOT_INTERNAL_APPLY_DEMO_PATCH=true` is set
2. **Verify Server Logs**: Look for "Demo mode enabled" messages in the server logs
3. **Check Mock Data**: Verify `mock_cluster_jobs.json5` contains jobs for your cluster

### Server Not Starting

If the server fails to start:

1. **Check Dependencies**: Ensure all required packages are installed
2. **Check Port**: Make sure port 8000 is not already in use
3. **Check Environment**: Verify you're in the correct directory and virtual environment

## Development

To modify the demo mode behavior:

1. **Edit `demo_mode.py`**: Main implementation file
2. **Add New Mock Data**: Add new JSON5 files and update the loading logic
3. **Extend Functionality**: Add new mock functions as needed

The demo mode automatically patches SkyPilot functions when `demo_mode.py` is imported, making it easy to extend and customize.
