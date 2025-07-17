# SkyPilot Demo Mode

This directory contains the demo mode implementation for SkyPilot, providing a read-only demonstration experience with realistic fake data.

## 🆕 ConfigMap-Based Demo (Recommended for Kubernetes)

For Kubernetes deployments, use the new ConfigMap-based demo chart that keeps all demo functionality self-contained without modifying the main SkyPilot chart:

**📁 Location**: [`demo/charts/skypilot-demo/`](charts/skypilot-demo/)
**📚 Documentation**: [ConfigMap Demo README](charts/skypilot-demo/README.md)
**📊 Quick Start**:
```bash
cd demo/charts/skypilot-demo
helm dependency update
helm install skypilot-demo .
```

**✅ Benefits**:
- 🔒 **Non-intrusive**: Main SkyPilot chart remains unchanged
- 🔄 **Hot-reloadable**: Update mock data via `helm upgrade`
- 📦 **Self-contained**: All demo code in `/demo` directory
- 🔗 **Zero duplication**: Symbolic links to original JSON5 files
- 📄 **Single source**: Edit files in `mock_data/` and changes reflect instantly
- ⚙️ **Easy customization**: Direct file editing with comments support

---

## 📜 Original Demo Mode (Local Development)

The original demo mode files are still available for local development and understanding the implementation:

## Directory Structure

```
demo/
├── charts/skypilot-demo/          # 🆕 Self-contained demo Helm chart
│   ├── mock_data -> ../../mock_data # Symbolic link to mock data
│   ├── values.yaml                # Demo chart configuration
│   ├── Chart.yaml                 # Chart metadata with SkyPilot dependency
│   └── templates/                 # Demo-specific Kubernetes templates
├── mock_data/                     # Mock data files (JSON5 format, single source)
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
│   ├── inference/                # Logs for inference cluster
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

### Option 1: Using Kubernetes (Recommended for Production-like Environment)

Here are the most commonly used commands for demo deployment:

```bash
# Deploy demo (no authentication required)
gcloud auth configure-docker us-central1-docker.pkg.dev
DOCKER_IMAGE=us-central1-docker.pkg.dev/skypilot-demo-465806/skypilot-demo/demo:v2
NAMESPACE=skypilot-demo
RELEASE=skypilot-demo

# Build and push the Docker image
docker buildx build --push --platform linux/amd64 -t $DOCKER_IMAGE -f Dockerfile .

# 🆕 NEW: Deploy using the demo chart (recommended)
cd demo/charts/skypilot-demo
helm dependency update

# Note: Use 'skypilot.' prefix because the main SkyPilot chart is a dependency
helm upgrade --install $RELEASE . \
   --namespace $NAMESPACE \
   --set skypilot.apiService.image=$DOCKER_IMAGE \
   --set skypilot.apiService.skipResourceCheck=true \
   --create-namespace \
   --wait
```

To only update the mock data, you can make the changes to the JSON5 files in `demo/mock_data/` and then run:

```bash
helm upgrade $RELEASE . \
   --namespace $NAMESPACE \
   --set skypilot.apiService.skipResourceCheck=true \
   --reuse-values \
   --wait
```

### Option 2: Using the Local API Server

1. **Start the API Server with Demo Mode**:
   ```bash
   cd /path/to/skypilot
   SKYPILOT_INTERNAL_APPLY_DEMO_PATCH=true python -m sky.server.server
   ```

2. **Access the Dashboard**:
   - Open your browser and go to `http://localhost:8000`
   - You'll see the SkyPilot dashboard with demo data

## Demo Data

The demo includes:

- **4 Clusters**: `dev-alice`, `training-multinode`, `inference`, `dev-cluster-alice`
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

## Kubernetes Deployment Details

The demo deployment uses a clean, direct approach:

1. **Chart Enhancement**: Added support for `extraEnv` in the API server deployment template to accept additional environment variables
2. **Demo chart** (`charts/skypilot-demo/`): Self-contained chart with ConfigMap-based mock data and sets `SKYPILOT_INTERNAL_APPLY_DEMO_PATCH=true` via `extraEnv`
3. **Direct Helm commands**: Simple `helm upgrade --install` command with appropriate flags
4. **Standard cleanup**: Standard `helm uninstall` command

This approach works because:
- The demo files are already included in the Docker image when building from source
- The chart enhancement allows setting the demo environment variable directly through Helm values
- Direct Helm commands bypass resource checks and focus on demo essentials
- No scripts, runtime patching, or complex workarounds needed

## Important Notes

1. **Read-Only Mode**: Demo mode blocks all write operations to prevent modifications
2. **No Authentication**: Demo deployment disables all authentication mechanisms (basic auth, OAuth2, service accounts) for easy access
3. **Resource Requirements**: Demo deployment uses 3 CPUs and 12 GB memory (bypasses chart's 4 CPU minimum with `skipResourceCheck=true`)
4. **Environment Variable**: Demo mode is enabled via `SKYPILOT_INTERNAL_APPLY_DEMO_PATCH=true` in the demo chart's `values.yaml`
5. **Hot Reloading**: Changes to JSON5 files are reflected immediately without server restart
6. **Realistic Data**: All data is designed to be realistic for demonstration purposes
7. **Kubernetes Requirements**: For Kubernetes deployment, you need `kubectl` and `helm` installed and configured
8. **Working Directory**: Run helm commands from the `demo/charts/skypilot-demo/` directory where the demo chart is located

## Troubleshooting

### Kubernetes Deployment Issues

If the Kubernetes deployment fails:

1. **Check Prerequisites**: Ensure `kubectl` and `helm` are installed and configured
2. **Verify Cluster Access**: Run `kubectl cluster-info` to verify cluster connectivity
3. **Check Namespace**: Ensure the namespace doesn't already exist or use a different one
4. **Verify Chart Path**: Ensure the chart exists at `../charts/skypilot` relative to the demo directory
5. **Check Deployment Logs**: Use `kubectl logs -n skypilot-demo deployment/skypilot-demo-api-server` to check for errors
6. **Verify Chart Directory**: Ensure you're running the command from the `demo/charts/skypilot-demo/` directory where the demo chart is located

### Environment Variable Not Set

If demo mode isn't working in Kubernetes:

1. **Verify Values File**: Check that `demo/charts/skypilot-demo/values.yaml` contains the demo environment variable in the `skypilot.apiService.extraEnv` section
2. **Check Pod Environment**: Verify the environment variable in the running pod:
   ```bash
   kubectl exec -n skypilot-demo deployment/skypilot-demo-api-server -- env | grep SKYPILOT_INTERNAL_APPLY_DEMO_PATCH
   ```
3. **Check Deployment**: Verify the environment variable was set during deployment:
   ```bash
   kubectl get deployment skypilot-demo-api-server -n skypilot-demo -o jsonpath='{.spec.template.spec.containers[0].env[?(@.name=="SKYPILOT_INTERNAL_APPLY_DEMO_PATCH")].value}'
   ```

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

