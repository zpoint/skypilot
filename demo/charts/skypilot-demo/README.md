# SkyPilot Demo Chart

This Helm chart deploys SkyPilot in demo mode with ConfigMap-based mock data, without modifying the main SkyPilot chart. It uses the main SkyPilot chart as a dependency and adds demo-specific functionality through ConfigMaps and deployment patches.

## Architecture

This demo chart:
- **Depends on the main SkyPilot chart** as a subchart dependency
- **Creates ConfigMaps** containing mock data in JSON5 format
- **Mounts ConfigMaps as files** using `extraVolumes`/`extraVolumeMounts` for optimal performance
- **Keeps all demo code** isolated in the `/demo` directory

## Quick Start

### 1. Initialize Dependencies

```bash
cd demo/charts/skypilot-demo
helm dependency update
```

### 2. Deploy Demo

```bash
# Basic deployment with default image
helm install skypilot-demo . \
  --namespace skypilot-demo \
  --create-namespace

# Or with custom Docker image (recommended for production)
DOCKER_IMAGE=us-central1-docker.pkg.dev/skypilot-demo-465806/skypilot-demo/demo:v2
helm install skypilot-demo . \
  --namespace skypilot-demo \
  --set skypilot.apiService.image=$DOCKER_IMAGE \
  --set skypilot.apiService.skipResourceCheck=true \
  --create-namespace
```

### 3. Access the Demo

```bash
kubectl port-forward -n skypilot-demo service/skypilot-demo-skypilot-api 8080:80
open http://localhost:8080
```

### 4. Update Mock Data

```bash
# Edit JSON5 files directly
vim mock_data/mock_users.json5
vim mock_data/mock_clusters.json5

# Upgrade with new data (no Docker rebuild!)
helm upgrade skypilot-demo .

# Data is automatically available via mounted files (~1ms access time)
```

## Files Structure

```
demo/charts/skypilot-demo/
├── Chart.yaml                    # Chart metadata with SkyPilot dependency
├── values.yaml                   # Demo configuration (no hardcoded data)
├── mock_data -> ../../mock_data  # Symbolic link to JSON5 mock data files
├── charts/                       # Downloaded dependencies (after helm dep update)
├── README.md                     # This file
└── templates/
    ├── demo-configmaps.yaml      # ConfigMaps reading from JSON5 files
    └── demo-deployment-patch.yaml # Deployment patch for demo volumes
```

## Configuration

### Configuration (values.yaml)

The values file contains:
- Demo enablement flag (`demo.enabled: true`)
- SkyPilot chart configuration under `skypilot` key
- No hardcoded mock data - all data loaded from JSON5 files

### Mock Data Files (mock_data -> ../../mock_data)

All mock data is accessed via symbolic link to the main demo directory:
- Single source of truth - no file duplication
- Supports JSON5 features like comments and trailing commas
- Changes to original files immediately reflected in chart
- Automatically loaded into ConfigMaps by Helm templates

### Custom Configuration

To customize the demo:

**Option 1: Edit JSON5 files directly**
```bash
# Edit the original mock data files (changes auto-reflected via symlink)
vim ../../mock_data/mock_users.json5
vim ../../mock_data/mock_clusters.json5

# Or edit via the symlink (same effect)
vim mock_data/mock_users.json5
vim mock_data/mock_clusters.json5
```

**Option 2: Create custom values file**
```yaml
# my-demo.yaml
demo:
  enabled: true

skypilot:
  apiService:
    # Customize SkyPilot settings
    resources:
      requests:
        cpu: "4"
        memory: "16Gi"
    # ... other SkyPilot configuration
```

```bash
helm install my-demo . -f my-demo.yaml
```

## Development Workflow

### 1. Update Mock Data

```bash
# Edit JSON5 files (via symlink or directly in main demo directory)
vim mock_data/mock_users.json5
# OR
vim ../../mock_data/mock_users.json5

# Redeploy (Helm automatically reads updated files via symlink)
helm upgrade skypilot-demo .
```

### 2. Test Configuration Changes

```bash
# Test template rendering
helm template skypilot-demo .

# Validate dependencies
helm dependency list
helm dependency update

# Check if JSON5 files are being read correctly
helm template skypilot-demo . | grep -A 5 "mock_users.json5"
```

### 3. Debug Deployment

```bash
# Check ConfigMaps
kubectl get configmaps -n skypilot-demo

# Check deployment patch
kubectl describe deployment skypilot-demo-api-server -n skypilot-demo

# Check logs
kubectl logs -n skypilot-demo -l app=skypilot-demo-api
```

## Advantages of This Approach

1. **🔒 Non-intrusive**: Main SkyPilot chart remains unchanged
2. **📦 Self-contained**: All demo functionality in `/demo` directory
3. **🔄 Hot-reloadable**: Update mock data via Helm upgrades
4. **🔗 Zero duplication**: Symbolic links eliminate file copying
5. **⚡ High performance**: ~1ms data access via mounted files
6. **🏗️ Maintainable**: Clear separation of concerns using standard Helm features
7. **🚀 Flexible**: Easy to customize for different demo scenarios

## Chart Dependencies

This chart depends on:
- `skypilot` (file://../../..) - The main SkyPilot Helm chart

Dependencies are managed automatically by Helm when you run `helm dependency update`.

## Environment Variables

The demo chart sets these environment variables in the SkyPilot API container:

- `SKYPILOT_INTERNAL_APPLY_DEMO_PATCH=true` - Enables demo mode

## Performance Optimization

The `demo_mode.py` uses a simple 2-tier loading approach:

1. **Mounted files** (~1ms) - Primary: Direct filesystem access to mounted ConfigMaps
2. **Embedded files** (~5ms) - Fallback: Local JSON5 files for development

This ensures optimal performance while maintaining simplicity and compatibility.

## ConfigMaps Created

When `demo.enabled: true`, this chart creates:

- `{release-name}-demo-users` - User accounts and roles
- `{release-name}-demo-clusters` - Cluster configurations
- `{release-name}-demo-jobs` - Managed job data
- `{release-name}-demo-cluster-jobs` - Cluster job data
- `{release-name}-demo-volumes` - Volume and storage data
- `{release-name}-demo-workspaces` - Workspace configurations
- `{release-name}-demo-infrastructure` - Infrastructure data

## Volume Mounts

The demo chart automatically mounts ConfigMaps as files using `extraVolumes`/`extraVolumeMounts`:

```
/etc/skypilot/demo/mock_data/mock_users.json5
/etc/skypilot/demo/mock_data/mock_clusters.json5
/etc/skypilot/demo/mock_data/mock_jobs.json5
/etc/skypilot/demo/mock_data/mock_cluster_jobs.json5
/etc/skypilot/demo/mock_data/mock_volumes.json5
/etc/skypilot/demo/mock_data/mock_workspaces.json5
/etc/skypilot/demo/mock_data/mock_infrastructure.json5
```

This provides optimal performance (~1ms access) with automatic hot reloading via Kubernetes.

## Troubleshooting

### Common Issues

1. **Dependencies not found**
   ```bash
   cd demo/charts/skypilot-demo
   helm dependency update
   ```

2. **ConfigMaps not updating**
   ```bash
   # Ensure you're upgrading, not installing
   helm upgrade skypilot-demo .
   ```

3. **Template errors**
   ```bash
   # Validate templates
   helm template skypilot-demo . --debug
   ```

### Debug Commands

```bash
# Check chart status
helm status skypilot-demo -n skypilot-demo

# Check all resources
kubectl get all -n skypilot-demo

# Check ConfigMap content
kubectl get configmap skypilot-demo-demo-users -o yaml

# Check container environment
kubectl exec -n skypilot-demo deploy/skypilot-demo-api-server -- env | grep DEMO
```

## Migration from Main Chart Modifications

If you previously modified the main charts, migrate by:

1. **Remove changes from main chart**
2. **Use this demo chart instead**
3. **Move custom values to demo chart format**

The demo code automatically falls back to embedded files if ConfigMaps aren't present, ensuring compatibility. 
