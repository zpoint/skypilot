# SkyPilot Demo Mode Mock Data

This directory contains the mock data files for SkyPilot's demo mode. The data has been organized into multiple focused JSON5 files for better maintainability and performance.

## File Structure

### Core Data Files

- **`mock_data/mock_users.json5`** - User accounts, roles, and authentication
- **`mock_data/mock_clusters.json5`** - Cluster configurations and YAML templates
- **`mock_data/mock_jobs.json5`** - Managed jobs (jobs controller)
- **`mock_data/mock_cluster_jobs.json5`** - Cluster-specific jobs (jobs running on clusters)
- **`mock_data/mock_volumes.json5`** - Volume and storage configurations
- **`mock_data/mock_workspaces.json5`** - Workspace configurations and permissions
- **`mock_data/mock_infrastructure.json5`** - Infrastructure data (clouds, GPUs, nodes)

### Configuration Files

- **`demo_mode.py`** - Main demo mode implementation
- **`README.md`** - This documentation file

### Log Files

- **`logs/`** - Directory containing realistic log files for cluster jobs
  - **`logs/dev-alice/`** - Logs for jobs on the dev-alice cluster
  - **`logs/training-multinode/`** - Logs for jobs on the training-multinode cluster
  - **`logs/inference-cluster/`** - Logs for jobs on the inference-cluster cluster
  - **`logs/dev-cluster-alice/`** - Logs for jobs on the dev-cluster-alice cluster

## Features

### Hot Reloading
All JSON5 files are loaded fresh on every API request, enabling real-time editing without server restarts. Simply edit any file and refresh your browser to see changes immediately.

### JSON5 Format
All files support JSON5 format with:
- Comments (`// single line` and `/* multi-line */`)
- Trailing commas
- Unquoted keys
- Multi-line strings

## Data Types

### Users (`mock_users.json5`)
- User accounts with IDs, names, and creation timestamps
- User roles (admin, user, etc.)
- Current user designation

### Clusters (`mock_clusters.json5`)
- Cluster configurations with cloud providers, regions, and resources
- YAML templates for different workload types
- Cluster status and metadata

### Jobs (`mock_jobs.json5`)
- Managed jobs from the jobs controller
- Job status, resources, and execution details
- Job scheduling and recovery information

### Cluster Jobs (`mock_cluster_jobs.json5`)
- Jobs running directly on clusters (not through jobs controller)
- Cluster-specific job logs and resources
- Job execution status and timing

### Volumes (`mock_volumes.json5`)
- Storage volumes and persistent volumes
- Volume configurations and attachments
- Storage class and access mode information

### Workspaces (`mock_workspaces.json5`)
- Workspace configurations and permissions
- Cloud provider settings per workspace
- Access control and user restrictions

### Infrastructure (`mock_infrastructure.json5`)
- Available cloud providers and regions
- GPU availability and node information
- Kubernetes cluster details and capacity

## Usage

### Editing Demo Data
1. Edit any JSON5 file in the `mock_data/` directory
2. Refresh your browser or make a new API request
3. Changes will be reflected immediately

### Editing Log Files
1. Edit any log file in the `logs/` directory
2. Log changes are reflected immediately when viewing job logs
3. Create new log files by adding entries to `mock_data/mock_cluster_jobs.json5`

### Adding New Data
1. Follow the existing data structure patterns
2. Use relative timestamps (negative offsets from current time)
3. Ensure consistent user_hash and cluster references

### Troubleshooting
- Check server logs for JSON5 parsing errors
- Validate JSON5 syntax online if needed
- Ensure all referenced IDs exist across files

## Data Relationships

- Users are referenced by `user_hash` in other files
- Clusters are referenced by `name` in jobs and volumes
- Workspaces are referenced by `workspace` field
- Cloud regions must match available infrastructure

## Time Handling

All timestamps use offset-based values:
- Negative values: relative to current time (e.g., -3600 = 1 hour ago)
- Positive values: absolute Unix timestamps
- null values: not set/applicable
- -1 values: special "never" indicator
