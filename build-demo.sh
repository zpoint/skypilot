#!/bin/bash

# SkyPilot Demo Build Script
# Generates patch and builds demo Docker image
# Usage: ./build-demo.sh [IMAGE_NAME] [TAG] [OPTIONS]
# Example: ./build-demo.sh skypilot-demo latest --push --registry berkeleyskypilot

set -e

# Default values
DEFAULT_IMAGE_NAME="skypilot-demo"
DEFAULT_TAG="latest"
DOCKERFILE="Dockerfile.demo"
PATCH_FILE="skypilot-demo-mode.patch"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Help function
show_help() {
    cat << EOF
SkyPilot Demo Build Script

USAGE:
    $0 [IMAGE_NAME] [TAG] [OPTIONS]

ARGUMENTS:
    IMAGE_NAME    Docker image name (default: $DEFAULT_IMAGE_NAME)
    TAG           Docker image tag (default: $DEFAULT_TAG)

OPTIONS:
    -h, --help              Show this help message
    -f, --dockerfile FILE   Use custom Dockerfile (default: $DOCKERFILE)
    -p, --patch FILE        Use custom patch file (default: $PATCH_FILE)
    --no-cache              Build without using Docker cache
    --dry-run               Generate patch but don't build image
    --push                  Push image to registry after build
    --registry REGISTRY     Docker registry to push to

EXAMPLES:
    $0                                      # Build $DEFAULT_IMAGE_NAME:$DEFAULT_TAG
    $0 my-demo v1.0                        # Build my-demo:v1.0
    $0 my-demo latest --push               # Build and push my-demo:latest
    $0 demo nightly --registry my.reg.io  # Build demo:nightly and push to registry

CI USAGE:
    # GitHub Actions
    - name: Build Demo
      run: ./build-demo.sh \${{ github.repository }}-demo \${{ github.sha }}

    # GitLab CI
    script:
      - ./build-demo.sh \$CI_PROJECT_NAME-demo \$CI_COMMIT_SHA --push
EOF
}

# Parse arguments
IMAGE_NAME="$DEFAULT_IMAGE_NAME"
TAG="$DEFAULT_TAG"
NO_CACHE=""
DRY_RUN=false
PUSH=false
REGISTRY=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -f|--dockerfile)
            DOCKERFILE="$2"
            shift 2
            ;;
        -p|--patch)
            PATCH_FILE="$2"
            shift 2
            ;;
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --push)
            PUSH=true
            shift
            ;;
        --registry)
            REGISTRY="$2"
            shift 2
            ;;
        -*)
            log_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
        *)
            if [[ -z "$IMAGE_NAME" || "$IMAGE_NAME" == "$DEFAULT_IMAGE_NAME" ]]; then
                IMAGE_NAME="$1"
            elif [[ -z "$TAG" || "$TAG" == "$DEFAULT_TAG" ]]; then
                TAG="$1"
            else
                log_error "Too many arguments: $1"
                echo "Use --help for usage information"
                exit 1
            fi
            shift
            ;;
    esac
done

# Construct full image name
if [[ -n "$REGISTRY" ]]; then
    FULL_IMAGE_NAME="$REGISTRY/$IMAGE_NAME:$TAG"
else
    FULL_IMAGE_NAME="$IMAGE_NAME:$TAG"
fi

# Start build process
log_info "SkyPilot Demo Build Starting..."
echo "Image: $FULL_IMAGE_NAME"
echo "Dockerfile: $DOCKERFILE"
echo "Patch: $PATCH_FILE"
echo ""

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    log_error "Not in a git repository"
    exit 1
fi

# Find the merge base (common ancestor) with master
MERGE_BASE=$(git merge-base HEAD master 2>/dev/null)
if [[ -z "$MERGE_BASE" ]]; then
    log_error "Cannot find common ancestor with master branch"
    exit 1
fi

# Check if we have changes to create a patch from (since merge-base)
if ! git diff --quiet "$MERGE_BASE" 2>/dev/null; then
    log_info "Changes detected since branching from master ($MERGE_BASE)"
else
    log_warning "No changes detected since branching from master - patch will be empty"
fi

# Generate the patch (only changes since merge-base, not all diffs from current master)
log_info "Generating demo mode patch from merge-base..."
if git format-patch "$MERGE_BASE" --stdout > "$PATCH_FILE" 2>/dev/null; then
    PATCH_SIZE=$(wc -c < "$PATCH_FILE")
    if [[ $PATCH_SIZE -gt 0 ]]; then
        log_success "Patch generated: $PATCH_FILE ($PATCH_SIZE bytes)"
        
        # Show patch summary
        ADDED_FILES=$(grep "^+++ b/" "$PATCH_FILE" | wc -l)
        MODIFIED_FILES=$(grep "^--- a/" "$PATCH_FILE" | wc -l)
        log_info "Patch summary: $MODIFIED_FILES modified, $ADDED_FILES added"
    else
        log_warning "Empty patch generated - no changes from master"
    fi
else
    log_error "Failed to generate patch from master branch"
    exit 1
fi

# Check if Dockerfile exists
if [[ ! -f "$DOCKERFILE" ]]; then
    log_error "Dockerfile not found: $DOCKERFILE"
    exit 1
fi

# Dry run mode - stop here
if [[ "$DRY_RUN" == "true" ]]; then
    log_info "Dry run mode - patch generated but skipping Docker build"
    exit 0
fi

# Build Docker image
log_info "Building Docker image..."
BUILD_START=$(date +%s)

if docker build $NO_CACHE -f "$DOCKERFILE" -t "$FULL_IMAGE_NAME" .; then
    BUILD_END=$(date +%s)
    BUILD_TIME=$((BUILD_END - BUILD_START))
    log_success "Docker image built successfully in ${BUILD_TIME}s"
    
    # Show image info
    IMAGE_SIZE=$(docker images --format "table {{.Size}}" "$FULL_IMAGE_NAME" | tail -1)
    log_info "Image size: $IMAGE_SIZE"
else
    log_error "Docker build failed"
    exit 1
fi

# Push image if requested
if [[ "$PUSH" == "true" ]]; then
    log_info "Pushing image to registry..."
    if docker push "$FULL_IMAGE_NAME"; then
        log_success "Image pushed successfully: $FULL_IMAGE_NAME"
    else
        log_error "Failed to push image"
        exit 1
    fi
fi

# Final success message
echo ""
log_success "Build completed successfully!"
echo "Image: $FULL_IMAGE_NAME"
echo ""
echo "To run the demo:"
echo "  docker run -p 46580:46580 $FULL_IMAGE_NAME"
echo ""
echo "To access:"
echo "  Dashboard: http://localhost:46580"
echo "  API: http://localhost:46580" 