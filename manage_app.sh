#!/bin/bash

# --- Script Configuration ---
BACKEND_COMPOSE_FILE="docker/backend.compose.yml"
FRONTEND_COMPOSE_FILE="docker/frontend.compose.yml"
NETWORK_NAME="nginx_proxy_network"

# Define the image names to build/push (These must match service names in your compose files)
BACKEND_IMAGE="osrogon/project10-backend" 
FRONTEND_IMAGE="osrogon/project10-frontend"
# ----------------------------

# Function to check for and create the external network
check_and_create_network() {
    echo "Checking for external network: $NETWORK_NAME..."
    
    # Check if the network exists (suppressing error output if it doesn't)
    docker network inspect "$NETWORK_NAME" >/dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        echo "✅ Network '$NETWORK_NAME' already exists."
    else
        echo "⚠️ Network '$NETWORK_NAME' not found. Creating it now..."
        # Create the network using the standard bridge driver
        docker network create --driver bridge "$NETWORK_NAME"
        
        if [ $? -eq 0 ]; then
            echo "✅ Network '$NETWORK_NAME' created successfully."
        else
            echo "❌ Error creating network '$NETWORK_NAME'. Aborting 'up' command."
            exit 1
        fi
    fi
}

# Function to build and push multi-architecture images
build_and_push() {
    local ARCHS="$1" # e.g., linux/amd64,linux/arm64
    local TAG="$2"   # e.g., latest

    if [ -z "$ARCHS" ]; then
        echo "❌ Error: Architectures not specified for build_push."
        echo "Usage: $0 build_push {amd|arm|all} [tag]"
        exit 1
    fi

    # Set platforms based on input
    case "$ARCHS" in
        amd) PLATFORMS="linux/amd64";;
        arm) PLATFORMS="linux/arm64";;
        all) PLATFORMS="linux/amd64,linux/arm64";;
        *)
            echo "❌ Invalid architecture parameter: '$ARCHS'. Use 'amd', 'arm', or 'both'."
            exit 1
            ;;
    esac

    # Set tag (default to latest)
    if [ -z "$TAG" ]; then
        TAG="latest"
    fi
    
    echo "--- Building and Pushing Images for Platforms: $PLATFORMS with tag: $TAG ---"
    
    # Check for and set up buildx builder if needed
    if ! docker buildx inspect multiarch-builder >/dev/null 2>&1; then
        echo "Setting up multi-architecture builder (multiarch-builder)..."
        docker buildx create --name multiarch-builder --use
    else
        docker buildx use multiarch-builder
    fi

    # 1. Build and Push Backend Image
    echo "🚀 Building and pushing **Backend** image: ${BACKEND_IMAGE}:${TAG}"
    # The --file and context paths should match your docker-compose service definitions
    docker buildx build \
        --platform "$PLATFORMS" \
        -f docker/backend.Dockerfile \
        --tag "${BACKEND_IMAGE}:${TAG}" \
        --push \
        .
    
    if [ $? -ne 0 ]; then
        echo "❌ Backend build/push failed. Aborting."
        exit 1
    fi

    # 2. Build and Push Frontend Image
    echo "🚀 Building and pushing **Frontend** image: ${FRONTEND_IMAGE}:${TAG}"
    docker buildx build \
        --platform "$PLATFORMS" \
        -f docker/frontend.Dockerfile \
        --tag "${FRONTEND_IMAGE}:${TAG}" \
        --push \
        .
        
    if [ $? -ne 0 ]; then
        echo "❌ Frontend build/push failed. Aborting."
        exit 1
    fi

    docker buildx use default
    echo "✅ All images built and pushed successfully to Docker Hub."
}

# --- Parameter Handling ---
ACTION=$1      # First parameter: up, down, or build_push
ARCH_OR_PULL=$2 # Second optional parameter: 'pull' (for up) or 'amd|arm|both' (for build_push)
TAG=$3         # Third optional parameter: image tag (for build_push)


# Check if the required parameter was provided
if [ -z "$ACTION" ]; then
    echo "Usage: $0 {up|down|build_push} [pull|arch] [tag]"
    echo "  up: Starts the Docker Compose services."
    echo "    (Optional) Add 'pull' as the second argument to pull the latest images first."
    echo "  down: Stops and removes the Docker Compose services."
    echo "  build_push: Builds multi-architecture images and pushes them to Docker Hub."
    echo "    Required: Second argument must be the architecture: **amd**, **arm**, or **all**."
    echo "    (Optional) Third argument is the Docker tag (e.g., 'v1.0.0'). Defaults to 'latest'."
    exit 1
fi

case "$ACTION" in
    up)
        # 1. Check for and create the network
        check_and_create_network

        # 2. Handle optional 'pull' command
        if [ "$ARCH_OR_PULL" == "pull" ]; then
            echo "Pulling latest images before starting services..."
            
            # Pull images for both backend and frontend compose files
            docker compose -f "$BACKEND_COMPOSE_FILE" pull 
            docker compose -f "$FRONTEND_COMPOSE_FILE" pull
            
            if [ $? -eq 0 ]; then
                echo "✅ Image pull successful."
            else
                echo "❌ Error pulling images. Continuing with local images."
            fi
        fi

        echo "Starting services..."
        # -d: Run containers in detached mode (in the background)
        # --build: Always rebuilds images, which is useful if source code changed
        echo "Starting Backend service..."
        docker compose -f "$BACKEND_COMPOSE_FILE" up -d --build
        
        echo "Starting Frontend service..."
        docker compose -f "$FRONTEND_COMPOSE_FILE" up -d --build
        
        if [ $? -eq 0 ]; then
            echo "✅ Services started successfully."
        else
            echo "❌ Error starting services."
        fi
        ;;
    
    down)
        echo "Stopping and removing services..."
        # Stop and remove frontend first, then backend
        docker compose -f "$FRONTEND_COMPOSE_FILE" down --rmi local -v
        docker compose -f "$BACKEND_COMPOSE_FILE" down --rmi local -v
        
        if [ $? -eq 0 ]; then
            echo "✅ Services stopped and removed successfully."
        else
            echo "❌ Error stopping services."
        fi
        ;;
        
    build_push)
        # Use the arguments for architecture and tag
        build_and_push "$ARCH_OR_PULL" "$TAG"
        ;;
        
    *)
        echo "Invalid parameter: '$ACTION'"
        echo "Usage: $0 {up|down|build_push} [pull|arch] [tag]"
        exit 1
        ;;
esac

exit 0