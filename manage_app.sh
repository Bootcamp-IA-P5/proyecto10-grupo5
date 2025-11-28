#!/bin/bash

# --- Script Configuration ---
BACKEND_COMPOSE_FILE="backend_compose.yml"
FRONTEND_COMPOSE_FILE="frontend_compose.yml"
NETWORK_NAME="nginx_proxy_network"
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

# --- Parameter Handling ---
ACTION=$1      # First parameter: up or down
PULL_IMAGES=$2 # Second optional parameter: 'pull'

# Check if the required parameter was provided
if [ -z "$ACTION" ]; then
    echo "Usage: $0 {up|down} [pull]"
    echo "  up: Starts the Docker Compose services."
    echo "    (Optional) Add 'pull' as the second argument to pull the latest images first."
    echo "  down: Stops and removes the Docker Compose services."
    exit 1
fi

case "$ACTION" in
    up)
        # 1. Check for and create the network
        check_and_create_network

        # 2. Handle optional 'pull' command
        if [ "$PULL_IMAGES" == "pull" ]; then
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
        
    *)
        echo "Invalid parameter: '$ACTION'"
        echo "Usage: $0 {up|down} [pull]"
        exit 1
        ;;
esac

exit 0