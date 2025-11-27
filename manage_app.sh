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
    
    # Check the exit code of the last command (docker network inspect)
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

# Check if the required parameter was provided
if [ -z "$1" ]; then
    echo "Usage: $0 {up|down}"
    echo "  up: Starts the Docker Compose services."
    echo "  down: Stops and removes the Docker Compose services."
    exit 1
fi

ACTION=$1

case "$ACTION" in
    up)
        # Run the network check function before starting services
        check_and_create_network

        echo "Starting $APP_NAME services..."
        # -d: Run containers in detached mode (in the background)
        # --build: Rebuild images if they have changed
        docker compose -f "$BACKEND_COMPOSE_FILE" up -d --build
        docker compose -f "$FRONTEND_COMPOSE_FILE" up -d --pull always
        
        if [ $? -eq 0 ]; then
            echo "✅ Services started successfully."
        else
            echo "❌ Error starting services."
        fi
        ;;
    
    down)
        echo "Stopping and removing services..."
        # --rmi all: Removes all images created by the build process
        # -v: Removes volumes declared in the Compose file
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
        echo "Usage: $0 {up|down}"
        exit 1
        ;;
esac

exit 0