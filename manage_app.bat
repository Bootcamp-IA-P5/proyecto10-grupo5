@echo off
REM Windows batch equivalent of manage_app.sh

REM --- Script Configuration ---
set "BACKEND_COMPOSE_FILE=docker/backend_compose.yml"
set "FRONTEND_COMPOSE_FILE=docker/frontend_compose.yml"
set "NETWORK_NAME=nginx_proxy_network"
REM ----------------------------
:usage
echo Usage: %~nx0 {up|down} [pull]
echo  up: Starts the Docker Compose services.
echo    (Optional) Add "pull" as the second argument to pull the latest images first.
echo  down: Stops and removes the Docker Compose services.
exit /B 1
:check_and_create_network
echo Checking for external network: %NETWORK_NAME%...
docker network inspect "%NETWORK_NAME%" >NUL 2>&1IF %ERRORLEVEL%==0 (    echo Network "%NETWORK_NAME%" already exists.) ELSE (    echo Network "%NETWORK_NAME%" not found. Creating it now...    docker network create --driver bridge "%NETWORK_NAME%"    IF %ERRORLEVEL%==0 (        echo Network "%NETWORK_NAME%" created successfully.    ) ELSE (        echo Error creating network "%NETWORK_NAME%". Aborting.        exit /B 1    ))
exit /B 0
:: --- Parameter Handling ---
IF "%~1"=="" (    call :usage)

set "ACTION=%~1"
set "PULL_IMAGES=%~2"

if /I "%ACTION%"=="up" goto :up
if /I "%ACTION%"=="down" goto :down

echo Invalid parameter: '%ACTION%'
call :usage

:: --- Up ---
:up
call :check_and_create_network

if /I "%PULL_IMAGES%"=="pull" (    echo Pulling latest images before starting services...    docker compose -f "%BACKEND_COMPOSE_FILE%" pull    IF %ERRORLEVEL%==0 (        echo Pulled backend images.    ) ELSE (        echo Error pulling backend images. Continuing with local images.    )    docker compose -f "%FRONTEND_COMPOSE_FILE%" pull    IF %ERRORLEVEL%==0 (        echo Pulled frontend images.    ) ELSE (        echo Error pulling frontend images. Continuing with local images.
    ))

echo Starting services...

echo Starting Backend service...
docker compose -f "%BACKEND_COMPOSE_FILE%" up -d --build
IF %ERRORLEVEL% NEQ 0 (    echo Error starting backend service.)

echo Starting Frontend service...
docker compose -f "%FRONTEND_COMPOSE_FILE%" up -d --build
IF %ERRORLEVEL%==0 (    echo Services started successfully.) ELSE (    echo Error starting services.)
exit /B 0

:: --- Down ---
:down
echo Stopping and removing services...

docker compose -f "%FRONTEND_COMPOSE_FILE%" down --rmi local -v
IF %ERRORLEVEL% NEQ 0 (    echo Error stopping frontend service.)
docker compose -f "%BACKEND_COMPOSE_FILE%" down --rmi local -v
IF %ERRORLEVEL%==0 (    echo Services stopped and removed successfully.) ELSE (    echo Error stopping services.)
exit /B 0