#!/bin/bash

USERID=`id -u`
VERSION=`git rev-parse --short HEAD`

function build() {
    docker build -t seam_carving:$VERSION --build-arg USERID=$USERID .
    docker tag seam_carving:$VERSION seam_carving:latest
}

function build_zh() {
    docker build -t seam_carving_zh:$VERSION --build-arg USERID=$USERID -f CN.Dockerfile .
    docker tag seam_carving_zh:$VERSION seam_carving_zh:latest
}

function clear() {
    docker rmi seam_carving:$VERSION
    docker rmi seam_carving:latest
}

main() {
    if [ "$1" = "build" ]; then
        build
    elif [ "$1" = "build_zh" ]; then
        build_zh
    elif [ "$1" = "clear" ]; then
        clear
    else
        echo "Usage: $0 build|clear"
    fi
}

main "$@"
