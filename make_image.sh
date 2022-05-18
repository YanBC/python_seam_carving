#!/bin/bash

USERID=`id -u`
VERSION=`git rev-parse --short HEAD`

function build() {
    docker build -t seam_carving:$VERSION --build-arg USERID=$USERID \
        .
    docker tag seam_carving:$VERSION seam_carving:latest
}

function build_gpu() {
    docker build -t seam_carving_cuda:$VERSION --build-arg USERID=$USERID \
        -f cuda.Dockerfile .
    docker tag seam_carving_cuda:$VERSION seam_carving_cuda:latest
}

function clear() {
    docker rmi seam_carving:$VERSION
    docker rmi seam_carving:latest
    docker rmi seam_carving_cuda:$VERSION
    docker rmi seam_carving_cuda:latest
}

main() {
    if [ "$1" = "build" ]; then
        build
    elif [ "$1" = "build_gpu" ]; then
        build_gpu
    elif [ "$1" = "clear" ]; then
        clear
    else
        echo "Usage: $0 build|build_gpu|clear"
    fi
}

main "$@"
