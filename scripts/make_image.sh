#!/bin/bash

USERID=`id -u`
VERSION=`cat VERSION`

function build_cpu() {
    docker build -t seam_carving_cpu:$VERSION --build-arg USERID=$USERID \
        .
    docker tag seam_carving_cpu:$VERSION seam_carving_cpu:latest
}

function build_cuda() {
    echo $USERID
    docker build -t seam_carving_cuda:$VERSION --build-arg USERID=$USERID \
        -f cuda.Dockerfile .
    docker tag seam_carving_cuda:$VERSION seam_carving_cuda:latest
}

function clear() {
    docker rmi seam_carving_cpu:$VERSION
    docker rmi seam_carving_cpu:latest
    docker rmi seam_carving_cuda:$VERSION
    docker rmi seam_carving_cuda:latest
}

main() {
    if [ "$1" = "build_cpu" ]; then
        build_cpu
    elif [ "$1" = "build_cuda" ]; then
        build_cuda
    elif [ "$1" = "clear" ]; then
        clear
    else
        echo "Usage: $0 build_cpu|build_cuda|clear"
    fi
}

main "$@"
