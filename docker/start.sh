#!/usr/bin/env bash
if [ -z "$1" ]
	then
		GPU="all"
	else
		GPU="device=${1}"
fi
docker run -v ~/edhr:/workspace --env-file docker/env.list --rm --gpus $GPU --shm-size 16G -it edhr /bin/bash
