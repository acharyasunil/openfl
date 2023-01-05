#!/bin/bash
set -e
ENVOY_NAME=$1
ENVOY_CONF=$2

# export TF_FORCE_GPU_ALLOW_GROWTH=true
export CUDA_VISIBLE_DEVICES="-1"

fx envoy start -n "$ENVOY_NAME" --disable-tls --envoy-config-path "$ENVOY_CONF" -dh localhost -dp 50051
