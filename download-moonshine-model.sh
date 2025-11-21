#!/bin/bash -ex

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT_DIR=$(dirname $SCRIPTS_DIR)
MODELS_ROOT_DIR=${REPO_ROOT_DIR}/app/src/main/assets/models/moonshine

MODEL_TYPE=$1
MODEL_LANGUAGE=$2

if [ -z "$MODEL_LANGUAGE" ] || [ -z "$MODEL_TYPE" ]; then
    echo "Usage: $0 <model-language> <model-type>"
    exit 1
fi

if [ "$MODEL_TYPE" != "tiny" ] && [ "$MODEL_TYPE" != "base" ]; then
    echo "Invalid model type: $MODEL_TYPE"
    exit 1
fi

MODEL_NAME="${MODEL_LANGUAGE}-${MODEL_TYPE}"
if [ "$MODEL_LANGUAGE" == "en" ]; then
    HF_PATH="UsefulSensors/moonshine-${MODEL_TYPE}"
else
    HF_PATH="UsefulSensors/moonshine-${MODEL_TYPE}-${MODEL_LANGUAGE}"
fi

optimum-cli export onnx --model $HF_PATH $MODEL_NAME