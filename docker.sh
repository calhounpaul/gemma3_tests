#!/bin/bash

docker kill vllm-openai
docker rm vllm-openai

MODEL_ID="google/gemma-3-27b-it"

docker run --runtime nvidia --gpus all \
    --name vllm-openai \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env HUGGING_FACE_HUB_TOKEN=$(cat hf_token.txt) \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model $MODEL_ID \
    --gpu-memory-utilization 0.93 \
    --quantization bitsandbytes \
    --load-format bitsandbytes \
    --tensor-parallel-size 2 \
    --tokenizer-mode 'auto' \
    --limit-mm-per-prompt 'image=8,video=2' \
    --max-model-len 8192