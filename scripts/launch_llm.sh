if [ -z "$MODEL_NAME" ]; then
    echo "Error: MODEL_NAME environment variable is not set."
    exit 1
fi

vllm serve ${MODEL_NAME}  \
    --dtype auto \
    --api-key token-abc123 \
    --max_model_len 8192 \
    --data-parallel-size 8 \
    --tensor-parallel-size 1  \
    --enable-expert-parallel
