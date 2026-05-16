#!/bin/bash
#SBATCH --job-name=dose-annot
#SBATCH --output=<your folder>
#SBATCH --error=<your folder>
#SBATCH --qos=h200_comm_shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --time=72:00:00
#SBATCH --array=0-7

set -euo pipefail

echo "Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID started on $(hostname) at $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

PYTHON=<your folder>
PORT=$((10001 + SLURM_ARRAY_TASK_ID))
NUM_SHARDS=8
SHARD_ID=$SLURM_ARRAY_TASK_ID
DP_SIZE=8

# Disable proxy, force offline, skip CuDNN check
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export SGLANG_DISABLE_CUDNN_CHECK=1

# Copy model to local /tmp on first use (per node)
MODEL_SRC="<your folder>"
MODEL="/tmp/LlavaGuard-v1.2-7B-OV"
if [ ! -d "$MODEL" ]; then
    echo "Copying model to $MODEL (resolving symlinks)..."
    cp -rL "$MODEL_SRC" "$MODEL"
    echo "Model copy done."
fi

# Per-shard output directory
OUTPUT_DIR="<your folder>"
mkdir -p "$OUTPUT_DIR"

# Launch SGLang server in background
echo "Starting SGLang server on port $PORT with dp_size=$DP_SIZE..."
$PYTHON -m sglang.launch_server \
    --model-path "$MODEL" \
    --host 127.0.0.1 \
    --port $PORT \
    --dp-size $DP_SIZE \
    --chat-template chatml-llava \
    --api-key sk-123456 &

SERVER_PID=$!

# Wait for server to be ready (up to 10 min)
echo "Waiting for server to start..."
for i in $(seq 1 120); do
    if curl -s http://127.0.0.1:${PORT}/health > /dev/null 2>&1; then
        echo "Server ready after $((i * 5))s"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "Server process died, check stderr"
        exit 1
    fi
    sleep 5
done

if ! curl -s http://127.0.0.1:${PORT}/health > /dev/null 2>&1; then
    echo "Server failed to start within timeout"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi

# Run annotation script with sharding
cd <your folder>

$PYTHON experiments/dose_response/entrypoint_annotate_training_data.py \
    --download-dir <your folder> \
    --output-dir "$OUTPUT_DIR" \
    --batch-size 1000 \
    --dp-size $DP_SIZE \
    --port $PORT \
    --num-shards $NUM_SHARDS \
    --shard-id $SHARD_ID

# Cleanup
echo "Shutting down server..."
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

echo "Shard $SHARD_ID completed at $(date)"
