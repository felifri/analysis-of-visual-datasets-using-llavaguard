#!/bin/bash
#SBATCH --job-name=dose-annotate
#SBATCH --output=<your folder>
#SBATCH --error=<your folder>
#SBATCH --qos=h200_comm_shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --time=72:00:00

set -euo pipefail

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi

PYTHON=<your folder>
PORT=10001
MODEL="/tmp/LlavaGuard-v1.2-7B-OV"
DP_SIZE=8

# Disable proxy so HF hub falls back to cache without timeout delays
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export SGLANG_DISABLE_CUDNN_CHECK=1

# Copy model snapshot to a flat directory (avoids symlink issues with offline mode)
MODEL_SRC="<your folder>"
MODEL="/tmp/LlavaGuard-v1.2-7B-OV"
if [ ! -d "$MODEL" ]; then
    echo "Copying model to $MODEL (resolving symlinks)..."
    cp -rL "$MODEL_SRC" "$MODEL"
    echo "Model copy done."
fi

# Launch SGLang server in background (use all 8 GPUs with dp_size=8)
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

# Verify server is actually responding
if ! curl -s http://127.0.0.1:${PORT}/health > /dev/null 2>&1; then
    echo "Server failed to start within timeout"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi

# Run annotation script
cd <your folder>

$PYTHON experiments/dose_response/entrypoint_annotate_training_data.py \
    --download-dir <your folder> \
    --output-dir <your folder> \
    --batch-size 1000 \
    --dp-size $DP_SIZE \
    --port $PORT

# Cleanup
echo "Shutting down server..."
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

echo "Job completed at $(date)"
