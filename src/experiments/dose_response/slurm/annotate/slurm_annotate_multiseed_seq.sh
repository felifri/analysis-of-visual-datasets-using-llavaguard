#!/bin/bash
#SBATCH --job-name=dose-annot-ms-seq
#SBATCH --output=<your folder>
#SBATCH --error=<your folder>
#SBATCH --qos=h200_comm_shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --time=24:00:00

# Annotate multi-seed images SEQUENTIALLY to avoid disk quota issues.
# Processes one (condition, seed) at a time: extract → annotate → cleanup → next.

set -euo pipefail

PYTHON=<your folder>
PORT=10001
DP_SIZE=8

# Disable proxy, force offline
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export SGLANG_DISABLE_CUDNN_CHECK=1

# Copy LlavaGuard model to local /tmp on first use
MODEL_SRC="<your folder>"
LLAVAGUARD="/tmp/LlavaGuard-v1.2-7B-OV"
if [ ! -d "$LLAVAGUARD" ]; then
    echo "Copying LlavaGuard model to $LLAVAGUARD..."
    cp -rL "$MODEL_SRC" "$LLAVAGUARD"
    echo "Model copy done."
fi

# Launch SGLang server once for all annotations
echo "Starting SGLang server on port $PORT with dp_size=$DP_SIZE..."
$PYTHON -m sglang.launch_server \
    --model-path "$LLAVAGUARD" \
    --host 127.0.0.1 \
    --port $PORT \
    --dp-size $DP_SIZE \
    --chat-template chatml-llava \
    --api-key sk-123456 &

SERVER_PID=$!

echo "Waiting for server to start..."
for i in $(seq 1 120); do
    if curl -s http://127.0.0.1:${PORT}/health > /dev/null 2>&1; then
        echo "Server ready after $((i * 5))s"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "Server process died"
        exit 1
    fi
    sleep 5
done

if ! curl -s http://127.0.0.1:${PORT}/health > /dev/null 2>&1; then
    echo "Server failed to start"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi

cd <your folder>

# Process each (condition, seed) sequentially
for CONDITION in C1 C2; do
    for SEED in 137 314 789 1331; do
        MODEL_ID="dose_${CONDITION}_seed_${SEED}"
        GEN_DIR="<your folder>"
        ANNOT_DIR="<your folder>"

        # Skip if parquet already exists
        if ls "${ANNOT_DIR}"/*.parquet 1>/dev/null 2>&1; then
            echo "=== Skipping ${MODEL_ID}: parquet already exists ==="
            continue
        fi

        echo ""
        echo "============================================================"
        echo "Processing ${MODEL_ID} at $(date)"
        echo "============================================================"

        mkdir -p "$ANNOT_DIR"

        # Unpack images from tar
        if [ -f "${GEN_DIR}/images.tar" ] && [ ! -d "${GEN_DIR}/images" ]; then
            echo "Unpacking images from ${GEN_DIR}/images.tar..."
            tar -xf "${GEN_DIR}/images.tar" -C "$GEN_DIR"
        fi

        if [ ! -d "${GEN_DIR}/images" ]; then
            echo "ERROR: No images directory found at ${GEN_DIR}/images. Skipping."
            continue
        fi

        # Run annotation
        $PYTHON -c "
import asyncio, io, json, logging, math, os, sys, time, base64, traceback
from glob import glob
from random import uniform

import openai
from PIL import Image
from tqdm.asyncio import tqdm

sys.path.insert(0, '.')
from util.annotation_utils import compress_annotations, save_json_annotations
from util.policy import POLICY_DEFAULT

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', force=True)
logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logger = logging.getLogger('annotate_multiseed')

def encode_image(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

async def request_batch(inputs, base_url, api_key='sk-123456', retries=3, timeout=300):
    async with openai.AsyncOpenAI(api_key=api_key, base_url=base_url) as client:
        hp = {'temperature': 0.2, 'top_p': 0.95, 'max_tokens': 500}
        async def fetch(inp, attempt=1):
            b64 = encode_image(inp['image'])
            try:
                resp = await asyncio.wait_for(
                    client.chat.completions.create(
                        model='default',
                        messages=[{'role': 'user', 'content': [
                            {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64}'}},
                            {'type': 'text', 'text': inp['prompt']},
                        ]}],
                        **hp,
                    ), timeout=timeout)
                return resp.choices[0].message.content.strip()
            except (openai.APITimeoutError, asyncio.TimeoutError):
                if attempt <= retries:
                    await asyncio.sleep(uniform(2, 5) * attempt)
                    return await fetch(inp, attempt + 1)
                return ''
            except Exception:
                return f'Error:\n{traceback.format_exc()}'
        results = await tqdm.gather(*[fetch(inp) for inp in inputs])
        return [r for r in results if r is not None]

images_dir = '${GEN_DIR}/images'
output_dir = '${ANNOT_DIR}'
port = ${PORT}
base_url = f'http://127.0.0.1:{port}/v1'
batch_size = 1000

paths = sorted(glob(os.path.join(images_dir, '*.jpg')))
names = [os.path.splitext(os.path.basename(p))[0] for p in paths]

# Skip already annotated
existing = set()
if os.path.isdir(output_dir):
    for f in os.listdir(output_dir):
        if f.endswith('.json') or f.endswith('.txt'):
            existing.add(f.rsplit('.', 1)[0])

todo_paths = [p for p, n in zip(paths, names) if n not in existing]
todo_names = [n for n in names if n not in existing]

logger.info(f'${MODEL_ID}: {len(todo_paths)} to annotate ({len(existing)} exist, {len(paths)} total)')

total_invalid = 0
for i in range(0, len(todo_paths), batch_size):
    batch_p = todo_paths[i:i+batch_size]
    batch_n = todo_names[i:i+batch_size]
    inputs = [{'image': p, 'prompt': POLICY_DEFAULT} for p in batch_p]
    annotations = asyncio.run(request_batch(inputs, base_url))
    invalid = save_json_annotations(annotations, output_dir, batch_n)
    total_invalid += len(invalid)
    logger.info(f'${MODEL_ID}: {i+len(batch_p)}/{len(todo_paths)} ({total_invalid} invalid)')

# Compress to parquet
compress_annotations(output_dir, os.path.dirname(output_dir), parquet_size=100000)
logger.info(f'${MODEL_ID}: annotation complete')
"

        # Remove unpacked images to free space for next iteration
        echo "Cleaning up ${GEN_DIR}/images..."
        rm -rf "${GEN_DIR}/images"
        echo "${MODEL_ID} completed at $(date)"
    done
done

# Shutdown server
echo "Shutting down server..."
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

echo "All multi-seed annotations completed at $(date)"
