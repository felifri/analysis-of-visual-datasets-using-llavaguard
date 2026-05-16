#!/bin/bash
# Annotate C5 SFT generated images with LlavaGuard
#
#SBATCH --job-name=annot-c5-sft
#SBATCH --output=<your folder>
#SBATCH --error=<your folder>
#SBATCH --qos=h200_dream_high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00

set -euo pipefail

CONDITION=C5
echo "Annotating sft/${CONDITION} on $(hostname) at $(date)"
nvidia-smi --query-gpu=index,name --format=csv,noheader

PYTHON=<your folder>
PORT=10001
DP_SIZE=8

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export SGLANG_DISABLE_CUDNN_CHECK=1
export PYTHONNOUSERSITE=1

MODEL_SRC="<your folder>"
LLAVAGUARD="/tmp/LlavaGuard-v1.2-7B-OV"
if [ ! -d "$LLAVAGUARD" ]; then
    cp -rL "$MODEL_SRC" "$LLAVAGUARD"
fi

GEN_DIR="<your folder>"
ANNOT_DIR="<your folder>"
mkdir -p "$ANNOT_DIR"

if [ -f "${GEN_DIR}/images.tar" ] && [ ! -d "${GEN_DIR}/images" ]; then
    tar -xf "${GEN_DIR}/images.tar" -C "$GEN_DIR"
fi

if [ ! -d "${GEN_DIR}/images" ]; then
    echo "ERROR: No images at ${GEN_DIR}/images"
    exit 1
fi

$PYTHON -m sglang.launch_server \
    --model-path "$LLAVAGUARD" \
    --host 127.0.0.1 \
    --port $PORT \
    --dp-size $DP_SIZE \
    --chat-template chatml-llava \
    --api-key sk-123456 &

SERVER_PID=$!

for i in $(seq 1 120); do
    if curl -s http://127.0.0.1:${PORT}/health > /dev/null 2>&1; then
        echo "Server ready after $((i * 5))s"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "Server process died"; rm -rf "${GEN_DIR}/images"; exit 1
    fi
    sleep 5
done

if ! curl -s http://127.0.0.1:${PORT}/health > /dev/null 2>&1; then
    kill $SERVER_PID 2>/dev/null || true; rm -rf "${GEN_DIR}/images"; exit 1
fi

cd <your folder>

$PYTHON -c "
import asyncio, json, logging, os, sys, time, base64, traceback
from glob import glob
from random import uniform
import openai
from tqdm.asyncio import tqdm
sys.path.insert(0, '.')
from util.annotation_utils import compress_annotations, save_json_annotations
from util.policy import POLICY_DEFAULT

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', force=True)
logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logger = logging.getLogger('annotate_sft')

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
base_url = f'http://127.0.0.1:${PORT}/v1'
batch_size = 1000

paths = sorted(glob(os.path.join(images_dir, '*.jpg')))
names = [os.path.splitext(os.path.basename(p))[0] for p in paths]
existing = set()
if os.path.isdir(output_dir):
    for f in os.listdir(output_dir):
        if f.endswith('.json') or f.endswith('.txt'):
            existing.add(f.rsplit('.', 1)[0])

todo_paths = [p for p, n in zip(paths, names) if n not in existing]
todo_names = [n for n in names if n not in existing]
logger.info(f'sft/C5: {len(todo_paths)} to annotate ({len(existing)} exist, {len(paths)} total)')

total_invalid = 0
for i in range(0, len(todo_paths), batch_size):
    batch_p = todo_paths[i:i+batch_size]
    batch_n = todo_names[i:i+batch_size]
    inputs = [{'image': p, 'prompt': POLICY_DEFAULT} for p in batch_p]
    annotations = asyncio.run(request_batch(inputs, base_url))
    invalid = save_json_annotations(annotations, output_dir, batch_n)
    total_invalid += len(invalid)
    logger.info(f'sft/C5: {i+len(batch_p)}/{len(todo_paths)} ({total_invalid} invalid)')

compress_annotations(output_dir, os.path.dirname(output_dir), parquet_size=100000)
logger.info('sft/C5: annotation complete')
"

kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
rm -rf "${GEN_DIR}/images"

echo "sft/C5 annotation completed at $(date)"
