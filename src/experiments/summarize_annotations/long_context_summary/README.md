## How to run Llama 4 Models

### Start a Server
* `screen -U -S llama-server`
* `conda deactivate && conda activate /pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/envs/sglang`
* `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m sglang.launch_server --model-path meta-llama/Llama-4-Scout-17B-16E-Instruct --tp 8 --context-length 128000 --disable-cuda-graph`
    * runs, but has unusable output after ca. 20K tokens
    * Maybe giving it 8 GPUs and enabling cuda-graph can improve some things?

## Send Requests with a Client
* `screen -U -S llama-client`
* `conda deactivate && conda activate /pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/envs/sglang`
* `python /pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/LlavaGuard/src/experiments/summarize_annotations/long_context_summary/long_context_summary.py`

## Play Around in a Juypter Notebook
* /pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/LlavaGuard/src/experiments/summarize_annotations/long_context_summary/long_context_summary.ipynb