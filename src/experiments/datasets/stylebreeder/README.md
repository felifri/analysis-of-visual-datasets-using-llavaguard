# LlavaGuard Inference on the Stylebreeder Dataset

This experiment downloads the Stylebreeder dataset from Huggingface. It is split into a 2M part where the images are already included and the full 4M part where images need to be downloaded afterwards. The 2M split are the entries with the lowest NSFW scores, so naturally we need to work with the full 4M split.

## Download
* Run `conda deactivate && conda activate /pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/envs/dataset_download`
* Run `python entrypoint_download.py` to download using 128 workers and 50 img/s per worker via the artbreeder CDN

## Inference
* Run `conda deactivate && conda activate /pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/envs/sglang`
* Use 2 servers running LlavaGuard-7B on 4 GPUs each (after 4 GPUs the sglang servers become less efficient)
    * Set `port=10000` in `entrypoint_inference_server.py`
    * Open a new `screen` session by running `screen -S server1`
    * Run `CUDA_VISIBLE_DEVICES=0,1,2,3 python entrypoint_inference_server.py`
    * Detach from the `screen` session (`Ctrl+A`, then `D`)
    * Open a new `screen` session by running `screen -S server2`
    * Set `port=10001` in `entrypoint_inference_server.py`
    * Run `CUDA_VISIBLE_DEVICES=4,5,6,7 python entrypoint_inference_server.py`
    * Detach from the `screen` session (`Ctrl+A`, then `D`)
* Use 2 clients to send requests to the servers
    * Choose an output directory for the run like `results/7B_25_01_23_01`
    * Set `port=10000` in `entrypoint_inference_client.py`
    * Set `chunk_ids_to_process = range(0, 200)` in `entrypoint_inference_client.py`
    * Open a new `screen` session by running `screen -S client1`
    * Run `time python entrypoint_inference_client.py`
    * Detach from the `screen` session (`Ctrl+A`, then `D`)
    * Set `port=10001` in `entrypoint_inference_client.py`
    * Set `chunk_ids_to_process = range(200, 400)` in `entrypoint_inference_client.py`
    * Open a new `screen` session by running `screen -S client2`
    * Run `time python entrypoint_inference_client.py` in a new terminal and leave open
    * Detach from the `screen` session (`Ctrl+A`, then `D`)
* Monitor inference runs
    * See GPU utilization by running `watch -n 10 nvidia-smi`
    * List `screen` sessions (`screen -ls`)
    * Re-attach if necessary (`screen -r session_name`)


## Troubleshooting
* If the inference failed at some point, check the `stats` folder in the output dir to quickly see information about each chunk
    * The last chunk will probably be only partially available
    * Delete the stats file for it as well as its directory in the `annotations` dir
    * Restart the servers and clients, but make sure to adapt the `chunk_ids_to_process` to exclude successful chunks
* There are cases when writing the `stats` json file itself procuded an error
    * In this case, the stats file itself will be skipped
    * The `annotations` dir of the chunk should be available though
    * Count the number of annotations in that dir by running `find . -mindepth 1 -type f -name "*.json" -printf x | wc -c`