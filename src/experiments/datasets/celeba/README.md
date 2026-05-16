# LlavaGuard Inference on the CelebA Dataset

This experiment downloads the CelebA dataset from Huggingface. It then evaluates its safety using LlavaGuard.

## Download Dataset
* Run `conda deactivate && conda activate /pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/envs/dataset_download`
* Run `python src/experiments/datasets/celeba/entrypoint_download.py download` to download via the huggingface `datasets` library
* Run `python src/experiments/datasets/celeba/entrypoint_download.py validate` to check if all images can be accessed without error

## Create Annotations
* Use 1 server running LlavaGuard-7B on 2 GPUs each (after 4 GPUs the sglang servers become less efficient)
    * Set `port=10000` in `entrypoint_inference_server.py`
    * Open a new `screen` session by running `screen -U -S server`
    * Run `conda deactivate && conda activate /pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/envs/sglang`
    * Run `CUDA_VISIBLE_DEVICES=0,1 python src/experiments/datasets/celeba/entrypoint_inference_server.py`
    * Detach from the `screen` session (`Ctrl+A`, then `D`)
* Use 1 client to send requests to the servers
    * Choose an output directory for the run like `results/7B_25_01_31_01`
    * Set `port=10000` in `entrypoint_inference_client.py`
    * Open a new `screen` session by running `screen -U -S client`
    * Run `conda deactivate && conda activate /pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/envs/sglang`
    * Run `python src/experiments/datasets/celeba/entrypoint_inference_client.py`
    * Detach from the `screen` session (`Ctrl+A`, then `D`)
* Monitor inference runs
    * See GPU utilization by running `watch -n 1 nvidia-smi`
    * List `screen` sessions (`screen -ls`)
    * Re-attach if necessary (`screen -r session_name`)

## Summarise Annotations as Parquet Files
* Set `annotation_dir`, `output_dir` and `splits` in `entrypoint_compress_annotations.py`
* Open a new `screen` session by running `screen -U -S compress_annotations`
* Run `conda deactivate && conda activate /pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/envs/dataset_download`
* Run `python src/experiments/datasets/celeba/entrypoint_compress_annotations.py`


## Troubleshooting
* If the inference failed at some point, check the `stats` folder in the output dir to quickly see information about each chunk
    * The last chunk will probably be only partially available
    * Delete the stats file for it as well as its directory in the `annotations` dir
    * Restart the servers and clients, but make sure to adapt the `chunk_ids_to_process` to exclude successful chunks
* There are cases when writing the `stats` json file itself procuded an error
    * In this case, the stats file itself will be skipped
    * The `annotations` dir of the chunk should be available though
    * Count the number of annotations in that dir by running `find . -mindepth 1 -type f -name "*.json" -printf x | wc -c`