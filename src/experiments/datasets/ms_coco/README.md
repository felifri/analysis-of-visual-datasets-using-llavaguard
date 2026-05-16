# LlavaGuard Inference on the MS COCO Dataset

This experiment downloads the MS COCO dataset from their servers. It then evaluates its safety using LlavaGuard.

## Download Dataset
* Give execute permissions to download script by running `chmod +x src/experiments/datasets/ms_coco/download.sh`
* Open a new `screen` session by running `screen -U -S download_ms_coco`
* Run `src/experiments/datasets/ms_coco/download.sh`
* Detach from the `screen` session (`Ctrl+A`, then `D`)

## Create Annotations
* Use 1 server running LlavaGuard-7B on 4 GPUs each (after 4 GPUs the sglang servers become less efficient)
    * Set `port=10000` in `entrypoint_inference_server.py`
    * Open a new `screen` session by running `screen -U -S server_ms_coco`
    * Run `conda deactivate && conda activate /pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/envs/sglang`
    * Run `CUDA_VISIBLE_DEVICES=0,1,2,3 python src/experiments/datasets/ms_coco/entrypoint_inference_server.py`
    * Detach from the `screen` session (`Ctrl+A`, then `D`)
* Use 1 client to send requests to the servers
    * Choose an output directory for the run like `results/7B_25_01_31_01`
    * Set `port=10000` in `entrypoint_inference_client.py`
    * Open a new `screen` session by running `screen -U -S client_ms_coco`
    * Run `conda deactivate && conda activate /pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/envs/sglang`
    * Run `python src/experiments/datasets/ms_coco/entrypoint_inference_client.py`
    * Detach from the `screen` session (`Ctrl+A`, then `D`)
* Monitor inference runs
    * See GPU utilization by running `watch -n 1 nvidia-smi`
    * List `screen` sessions (`screen -ls`)
    * Re-attach if necessary (`screen -r session_name`)

## Summarise Annotations as Parquet Files
* Set `annotation_dir`, `output_dir` and `splits` in `entrypoint_compress_annotations.py`
* Open a new `screen` session by running `screen -U -S compress_annotations`
* Run `conda deactivate && conda activate /pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/envs/dataset_download`
* Run `python src/experiments/datasets/ms_coco/entrypoint_compress_annotations.py`
* Once successfully saved to .parquet format, we can archive and delete the original annotations:
    * Archive annotations without compression by running `tar -cvf annotations.tar annotation_dir`
    * Delete the original annotations by running `rm -r annotation_dir`


## Troubleshooting
* If the inference failed at some point, check the `stats` folder in the output dir to quickly see information about each chunk
    * The last chunk will probably be only partially available
    * Delete the stats file for it as well as its directory in the `annotations` dir
    * Restart the servers and clients, but make sure to adapt the `chunk_ids_to_process` to exclude successful chunks
* There are cases when writing the `stats` json file itself procuded an error
    * In this case, the stats file itself will be skipped
    * The `annotations` dir of the chunk should be available though
    * Count the number of annotations in that dir by running `find . -mindepth 1 -type f -name "*.json" -printf x | wc -c`