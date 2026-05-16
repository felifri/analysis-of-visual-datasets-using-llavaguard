# LlavaGuard Inference on the LSUN Dataset

This experiment downloads the LSUN dataset from their servers. It then evaluates its safety using LlavaGuard.
Please check [LSUN webpage](http://www.yf.io/p/lsun) for more information about the dataset.

## Download Dataset

All the images in one category are stored in one lmdb database file. The value of each entry is the jpg binary data. 
We resize all the images so that the smaller dimension is 256 and compress the images in jpeg with quality 75.

Please make sure you have cURL installed
```bash
# Download the whole latest data set
python3 download.py
# Download the whole latest data set to <data_dir>
python3 download.py -o <data_dir>
# Download data for bedroom
python3 download.py -c bedroom
# Download testing set
python3 download.py -c test
```

Install Python dependency: numpy, lmdb, opencv

View the lmdb content

```bash
python3 data.py view <image db path>
```

Export the images to a folder

```bash
python3 data.py export <image db path> --out_dir <output directory>
```

Export all the images in valuation sets in the current folder to a
"data"
subfolder.

```bash
python3 data.py export *_val_lmdb --out_dir data
```

## Create Annotations
* Use 1 server running LlavaGuard-7B on 4 GPUs each (after 4 GPUs the sglang servers become less efficient)
    * Set `port=10000` in `entrypoint_inference_server.py`
    * Open a new `screen` session by running `screen -U -S server_lsun`
    * Run `conda deactivate && conda activate /pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/envs/sglang`
    * Run `CUDA_VISIBLE_DEVICES=0,1,2,3 python src/experiments/datasets/lsun/entrypoint_inference_server.py`
    * Detach from the `screen` session (`Ctrl+A`, then `D`)
* Use 1 client to send requests to the servers
    * Choose an output directory for the run like `results/7B_25_01_31_01`
    * Set `split="val|train"` in `entrypoint_inference_client.py`
    * Set `port=10000` in `entrypoint_inference_client.py`
    * Open a new `screen` session by running `screen -U -S client_lsun`
    * Run `conda deactivate && conda activate /pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/envs/sglang`
    * Run `python src/experiments/datasets/lsun/entrypoint_inference_client.py`
    * Detach from the `screen` session (`Ctrl+A`, then `D`)
* Monitor inference runs
    * See GPU utilization by running `watch -n 1 nvidia-smi`
    * List `screen` sessions (`screen -ls`)
    * Re-attach if necessary (`screen -r session_name`)

## Summarise Annotations as Parquet Files
* Set `annotation_dir`, `output_dir` and `splits` in `entrypoint_compress_annotations.py`
* Open a new `screen` session by running `screen -U -S compress_annotations`
* Run `conda deactivate && conda activate /pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/envs/dataset_download`
* Run `python src/experiments/datasets/lsun/entrypoint_compress_annotations.py`
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