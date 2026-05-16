# LlavaGuard Inference on the DataComp-1B Dataset

This experiment expects the DataComp-1B dataset to already be downloaded and its images to be in chunked folders. Workers can be started in parallel with a range of folders assigned to each.

## Prerequisites
* Ensure that a `conda` environment with the following dependencies is present:
  * `asyncio`
  * `pillow`
  * `pyarrow`
  * `rtpt`
  * `safetensors`
  * `screen`
  * `sglang`
  * `sglang_router`
  * `torch`
  * `torchvision`
  * `tqdm`
  * `transformers`
* Set `base_image_dir` and `base_output_dir` in `inference_client.py`

## Inference
* Use 2 servers running LlavaGuard-7B on 4 GPUs each (after 4 GPUs the sglang servers become less efficient)
    * Set `port=10000` in `inference_server.py`
    * Open a new `screen` session by running `screen -S server1`
    * Run `CUDA_VISIBLE_DEVICES=0,1,2,3 python inference_server.py`
    * Detach from the `screen` session (`Ctrl+A`, then `D`)
    * Open a new `screen` session by running `screen -S server2`
    * Set `port=10001` in `inference_server.py`
    * Run `CUDA_VISIBLE_DEVICES=4,5,6,7 python inference_server.py`
    * Detach from the `screen` session (`Ctrl+A`, then `D`)
* Use 2 clients to send requests to the servers
    * Choose an output directory for the run like `results/7B_25_01_23_01`
    * Set `port=10000` in `inference_client.py`
    * Set `chunk_ids_to_process = range(0, 200)` in `inference_client.py`
    * Open a new `screen` session by running `screen -L -Logfile path_to_logfile -S client1`
    * Run `time python inference_client.py`
    * Detach from the `screen` session (`Ctrl+A`, then `D`)
    * Set `port=10001` in `inference_client.py`
    * Set `chunk_ids_to_process = range(200, 400)` in `inference_client.py`
    * Open a new `screen` session by running `screen -L -Logfile path_to_logfile -S client2`
    * Run `time python inference_client.py` in a new terminal and leave open
    * Detach from the `screen` session (`Ctrl+A`, then `D`)
* Monitor inference runs
    * See GPU utilization by running `watch -n 1 nvidia-smi`
    * List `screen` sessions (`screen -ls`)
    * Re-attach if necessary (`screen -r session_name`)