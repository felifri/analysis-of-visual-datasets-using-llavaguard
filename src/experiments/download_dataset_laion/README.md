# Instructions

https://github.com/rom1504/img2dataset/blob/main/dataset_examples/laion5B.md

0. Run `conda activate /pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/envs/dataset_download`
0. Download .parquet files (contains urls to the images)
    * Request access to https://huggingface.co/datasets/laion/relaion2B-en-research
    * Create finegrained access token for HF and supply it as env variable
    * `cd /pfss/mlde/workspaces/mlde_wsp_Shared_Datasets/laion2B-en/`
    * `for i in {00000..00127}; do wget --header="Authorization: Bearer $HF_READ_TOKEN" https://huggingface.co/datasets/laion/relaion2B-en-research/resolve/main/part-$i-b31ba513-fc6b-4450-9ba4-a1bba183f408-c000.snappy.parquet; done`
0. Inspect one .parquet file to learn about column structure 
    * `cd /pfss/mlde/workspaces/mlde_wsp_Shared_Datasets/laion2B-en/metadata`
    * `parquet-tools inspect part-00000
-b31ba513-fc6b-4450-9ba4-a1bba183f408-c000.snappy.parquet`
0. Download the images
    * Install img2dataset
    * Run `python src/experiments/download_dataset_laion/entrypoint.py`

## Available Columns in .parquet files

* url
* similarity
* hash
* pwatermark
* punsafe
* caption
* key
* status
* error_message
* width
* height
* original_width
* original_height
* exif
* md5