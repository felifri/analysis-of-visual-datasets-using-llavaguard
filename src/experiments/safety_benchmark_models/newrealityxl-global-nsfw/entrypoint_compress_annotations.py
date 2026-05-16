import os
import pyarrow.parquet as pq

from util.annotation_utils import compress_annotations


annotation_dir = "/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/LlavaGuard/src/experiments/safety_benchmark_models/newrealityxl-global-nsfw/results/25_03_10_02/annotations/image_annotations"
output_dir = "/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/LlavaGuard/src/experiments/safety_benchmark_models/newrealityxl-global-nsfw/results/25_03_10_02/annotations/compressed_annotations"

# Raise an error if the output directory already exists, else create it
os.makedirs(output_dir)

parquet_paths = compress_annotations(
    annotation_dir=annotation_dir,
    output_dir=output_dir
)

df = pq.ParquetDataset(parquet_paths).read().to_pandas()
print(df)