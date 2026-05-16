import os
import pyarrow.parquet as pq

from util.annotation_utils import compress_annotations


base_annotation_dir = "/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/LlavaGuard/src/experiments/datasets/lsun/results/7B_25_03_21_01/train"
output_dir = "/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/LlavaGuard/src/experiments/datasets/lsun/results/7B_25_03_21_01/train_compressed"

# Raise an error if the output directory already exists, else create it
os.makedirs(output_dir, exist_ok=True)

parquet_paths = []

for split in ['tower']:
    print(f"Compressing annotations of split {split}...")
    annotation_dir = os.path.join(base_annotation_dir, split)

    parquet_paths.extend(compress_annotations(
        annotation_dir=annotation_dir,
        output_dir=output_dir
    ))

df = pq.ParquetDataset(parquet_paths).read().to_pandas()
print(df)