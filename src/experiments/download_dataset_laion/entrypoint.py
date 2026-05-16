import dns.resolver
from img2dataset import download
import os

# time img2dataset --url_list=test_10000.parquet --image_size=336 --output_folder=data_336 --output_format="files" --input_format "parquet" --url_col "URL" --caption_col "TEXT" --enable_wandb False --number_sample_per_shard 1000

def download_images():
    output_dir = os.path.abspath("/pfss/mlde/workspaces/mlde_wsp_Shared_Datasets/laion2B-en/data_dnspython/worker_1")

    download(
        processes_count=32,
        thread_count=256,
        url_list="/pfss/mlde/workspaces/mlde_wsp_Shared_Datasets/laion2B-en/metadata/worker_1",
        image_size=336,
        resize_only_if_bigger=True,
        resize_mode="border",
        skip_reencode=True,
        output_folder=output_dir,
        output_format="webdataset", # Use webdataset for more than 1M images -> compresses each shard as .tar
        input_format="parquet",
        url_col="url",
        caption_col="caption",
        enable_wandb=False, # maybe good for later, but don't have weights & biases right now
        number_sample_per_shard=10000,
        distributor="multiprocessing",
        save_additional_columns=["punsafe", "similarity", "pwatermark"],
        oom_shard_count=6,
        compute_hash="md5",
        verify_hash=["md5", "md5"],
    )

if __name__ == '__main__':
    dns.resolver.override_system_resolver("/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/LlavaGuard/src/experiments/download_dataset_laion/resolv.conf")

    download_images()

    dns.resolver.restore_system_resolver()
