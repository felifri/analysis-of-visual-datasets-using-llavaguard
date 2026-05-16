import os
import sys
from datasets import load_dataset
import logging
import warnings
import webdataset as wds
from huggingface_hub import get_token
import fiftyone as fo
import fiftyone.zoo as foz

print(os.environ['HF_HOME'])

if "finngu" in os.environ['HF_HOME']:
    print("HF configured locally")
else:
    print("HF uses shared dir")

dataset_name = "dalle-mini/open-images"
splits = ["train", "validation", "test"]

def download_dataset():
    logging.basicConfig(
        filename='/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/LlavaGuard/src/experiments/datasets/open-images-v6/download_dataset.log', 
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s %(message)s'
    )
    logger = logging.getLogger(__name__)

    logger.info(f"Downloading '{dataset_name}'...")

    # dataset = foz.load_zoo_dataset(
    #     "open-images-v6",
    #     # split="validation",  # if not provided, all splits are downloaded
    #     label_types=[],
    #     seed=42069,
    #     shuffle=False,
    #     max_samples=100000,
    # )

    # logger.info(f"Download complete.")

    # dataset.export(export_dir="/pfss/mlde/workspaces/mlde_wsp_Shared_Datasets/open-images-v6")

    hf_token = get_token()
    url = f"https://huggingface.co/datasets/dalle-mini/open-images/resolve/main/data/valid/{{00000..00008}}.tar"
    # url = f"https://huggingface.co/datasets/dalle-mini/open-images/resolve/main/data/test/{{00000..00024}}.tar"
    # url = f"https://huggingface.co/datasets/dalle-mini/open-images/resolve/main/data/train/{{00000..01802}}.tar"
    url = f"pipe:curl -s -L {url} -H 'Authorization:Bearer {hf_token}'"
    
    def is_valid_sample(sample):
        return sample is not None and all(value is not None for value in sample.values())

    try:
        dataset = (
            wds.WebDataset(url, shardshuffle=False)
            .decode()
            .map(lambda x: x if is_valid_sample(x) else None)
            .to_tuple("__key__", "jpg")
            .batched(1000)
        )

        logger.info(f"Dataset loaded successfully.")

        for batch in dataset:
            keys, images = batch
            logger.info(f"Processed batch: {keys}")
            break

        logger.info("Successfully downloaded and processed all tar files.")
    except Exception as e:
        logger.error(f"Failed to download or process dataset: {e}")
        raise

    # # dataset = load_dataset(dataset_name)

    # # logger.info("Successfully loaded all arrow files.")

    # # logger.info(dataset['train'][:1])

    # # warnings.filterwarnings("error")

    # # for split in splits:
    # #     ds = dataset[split]

    # #     logger.info(f"Full split '{split}' of dataset includes {len(ds)} images.")

    # #     for idx in range(len(ds)):
    # #         try:
    # #             ds[idx]['jpg'].load()
    # #             ds[idx]['jpg'].close()
    # #         except Exception as e:
    # #             logger.error(f"{idx}: {e}")
            
    # #         if (idx + 1) % 1000 == 0 and idx != 0:
    # #             logger.info(f"\t{idx + 1} images done.")
        
    # #     logger.info(f"Successfully validated all {len(ds)} images of split '{split}'.")

    # warnings.resetwarnings()


def validate_download():
    logging.basicConfig(
        filename='/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/LlavaGuard/src/experiments/datasets/open-images-v6/inspect_corrupted_indices.log', 
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s %(message)s'
    )
    logger = logging.getLogger(__name__)

    warnings.filterwarnings("error")

    with open("/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/LlavaGuard/src/experiments/datasets/open-images-v6/corrupted_image_indices.txt", "r") as f:
        corrupted_indices = [int(line.strip()) for line in f.readlines()]

        for split in splits:
            ds = load_dataset("pixparse/open-images-v6-wds", split=split)

            logger.info(f"Full split '{split}' of dataset includes {len(ds)} images.")
            logger.info(f"Excluding {len(corrupted_indices)} corrupted images from split '{split}'...")

            ds_valid_subset = ds.select((i for i in range(len(ds)) if i not in set(corrupted_indices)))
            logger.info(f"Successfully selected subset with {len(ds_valid_subset)} images.")

            logger.info("Testing if invalid images are excluded from the dataset...")

            for idx in corrupted_indices:
                possibly_corrupted_row = ds_valid_subset[idx]

                try:
                    possibly_corrupted_row['jpg'].load()
                    possibly_corrupted_row['jpg'].close()
                except Exception as e:
                    logger.error(f"{idx}: {e}")

            logger.info(f"Validation of split '{split}' completed.")

    warnings.resetwarnings()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == "download":
        download_dataset()
    elif len(sys.argv) > 1 and sys.argv[1] == "validate":
        validate_download()
    else:
        print("No command line argument provided. Provide one of 'download' or 'validate'.")