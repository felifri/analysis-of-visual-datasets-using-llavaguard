import os
import sys
from datasets import load_dataset
import logging
import warnings

print(os.environ['HF_HOME'])

if "finngu" in os.environ['HF_HOME']:
    print("HF configured locally")
else:
    print("HF uses shared dir")

def download_dataset():
    dataset = load_dataset("flwrlabs/celeba")

    print("Successfully loaded all arrow files.")
    # print(dataset['4M_full'].filter(lambda row: row['image_id'] == 295543312)[:1])

    # Available splits: "train", "valid", "test"
    print(dataset["train"][:10])


def validate_download():
    dataset = load_dataset("flwrlabs/celeba")
    splits = ["train", "valid", "test"]

    logging.basicConfig(
        filename='/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/LlavaGuard/src/experiments/datasets/celeba/validate_download.log', 
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s %(message)s'
    )
    logger = logging.getLogger(__name__)

    warnings.filterwarnings("error")

    for split in splits:
        ds = dataset[split]
        ds_len = len(ds)

        logger.info(f"Validating all {ds_len} images of split '{split}'...")

        for idx in range(ds_len):
            try:
                ds[idx]['image'].load()
                ds[idx]['image'].close()
            except Exception as e:
                logger.error(f"{idx}: {e}")
            
            if (idx + 1) % 1000 == 0 and idx != 0:
                logger.info(f"\t{idx + 1} images done.")
        
        logger.info(f"Successfully validated all {ds_len} images of split '{split}'.")

    warnings.resetwarnings()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == "download":
        download_dataset()
    elif len(sys.argv) > 1 and sys.argv[1] == "validate":
        validate_download()
    else:
        print("No command line argument provided. Provide one of 'download' or 'validate'.")