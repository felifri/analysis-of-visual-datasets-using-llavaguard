import os
import sys
from datasets import load_dataset, Image
import logging
import warnings

print(os.environ['HF_HOME'])

if "finngu" in os.environ['HF_HOME']:
    print("HF configured locally")
else:
    print("HF uses shared dir")

def download_dataset():
    dataset = load_dataset("pixparse/cc12m-wds")

    print("Successfully loaded all arrow files.")
    # print(dataset['4M_full'].filter(lambda row: row['image_id'] == 295543312)[:1])

    # Available splits: "train"
    print(dataset["train"][:10])


def validate_download():
    splits = ["train"]

    logging.basicConfig(
        filename='/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/LlavaGuard/src/experiments/datasets/cc12m/inspect_corrupted_indices.log', 
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s %(message)s'
    )
    logger = logging.getLogger(__name__)

    warnings.filterwarnings("error")

    with open("/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/LlavaGuard/src/experiments/datasets/cc12m/corrupted_image_indices.txt", "r") as f:
        corrupted_indices = [int(line.strip()) for line in f.readlines()]

        for split in splits:
            ds = load_dataset("pixparse/cc12m-wds", split=split)

            logger.info(f"Full split '{split}' of dataset includes {len(ds)} images.")
            logger.info(f"Excluding {len(corrupted_indices)} corrupted images from split '{split}'...")

            ds_valid_subset = ds.select((i for i in range(len(ds)) if i not in set(corrupted_indices)))
            logger.info(f"Successfully selected subset with {len(ds_valid_subset)} images.")

            # ds = ds.cast_column("jpg", Image(decode=False)).filter(lambda _, idx: idx not in set(corrupted_indices), with_indices=True)
            # ds = ds.filter(lambda _, idx: idx not in set(corrupted_indices), with_indices=True)

            logger.info("Testing if invalid images are excluded from the dataset...")

            for idx in corrupted_indices:
                # possibly_corrupted_subset = ds.skip(max(0, idx - 1)).take(1)  # .cast_column("jpg", Image(decode=True))

                possibly_corrupted_row = ds_valid_subset[idx]

                # for possibly_corrupted_row in possibly_corrupted_subset:
                try:
                    possibly_corrupted_row['jpg'].load()
                    possibly_corrupted_row['jpg'].close()
                except Exception as e:
                    logger.error(f"{idx}: {e}")

            logger.info(f"Validation of split '{split}' completed.")

            # for idx in range(len(ds)):
            #     try:
            #         ds[idx]['jpg'].load()
            #         ds[idx]['jpg'].close()
            #     except Exception as e:
            #         logger.error(f"{idx}: {e}")
                
            #     if (idx + 1) % 1000 == 0 and idx != 0:
            #         logger.info(f"\t{idx + 1} images done.")
            
            # logger.info(f"Successfully validated all {len(ds)} images of split '{split}'.")

    warnings.resetwarnings()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == "download":
        download_dataset()
    elif len(sys.argv) > 1 and sys.argv[1] == "validate":
        validate_download()
    else:
        print("No command line argument provided. Provide one of 'download' or 'validate'.")