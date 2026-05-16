import os
from datasets import load_dataset, IterableDataset
import math
import logging
import warnings

print(os.environ['HF_HOME'])

if "finngu" in os.environ['HF_HOME']:
    print("HF configured locally")
else:
    print("HF uses shared dir")
    print("Aborting.")
    exit()

def download_dataset():
    dataset = load_dataset("ILSVRC/imagenet-1k")

    print("Successfully loaded all arrow files.")

    print(dataset)

    # Available splits: "train", "validation", "test"
    # print(dataset["train"][:2])
    # print(len(dataset["train"]))

    splits = ["train", "validation", "test"]
    chunk_size = 10000
    batch_size = 1000

    for split in splits:
        len_split = len(dataset[split])
        num_chunks = math.ceil(len_split / chunk_size)
        chunk_ids_to_process = range(0, num_chunks // 2)  # range(num_chunks // 2, num_chunks)
        iterable_ds = load_dataset("ILSVRC/imagenet-1k", split=split).to_iterable_dataset()

        if not isinstance(iterable_ds, IterableDataset):
            raise ValueError("Dataset must be an IterableDataset for better performance.")
        
        iterable_ds = iterable_ds.map(
            function=lambda x, idx: {'image': x['image'], 'image_name': f"{str(idx).zfill(int(math.log10(len_split)) + 1)}.jpg"}, 
            with_indices=True,
            remove_columns=['label']
        )

        for chunk_idx in chunk_ids_to_process:
            print(f"Processing chunk {chunk_idx + 1}/{num_chunks}")
            len_chunk = min(chunk_size, len_split - chunk_idx * chunk_size)
            print(f"Chunk has {len_chunk} entries")

            try:
                chunk = iterable_ds.skip(chunk_idx * chunk_size).take(chunk_size)
            except IndexError:
                print(f"\tChunk {chunk_idx} is out of range. Skipping.")
                continue

            for batch_idx, batch in enumerate(chunk.iter(batch_size)):
                print(f"\tProcessing batch {batch_idx}")
                print([{'image': image, 'image_name': image_name, 'policy': "test"} for image, image_name in zip(batch['image'], batch['image_name'])])

def validate_download():
    dataset = load_dataset("ILSVRC/imagenet-1k")
    splits = ["train", "validation", "test"]

    logging.basicConfig(
        filename='/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/LlavaGuard/src/experiments/datasets/imagenet/validate_download.log', 
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s %(message)s'
    )
    logger = logging.getLogger(__name__)

    warnings.filterwarnings("error")

    for split in splits:
        logger.info(f"Validating all images of split '{split}'...")

        # iterable_ds = dataset[split].to_iterable_dataset()

        # for idx, example in enumerate(iterable_ds):
        #     try:
        #         img = example['image']
        #     except Exception as e:
        #         logger.error(f"{idx}: {e}")
            
        #     if idx % 10000 == 0:
        #         logger.info(f"\t{idx} images done.")

        ds = dataset[split]
        ds_len = len(ds)

        for idx in range(ds_len):
            try:
                ds[idx]['image'].load()
                ds[idx]['image'].close()
            except Exception as e:
                logger.error(f"{idx}: {e}")
            
            if idx % 10000 == 0:
                logger.info(f"\t{idx} images done.")

    warnings.resetwarnings()


if __name__ == '__main__':
    # download_dataset()
    validate_download()