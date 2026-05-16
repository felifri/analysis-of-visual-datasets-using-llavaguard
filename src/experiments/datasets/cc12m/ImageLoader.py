from experiments.datasets.BaseImageLoader import BaseImageLoader
from datasets import load_dataset, Image
import pyarrow.dataset as pds
import pyarrow.compute as pc

class ImageLoader(BaseImageLoader):
    def __init__(self):
        # corrupted_indices = []
        # with open("/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/LlavaGuard/src/experiments/datasets/cc12m/corrupted_image_indices.txt", "r") as f:
        #     corrupted_indices = [int(line.strip()) for line in f.readlines()]
            
        # print(f"Loading dataset and excluding {len(corrupted_indices)} invalid indices")
        
        # Using Dataset as well as IterableDataset results in storing it twice on disk.
        # If storage is low, the Dataset can be deleted once the IterableDataset is created.
        # rm -r /pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/HF_HOME/datasets/pixparse___cc12m-wds
        # The IterableDataset is stored in the 'hub' folder and not affected by the above command.
        dataset = load_dataset("pixparse/cc12m-wds", split="train")
        # dataset = dataset.select((i for i in range(len(dataset)) if i not in set(corrupted_indices)))

        self.split_names = ["train"]
        self.dataset_splits = {
            "train": dataset
        }
        self.img_col = "jpg"

    def load_image(self, key: str, split: list[str] | None = None) -> list[(str, Image | None)]:
        split = "train"
        dataset = self.dataset_splits[split]

        expr = pc.field("__key__") == key

        filtered = dataset.with_format("arrow").filter(
            lambda t: pds.dataset(t).to_table(columns={"mask": expr})[0].to_numpy(),
            batched=True,
        ).with_format(None)

        if (len(filtered) != 1):
            print(f"Found {len(filtered)} entries for key {key} in split {split}.")
            return [(split, None)]

        return [(split, filtered[0][self.img_col])]
    
    def load_images_batch(self, keys: list[str]) -> list[(str, Image | None)]:
        split = "train"
        dataset = self.dataset_splits[split]

        expr = pc.field("__key__").isin(keys)

        filtered = dataset.with_format("arrow").filter(
            lambda t: pds.dataset(t).to_table(columns={"mask": expr})[0].to_numpy(),
            batched=True,
        ).with_format(None)

        results = []
        for key in keys:
            entry = filtered.filter(lambda entry: entry["__key__"] == key)
            if (len(entry) != 1):
                print(f"Found {len(entry)} entries for key {key}.")
                results.append((key, None))
            else:
                results.append((key, entry[0][self.img_col]))

        return results
