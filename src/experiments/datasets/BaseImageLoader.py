from datasets import load_dataset, Image

class BaseImageLoader:
    split_names = []
    dataset_splits = {}

    def __init__(self, dataset_name: str, split_names: list[str], img_col: str = "image"):
        self.split_names = split_names
        self.dataset_splits = {
            split: load_dataset(dataset_name, split=split) for split in split_names
        }
        self.img_col = img_col

    def load_image(self, idx: int | str, split: list[str] | None = None) -> list[Image | None]:
        images = []
        splits = split if split else self.split_names

        for split in splits:
            dataset = self.dataset_splits[split]

            # Remove leading zeroes from idx if it is a string
            idx = int(idx)
            
            try:
                images.append(dataset[idx][self.img_col])
            except IndexError:
                images.append(None)

        return images
