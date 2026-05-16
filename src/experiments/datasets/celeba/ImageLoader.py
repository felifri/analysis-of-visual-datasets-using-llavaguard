from experiments.datasets.BaseImageLoader import BaseImageLoader

class ImageLoader(BaseImageLoader):
    def __init__(self):
        super().__init__(
            dataset_name="flwrlabs/celeba", 
            split_names=["train", "valid", "test"]
        )
