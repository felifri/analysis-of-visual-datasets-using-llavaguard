from experiments.datasets.BaseImageLoader import BaseImageLoader

class ImageLoader(BaseImageLoader):
    def __init__(self):
        super().__init__(
            dataset_name="uoft-cs/cifar10", 
            split_names=["train", "test"], 
            img_col="img"
        )
