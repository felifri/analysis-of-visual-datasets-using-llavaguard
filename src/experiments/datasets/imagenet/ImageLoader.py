from experiments.datasets.BaseImageLoader import BaseImageLoader

class ImageLoader(BaseImageLoader):
    def __init__(self):        
        super().__init__(
            dataset_name="ILSVRC/imagenet-1k", 
            split_names=["train", "validation", "test"]
        )
