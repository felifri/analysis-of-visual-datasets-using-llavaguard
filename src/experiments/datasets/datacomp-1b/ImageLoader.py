from experiments.datasets.BaseImageLoader import BaseImageLoader
from datasets import Image

class ImageLoader(BaseImageLoader):
    def __init__(self):        
        pass

    def load_image(self, idx: int | str, split: list[str] | None = None) -> list[Image | None]:
        raise NotImplementedError("We don't have access to the images as they are stored on the DFKI cluster. Ask Niharika.")