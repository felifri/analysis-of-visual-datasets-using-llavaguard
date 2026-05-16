from experiments.datasets.BaseImageLoader import BaseImageLoader
from PIL import Image
import os

class ImageLoader(BaseImageLoader):
    def __init__(self):
        self.base_img_dir = "/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/HF_HOME/datasets/ms-coco/images"
        self.split_names = ["train2017", "test2017", "val2017", "unlabeled2017"]

    def load_image(self, idx: int | str, split: list[str] | None = None) -> list[(str, Image.Image | None)]:
        splits = split if split else self.split_names
        
        images = []
        for split in splits:
            img_path = os.path.join(self.base_img_dir, split, f"{idx}.jpg")
            
            if os.path.exists(img_path):
                images.append((split, Image.open(img_path)))
            else:
                images.append((split, None))

        return images
